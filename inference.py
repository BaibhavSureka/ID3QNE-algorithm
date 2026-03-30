from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from client import SepsisTreatmentEnv
from models import SepsisAction, SepsisObservation


OUTPUT_DIR = Path("outputs")
MAX_STEPS_PER_TASK = {"easy": 8, "medium": 12, "hard": 16}


def heuristic_action(observation: SepsisObservation) -> SepsisAction:
    severity = observation.severity_proxy
    shock = observation.vitals.get("Shock_Index", 0.0)
    mean_bp = observation.vitals.get("MeanBP", 0.0)
    visible_labs = observation.visible_labs
    requested_labs = set(observation.requested_labs)

    if not observation.requested_labs:
        return SepsisAction(
            action_type="request_lab",
            suspect_sepsis=severity >= 1.0 or shock > 0.1 or mean_bp < 0.0,
            lab_type="lactate",
            rationale="Start the workup with lactate under partial observability.",
        )
    if "wbc" not in requested_labs and severity >= 0.75:
        return SepsisAction(
            action_type="request_lab",
            suspect_sepsis=True,
            lab_type="wbc",
            rationale="Follow the initial sepsis workup with WBC.",
        )
    if severity >= 1.5 and "creatinine" not in requested_labs:
        return SepsisAction(
            action_type="request_lab",
            suspect_sepsis=True,
            lab_type="creatinine",
            rationale="Check renal dysfunction before escalating treatment.",
        )

    lactate = visible_labs.get("lactate", 0.0) or 0.0
    if severity < 0.8 and mean_bp >= -0.1:
        treatment_type = "monitor"
    elif severity >= 2.0 or mean_bp < -0.2:
        treatment_type = "combination" if lactate > 0.25 else "vasopressors"
    elif shock > 0.15 or severity >= 1.1:
        treatment_type = "fluids"
    else:
        treatment_type = "monitor"

    return SepsisAction(
        action_type="request_treatment",
        suspect_sepsis=severity >= 1.0 or lactate > 0.25,
        treatment_type=treatment_type,
        rationale="Deterministic staged baseline for lab workup and treatment selection.",
    )


def build_prompt(observation: SepsisObservation) -> str:
    return (
        "You are controlling a sequential sepsis management simulator.\n"
        f"Task: {observation.task_description}\n"
        f"Step: {observation.step_index + 1}/{observation.max_steps}\n"
        f"Severity proxy: {observation.severity_proxy:.2f}\n"
        f"Mortality flag in logged trajectory: {observation.mortality_risk_flag}\n"
        f"Demographics: {json.dumps(observation.demographics)}\n"
        f"Vitals: {json.dumps(observation.vitals)}\n"
        f"Context features: {json.dumps(observation.context_features)}\n"
        f"Visible labs: {json.dumps(observation.visible_labs)}\n"
        f"Requested labs so far: {json.dumps(observation.requested_labs)}\n"
        "Return JSON with keys action_type, suspect_sepsis, lab_type, treatment_type, rationale. "
        "action_type must be one of request_lab, request_treatment, monitor."
    )


def model_action(client: OpenAI | None, model_name: str | None, observation: SepsisObservation) -> SepsisAction:
    if client is None or not model_name:
        return heuristic_action(observation)

    messages = [
        {"role": "system", "content": "Return only valid JSON for a sepsis management action."},
        {"role": "user", "content": build_prompt(observation)},
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        content = completion.choices[0].message.content or ""
        payload = json.loads(content)
        return SepsisAction(**payload)
    except Exception:
        return heuristic_action(observation)


def run_task(task_id: str, client: OpenAI | None, model_name: str | None) -> dict:
    env = SepsisTreatmentEnv(base_url=os.getenv("ENV_BASE_URL"), task_id=task_id)
    result = env.reset()
    observation = result.observation
    final_info = result.info

    for _ in range(MAX_STEPS_PER_TASK[task_id]):
        action = model_action(client, model_name, observation)
        result = env.step(action)
        observation = result.observation
        final_info = result.info
        if result.done:
            break

    state = env.state()
    env.close()
    metrics = final_info.get("metrics", {})
    return {
        "task_id": task_id,
        "episode_id": state.episode_id,
        "score": metrics.get("score", 0.0),
        "avg_reward": metrics.get("avg_reward", 0.0),
        "detection": metrics.get("detection", 0.0),
        "lab_workup": metrics.get("lab_workup", 0.0),
        "treatment": metrics.get("treatment", 0.0),
        "timeliness": metrics.get("timeliness", 0.0),
        "stability": metrics.get("stability", 0.0),
        "safety": metrics.get("safety", 0.0),
        "safety_violation_rate": metrics.get("safety_violation_rate", 0.0),
        "outcome": metrics.get("outcome", 0.0),
        "steps": metrics.get("steps", state.step_count),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    llm_client = None
    if api_base_url and model_name and hf_token:
        llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)

    results = [run_task(task_id, llm_client, model_name) for task_id in ["easy", "medium", "hard"]]
    summary = {
        "results": results,
        "mean_score": round(sum(item["score"] for item in results) / len(results), 4),
        "model_name": model_name or "heuristic-baseline",
    }
    output_path = OUTPUT_DIR / "baseline_scores.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
