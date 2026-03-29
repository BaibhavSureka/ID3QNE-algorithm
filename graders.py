from __future__ import annotations

from typing import Any

from tasks import TaskConfig


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def grade_episode(task: TaskConfig, metrics: dict[str, Any]) -> float:
    reward_score = _clamp((metrics.get("avg_reward", 0.0) + 1.0) / 2.0)
    agreement_score = _clamp(metrics.get("agreement_rate", 0.0))
    safety_score = _clamp(1.0 - metrics.get("safety_violation_rate", 0.0))
    terminal_score = _clamp(metrics.get("terminal_success", 0.0))

    weights = task.score_weights
    score = (
        weights.get("reward", 0.0) * reward_score
        + weights.get("agreement", 0.0) * agreement_score
        + weights.get("safety", 0.0) * safety_score
        + weights.get("terminal", 0.0) * terminal_score
    )
    return round(_clamp(score), 4)


def summarize_episode(total_reward: float, state_history: list[dict[str, Any]], terminal_outcome: str) -> dict[str, Any]:
    step_count = max(len(state_history), 1)
    agreement_rate = sum(item.get("agreement_score", 0.0) for item in state_history) / step_count
    safety_violations = sum(1 for item in state_history if item.get("unsafe", False))
    avg_reward = total_reward / step_count
    terminal_success = 1.0 if terminal_outcome == "survived" else 0.0
    return {
        "steps": step_count,
        "avg_reward": avg_reward,
        "agreement_rate": agreement_rate,
        "safety_violation_rate": safety_violations / step_count,
        "terminal_success": terminal_success,
    }
