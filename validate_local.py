from __future__ import annotations

from fastapi.testclient import TestClient

from client import SepsisTreatmentEnv
from models import SepsisAction
from server.app import app


def main() -> None:
    env = SepsisTreatmentEnv(task_id="easy")
    reset_result = env.reset()
    assert reset_result.observation.task_id == "easy"
    step_result = env.step(
        SepsisAction(
            action_type="request_lab",
            suspect_sepsis=True,
            lab_type="lactate",
            rationale="smoke",
        )
    )
    assert step_result.reward is not None
    state = env.state()
    assert state.step_count == 1
    env.close()

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/metadata").status_code == 200
    reset_response = client.post("/reset", json={"task_id": "medium"})
    assert reset_response.status_code == 200
    step_response = client.post(
        "/step",
        json={
            "action_type": "request_treatment",
            "suspect_sepsis": True,
            "treatment_type": "fluids",
            "rationale": "smoke",
        },
    )
    assert step_response.status_code == 200
    state_response = client.get("/state")
    assert state_response.status_code == 200

    print("Local validation passed.")


if __name__ == "__main__":
    main()
