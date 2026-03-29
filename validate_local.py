from __future__ import annotations

from fastapi.testclient import TestClient

from client import SepsisTreatmentEnv
from models import SepsisAction
from server.app import app


def main() -> None:
    env = SepsisTreatmentEnv(task_id="easy")
    reset_result = env.reset()
    assert reset_result.observation.task_id == "easy"
    step_result = env.step(SepsisAction(fluid_bin=1, pressor_bin=0, rationale="smoke"))
    assert step_result.reward is not None
    state = env.state()
    assert state.step_count == 1
    env.close()

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/metadata").status_code == 200
    reset_response = client.post("/reset", json={"task_id": "medium"})
    assert reset_response.status_code == 200
    step_response = client.post("/step", json={"fluid_bin": 2, "pressor_bin": 1, "rationale": "smoke"})
    assert step_response.status_code == 200
    state_response = client.get("/state")
    assert state_response.status_code == 200

    print("Local validation passed.")


if __name__ == "__main__":
    main()
