from __future__ import annotations

from typing import Any

import requests

from models import SepsisAction, SepsisObservation, SepsisState
from openenv_compat import EnvClient, OPENENV_AVAILABLE, StepResult
from server.sepsis_environment import SepsisTreatmentEnvironment


class SepsisTreatmentEnv(EnvClient):
    def __init__(self, base_url: str | None = None, task_id: str = "easy"):
        if base_url is not None:
            super().__init__(base_url=base_url)
        else:
            self.base_url = None
            if OPENENV_AVAILABLE:
                self._provider = None
                self._ws = None
        self.task_id = task_id
        self._local_env = None if base_url else SepsisTreatmentEnvironment(task_id=task_id)

    def _step_payload(self, action: SepsisAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SepsisObservation]:
        return StepResult(
            observation=SepsisObservation(**payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SepsisState:
        return SepsisState(**payload)

    def reset(self) -> StepResult[SepsisObservation]:
        if self._local_env is not None:
            observation = self._local_env.reset(task_id=self.task_id)
            return StepResult(
                observation=observation,
                reward=0.0,
                done=False,
                info={"tasks": self._local_env.available_tasks()},
            )

        response = requests.post(f"{self.base_url.rstrip('/')}/reset", json={"task_id": self.task_id}, timeout=30)
        response.raise_for_status()
        return self._parse_result(response.json())

    def step(self, action: SepsisAction) -> StepResult[SepsisObservation]:
        if self._local_env is not None:
            observation = self._local_env.step(action)
            return StepResult(
                observation=observation,
                reward=observation.reward,
                done=observation.done,
                info={"metrics": self._local_env.current_metrics()},
            )

        action_payload = self._step_payload(action)
        response = requests.post(f"{self.base_url.rstrip('/')}/step", json=action_payload, timeout=30)
        if response.status_code == 422:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/step",
                json={"action": action_payload},
                timeout=30,
            )
        response.raise_for_status()
        return self._parse_result(response.json())

    def state(self) -> SepsisState:
        if self._local_env is not None:
            return self._local_env.state
        response = requests.get(f"{self.base_url.rstrip('/')}/state", timeout=30)
        response.raise_for_status()
        return self._parse_state(response.json())

    def metadata(self) -> dict[str, Any]:
        if self._local_env is not None:
            return self._local_env.metadata()
        response = requests.get(f"{self.base_url.rstrip('/')}/metadata", timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._local_env = None
