from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel


OPENENV_AVAILABLE = False

try:
    from openenv.core.client_types import StepResult as OpenEnvStepResult
    from openenv.core.env_client import EnvClient as OpenEnvEnvClient
    from openenv.core.env_server import create_app as openenv_create_app
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState

    Action = OpenEnvAction
    Observation = OpenEnvObservation
    State = OpenEnvState
    Environment = OpenEnvEnvironment
    EnvClient = OpenEnvEnvClient
    create_app = openenv_create_app
    OPENENV_AVAILABLE = True
except Exception:
    create_app = None

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        pass

    class State(BaseModel):
        episode_id: str
        step_count: int = 0

    class Environment:
        def reset(self):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        @property
        def state(self):
            raise NotImplementedError

    TObservation = TypeVar("TObservation")

    class EnvClient(Generic[TObservation]):
        def __init__(self, base_url: str | None = None, **_: Any):
            self.base_url = base_url

        def close(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()


TObservation = TypeVar("TObservation")


@dataclass
class StepResult(Generic[TObservation]):
    observation: TObservation
    reward: Optional[float] = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Action",
    "Observation",
    "State",
    "Environment",
    "EnvClient",
    "StepResult",
    "OPENENV_AVAILABLE",
    "create_app",
]
