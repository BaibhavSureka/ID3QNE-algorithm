from __future__ import annotations

from typing import Any

from pydantic import Field

from openenv_compat import Action, Observation, State


class SepsisAction(Action):
    fluid_bin: int = Field(..., ge=0, le=4, description="Discrete IV fluid dose bin.")
    pressor_bin: int = Field(..., ge=0, le=4, description="Discrete vasopressor dose bin.")
    rationale: str = Field(default="", description="Optional reasoning for the treatment decision.")

    @property
    def action_index(self) -> int:
        return int(self.fluid_bin * 5 + self.pressor_bin)


class SepsisObservation(Observation):
    episode_id: str = Field(..., description="Unique episode identifier.")
    task_id: str = Field(..., description="Current task id.")
    task_description: str = Field(..., description="Human-readable task goal.")
    patient_id: int = Field(..., description="ICU stay id for the current patient trajectory.")
    step_index: int = Field(..., description="Current step index within the episode.")
    max_steps: int = Field(..., description="Maximum number of steps in this task episode.")
    severity_proxy: float = Field(..., description="Proxy severity score used for reward shaping.")
    mortality_risk_flag: int = Field(..., description="Binary mortality flag from the logged stay.")
    features: dict[str, float] = Field(default_factory=dict, description="Normalized patient features.")
    allowed_actions: list[int] = Field(default_factory=lambda: list(range(25)), description="Valid action ids.")
    cumulative_reward: float = Field(default=0.0, description="Running reward total.")
    last_reward: float = Field(default=0.0, description="Reward from the previous step.")
    done: bool = Field(default=False, description="Whether the episode has ended.")
    reward: float | None = Field(default=None, description="Current reward for OpenEnv compatibility.")


class SepsisState(State):
    task_id: str = Field(..., description="Current task id.")
    current_stay_id: int = Field(..., description="Current ICU stay id.")
    step_count: int = Field(default=0, description="Number of actions taken in this episode.")
    max_steps: int = Field(default=0, description="Maximum steps in this episode.")
    cumulative_reward: float = Field(default=0.0, description="Running reward total.")
    agreement_sum: float = Field(default=0.0, description="Accumulated clinician agreement score.")
    safety_violations: int = Field(default=0, description="Count of unsafe actions.")
    terminal_outcome: str = Field(default="ongoing", description="survived, died, or ongoing")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Per-step action trace.")
