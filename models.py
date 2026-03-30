from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from openenv_compat import Action, Observation, State


class SepsisAction(Action):
    action_type: Literal["request_lab", "request_treatment", "monitor"] = Field(
        ..., description="High-level agent action for the current clinical step."
    )
    suspect_sepsis: bool = Field(
        default=False,
        description="Whether the agent believes the current presentation is consistent with sepsis.",
    )
    lab_type: str | None = Field(
        default=None,
        description="Lab to request when action_type=request_lab. Supported values: lactate, wbc, creatinine, bicarbonate, platelets, bilirubin.",
    )
    treatment_type: str | None = Field(
        default=None,
        description="Treatment plan when action_type=request_treatment. Supported values: monitor, fluids, vasopressors, combination.",
    )
    rationale: str = Field(default="", description="Optional reasoning for the clinical decision.")

    @property
    def action_index(self) -> int:
        action_map = {"request_lab": 0, "request_treatment": 1, "monitor": 2}
        lab_map = {
            None: 0,
            "lactate": 1,
            "wbc": 2,
            "creatinine": 3,
            "bicarbonate": 4,
            "platelets": 5,
            "bilirubin": 6,
        }
        treatment_map = {
            None: 0,
            "monitor": 1,
            "fluids": 2,
            "vasopressors": 3,
            "combination": 4,
        }
        return int((action_map[self.action_type] * 100) + (lab_map.get(self.lab_type, 0) * 10) + treatment_map.get(self.treatment_type, 0))

    @model_validator(mode="after")
    def validate_payload(self) -> "SepsisAction":
        if self.action_type == "request_lab" and not self.lab_type:
            raise ValueError("lab_type is required when action_type='request_lab'.")
        if self.action_type == "request_treatment" and not self.treatment_type:
            raise ValueError("treatment_type is required when action_type='request_treatment'.")
        if self.action_type != "request_lab":
            object.__setattr__(self, "lab_type", None)
        if self.action_type != "request_treatment":
            object.__setattr__(self, "treatment_type", None)
        return self


class SepsisObservation(Observation):
    episode_id: str = Field(..., description="Unique episode identifier.")
    task_id: str = Field(..., description="Current task id.")
    task_description: str = Field(..., description="Human-readable task goal.")
    patient_id: int = Field(..., description="ICU stay id for the current patient trajectory.")
    step_index: int = Field(..., description="Current step index within the episode.")
    max_steps: int = Field(..., description="Maximum number of steps in this task episode.")
    severity_proxy: float = Field(..., description="Proxy severity score used for reward shaping.")
    mortality_risk_flag: int = Field(..., description="Binary mortality flag from the logged stay.")
    demographics: dict[str, float] = Field(default_factory=dict, description="Static or slowly changing patient context.")
    vitals: dict[str, float] = Field(default_factory=dict, description="Always-visible bedside measurements.")
    context_features: dict[str, float] = Field(default_factory=dict, description="Visible non-lab features derived from the logged state.")
    visible_labs: dict[str, float | None] = Field(default_factory=dict, description="Labs revealed so far by explicit requests.")
    requested_labs: list[str] = Field(default_factory=list, description="Ordered list of labs the agent has already requested.")
    available_lab_options: list[str] = Field(default_factory=list, description="Supported lab request types.")
    available_treatment_options: list[str] = Field(default_factory=list, description="Supported treatment request types.")
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
    safety_violations: int = Field(default=0, description="Count of unsafe actions.")
    terminal_outcome: str = Field(default="ongoing", description="survived, died, or ongoing")
    requested_labs: list[str] = Field(default_factory=list, description="Lab requests made in the episode.")
    visible_labs: dict[str, float | None] = Field(default_factory=dict, description="Latest revealed lab values.")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Per-step action trace.")
