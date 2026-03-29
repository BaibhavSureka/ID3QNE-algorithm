from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graders import grade_episode, summarize_episode
from models import SepsisAction, SepsisObservation, SepsisState
from openenv_compat import Environment
from tasks import TaskConfig, build_task_catalog


ROOT = Path(__file__).resolve().parent.parent
ENV_DATA_DIR = ROOT / "env_data"
DATASET_PATH = ENV_DATA_DIR / "processed_demo_dataset.pkl"
FEATURES_PATH = ENV_DATA_DIR / "selected_features.json"


def load_processed_assets() -> tuple[pd.DataFrame, list[str]]:
    if DATASET_PATH.exists() and FEATURES_PATH.exists():
        dataset = pd.read_pickle(DATASET_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as handle:
            selected_features = json.load(handle)
        return dataset, selected_features
    raise FileNotFoundError(
        "Missing env_data assets. Expected env_data/processed_demo_dataset.pkl "
        "and env_data/selected_features.json."
    )


def build_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        dataset.groupby("icustay_id")
        .agg(
            length=("bin_idx", "count"),
            mean_severity=("severity_proxy", "mean"),
            max_severity=("severity_proxy", "max"),
            mortality=("mortality", "max"),
        )
        .reset_index()
        .sort_values("icustay_id")
        .reset_index(drop=True)
    )
    return grouped


class SepsisTreatmentEnvironment(Environment):
    def __init__(self, task_id: str = "easy"):
        self.dataset, self.selected_features = load_processed_assets()
        self.summary = build_summary(self.dataset)
        self.task_catalog = build_task_catalog(self.summary)
        self._task_cycle = {task_key: 0 for task_key in self.task_catalog}
        self.default_task_id = task_id if task_id in self.task_catalog else "easy"
        self._episode = pd.DataFrame()
        self._cursor = 0
        self._task: TaskConfig = self.task_catalog[self.default_task_id]
        self._metrics: dict[str, Any] = {}
        self._state = SepsisState(
            episode_id=str(uuid.uuid4()),
            task_id=self.default_task_id,
            current_stay_id=-1,
            step_count=0,
            max_steps=0,
            cumulative_reward=0.0,
            agreement_sum=0.0,
            safety_violations=0,
            terminal_outcome="ongoing",
            history=[],
        )

    def available_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": task.task_id,
                "title": task.title,
                "description": task.description,
                "min_steps": task.min_steps,
                "max_steps": task.max_steps,
                "num_episodes": len(task.preferred_stay_ids),
            }
            for task in self.task_catalog.values()
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "sepsis-openenv",
            "description": "Offline sepsis treatment environment built from the MIMIC-III demo cohort.",
            "tasks": self.available_tasks(),
            "selected_features": self.selected_features,
            "action_space": {
                "fluid_bin": "0-4",
                "pressor_bin": "0-4",
                "total_actions": 25,
            },
        }

    def _select_episode(self, task: TaskConfig) -> pd.DataFrame:
        stay_ids = task.preferred_stay_ids
        if not stay_ids:
            stay_ids = tuple(int(stay_id) for stay_id in self.summary["icustay_id"].tolist())
        index = self._task_cycle[task.task_id] % len(stay_ids)
        self._task_cycle[task.task_id] += 1
        stay_id = stay_ids[index]
        episode = self.dataset[self.dataset["icustay_id"] == stay_id].sort_values("bin_idx").reset_index(drop=True)
        episode = episode.head(task.max_steps).copy()
        if len(episode) < task.min_steps:
            episode = self.dataset[self.dataset["icustay_id"] == stay_id].sort_values("bin_idx").reset_index(drop=True)
        return episode

    def _make_observation(self, row: pd.Series, *, reward: float, done: bool) -> SepsisObservation:
        return SepsisObservation(
            episode_id=self._state.episode_id,
            task_id=self._task.task_id,
            task_description=self._task.description,
            patient_id=int(row["icustay_id"]),
            step_index=int(self._cursor),
            max_steps=int(len(self._episode)),
            severity_proxy=float(row["severity_proxy"]),
            mortality_risk_flag=int(row["mortality"]),
            features={name: float(row[name]) for name in self.selected_features},
            cumulative_reward=float(self._state.cumulative_reward),
            last_reward=float(reward),
            done=done,
            reward=float(reward),
        )

    def _compute_reward(
        self,
        row: pd.Series,
        next_row: pd.Series | None,
        action: SepsisAction,
        is_terminal: bool,
    ) -> tuple[float, dict[str, Any]]:
        clinician_fluid = int(row["fluid_bin"])
        clinician_pressor = int(row["pressor_bin"])
        fluid_gap = abs(action.fluid_bin - clinician_fluid)
        pressor_gap = abs(action.pressor_bin - clinician_pressor)
        agreement = max(0.0, 1.0 - ((fluid_gap + pressor_gap) / 8.0))

        severity_now = float(row["severity_proxy"])
        severity_next = float(next_row["severity_proxy"]) if next_row is not None else severity_now
        progress = float(0.15 * np.tanh(severity_now - severity_next))

        safety_penalty = 0.0
        unsafe = False
        if severity_now <= 1.0 and action.pressor_bin >= clinician_pressor + 2:
            safety_penalty += 0.35
            unsafe = True
        if severity_now >= 2.0 and action.pressor_bin == 0 and clinician_pressor >= 2:
            safety_penalty += 0.25
            unsafe = True
        if action.fluid_bin >= clinician_fluid + 3:
            safety_penalty += 0.10
            unsafe = True
        if severity_now >= 2.0 and action.fluid_bin == 0 and clinician_fluid >= 2:
            safety_penalty += 0.10
            unsafe = True

        reward = (0.6 * agreement) + progress - safety_penalty
        if is_terminal:
            reward += 0.40 if int(row["mortality"]) == 0 else -0.40

        reward = float(np.clip(reward, -1.0, 1.0))
        return reward, {
            "agreement_score": round(agreement, 4),
            "progress_score": round(progress, 4),
            "safety_penalty": round(safety_penalty, 4),
            "unsafe": unsafe,
        }

    def reset(self, task_id: str | None = None) -> SepsisObservation:
        chosen_task = task_id or self.default_task_id
        if chosen_task not in self.task_catalog:
            chosen_task = self.default_task_id
        self._task = self.task_catalog[chosen_task]
        self._episode = self._select_episode(self._task)
        self._cursor = 0
        current_row = self._episode.iloc[self._cursor]
        self._state = SepsisState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task.task_id,
            current_stay_id=int(current_row["icustay_id"]),
            step_count=0,
            max_steps=int(len(self._episode)),
            cumulative_reward=0.0,
            agreement_sum=0.0,
            safety_violations=0,
            terminal_outcome="ongoing",
            history=[],
        )
        self._metrics = {}
        return self._make_observation(current_row, reward=0.0, done=False)

    def step(self, action: SepsisAction) -> SepsisObservation:
        if self._episode.empty:
            return self.reset(self.default_task_id)

        row = self._episode.iloc[self._cursor]
        next_index = self._cursor + 1
        next_row = self._episode.iloc[next_index] if next_index < len(self._episode) else None
        done = next_row is None

        reward, details = self._compute_reward(row, next_row, action, done)
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.agreement_sum += details["agreement_score"]
        if details["unsafe"]:
            self._state.safety_violations += 1
        if done:
            self._state.terminal_outcome = "survived" if int(row["mortality"]) == 0 else "died"

        history_row = {
            "step": int(self._cursor),
            "action_index": action.action_index,
            "fluid_bin": action.fluid_bin,
            "pressor_bin": action.pressor_bin,
            "agreement_score": details["agreement_score"],
            "progress_score": details["progress_score"],
            "safety_penalty": details["safety_penalty"],
            "unsafe": details["unsafe"],
            "reward": round(reward, 4),
        }
        self._state.history.append(history_row)

        if not done:
            self._cursor = next_index
            observation_row = self._episode.iloc[self._cursor]
        else:
            observation_row = row

        episode_metrics = summarize_episode(
            total_reward=float(self._state.cumulative_reward),
            state_history=self._state.history,
            terminal_outcome=self._state.terminal_outcome,
        )
        episode_metrics["score"] = grade_episode(self._task, episode_metrics)
        self._metrics = episode_metrics
        return self._make_observation(observation_row, reward=reward, done=done)

    def current_metrics(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "task_id": self._task.task_id,
            "current_stay_id": self._state.current_stay_id,
            "cumulative_reward": round(float(self._state.cumulative_reward), 4),
            "safety_violations": int(self._state.safety_violations),
        }

    @property
    def state(self) -> SepsisState:
        return self._state
