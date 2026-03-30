from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    title: str
    description: str
    min_steps: int
    max_steps: int
    preferred_stay_ids: tuple[int, ...]
    score_weights: dict[str, float]


def _ordered_stays(summary: pd.DataFrame, selector) -> tuple[int, ...]:
    subset = summary[selector(summary)].sort_values(["mean_severity", "length", "mortality", "icustay_id"])
    return tuple(int(stay_id) for stay_id in subset["icustay_id"].tolist())


def build_task_catalog(summary: pd.DataFrame) -> dict[str, TaskConfig]:
    easy_ids = _ordered_stays(
        summary,
        lambda df: (df["mean_severity"] <= 1.0) & (df["length"] >= 8),
    )
    medium_ids = _ordered_stays(
        summary,
        lambda df: (df["mean_severity"] >= 0.8) & (df["max_severity"] >= 2.0) & (df["length"] >= 8),
    )
    hard_ids = _ordered_stays(
        summary,
        lambda df: (df["max_severity"] >= 3.0) & (df["length"] >= 8),
    )

    if not hard_ids:
        hard_ids = medium_ids[-4:]
    if not medium_ids:
        medium_ids = easy_ids[-6:]
    if not easy_ids:
        easy_ids = tuple(int(stay_id) for stay_id in summary.sort_values("icustay_id")["icustay_id"].head(6).tolist())

    return {
        "easy": TaskConfig(
            task_id="easy",
            title="Early Sepsis Workup",
            description=(
                "Identify likely sepsis early and request the most informative initial labs from partial bedside data."
            ),
            min_steps=6,
            max_steps=8,
            preferred_stay_ids=easy_ids,
            score_weights={"detection": 0.35, "lab_workup": 0.35, "timeliness": 0.20, "safety": 0.10},
        ),
        "medium": TaskConfig(
            task_id="medium",
            title="Diagnosis And Early Treatment",
            description=(
                "Use iterative lab requests to confirm deterioration and start an appropriate treatment plan early."
            ),
            min_steps=8,
            max_steps=12,
            preferred_stay_ids=medium_ids,
            score_weights={
                "detection": 0.20,
                "lab_workup": 0.20,
                "treatment": 0.30,
                "timeliness": 0.20,
                "safety": 0.10,
            },
        ),
        "hard": TaskConfig(
            task_id="hard",
            title="Full Sepsis Management",
            description=(
                "Balance workup, treatment escalation, and stabilization across longer unstable sepsis trajectories "
                "while avoiding unsafe actions."
            ),
            min_steps=8,
            max_steps=16,
            preferred_stay_ids=hard_ids,
            score_weights={
                "detection": 0.15,
                "lab_workup": 0.15,
                "treatment": 0.20,
                "stability": 0.20,
                "safety": 0.10,
                "outcome": 0.20,
            },
        ),
    }
