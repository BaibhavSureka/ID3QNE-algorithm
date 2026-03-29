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
            title="Stabilize Mild Sepsis",
            description=(
                "Recommend conservative IV fluid and vasopressor bins for low-severity sepsis trajectories "
                "while avoiding unnecessary escalation."
            ),
            min_steps=6,
            max_steps=8,
            preferred_stay_ids=easy_ids,
            score_weights={"reward": 0.45, "agreement": 0.35, "safety": 0.20},
        ),
        "medium": TaskConfig(
            task_id="medium",
            title="Manage Mixed-Severity Sepsis",
            description=(
                "Track the clinician policy on medium-complexity sepsis cases while maintaining safe dosing "
                "decisions over a longer trajectory."
            ),
            min_steps=8,
            max_steps=12,
            preferred_stay_ids=medium_ids,
            score_weights={"reward": 0.40, "agreement": 0.35, "safety": 0.15, "terminal": 0.10},
        ),
        "hard": TaskConfig(
            task_id="hard",
            title="Handle Unstable Sepsis Cases",
            description=(
                "Treat higher-severity, less stable sepsis trajectories with sensible escalation while limiting "
                "unsafe divergence from the logged clinician policy."
            ),
            min_steps=8,
            max_steps=16,
            preferred_stay_ids=hard_ids,
            score_weights={"reward": 0.35, "agreement": 0.25, "safety": 0.15, "terminal": 0.25},
        ),
    }
