---
title: Sepsis OpenEnv
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
tags:
  - openenv
  - healthcare
  - offline-rl
  - sepsis
---

# Sepsis OpenEnv

`Sepsis OpenEnv` is a real-world offline sepsis treatment environment built for the OpenEnv hackathon workflow. It exposes a standard agent loop through `reset()`, `step()`, and `state()` and scores treatment decisions on logged ICU trajectories.

It is designed to satisfy the Round 1 submission requirements:

- real-world task: ICU sepsis treatment decisions
- typed models for action, observation, and state
- 3 graded tasks: `easy`, `medium`, `hard`
- meaningful dense rewards with safety penalties
- reproducible baseline `inference.py`
- Dockerized server for local and Hugging Face deployment

## What The Environment Simulates

At each step, the agent recommends:

- an IV fluid dose bin `0..4`
- a vasopressor dose bin `0..4`

The environment advances along a logged patient trajectory and rewards the agent for:

- matching sensible treatment intensity
- making safe decisions for lower-severity cases
- handling more unstable cases without obviously harmful escalation

This is an offline environment built from a compact processed bundle derived from the local MIMIC-III demo cohort. It is inspired by the WD3QNE sepsis-treatment paper, but the environment itself is purpose-built for OpenEnv evaluation rather than paper reproduction.

## Tasks

Task definitions live in [tasks.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/tasks.py).

- `easy`
  Conservative treatment on mild sepsis trajectories with short horizons.
- `medium`
  Mixed-severity cases with longer trajectories and stronger agreement pressure.
- `hard`
  Higher-severity cases where sensible escalation and terminal-outcome handling matter more.

Each task has a deterministic grader in [graders.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/graders.py) that returns a score in `[0.0, 1.0]`.

## Action Space

Defined in [models.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/models.py#L10).

- `fluid_bin`: integer `0..4`
- `pressor_bin`: integer `0..4`
- total discrete combinations: `25`

## Observation Space

Defined in [models.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/models.py#L20).

Each observation contains:

- task id and task description
- current patient trajectory id
- current step and max steps
- severity proxy
- mortality flag from the logged stay
- normalized feature dictionary for the selected 37 features
- current cumulative reward and last reward

The hidden logged clinician action is intentionally not exposed in observations.

## Reward Design

The reward function is dense, not purely terminal.

Per step:

- positive signal for plausible agreement with the hidden logged clinician action
- small progress bonus when the next logged state becomes less severe
- penalties for unsafe escalation or obviously insufficient treatment

At the end of the episode:

- bonus for survival trajectories
- penalty for death trajectories

This gives the agent useful partial-progress feedback throughout the trajectory.

## Core Files

- [openenv.yaml](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/openenv.yaml): OpenEnv metadata
- [models.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/models.py): typed action / observation / state models
- [tasks.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/tasks.py): task catalog
- [graders.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/graders.py): deterministic graders
- [client.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/client.py): client wrapper
- [server/app.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/server/app.py): FastAPI app
- [server/sepsis_environment.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/server/sepsis_environment.py): environment implementation
- [inference.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/inference.py): baseline runner
- [validate_local.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/validate_local.py): local smoke checks
- [prepare_submission.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/prepare_submission.py): creates a clean submission bundle

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the local validation:

```bash
python validate_local.py
```

Start the environment server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Quick checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata
```

## Baseline Inference

The required root-level baseline script is [inference.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/inference.py).

Run locally:

```bash
python inference.py
```

It writes reproducible scores to:

- [outputs/baseline_scores.json](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/outputs/baseline_scores.json)

If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are present, the script uses the OpenAI client with those variables. If not, it falls back to a deterministic heuristic policy so the baseline still runs reproducibly.

## Docker

Build:

```bash
docker build -t sepsis-openenv .
```

Run:

```bash
docker run -p 8000:8000 sepsis-openenv
```

The container should expose a working `/health` endpoint and respond to `/reset`.

## Submission Bundle

To prepare a clean hackathon-ready bundle without the larger local research folders, run:

```bash
python prepare_submission.py
```

This creates:

- `submission_bundle/`

with only the files needed for the environment runtime and submission packaging.

## Runtime Assets

The submission runtime uses the small preprocessed assets in:

- `env_data/processed_demo_dataset.pkl`
- `env_data/selected_features.json`

This keeps the environment lightweight enough for the hackathon resource limits.

## Known Caveat

The repo was structured for OpenEnv submission, but the official `openenv validate` CLI was not available in this session. Local validation, baseline runs, HTTP checks, and Docker build/run all passed.

## Inspiration

Wu, X., Li, R., He, Z. et al. *A value-based deep reinforcement learning model with human expertise in optimal treatment of sepsis.* npj Digital Medicine 6, 15 (2023). https://doi.org/10.1038/s41746-023-00755-5
