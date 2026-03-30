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

`Sepsis OpenEnv` is a real-world sequential sepsis management environment built for the OpenEnv hackathon workflow. It exposes a standard agent loop through `reset()`, `step()`, and `state()` and evaluates how well an agent gathers information, chooses treatment, and manages a logged ICU trajectory under partial observability.

It is designed to satisfy the Round 1 submission requirements:

- real-world task: ICU sepsis workup and treatment decisions
- typed models for action, observation, and state
- 3 graded tasks: `easy`, `medium`, `hard`
- meaningful dense rewards with safety penalties
- reproducible baseline `inference.py`
- Dockerized server for local and Hugging Face deployment

## What The Environment Simulates

At each step, the agent can:

- request one lab from a small clinically meaningful set
- request one treatment plan from a small sepsis-management action set
- optionally mark the current state as suspected sepsis

The environment advances along a logged patient trajectory and rewards the agent for:

- detecting likely sepsis early
- requesting informative labs instead of repeatedly querying low-value tests
- selecting treatment plans that fit the hidden severity pattern in the logged stay
- avoiding obviously unsafe escalation or under-treatment

This is an offline environment built from a compact processed bundle derived from the local MIMIC-III demo cohort. It is inspired by the WD3QNE sepsis-treatment paper, but the environment itself is purpose-built for OpenEnv evaluation rather than paper reproduction.

## Tasks

Task definitions live in [tasks.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/tasks.py).

- `easy`
  Early sepsis workup from partial bedside data with an emphasis on timely lab selection.
- `medium`
  Diagnosis plus early treatment initiation after iterative lab requests.
- `hard`
  Full sepsis management across longer unstable trajectories with stabilization and outcome pressure.

Each task has a deterministic grader in [graders.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/graders.py) that returns a score in `[0.0, 1.0]`.

## Action Space

Defined in [models.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/models.py#L10).

- `action_type`: `request_lab`, `request_treatment`, or `monitor`
- `suspect_sepsis`: boolean detection signal
- `lab_type`: one of `lactate`, `wbc`, `creatinine`, `bicarbonate`, `platelets`, `bilirubin`
- `treatment_type`: one of `monitor`, `fluids`, `vasopressors`, `combination`

## Observation Space

Defined in [models.py](/C:/Users/Baibhav%20Sureka/Videos/ID3QNE-algorithm/models.py#L20).

Each observation contains:

- task id and task description
- current patient trajectory id
- current step and max steps
- severity proxy
- mortality flag from the logged stay
- demographics and always-visible vitals
- visible non-lab context features
- only the labs explicitly requested so far
- current cumulative reward and last reward

Hidden logged treatment choices and unrevealed labs are intentionally not exposed in observations.

## Reward Design

The reward function is dense, not purely terminal.

Per step:

- positive signal for early sepsis suspicion on high-risk states
- reward for requesting priority labs that fit the current presentation
- reward for selecting treatment plans that match the hidden severity pattern
- progress bonus when the next logged state becomes less severe
- penalties for duplicate low-value labs, unsafe escalation, or obvious under-treatment

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

If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are present, the script uses the OpenAI client with those variables. If not, it falls back to a deterministic staged baseline that requests key labs first and then selects a treatment plan, so the benchmark still runs reproducibly.

Current deterministic baseline scores from the local run:

- `easy`: `0.93`
- `medium`: `0.9062`
- `hard`: `0.8886`
- mean score: `0.9083`

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
