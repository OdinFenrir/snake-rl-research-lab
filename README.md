# Snake RL Research Lab

## Overview
An interactive Snake RL lab built with `pygame` and PPO.  
The goal is simple: train an agent, watch it play live, measure failures, and iteratively improve behavior with reproducible data.

## Tech Stack
- Python 3.12
- pygame
- stable-baselines3
- sb3-contrib
- NumPy / pandas-based analysis scripts

## Features
- Live training dashboard with run telemetry
- PPO + controller arbitration logic
- Holdout suite and worst-seed diagnostics
- Artifact generation for training and agent performance reports
- Model manager and experiment isolation workflows

## Screenshots
- Live UI screenshots are in `docs/assets/` (see Demo section below).

## Run Locally
```bat
setup_env.bat
run.bat
```

## Roadmap
- [ ] Add a concise benchmark table near the top of README
- [ ] Add a quick architecture diagram image
- [ ] Add a short GIF of live training + agent play

## What This Demonstrates

- Built a complete desktop ML application with live UI, persistence, and worker-thread training
- Designed regression tooling for worst-case seed analysis and focused debugging traces
- Enforced evaluation discipline with paired holdout runs (PPO-only vs controller-on) and contamination guards
- Implemented learned controller memory (online arbiter + clustered tactic memory) that persists across sessions
- Handled failure recovery for local persisted state with atomic writes and rollback

## Documentation

Use these docs directly from the main page:

- [README](README.md) - product overview, setup, UI flow, and validation commands
- [Architecture](ARCHITECTURE.md) - runtime modules, decision stack, data/artifact contracts
- [Operating Rules](OPERATING_RULES.md) - experiment isolation and comparison discipline
- [Trusted Baselines](TRUSTED_BASELINES.md) - benchmark trust boundary and baseline references
- [Changelog](CHANGELOG.md) - chronological project changes
- [Report Tooling Contract](docs/REPORT_TOOLING_CONTRACT.md) - canonical paths, latest semantics, retention, CLI/failure rules

## Demo

### Live Training UI

Menu 1:

![Snake Frame menu 1](docs/assets/live_training_ui_menu1.png)

Menu 2:

![Snake Frame menu 2](docs/assets/live_training_ui_menu2.png)

Menu 3:

![Snake Frame menu 3](docs/assets/live_training_ui_menu3.png)

Menu 4:

![Snake Frame menu 4](docs/assets/live_training_ui_menu4.png)

## Project Overview

This project is a full training + evaluation environment for reinforcement learning experiments on Snake:

- Learning core:
  - PPO with action masking (`stable-baselines3` + `sb3-contrib`)
  - Gym-style environment (`snake_frame/ppo_env.py`)
- Runtime intelligence:
  - policy inference from PPO
  - dynamic safety/controller arbitration (`snake_frame/gameplay_controller.py`)
  - learned controller memory:
    - `arbiter_model.json` (online learned arbitration)
    - `tactic_memory.json` (clustered tactic memory)
- Experiment loop:
  - train in app UI
  - run holdout suites (`ppo_only` vs `controller_on`)
  - isolate worst seeds
  - capture per-step traces
  - patch and re-validate

## How It Works (For Enthusiasts)

At each game decision:
1. The PPO model predicts an action and confidence.
2. The controller scores local risk (danger, space viability, food pressure, loop signals).
3. The system either trusts PPO or applies controller logic (`escape` / `space_fill` behavior).
4. Outcomes are logged into telemetry and artifacts.

### Decision Stack

<center>

```
                         GAME STATE
              (snake body, head, food, grid)
                            |
                            v
                    OBSERVATION LAYER
              (31-dim: extended, path, tail, free_space, trend)
                            |
                            v
                       PPO INFERENCE
              (observation -> action logits + confidence)
                            |
                            v
                     RISK EVALUATION
              - Danger: immediate death if move?
              - Space: would this trap the snake?
              - Food pressure: how far from food?
              - Cycle signals: are we looping?
              - Tail reachability: can we reach tail?
                            |
                            v
                 CONTROLLER ARBITRATION
    +---------------+---------------+---------------+
    |     PPO       |    ESCAPE     |  SPACE_FILL   |
    |   (trust)     |   (cycles)    | (no progress) |
    +---------------+---------------+---------------+
                            |
                            v
                    LEARNED MEMORY
        (arbiter_model.json + tactic_memory.json)
                            |
                            v
                     FINAL ACTION
```

</center>

### Controller Modes

- **PPO** (default): Trust the model's prediction when confidence is high and risk is low
- **ESCAPE**: Activated on detected cycle/repeat patterns - breaks loops
- **SPACE_FILL**: Activated on no-progress - fills space efficiently to survive longer

The controller learns from experience via `arbiter_model.json` (online) and `tactic_memory.json` (clustered patterns).

During training:
1. PPO runs in vectorized environments.
2. Evaluation/checkpoints are saved under `state/ppo/<experiment_name>/`. The default baseline path is `state/ppo/baseline/`.
3. You can watch live behavior in the same app while training runs.
4. Post-run, focused seed tools identify exactly where controller behavior underperforms.

## Core Features

- Live dashboard with training KPIs, run KPIs, and risk/intervention counters
- Save/load/delete model lifecycle controls in UI
- Determinism and smoke-performance validation tools
- Worst-seed gate + focused per-step trace pipeline for factual debugging
- Clean local artifact model for iterative tuning

## Quick Start (Windows)

1. `setup_env.bat`
2. `run.bat`

Environment defaults:
- Python `3.12`
- Virtual environment at `.venv`
- Locked dependencies from `requirements-lock.txt`

## Reproducibility & Data

After cloning:
1. `setup_env.bat`
2. `run_dashboard.bat` for CI-equivalent validation

Not versioned by design (local experiment data):
- `state/` (local models/checkpoints/UI state)
- `artifacts/` (generated diagnostics/evals/reports)

### Experiment Isolation

- Baseline runs use the default experiment path: `state/ppo/baseline/`
- New experiments should use a distinct `experiment_name`
- Before training into the baseline path, preserve both:
  - the baseline model directory under `state/ppo/baseline/`
  - the matching suite artifact under `artifacts/live_eval/suites/`

For trustworthy comparisons, cite:
- git commit
- suite artifact (`artifacts/live_eval/suites/suite_*.json`)
- matching `metadata.json`
- experiment name

## Main Controls (In App)

- Workspace entry screen:
  - `Live Training` (main game/training UI)
  - `Analysis Tools` (report runners + in-app output viewer)
  - `Model Manager` (promote/archive/recover/delete model workflows)
  - `Application Settings` (opens options panel)
  - `Esc` from live app returns to workspace menu
- `Start Train` / `Stop Train`
- `Save` / `Load` / `Delete`
- `Start Manual` or `Start Agent` (context-aware) / `Stop Game` / `Restart`
- Right panel tabs: `Train`, `Run`, `Debug`
- Session block includes persistent model save status (`Saved: ...`)
- Options panel groups:
  - `Training`: Reward Shaping, Safe Space Bias, Tail Trend Assist
  - `Visual`: Theme, Board Style, Snake Look, Fog Level
  - `Playback Speed`: Slower / Faster
  - `Model Checks`: Run Full Evaluation, Run Holdout Check, Eval mode toggle (PPO vs Controller)
  - `Debug & Tools`: Debug Overlay, Reachability Overlay, Export Diagnostics

### Analysis Tools Menu (In App)

- `Training Quality Report` (single model)
- `Agent Runtime Report` (single model)
- `Model vs Model Compare` (Model 1 vs Model 2)
- `Report Artifact Manager` (retain `latest` + last N stamped files)
- `Purge Report Artifacts` (hard-delete canonical report artifacts)
- `Failure Replay`
- `Evaluation Suite`
- `Policy 3D Explorer`
- `Model Graph (Netron)`

Model selector behavior:
- Single-model tools use one selector (`Model`).
- Compare tool uses two selectors (`Model 1`, `Model 2`).

Execution model:
- In-app Analysis Tools execute Python scripts directly (no `.bat` dependency for app workflow).
- Root `.bat` files remain as optional helper entrypoints for manual terminal runs.

### Model Safety Workflow

- Startup mode is detached by default:
  - `Experiment: New (not loaded)`
  - `Model: none`
  - `Saved: no model on disk`
  - App does not auto-bind to previous experiment on startup.
- A real experiment is bound only after explicit `Load` or `Save`.
- `Save`: prompts for experiment name and saves into `state/ppo/<experiment_name>/`
- `Load`/`Delete`: open a folder picker under `state/ppo/` to target an explicit experiment
- `Save` is blocked when no model is loaded/trained
- This prevents accidental overwrite of a different experiment's `last_model.zip`
- Model Manager destructive baseline actions are guarded by quick reliability gate checks.

## Persistence

Saved artifacts (default baseline path shown):
- `state/ui_state.json`
- `state/ppo/<experiment_name>/*`
- `state/ppo/<experiment_name>/metadata.json`
- `state/ppo/<experiment_name>/arbiter_model.json`
- `state/ppo/<experiment_name>/tactic_memory.json`

The app starts with the `baseline` experiment loaded.
Use `Create` to make a new experiment, `Load` to load an existing one, or `Save` to persist changes.

Metadata captures run IDs, timesteps, configs, provenance, and eval summaries for future tuning.

## Evaluation & Diagnostics

Core artifacts:
- Holdout suite: `artifacts/live_eval/suites/latest_suite.json`
- Focused worst-seed report: `artifacts/live_eval/worst10_latest.json`
- Focused per-step traces: `artifacts/live_eval/focused_traces/<timestamp>_<tag>/seed_<seed>.jsonl`
- Controller gate result: `artifacts/live_eval/controller_candidate_gate.json`

Useful scripts:
- `scripts/worst_seed_gate.py`
- `scripts/focused_controller_trace.py`
- `scripts/post_run_suite.py`
- `scripts/controller_candidate_gate.py`
- `scripts/blind_spot_replay.py`
- `scripts/blind_spot_replay_view.py`
- `scripts/training_input/build_training_input_report.py` (training-input-only report)
- `scripts/agent_performance/build_agent_performance_report.py` (agent-performance-only report)
- `scripts/phase3_compare/build_model_agent_compare_report.py` (Phase 3 pairwise model+agent compare report)

Controller suite contract (automatic):
- `controller_on` summaries now emit:
  - `mean_interventions_pct`
  - `controller_telemetry_rows` (per-seed decisions/interventions/interventions_pct)
- `latest_suite.json` now carries `comparison.mean_interventions_pct` for gate enforcement.

Dashboard acceptance path:
1. Lint
2. Tests
3. Render regression
4. Smoke median perf gate
5. Determinism drift check
6. Controller candidate gate (hard fail on reject)

Blind-spot replay one-shot (Windows):
- `run_blind_spot_replay.bat`
- Generates:
  - `artifacts/live_eval/blind_spot_replay_latest.json`
  - `artifacts/live_eval/blind_spot_replay_latest.html`

Training-input report one-shot (Windows):
- `run_training_input_report.bat`
- Generates:
  - `artifacts/training_input/training_input_latest.json`
  - `artifacts/training_input/training_input_latest.md`
  - `artifacts/training_input/training_input_checkpoint_vecnorm_latest.csv`
  - `artifacts/training_input/training_input_timeline_latest.json`
  - `artifacts/training_input/training_input_timeline_latest.md`
  - `artifacts/training_input/training_input_timeline_latest.csv`
  - `artifacts/training_input/training_input_dashboard_latest.html` (interactive charts)
  - `artifacts/reports/reports_hub_latest.md` (single organized hub)
  - `artifacts/reports/reports_hub_latest.txt` (plain-text copy/paste)

Agent-performance report one-shot (Windows):
- `run_agent_performance_report.bat`
- Generates:
  - `artifacts/agent_performance/agent_performance_latest.json`
  - `artifacts/agent_performance/agent_performance_latest.md`
  - `artifacts/agent_performance/agent_performance_rows_latest.csv`
  - `artifacts/agent_performance/agent_performance_dashboard_latest.html` (interactive charts)
  - `artifacts/reports/reports_hub_latest.md` (single organized hub)
  - `artifacts/reports/reports_hub_latest.txt` (plain-text copy/paste)

Model+agent compare report one-shot (Windows):
- `run_model_agent_compare_report.bat <left_experiment> <right_experiment>`
- If args are omitted, it prompts for both experiment folder names (under `state/ppo/`)
- Example:
  - `run_model_agent_compare_report.bat baseline Test_1`
- Generates:
  - `artifacts/phase3_compare/model_agent_compare_latest.json`
  - `artifacts/phase3_compare/model_agent_compare_latest.md`
  - `artifacts/phase3_compare/model_agent_compare_rows_latest.csv`
  - `artifacts/phase3_compare/model_agent_compare_dashboard_latest.html` (interactive charts)
  - `artifacts/reports/reports_hub_latest.md` (single organized hub)
  - `artifacts/reports/reports_hub_latest.txt` (plain-text copy/paste)

### Artifact Retention Policy

To reduce noise from repeated analysis on the same run, report generators keep:
- `latest` aliases (always)
- last **N** stamped files per output type (default `N=5`)

Supported flag:
- `--retain-stamped <N>` on report generator scripts (`training_input`, `agent_performance`, `phase3_compare`, including dashboard builders)

Example:
- `run_training_input_report.bat --artifact-dir state\ppo\Test_1 --retain-stamped 3`

Notes:
- Retention prunes only stamped outputs for that report family.
- `latest` files are never pruned.

### Report Contract + Legacy Timeline

- Canonical report locations and required outputs are defined in `docs/REPORT_TOOLING_CONTRACT.md`.
- App workflow uses direct Python entrypoints; root `.bat` files remain compatibility helpers for manual runs.
- Legacy write support ends: **2026-06-30 23:59 UTC**
- Legacy read-fallback support removed: **2026-07-31 23:59 UTC**
- After **2026-07-31 23:59 UTC**, old fallback paths are hard errors with migration hints.

## Evaluation Protocol + Baseline Tracking

Protocol for comparable controller-vs-PPO checks:
1. Use fixed holdout seeds `17001-17030`.
2. Run paired evaluation with the same model selector (`last`): `ppo_only` then `controller_on`.
3. Keep controller learning disabled during holdout eval (prevents eval contamination/drift).
4. Compare paired seed deltas (`controller - ppo`) and aggregate means.
5. Re-check repeatability on worst-10 seeds from `artifacts/live_eval/worst10_latest.json`.

For the current canonical baseline package and trust boundary, use:
- `TRUSTED_BASELINES.md`
- the latest suite under `artifacts/live_eval/suites/`
- matching `state/ppo/<experiment_name>/metadata.json`

Previous numeric baseline values are archived as historical context only.  
Generate new baseline metrics from fresh suites in the `baseline` experiment before making benchmark claims.

## Validation Commands (Local)

- Lint:
  - `.venv\Scripts\python.exe -m ruff check scripts snake_frame tests main.py`
- Full tests:
  - `.venv\Scripts\python.exe -m pytest -q`
- Quick reliability gate:
  - `.venv\Scripts\python.exe scripts\quick_gate.py --cycles 5`
- Determinism:
  - `.venv\Scripts\python.exe scripts\validate_determinism.py --baseline tests\baselines\deterministic_windows.json`
- Smoke median gate:
  - `.venv\Scripts\python.exe scripts\smoke_gate_median.py --runs 3 --train-steps 2048 --game-steps 300 --ppo-profile fast --max-frame-p95-ms 40 --max-frame-avg-ms 34 --max-frame-jitter-ms 8 --max-inference-p95-ms 12 --min-training-steps-per-sec 250`
- Worst-seed gate:
  - `.venv\Scripts\python.exe scripts\worst_seed_gate.py --suite artifacts\live_eval\suites\latest_suite.json --top-n 10 --enforce --max-worse-count 8 --min-mean-delta -25`

## CI / Automation

GitHub Actions workflow definitions are in place for fast validation on push/PR. Full validation runs locally:
- Linting (ruff)
- Unit and integration tests
- Render regression checks

Full ML training/evaluation gates remain local because they are runtime-heavy and hardware-sensitive (long runtimes and timing-sensitive smoke gates on local hardware profiles).

Local validation commands are documented above.

## Project Structure

- `main.py`
- `snake_frame/` core app + training + controller modules
- `scripts/` diagnostics/eval/research tooling
- `tests/` automated suite
- `ARCHITECTURE.md` deep architecture notes
