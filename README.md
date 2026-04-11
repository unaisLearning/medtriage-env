---
title: MedTriageEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - agent-environment
---

# MedTriageEnv
### OpenEnv for Emergency Department Triage

MedTriageEnv is an OpenEnv-compatible reinforcement learning environment for emergency department triage. The agent has to assess acuity, prioritize patients, order diagnostics, and react to deterioration while the episode is still moving.

It is not a toy benchmark. The environment is meant to model a real workflow where decisions happen step by step, the state changes over time, and partial progress should still matter.

MedTriageEnv includes:
- single-patient ESI classification
- multi-patient urgency ranking
- dynamic deterioration management

The goal is to give agents a realistic triage loop to learn from and evaluate against.

---

## Context

Every minute in an emergency department, nurses make triage decisions that affect who gets seen first. If the call is wrong, a patient can worsen while waiting. Open, reproducible training environments for triage are still rare, and this project tries to close that gap.

- Built on **ESI v4** — the actual triage algorithm used in US hospitals
- **12 seeded clinical scenarios** covering sepsis, chest pain, trauma, stroke, and more
- **Deterministic graders** — reproducible scoring
- **Curriculum difficulty** — easy single-patient to medium multi-patient to hard deterioration

---

## 3 Tasks — Easy to Hard

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| task1_single_patient | Easy | Assign correct ESI level (1-5) to one patient | 5 |
| task2_multi_patient | Medium | Rank 5 simultaneous patients by urgency | 8 |
| task3_dynamic_deterioration | Hard | Manage a patient whose vitals deteriorate over 10 steps | 10 |

Task 3 is genuinely hard — the agent must detect deterioration trends across steps and escalate at exactly the right moment. Escalating too early is penalised, and waiting too long incurs missed-deterioration penalties.

---

## Baseline Scores

Current checked-in baseline artifact from [`baseline_results.json`](/Users/unais/Downloads/medtriage_env/baseline_results.json) reports the following scores:

| Task | Seed 42 | Seed 7 | Seed 99 | Average |
|------|---------|--------|---------|---------|
| Task 1 — Single patient ESI | 0.94 | 0.60 | 0.60 | 0.73 |
| Task 2 — Multi-patient ranking | 0.94 | 0.94 | 0.94 | 0.94 |
| Task 3 — Dynamic deterioration | 0.35 | 0.05 | 0.05 | 0.15 |
| Overall | | | | 0.61 |

The main point is that weak policies should score poorly, while better triage decisions should score higher.

---

## Reward Function

Dense per-step rewards — not sparse end-of-episode only:

| Event | Reward |
|-------|--------|
| Correct ESI assignment | +0.50 |
| ESI off by 1 | +0.20 |
| ESI off by 2+ | -0.15 x delta |
| Appropriate diagnostic ordered | +0.05 to +0.20 |
| Escalation during deterioration | +0.35 |
| O2 given when SpO2 < 95% | +0.10 |
| Missed deterioration | -0.20 |
| NOOP taken | -0.05 |

---

## Action Space (18 discrete actions)

| Category | Actions |
|----------|---------|
| ESI Assignment | ASSIGN_ESI_1 through ASSIGN_ESI_5 |
| Diagnostics | ORDER_ECG, ORDER_LABS, ORDER_XRAY, ORDER_CT |
| Escalation | CALL_PHYSICIAN, ACTIVATE_TRAUMA, TRANSFER_ICU |
| Supportive Care | ADMINISTER_O2, IV_ACCESS, PAIN_MANAGEMENT |
| Flow Control | DISCHARGE, REASSESS, NOOP |

---

## Observation Space

Each step the agent receives a rich clinical summary with medical alerts:

```
[Step 2] *** PATIENT DETERIORATING - ESCALATE NOW ***
  55M via walk-in | c/o: fever, productive cough, shortness of breath
  HR 128 | BP 95/65 | SpO2 88% | RR 28 | Temp 39.8C | GCS 15 | Pain 6/10
  PMH: COPD | Meds: salbutamol | Allergies: NKDA
  ALERTS: CRITICAL: SpO2 < 90% | CRITICAL: Tachypnea RR > 25
  TASK: Monitor and manage this patient. Escalate if deteriorating.
```

Structured info also includes named actions, steps remaining, and scoring hints.

---

## Quick Start

```bash
git clone https://github.com/unaisLearning/medtriage-env
cd medtriage-env
pip install -e .
uvicorn medtriage_env.server.app:app --host 0.0.0.0 --port 7860
```

Docker:
```bash
docker build -t medtriage-env .
docker run -p 7860:7860 medtriage-env
```

Run inference:
```bash
export HF_TOKEN="hf_your_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python3 inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Server health check |
| /reset | POST | Start new episode |
| /step | POST | Take action |
| /state | GET | Current episode state |
| /tasks | GET | List all tasks |
| /actions | GET | List all 18 actions |
| /docs | GET | Interactive Swagger UI |

---

## Project Structure

```
medtriage_env/
├── Dockerfile
├── inference.py
├── openenv.yaml
├── test_medtriage.py        48 tests — all passing
└── medtriage_env/
    ├── models.py            Pydantic models
    ├── scenarios.py         12 seeded clinical scenarios
    ├── graders.py           ESI v4, Kendall tau-b, composite graders
    └── server/
        ├── app.py           FastAPI server
        └── environment.py   Core reset/step/state logic
```

---

## Use Case

Unlike static medical QA benchmarks, MedTriageEnv is dynamic — patient states change, conditions deteriorate, and the agent must reason across multiple steps under time pressure. This mirrors real clinical decision-making in ways no existing benchmark does.

This gives us a realistic, reproducible setup for training and testing triage agents. There are very few open-source environments for this use case, and even fewer with medically valid ground truth.

---

Built solo for the Meta x HuggingFace OpenEnv Hackathon, April 2026.
Open-source emergency department triage RL environment built on real medical algorithms.
