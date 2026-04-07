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

# MedTriageEnv 🏥

**Emergency Department Triage Simulator — OpenEnv Compatible RL Environment**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)

---

## What Is This?

MedTriageEnv simulates an **emergency department triage desk**. An AI agent acts as a triage nurse — assessing incoming patients, assigning acuity levels, ordering diagnostics, and responding when patients deteriorate.

This is a task humans do every day, under time pressure, with real consequences. Emergency nurses triage over **136 million ED visits per year** in the US alone. Getting acuity wrong means critical patients wait too long, or limited resources are burned on non-urgent cases.

**Why this makes a great RL environment:**
- Rich, dense reward signal at every step (not sparse end-of-episode)
- Naturally progressive difficulty (single patient → multi-patient → deteriorating patient)
- Medically grounded ground truth (ESI v4 — the standard used in real EDs worldwide)
- Novel domain — no existing OpenEnv environment covers healthcare

---

## Tasks

### Task 1 — Single Patient ESI Classification (`easy`)

The agent sees one patient with full vitals and clinical context. It must assign the correct **Emergency Severity Index** (ESI 1–5) level and order appropriate diagnostics.

| ESI | Name | Criteria |
|-----|------|----------|
| 1 | Resuscitation | Immediate life-saving intervention required |
| 2 | Emergent | High risk / severe distress / altered mental status |
| 3 | Urgent | Stable, needs multiple resources |
| 4 | Less Urgent | Stable, needs one resource |
| 5 | Non-Urgent | No resources needed |

**Grader:** Partial credit by ESI distance. Exact match = 1.0, off-by-1 = 0.60, off-by-2 = 0.25, off-by-3+ = 0.0. Bonus up to +0.15 for clinically appropriate diagnostics (e.g. ECG + troponin for chest pain).

**Max steps:** 5 | **Expected baseline:** 0.65

---

### Task 2 — Multi-Patient Prioritisation (`medium`)

Five patients arrive simultaneously. The agent must rank them from most urgent to least urgent by submitting a `patient_rankings` list.

**Grader:** Kendall's tau-b rank correlation between agent ranking and ground-truth ESI ranking, normalised to [0, 1]. Bonus +0.10 if the most critical patient (ESI 1) is ranked first. Penalty −0.15 if critical patient is not in the top 2.

**Max steps:** 8 | **Expected baseline:** 0.45

---

### Task 3 — Dynamic Deterioration Response (`hard`)

A single patient's vitals deteriorate over 10 steps. The agent must detect the deterioration, escalate care appropriately, order diagnostics, and assign the correct ESI — all under time pressure.

**Grader:** Weighted composite score:
- ESI accuracy: **30%** — correct ESI assignment
- Escalation timeliness: **35%** — how quickly the agent escalated after deterioration onset
- Diagnostic coverage: **20%** — appropriate tests for the complaint
- Step efficiency: **15%** — fewer steps to correct action = higher score

**Max steps:** 10 | **Expected baseline:** 0.30

---

## Action Space

18 discrete actions covering the full triage workflow:

| ID | Action | Category |
|----|--------|----------|
| 1–5 | `ASSIGN_ESI_1` to `ASSIGN_ESI_5` | ESI assignment |
| 6 | `ORDER_ECG` | Diagnostics |
| 7 | `ORDER_LABS` | Diagnostics |
| 8 | `ORDER_XRAY` | Diagnostics |
| 9 | `ORDER_CT` | Diagnostics |
| 10 | `CALL_PHYSICIAN` | Escalation |
| 11 | `ACTIVATE_TRAUMA` | Escalation |
| 12 | `TRANSFER_ICU` | Escalation |
| 13 | `ADMINISTER_O2` | Supportive care |
| 14 | `IV_ACCESS` | Supportive care |
| 15 | `PAIN_MANAGEMENT` | Supportive care |
| 16 | `DISCHARGE` | Flow control |
| 17 | `REASSESS` | Flow control |
| 18 | `NOOP` | No action (penalised if overused) |

---

## Observation Space

Each observation is a structured JSON with:

```json
{
  "task_id": "task1_single_patient",
  "step": 0,
  "max_steps": 5,
  "done": false,
  "patients": [
    {
      "patient_id": "P001",
      "age": 58,
      "sex": "M",
      "chief_complaint": "crushing chest pain radiating to left arm, diaphoretic",
      "vitals": {
        "heart_rate": 112,
        "systolic_bp": 88,
        "diastolic_bp": 58,
        "respiratory_rate": 24,
        "spo2": 93,
        "temperature": 37.1,
        "gcs": 15,
        "pain_score": 10
      },
      "pmh": ["hypertension", "hyperlipidaemia", "type-2 diabetes"],
      "medications": ["metformin", "amlodipine", "atorvastatin"],
      "allergies": ["penicillin"],
      "arrival_mode": "ambulance",
      "test_results": {}
    }
  ],
  "clinical_summary": "[Step 0]\n  58M via ambulance | c/o: crushing chest pain...",
  "legal_actions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 18],
  "step_reward": 0.0,
  "cumulative_reward": 0.0,
  "time_pressure_flag": true,
  "resource_constraints": {"trauma_bays": 1, "icu_beds": 1, "physicians_available": 2},
  "info": {}
}
```

---

## Reward Function

Rewards are **dense** — the agent receives signal at every step, not just at episode end.

### Step-level rewards

| Event | Reward |
|-------|--------|
| Correct ESI assignment (exact match) | +0.50 |
| ESI off by 1 | +0.20 |
| ESI off by 2+ | −0.15 × delta |
| Appropriate diagnostic ordered | +0.05 to +0.20 |
| Appropriate escalation during deterioration | +0.35 |
| O2 given when SpO2 < 95% | +0.10 |
| O2 given when SpO2 ≥ 95% (not indicated) | −0.05 |
| NOOP taken | −0.05 |
| Missed deterioration (NOOP/REASSESS during decline) | −0.20 |
| Terminal event (patient in extremis) | −0.30 |

### Episode-level grader (0.0–1.0)

Each episode is graded by the task-specific grader at termination. This score goes into `state().task_score` and `info["final_score"]`.

---

## Baseline Scores

Baseline agent: rule-based heuristic (vital sign thresholds). LLM agent scores (gpt-4o-mini equivalent) expected to be higher.

| Task | Seeds | Scores | Average |
|------|-------|--------|---------|
| Task 1 — Single patient | [42, 7, 99] | [0.65, 1.0, 1.0] | **0.88** |
| Task 2 — Multi-patient | [42, 7, 99] | [0.60, 0.40, 1.0] | **0.67** |
| Task 3 — Deterioration | [42, 7, 99] | [0.92, 0.92, 0.82] | **0.88** |
| **Overall** | | | **0.81** |

---

## Project Structure

```
medtriage_env/
├── __init__.py
├── models.py          # Pydantic models: Action, Observation, State
├── scenarios.py       # Seeded patient scenario generator (12 clinical cases)
├── graders.py         # Deterministic graders for all 3 tasks
├── client.py          # HTTP client + in-process local client
├── openenv.yaml       # OpenEnv metadata spec
└── server/
    ├── __init__.py
    ├── app.py          # FastAPI server (reset/step/state/health endpoints)
    ├── environment.py  # Core environment logic
    └── requirements.txt

inference.py            # ← Mandatory baseline inference script
Dockerfile
README.md
pyproject.toml
```

---

## Setup & Usage

### Option 1 — Local (no Docker)

```bash
# Install
git clone https://github.com/YOUR_USERNAME/medtriage-env
cd medtriage-env
pip install -e .

# Start server
uvicorn medtriage_env.server.app:app --host 0.0.0.0 --port 8000

# In another terminal — use the Python client
python -c "
from medtriage_env.client import MedTriageEnv
from medtriage_env.models import MedTriageAction, TriageAction

env = MedTriageEnv(base_url='http://localhost:8000', task_id='task1_single_patient')
obs = env.reset(seed=42)
print(obs.clinical_summary)

result = env.step(MedTriageAction(action=TriageAction.ORDER_ECG))
result = env.step(MedTriageAction(action=TriageAction.ASSIGN_ESI_2))
print('Score:', result.info.get('final_score'))
env.close()
"
```

### Option 2 — Docker

```bash
# Build
docker build -t medtriage-env:latest .

# Run (Task 1 by default)
docker run -p 8000:8000 medtriage-env:latest

# Run a specific task
docker run -p 8000:8000 -e MEDTRIAGE_TASK=task3_dynamic_deterioration medtriage-env:latest

# Health check
curl http://localhost:8000/health
# → {"status":"ok","environment":"MedTriageEnv","version":"1.0.0"}
```

### Option 3 — Hugging Face Spaces

```bash
# The HF Space is already live at:
# https://huggingface.co/spaces/YOUR_USERNAME/medtriage-env

# Use from Python
from medtriage_env.client import MedTriageEnv
env = MedTriageEnv(base_url="https://YOUR_USERNAME-medtriage-env.hf.space")
obs = env.reset(seed=42, task_id="task1_single_patient")
```

---

## Running the Baseline Inference Script

```bash
# Set credentials
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Run (< 10 minutes on cpu-basic)
python inference.py
```

Output:
```
=================================================================
  MedTriageEnv — Baseline Inference
  Model : meta-llama/Llama-3.3-70B-Instruct
  API   : https://router.huggingface.co/v1
=================================================================

─────────────────────────────────────────────────────────────────
  TASK: task1_single_patient
─────────────────────────────────────────────────────────────────
  Episode: task=task1_single_patient, seed=42
  Step 1: ORDER_LABS                reward=+0.100
  Step 2: ASSIGN_ESI_2              reward=+0.500  [DONE]
  Final score: 1.0000
  ...

=================================================================
  BASELINE SCORES
=================================================================
  Task 1 — Single patient ESI   (easy)   0.8833  [█████████████████████████     ]
  Task 2 — Multi-patient ranking (medium) 0.6667  [████████████████████          ]
  Task 3 — Dynamic deterioration (hard)   0.8817  [█████████████████████████     ]

  Overall mean: 0.8106
  Runtime:      142.3s
=================================================================

  Full results written to: baseline_results.json
```

---

## API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check |
| `POST` | `/reset` | `{"task_id": "...", "seed": 42}` | Start new episode |
| `POST` | `/step` | `{"action": 2, "reasoning": "..."}` | Execute action |
| `GET` | `/state` | — | Episode metadata |
| `GET` | `/tasks` | — | List tasks |
| `GET` | `/actions` | — | List all actions |
| `GET` | `/docs` | — | Swagger UI |

### Example: Task 2 — submit patient ranking

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": 17,
    "patient_rankings": ["P005", "P001", "P003", "P002", "P004"],
    "reasoning": "P005 unresponsive, ESI 1 — ranked first"
  }'
```

---

## Clinical Background

The **Emergency Severity Index (ESI)** is a five-level triage algorithm used in over 75% of US emergency departments. It classifies patients by acuity (severity) and expected resource consumption. The algorithm is validated, reproducible, and widely taught to ED nurses and physicians.

Our ground-truth ESI computation follows the **ESI v4 algorithm**:
1. Is the patient dying? → ESI 1
2. Is the patient high-risk, severely distressed, or confused? → ESI 2
3. How many resources will the patient need? → ESI 3 (≥2), ESI 4 (1), ESI 5 (0)

This makes the grading deterministic, clinically meaningful, and defensible — judges can verify the ground truth against real triage textbooks.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@software{medtriage_env_2026,
  title   = {MedTriageEnv: Emergency Department Triage Simulator for RL Agent Evaluation},
  year    = {2026},
  url     = {https://huggingface.co/spaces/YOUR_USERNAME/medtriage-env},
  note    = {OpenEnv-compatible environment submitted to Meta PyTorch Hackathon 2026}
}
```
