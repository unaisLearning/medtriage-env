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
### Emergency Department Triage Simulator — OpenEnv RL Environment

> **136 million ED visits** occur annually in the US. Triage errors contribute to **1 in 4 preventable adverse events** in hospitals. MedTriageEnv is the first open-source RL environment that trains AI agents to make life-critical triage decisions.

---

## What It Does

An AI agent acts as an emergency department triage nurse. It reads patient vitals, chief complaints, and medical history — then decides urgency levels, orders diagnostics, and responds to deteriorating patients. Built on the **ESI v4 algorithm** — the validated triage standard used in real US hospitals.

---

## 3 Tasks — Easy to Hard

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| task1_single_patient | Easy | Assign correct ESI level (1-5) to one patient | 5 |
| task2_multi_patient | Medium | Rank 5 simultaneous patients by urgency | 8 |
| task3_dynamic_deterioration | Hard | Manage a patient whose vitals deteriorate over 10 steps | 10 |

---

## Baseline Scores

Model: meta-llama/Llama-3.1-8B-Instruct via HuggingFace Router

| Task | Scores (seeds 42,7,99) | Average |
|------|----------------------|---------|
| Task 1 — Single patient ESI | [0.0, 1.0, 0.05] | 0.35 |
| Task 2 — Multi-patient ranking | [0.8, 0.5, 0.5] | 0.60 |
| Task 3 — Dynamic deterioration | [0.63, 0.23, 0.05] | 0.30 |
| Overall | | 0.42 |

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
| /actions | GET | List all valid actions |

---

## Real-World Impact

Unlike static medical QA benchmarks, MedTriageEnv is dynamic — patient states change, conditions deteriorate, and the agent must reason across multiple steps under time pressure. This mirrors real clinical decision-making in ways no existing benchmark does.

136 million ED visits per year. 1 in 4 preventable adverse events linked to triage errors. An agent trained on this environment has direct clinical value.

---

Built for the Meta x HuggingFace OpenEnv Hackathon, April 2026
