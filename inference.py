"""
inference.py — MedTriageEnv Baseline Inference Script
======================================================

MANDATORY submission file. Runs an LLM agent against all 3 tasks and
produces reproducible baseline scores.

Environment variables required:
  API_BASE_URL   — LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — Model identifier   (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       — HuggingFace / API key

Usage:
  python inference.py

Runtime: < 10 minutes on cpu-basic (well within 20-min limit).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from medtriage_env.client import MedTriageEnvLocal
from medtriage_env.models import MedTriageAction, MedTriageObservation, TriageAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN: str     = os.getenv("HF_TOKEN") or ""
MODEL_NAME: str   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS_PER_TASK = {
    "task1_single_patient":       5,
    "task2_multi_patient":        8,
    "task3_dynamic_deterioration":10,
}
SEEDS            = [42, 7, 99]   # 3 seeds per task → 9 episodes total
TEMPERATURE      = 0.1            # Low temp for reproducibility
MAX_TOKENS       = 512
FALLBACK_ACTION  = TriageAction.REASSESS.value

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency department (ED) triage nurse.
You will receive a clinical scenario and must choose the best action.

TASK TYPES:
- task1_single_patient: Assign ESI level (1=most urgent, 5=least urgent) to one patient.
- task2_multi_patient:  Rank 5 patients from most to least urgent.
- task3_dynamic_deterioration: Manage a deteriorating patient over multiple steps.

AVAILABLE ACTIONS (use the integer ID):
1=ASSIGN_ESI_1  2=ASSIGN_ESI_2  3=ASSIGN_ESI_3  4=ASSIGN_ESI_4  5=ASSIGN_ESI_5
6=ORDER_ECG     7=ORDER_LABS    8=ORDER_XRAY    9=ORDER_CT
10=CALL_PHYSICIAN  11=ACTIVATE_TRAUMA  12=TRANSFER_ICU
13=ADMINISTER_O2   14=IV_ACCESS        15=PAIN_MANAGEMENT
16=DISCHARGE    17=REASSESS     18=NOOP

ESI GUIDE:
ESI 1: Requires IMMEDIATE life-saving intervention (intubation, CPR, defibrillation)
       → GCS ≤8, SpO2 <85%, HR <40 or >180, BP <70, RR <6 or >35
ESI 2: High-risk, high-acuity. Severe pain, confused, or high-risk complaint
       → Chest pain, stroke sx, anaphylaxis, sepsis, SpO2 <90, HR >150, BP <90
ESI 3: Stable but needs multiple resources (labs, imaging, IV)
       → Abdominal pain, pneumonia, trauma, BP 90–140 with symptoms
ESI 4: Stable, needs one resource (X-ray, urine dip, wound care)
       → Sprain, sore throat, minor laceration, simple fracture
ESI 5: Stable, no resources needed (counselling, prescription refill only)

RESPONSE FORMAT — always reply with a JSON object only, no prose:
{
  "action": <integer 1-18>,
  "reasoning": "<one sentence clinical justification>",
  "patient_rankings": ["P001","P002",...] // ONLY for task2, most urgent first
}
""").strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(obs: MedTriageObservation, step: int) -> str:
    """Convert observation into a structured clinical prompt."""
    lines = [
        f"TASK: {obs.task_id}",
        f"STEP: {step + 1} / {obs.max_steps}",
        f"LEGAL ACTIONS: {obs.legal_actions}",
        "",
        "CLINICAL SITUATION:",
        obs.clinical_summary,
    ]
    if obs.time_pressure_flag:
        lines.append("\n⚠ TIME PRESSURE — patient in immediate danger!")
    if obs.resource_constraints:
        lines.append(f"\nRESOURCES: {obs.resource_constraints}")
    if obs.step_reward != 0.0:
        lines.append(f"\nLAST STEP REWARD: {obs.step_reward:+.3f}")
    if obs.task_id == "task2_multi_patient":
        lines.append("\nPatient IDs: " + ", ".join(p.patient_id for p in obs.patients))
        lines.append("Rank from MOST urgent to LEAST urgent using patient_rankings.")
    lines.append("\nRespond with JSON only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(
    raw: str,
    obs: MedTriageObservation,
) -> Tuple[int, Optional[List[str]], str]:
    """
    Parse LLM JSON response into (action_id, patient_rankings, reasoning).
    Falls back gracefully on malformed output.
    """
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        action_id    = int(data.get("action", FALLBACK_ACTION))
        rankings     = data.get("patient_rankings", None)
        reasoning    = str(data.get("reasoning", ""))

        # Validate action is legal
        if action_id not in obs.legal_actions:
            # Pick closest legal action
            esi_actions = [a for a in obs.legal_actions if 1 <= a <= 5]
            action_id = esi_actions[0] if esi_actions else obs.legal_actions[0]
            reasoning += " [action corrected to legal]"

        return action_id, rankings, reasoning

    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: scan for first digit
        import re
        nums = re.findall(r'\b([1-9]|1[0-8])\b', raw)
        for n in nums:
            if int(n) in obs.legal_actions:
                return int(n), None, f"fallback parse: {raw[:80]}"
        fallback = obs.legal_actions[0] if obs.legal_actions else FALLBACK_ACTION
        return fallback, None, f"parse failed — fallback to {fallback}"


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    task_id: str,
    seed: int,
    verbose: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run one full episode. Returns (final_score, info_dict).
    """
    env = MedTriageEnvLocal(task_id=task_id)
    obs = env.reset(seed=seed)
    max_steps = MAX_STEPS_PER_TASK[task_id]

    episode_info: Dict[str, Any] = {
        "task_id":    task_id,
        "seed":       seed,
        "steps":      [],
        "final_score": None,
    }

    print(f'[START] {{"task_id": "{task_id}", "seed": {seed}, "model": "{MODEL_NAME}"}}', flush=True)
    if verbose:
        print(f"\n  Episode: task={task_id}, seed={seed}")
        print(f"  Clinical: {obs.clinical_summary.splitlines()[0]}")

    total_reward = 0.0

    for step in range(max_steps):
        if obs.done:
            break

        user_prompt = build_user_prompt(obs, step)

        # ── LLM call ──────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_response = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [WARNING] LLM call failed at step {step+1}: {exc}")
            raw_response = json.dumps({"action": FALLBACK_ACTION, "reasoning": "api error"})

        # ── Parse ─────────────────────────────────────────────────────
        action_id, rankings, reasoning = parse_llm_response(raw_response, obs)

        # ── Build typed action ─────────────────────────────────────────
        action = MedTriageAction(
            action=TriageAction(action_id),
            patient_rankings=rankings,
            reasoning=reasoning,
        )

        # ── Step environment ───────────────────────────────────────────
        result = env.step(action)
        total_reward += result.reward
        obs = result.observation

        step_log = {
            "step":      step + 1,
            "action":    action_id,
            "action_name": TriageAction(action_id).name,
            "reasoning": reasoning[:100],
            "reward":    round(result.reward, 4),
            "done":      result.done,
        }
        episode_info["steps"].append(step_log)
        print(f'[STEP] {{"step": {step+1}, "action": "{TriageAction(action_id).name}", "reward": {round(result.reward, 4)}, "done": {str(result.done).lower()}}}', flush=True)

        if verbose:
            print(
                f"  Step {step+1}: {TriageAction(action_id).name:25s} "
                f"reward={result.reward:+.3f}  "
                f"{'[DONE]' if result.done else ''}"
            )

        if result.done:
            final_score = result.info.get("final_score")
            if final_score is not None:
                episode_info["final_score"] = final_score
                if verbose:
                    print(f"  Final score: {final_score:.4f}")
            print(f'[END] {{"task_id": "{task_id}", "seed": {seed}, "score": {round(float(episode_info["final_score"] or 0.0), 4)}}}', flush=True)
            break

    # Retrieve final score from state if not in last step info
    if episode_info["final_score"] is None:
        try:
            state = env.state()
            episode_info["final_score"] = state.task_score or 0.0
        except Exception:
            episode_info["final_score"] = max(0.0, min(1.0, total_reward / max_steps))
        # Print [END] for max_steps case where done was never True
        print(f"[END] {{"task_id": "{task_id}", "seed": {seed}, "score": {round(float(episode_info["final_score"] or 0.0), 4)}}}", flush=True)

    env.close()
    return float(episode_info["final_score"] or 0.0), episode_info


# ---------------------------------------------------------------------------
# Main — runs all 3 tasks × 3 seeds = 9 episodes
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  MedTriageEnv — Baseline Inference")
    print(f"  Model : {MODEL_NAME}")
    print(f"  API   : {API_BASE_URL}")
    print("=" * 65)

    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN not set. Requests may fail.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "none")

    tasks = [
        "task1_single_patient",
        "task2_multi_patient",
        "task3_dynamic_deterioration",
    ]

    all_results: Dict[str, List[float]] = {t: [] for t in tasks}
    all_episodes: List[Dict] = []

    start_time = time.time()

    for task_id in tasks:
        print(f"\n{'─'*65}")
        print(f"  TASK: {task_id}")
        print(f"{'─'*65}")

        for seed in SEEDS:
            try:
                score, episode_info = run_episode(
                    client=client,
                    task_id=task_id,
                    seed=seed,
                    verbose=True,
                )
            except Exception as exc:
                print(f"  [ERROR] Episode failed (task={task_id}, seed={seed}): {exc}")
                score = 0.0
                episode_info = {"task_id": task_id, "seed": seed, "error": str(exc), "final_score": 0.0}

            all_results[task_id].append(score)
            all_episodes.append(episode_info)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 65}")
    print("  BASELINE SCORES")
    print(f"{'=' * 65}")

    task_labels = {
        "task1_single_patient":       "Task 1 — Single patient ESI   (easy)  ",
        "task2_multi_patient":        "Task 2 — Multi-patient ranking (medium)",
        "task3_dynamic_deterioration":"Task 3 — Dynamic deterioration (hard)  ",
    }

    scores_summary: Dict[str, float] = {}
    for task_id in tasks:
        scores = all_results[task_id]
        avg = sum(scores) / len(scores) if scores else 0.0
        scores_summary[task_id] = avg
        bar = "█" * int(avg * 30)
        print(f"  {task_labels[task_id]} {avg:.4f}  [{bar:<30}]")
        print(f"    seeds={SEEDS}, scores={[round(s,4) for s in scores]}")

    overall = sum(scores_summary.values()) / len(scores_summary)
    print(f"\n  Overall mean: {overall:.4f}")
    print(f"  Runtime:      {elapsed:.1f}s")
    print("=" * 65)

    # ── Write JSON results for automated evaluation ───────────────────
    results_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    output = {
        "model": MODEL_NAME,
        "seeds": SEEDS,
        "task_scores": {k: round(v, 4) for k, v in scores_summary.items()},
        "overall_mean": round(overall, 4),
        "runtime_seconds": round(elapsed, 1),
        "episodes": all_episodes,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results written to: {results_path}")


if __name__ == "__main__":
    main()
