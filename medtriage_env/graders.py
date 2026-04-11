"""
Graders for MedTriageEnv.

Each task gets a deterministic score in the open interval (0, 1).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from medtriage_env.models import MedTriageState, TriageAction


STRICT_SCORE_EPSILON = 1e-6


def _strict_unit_interval(value: float) -> float:
    """
    Keep final task scores away from the exact 0 and 1 endpoints.
    """
    return min(1.0 - STRICT_SCORE_EPSILON, max(STRICT_SCORE_EPSILON, value))


# ---------------------------------------------------------------------------
def _strict_clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by judges."""
    return max(0.001, min(0.999, float(score)))


# Task 1 Grader — Single patient ESI classification
# ---------------------------------------------------------------------------

class Task1Grader:
    """
    Grade the agent's ESI assignment for a single patient.

    Scoring:
      Exact match:         1.00
      Off by 1 level:      0.60  (common clinical disagreement zone)
      Off by 2 levels:     0.25  (clinically significant error)
      Off by ≥3 levels:    0.00  (dangerous under/over-triage)

    Additionally:
      +0.15 bonus if agent ordered appropriate diagnostics (ECG for chest pain etc.)
      −0.20 penalty if NOOP was the only action taken (no clinical engagement)
    """

    EXACT_SCORE   = 1.00
    OFF_BY_1      = 0.60
    OFF_BY_2      = 0.25
    OFF_BY_GE_3   = 0.00

    def grade(
        self,
        assigned_esi: Optional[int],
        ground_truth_esi: int,
        actions_taken: List[int],
        chief_complaint: str,
    ) -> Tuple[float, Dict]:
        """
        Returns (score, breakdown_dict).
        breakdown_dict contains component scores for transparency.
        """
        breakdown: Dict = {}

        # --- Primary ESI score ---
        if assigned_esi is None:
            esi_score = 0.0
            breakdown["esi_component"] = 0.0
            breakdown["esi_note"] = "No ESI assigned"
        else:
            delta = abs(assigned_esi - ground_truth_esi)
            if delta == 0:
                esi_score = self.EXACT_SCORE
            elif delta == 1:
                esi_score = self.OFF_BY_1
            elif delta == 2:
                esi_score = self.OFF_BY_2
            else:
                esi_score = self.OFF_BY_GE_3
            breakdown["esi_component"] = esi_score
            breakdown["esi_assigned"] = assigned_esi
            breakdown["esi_ground_truth"] = ground_truth_esi
            breakdown["esi_delta"] = delta

        # --- Diagnostic appropriateness bonus ---
        bonus = 0.0
        complaint_lower = chief_complaint.lower()
        has_ecg = TriageAction.ORDER_ECG.value in actions_taken
        has_labs = TriageAction.ORDER_LABS.value in actions_taken
        has_ct = TriageAction.ORDER_CT.value in actions_taken

        if "chest pain" in complaint_lower or "crushing" in complaint_lower:
            if has_ecg and has_labs:
                bonus = 0.15
                breakdown["diagnostic_bonus"] = "ECG + Labs ordered for chest pain (+0.15)"
            elif has_ecg or has_labs:
                bonus = 0.07
                breakdown["diagnostic_bonus"] = "Partial diagnostics ordered (+0.07)"
        elif "headache" in complaint_lower or "stroke" in complaint_lower:
            if has_ct:
                bonus = 0.15
                breakdown["diagnostic_bonus"] = "CT ordered for neuro complaint (+0.15)"
        else:
            if has_labs or has_ecg:
                bonus = 0.05
                breakdown["diagnostic_bonus"] = "Diagnostics ordered (+0.05)"

        # --- NOOP penalty ---
        only_noop = all(a == TriageAction.NOOP.value for a in actions_taken) if actions_taken else True
        noop_penalty = -0.20 if only_noop else 0.0
        breakdown["noop_penalty"] = noop_penalty

        # Avoid exact 0 or 1 here; the validator rejects endpoint scores.
        final = _strict_unit_interval(esi_score + bonus + noop_penalty)
        breakdown["final_score"] = round(final, 6)

        return final, breakdown


# ---------------------------------------------------------------------------
# Task 2 Grader — Multi-patient prioritisation
# ---------------------------------------------------------------------------

class Task2Grader:
    """
    Grade the agent's ranking of N patients by urgency.

    Uses a tie-aware pairwise ordering score from ground-truth ESI.
    Equal-ESI pairs are ignored.

    Additionally:
      +0.10 if critical (ESI 1) patient is ranked #1
      −0.15 if critical (ESI 1) patient is NOT ranked #1 or #2
    """

    def grade(
        self,
        agent_rankings: List[str],       # patient IDs, most urgent first
        ground_truth_rankings: List[str], # patient IDs, most urgent first (by ESI)
        esi_map: Dict[str, int],          # patient_id → ground_truth_esi
    ) -> Tuple[float, Dict]:
        """Returns (score, breakdown_dict)."""
        breakdown: Dict = {}

        if not agent_rankings or not ground_truth_rankings:
            score = _strict_unit_interval(0.0)
            return score, {"error": "Empty rankings provided", "final_score": round(score, 6)}

        patient_ids = [pid for pid in ground_truth_rankings if pid in esi_map]
        agent_rank = {pid: i for i, pid in enumerate(agent_rankings)}

        concordant = 0
        discordant = 0
        skipped_ties = 0
        n_pairs = 0
        for i in range(len(patient_ids)):
            for j in range(i + 1, len(patient_ids)):
                pid_i = patient_ids[i]
                pid_j = patient_ids[j]
                if pid_i not in agent_rank or pid_j not in agent_rank:
                    continue
                esi_i = esi_map[pid_i]
                esi_j = esi_map[pid_j]
                if esi_i == esi_j:
                    skipped_ties += 1
                    continue
                gt_order = esi_i < esi_j  # lower ESI means more urgent
                agent_order = agent_rank[pid_i] < agent_rank[pid_j]
                if gt_order == agent_order:
                    concordant += 1
                else:
                    discordant += 1
                n_pairs += 1

        if n_pairs == 0:
            tau = 0.0
        else:
            tau = (concordant - discordant) / n_pairs

        normalized_tau = (tau + 1.0) / 2.0
        breakdown["pairwise_tau"] = round(tau, 4)
        breakdown["pairwise_score"] = round(normalized_tau, 4)
        breakdown["ignored_tied_pairs"] = skipped_ties

        # Small bonus/penalty for the most critical patient
        critical_patients = [pid for pid, esi in esi_map.items() if esi == 1]
        bonus = 0.0
        if critical_patients:
            most_critical = critical_patients[0]
            if agent_rankings and agent_rankings[0] == most_critical:
                bonus += 0.10
                breakdown["critical_bonus"] = "Critical patient ranked #1 (+0.10)"
            elif most_critical not in agent_rankings[:2]:
                bonus -= 0.15
                breakdown["critical_penalty"] = "Critical patient not in top 2 (-0.15)"

        final = _strict_unit_interval(normalized_tau + bonus)
        breakdown["final_score"] = round(final, 6)

        return final, breakdown


# ---------------------------------------------------------------------------
# Task 3 Grader — Dynamic deterioration response
# ---------------------------------------------------------------------------

class Task3Grader:
    """
    Grade the agent's management of a deteriorating patient over 10 steps.

    Score components:
      1. ESI accuracy (30%)      — correct ESI assignment
      2. Escalation timeliness (35%) — did agent escalate before critical threshold?
      3. Diagnostic coverage (20%) — appropriate tests ordered
      4. Step efficiency (15%)   — fewer steps to correct action = higher score

    Final score is a weighted sum in the open interval (0, 1).
    """

    ESI_WEIGHT        = 0.30
    ESCALATION_WEIGHT = 0.35
    DIAGNOSTIC_WEIGHT = 0.20
    EFFICIENCY_WEIGHT = 0.15

    ESCALATION_ACTIONS = {
        TriageAction.CALL_PHYSICIAN.value,
        TriageAction.ACTIVATE_TRAUMA.value,
        TriageAction.TRANSFER_ICU.value,
        TriageAction.ASSIGN_ESI_1.value,
        TriageAction.ASSIGN_ESI_2.value,
    }

    DIAGNOSTIC_ACTIONS = {
        TriageAction.ORDER_ECG.value,
        TriageAction.ORDER_LABS.value,
        TriageAction.ORDER_XRAY.value,
        TriageAction.ORDER_CT.value,
    }

    def grade(
        self,
        assigned_esi: Optional[int],
        ground_truth_esi: int,
        actions_taken: List[int],            # all actions this episode, in order
        deterioration_step: int,             # step at which patient started deteriorating
        escalation_step: Optional[int],      # first step agent escalated (None if never)
        missed_deteriorations: int,          # count from state
        chief_complaint: str,
        max_steps: int = 10,
    ) -> Tuple[float, Dict]:
        """Returns (score, breakdown_dict)."""
        breakdown: Dict = {}

        # --- Component 1: ESI accuracy ---
        if assigned_esi is None:
            esi_score = 0.0
        else:
            delta = abs(assigned_esi - ground_truth_esi)
            esi_score = max(0.0, 1.0 - (delta * 0.4))
        breakdown["esi_score"] = round(esi_score, 4)

        # --- Component 2: Escalation timeliness ---
        # Earlier is better once the patient starts to worsen.
        if escalation_step is None:
            escalation_score = 0.0
            breakdown["escalation_note"] = "No escalation detected"
        else:
            steps_after = escalation_step - deterioration_step
            if steps_after < 0:
                escalation_score = max(0.2, 0.6 - (abs(steps_after) - 1) * 0.15)
                breakdown["escalation_note"] = "Premature escalation"
            elif steps_after == 0:
                escalation_score = 1.0
                breakdown["escalation_note"] = "Escalated at deterioration onset"
            elif steps_after == 1:
                escalation_score = 0.85
                breakdown["escalation_note"] = "Escalated 1 step after deterioration"
            elif steps_after == 2:
                escalation_score = 0.65
                breakdown["escalation_note"] = "Escalated 2 steps after deterioration"
            elif steps_after == 3:
                escalation_score = 0.40
                breakdown["escalation_note"] = "Escalated 3 steps after deterioration (late)"
            else:
                escalation_score = max(0.0, 0.25 - (steps_after - 4) * 0.05)
                breakdown["escalation_note"] = f"Escalated {steps_after} steps late"

        # Missed deterioration should hurt the score.
        missed_penalty = min(0.30, missed_deteriorations * 0.10)
        escalation_score = max(0.0, escalation_score - missed_penalty)
        breakdown["escalation_score"] = round(escalation_score, 4)
        breakdown["missed_deteriorations"] = missed_deteriorations

        # --- Component 3: Diagnostic coverage ---
        diagnostic_actions_taken = set(actions_taken) & self.DIAGNOSTIC_ACTIONS
        complaint_lower = chief_complaint.lower()

        required_diagnostics: set = set()
        if "chest pain" in complaint_lower or "crushing" in complaint_lower:
            required_diagnostics = {
                TriageAction.ORDER_ECG.value,
                TriageAction.ORDER_LABS.value,
            }
        elif "headache" in complaint_lower or "stroke" in complaint_lower:
            required_diagnostics = {TriageAction.ORDER_CT.value}
        elif "breath" in complaint_lower or "respiratory" in complaint_lower:
            required_diagnostics = {
                TriageAction.ORDER_XRAY.value,
                TriageAction.ORDER_LABS.value,
            }
        else:
            required_diagnostics = {TriageAction.ORDER_LABS.value}

        if required_diagnostics:
            covered = len(diagnostic_actions_taken & required_diagnostics)
            diagnostic_score = covered / len(required_diagnostics)
        else:
            diagnostic_score = 1.0 if diagnostic_actions_taken else 0.5
        breakdown["diagnostic_score"] = round(diagnostic_score, 4)

        # --- Component 4: Efficiency ---
        # Fewer steps to the right action gets a better score.
        escalation_step_val = escalation_step if escalation_step is not None else max_steps
        ideal_steps = deterioration_step + 1  # perfect: escalate immediately
        actual_steps = escalation_step_val
        efficiency_ratio = max(0.0, 1.0 - (actual_steps - ideal_steps) / max_steps)
        efficiency_score = efficiency_ratio
        breakdown["efficiency_score"] = round(efficiency_score, 4)

        # --- Weighted final ---
        final = (
            esi_score * self.ESI_WEIGHT
            + escalation_score * self.ESCALATION_WEIGHT
            + diagnostic_score * self.DIAGNOSTIC_WEIGHT
            + efficiency_score * self.EFFICIENCY_WEIGHT
        )
        final = _strict_unit_interval(final)
        breakdown["final_score"] = round(final, 6)
        breakdown["weights"] = {
            "esi": self.ESI_WEIGHT,
            "escalation": self.ESCALATION_WEIGHT,
            "diagnostic": self.DIAGNOSTIC_WEIGHT,
            "efficiency": self.EFFICIENCY_WEIGHT,
        }

        return final, breakdown
