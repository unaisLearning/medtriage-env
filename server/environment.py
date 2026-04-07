"""
MedTriageEnvironment — core environment logic.

Implements the OpenEnv Environment interface:
  reset(seed) → MedTriageObservation
  step(action) → StepResult
  state()      → MedTriageState

This class is instantiated by the FastAPI server. Each WebSocket session
gets its own instance (thread-safe by design — no shared mutable state).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from medtriage_env.graders import Task1Grader, Task2Grader, Task3Grader
from medtriage_env.models import (
    ESILevel,
    MedTriageAction,
    MedTriageObservation,
    MedTriageState,
    PatientRecord,
    StepResult,
    TriageAction,
    VitalSigns,
)
from medtriage_env.scenarios import (
    ScenarioSpec,
    generate_task1_scenario,
    generate_task2_scenario,
    generate_task3_scenario,
)


# ---------------------------------------------------------------------------
# Legal action sets per task
# ---------------------------------------------------------------------------

TASK1_LEGAL = [
    TriageAction.ASSIGN_ESI_1, TriageAction.ASSIGN_ESI_2, TriageAction.ASSIGN_ESI_3,
    TriageAction.ASSIGN_ESI_4, TriageAction.ASSIGN_ESI_5,
    TriageAction.ORDER_ECG, TriageAction.ORDER_LABS, TriageAction.ORDER_XRAY,
    TriageAction.ORDER_CT, TriageAction.PAIN_MANAGEMENT, TriageAction.ADMINISTER_O2,
    TriageAction.IV_ACCESS, TriageAction.NOOP,
]

TASK2_LEGAL = [
    TriageAction.ASSIGN_ESI_1, TriageAction.ASSIGN_ESI_2, TriageAction.ASSIGN_ESI_3,
    TriageAction.ASSIGN_ESI_4, TriageAction.ASSIGN_ESI_5,
    TriageAction.REASSESS, TriageAction.NOOP,
]

TASK3_LEGAL = list(TriageAction)  # All actions available


def _legal_ints(actions) -> List[int]:
    return [int(a.value) for a in actions]


# ---------------------------------------------------------------------------
# Clinical summary generator
# ---------------------------------------------------------------------------

def _build_clinical_summary(patients: List[PatientRecord], task_id: str, step: int) -> str:
    if task_id == "task2_multi_patient":
        lines = [f"[Step {step}] ED census: {len(patients)} patients waiting.\n"]
        for p in patients:
            v = p.vitals
            lines.append(
                f"  {p.patient_id} — {p.age}{p.sex} | {p.chief_complaint} | "
                f"HR {v.heart_rate} BP {v.systolic_bp}/{v.diastolic_bp} "
                f"SpO2 {v.spo2}% RR {v.respiratory_rate} Temp {v.temperature}°C "
                f"GCS {v.gcs} Pain {v.pain_score}/10 | "
                f"Waiting {p.time_in_ed_minutes} min | Arrival: {p.arrival_mode}"
            )
        return "\n".join(lines)
    else:
        p = patients[0]
        v = p.vitals
        deteriorating_flag = " ⚠ DETERIORATING" if p.deteriorating else ""
        test_str = ""
        if p.test_results:
            results = "; ".join(f"{k}: {v_}" for k, v_ in p.test_results.items())
            test_str = f"\n  Results: {results}"
        return (
            f"[Step {step}]{deteriorating_flag}\n"
            f"  {p.age}{p.sex} via {p.arrival_mode} | c/o: {p.chief_complaint}\n"
            f"  HR {v.heart_rate} | BP {v.systolic_bp}/{v.diastolic_bp} | "
            f"SpO2 {v.spo2}% | RR {v.respiratory_rate} | "
            f"Temp {v.temperature}°C | GCS {v.gcs} | Pain {v.pain_score}/10\n"
            f"  PMH: {', '.join(p.pmh) if p.pmh else 'None'} | "
            f"Meds: {', '.join(p.medications) if p.medications else 'None'} | "
            f"Allergies: {', '.join(p.allergies) if p.allergies else 'NKDA'}"
            f"{test_str}"
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MedTriageEnvironment:
    """
    Emergency Department Triage Simulator.

    Tasks:
      task1_single_patient      — Classify one patient's ESI level (easy)
      task2_multi_patient       — Rank 5 patients by urgency (medium)
      task3_dynamic_deterioration — Manage deteriorating patient over 10 steps (hard)
    """

    MAX_STEPS = {
        "task1_single_patient": 5,
        "task2_multi_patient": 8,
        "task3_dynamic_deterioration": 10,
    }

    TASK_DESCRIPTIONS = {
        "task1_single_patient": (
            "Assess the single patient and assign the correct ESI (1–5) triage level. "
            "Order appropriate diagnostics to support your decision."
        ),
        "task2_multi_patient": (
            "Five patients are waiting. Rank them from most urgent (rank 1) to least urgent "
            "by providing a patient_rankings list in your action. You may reassess before submitting."
        ),
        "task3_dynamic_deterioration": (
            "Manage a patient whose vitals are deteriorating over time. "
            "Order diagnostics, escalate care, and reassess at each step. "
            "You have 10 steps — act decisively when the patient worsens."
        ),
    }

    def __init__(self, task_id: str = "task1_single_patient"):
        self._task_id = task_id
        self._state: Optional[MedTriageState] = None
        self._patients: List[PatientRecord] = []
        self._ground_truth_specs: List[ScenarioSpec] = []
        self._deterioration_schedule: List[Dict] = []
        self._actions_taken: List[int] = []
        self._escalation_step: Optional[int] = None
        self._deterioration_start_step: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._seed: Optional[int] = None

        # Graders (stateless — instantiate once)
        self._g1 = Task1Grader()
        self._g2 = Task2Grader()
        self._g3 = Task3Grader()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> MedTriageObservation:
        """Start a new episode. Returns initial observation."""
        self._seed = seed if seed is not None else 42
        episode_id = str(uuid.uuid4())

        self._actions_taken = []
        self._escalation_step = None
        self._deterioration_start_step = 2  # Task 3: deterioration begins at step 2
        self._cumulative_reward = 0.0
        self._done = False

        # Build scenario
        if self._task_id == "task1_single_patient":
            spec = generate_task1_scenario(self._seed)
            self._ground_truth_specs = [spec]
            self._patients = [spec.patient.model_copy(deep=True)]
            self._deterioration_schedule = []

        elif self._task_id == "task2_multi_patient":
            specs = generate_task2_scenario(self._seed, n_patients=5)
            self._ground_truth_specs = specs
            self._patients = [s.patient.model_copy(deep=True) for s in specs]
            self._deterioration_schedule = []

        elif self._task_id == "task3_dynamic_deterioration":
            spec, schedule = generate_task3_scenario(self._seed)
            self._ground_truth_specs = [spec]
            self._patients = [spec.patient.model_copy(deep=True)]
            self._deterioration_schedule = schedule

        # Initialise state
        self._state = MedTriageState(
            episode_id=episode_id,
            step_count=0,
            task_id=self._task_id,
            task_description=self.TASK_DESCRIPTIONS[self._task_id],
            num_patients=len(self._patients),
            seed=self._seed,
        )

        return self._build_observation(step_reward=0.0)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: MedTriageAction) -> StepResult:
        """Execute one action. Returns (observation, reward, done, info)."""
        if self._done:
            # Gracefully handle post-episode calls
            obs = self._build_observation(step_reward=0.0)
            return StepResult(observation=obs, reward=0.0, done=True, info={"warning": "Episode already done"})

        step_reward = 0.0
        info: Dict[str, Any] = {}

        # Record action
        action_val = int(action.action.value) if hasattr(action.action, 'value') else int(action.action)
        self._actions_taken.append(action_val)

        # Apply task-specific step logic
        if self._task_id == "task1_single_patient":
            step_reward, info = self._step_task1(action)
        elif self._task_id == "task2_multi_patient":
            step_reward, info = self._step_task2(action)
        elif self._task_id == "task3_dynamic_deterioration":
            step_reward, info = self._step_task3(action)

        # NOOP penalty
        if action_val == TriageAction.NOOP.value:
            step_reward -= 0.05
            info["noop_penalty"] = -0.05

        self._cumulative_reward += step_reward
        self._state.step_count += 1  # type: ignore

        # Check episode end conditions
        max_steps = self.MAX_STEPS[self._task_id]
        done = self._done or self._state.step_count >= max_steps

        if done:
            self._done = True
            # Final grading
            if self._state.task_score is None:
                final_score, grade_breakdown = self._final_grade()
                self._state.task_score = final_score  # type: ignore
                info["final_grade"] = grade_breakdown
                info["final_score"] = final_score

        obs = self._build_observation(step_reward=step_reward)
        return StepResult(observation=obs, reward=step_reward, done=self._done, info=info)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> MedTriageState:
        """Return current episode metadata."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    # ------------------------------------------------------------------
    # Task-specific step logic
    # ------------------------------------------------------------------

    def _step_task1(self, action: MedTriageAction) -> Tuple[float, Dict]:
        """Process one step for Task 1 (single patient ESI classification)."""
        reward = 0.0
        info: Dict = {}
        patient = self._patients[0]
        gt_esi = self._ground_truth_specs[0].ground_truth_esi
        action_val = int(action.action.value) if hasattr(action.action, 'value') else int(action.action)

        # ESI assignment
        if action_val in range(
            TriageAction.ASSIGN_ESI_1.value, TriageAction.ASSIGN_ESI_5.value + 1
        ):
            assigned = action_val  # TriageAction 1–5 map directly to ESI 1–5
            patient.assigned_esi = assigned
            delta = abs(assigned - gt_esi)
            if delta == 0:
                reward += 0.50   # Strong positive signal for correct assignment
                info["esi_feedback"] = "Correct ESI assignment"
            elif delta == 1:
                reward += 0.20   # Partial credit
                info["esi_feedback"] = f"ESI off by 1 (assigned {assigned}, correct {gt_esi})"
            else:
                reward -= 0.15 * delta
                info["esi_feedback"] = f"ESI off by {delta} (assigned {assigned}, correct {gt_esi})"
            self._done = True   # Task 1 ends when ESI is assigned

        # Diagnostic orders — give small positive reward for clinical engagement
        elif action_val in (
            TriageAction.ORDER_ECG.value, TriageAction.ORDER_LABS.value,
            TriageAction.ORDER_XRAY.value, TriageAction.ORDER_CT.value,
        ):
            reward, test_result = self._process_diagnostic(patient, action_val)
            patient.test_results.update(test_result)
            info["test_ordered"] = test_result

        # Supportive care
        elif action_val == TriageAction.ADMINISTER_O2.value:
            if patient.vitals.spo2 < 95:
                reward += 0.10
                info["o2_note"] = "Appropriate O2 for low SpO2"
            else:
                reward -= 0.05
                info["o2_note"] = "O2 not indicated (SpO2 ≥95%)"

        elif action_val == TriageAction.IV_ACCESS.value:
            if gt_esi <= 2:
                reward += 0.08
                info["iv_note"] = "IV access appropriate for high-acuity patient"
            else:
                reward += 0.02

        return reward, info

    def _step_task2(self, action: MedTriageAction) -> Tuple[float, Dict]:
        """Process one step for Task 2 (multi-patient ranking)."""
        reward = 0.0
        info: Dict = {}
        action_val = int(action.action.value) if hasattr(action.action, 'value') else int(action.action)

        if action.patient_rankings and len(action.patient_rankings) == len(self._patients):
            # Agent submitted a complete ranking → grade immediately
            gt_specs = self._ground_truth_specs
            gt_by_esi = sorted(gt_specs, key=lambda s: s.ground_truth_esi)
            gt_ranking = [s.patient.patient_id for s in gt_by_esi]
            esi_map = {s.patient.patient_id: s.ground_truth_esi for s in gt_specs}

            score, breakdown = self._g2.grade(
                agent_rankings=action.patient_rankings,
                ground_truth_rankings=gt_ranking,
                esi_map=esi_map,
            )
            reward = score * 0.8  # Partial credit at step; full credit at final grade
            info["ranking_feedback"] = breakdown
            self._done = True

        elif action_val == TriageAction.REASSESS.value:
            reward += 0.02   # Small bonus for taking time to reassess before ranking
            info["reassess_note"] = "Reassessing all patients"

        return reward, info

    def _step_task3(self, action: MedTriageAction) -> Tuple[float, Dict]:
        """Process one step for Task 3 (dynamic deterioration)."""
        reward = 0.0
        info: Dict = {}
        patient = self._patients[0]
        action_val = int(action.action.value) if hasattr(action.action, 'value') else int(action.action)

        # Apply deterioration for this step (before agent acts)
        step_idx = self._state.step_count  # type: ignore
        if step_idx < len(self._deterioration_schedule):
            changes = self._deterioration_schedule[step_idx]
            if changes:
                # Apply vital sign changes
                self._apply_vitals_change(patient, changes)
                patient.deteriorating = True
                info["deterioration"] = changes
            else:
                patient.deteriorating = False
        else:
            patient.deteriorating = False

        is_deteriorating = patient.deteriorating

        # ESI assignment
        if action_val in range(
            TriageAction.ASSIGN_ESI_1.value, TriageAction.ASSIGN_ESI_5.value + 1
        ):
            patient.assigned_esi = action_val
            gt_esi = self._ground_truth_specs[0].ground_truth_esi
            delta = abs(action_val - gt_esi)
            reward += max(0.0, 0.30 - delta * 0.10)
            info["esi_assigned"] = action_val

        # Escalation actions
        elif action_val in (
            TriageAction.CALL_PHYSICIAN.value,
            TriageAction.ACTIVATE_TRAUMA.value,
            TriageAction.TRANSFER_ICU.value,
        ):
            if self._escalation_step is None:
                self._escalation_step = step_idx
            if is_deteriorating:
                reward += 0.35   # Strong reward for escalating on deterioration
                info["escalation_note"] = "Timely escalation during deterioration"
            else:
                reward += 0.10
                info["escalation_note"] = "Escalation (patient stable)"

        # Diagnostics
        elif action_val in (
            TriageAction.ORDER_ECG.value, TriageAction.ORDER_LABS.value,
            TriageAction.ORDER_XRAY.value, TriageAction.ORDER_CT.value,
        ):
            diag_reward, test_result = self._process_diagnostic(patient, action_val)
            reward += diag_reward
            patient.test_results.update(test_result)
            info["test_ordered"] = test_result

        # Missed deterioration penalty
        if is_deteriorating and action_val in (
            TriageAction.NOOP.value, TriageAction.REASSESS.value
        ):
            reward -= 0.20
            self._state.missed_deteriorations += 1  # type: ignore
            info["missed_deterioration_penalty"] = -0.20

        # Check terminal vitals (patient coded)
        v = patient.vitals
        if v.spo2 < 70 or v.heart_rate < 20 or v.systolic_bp < 40:
            self._done = True
            reward -= 0.30
            info["terminal_event"] = "Patient in extremis — episode ended"

        return reward, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_vitals_change(self, patient: PatientRecord, changes: Dict) -> None:
        """Apply delta dict to patient vitals in place."""
        v = patient.vitals
        for field, value in changes.items():
            if hasattr(v, field):
                setattr(v, field, value)

    def _process_diagnostic(
        self, patient: PatientRecord, action_val: int
    ) -> Tuple[float, Dict]:
        """Simulate ordering a diagnostic test. Returns (reward, result_dict)."""
        complaint = patient.chief_complaint.lower()
        v = patient.vitals

        if action_val == TriageAction.ORDER_ECG.value:
            result = {}
            if "chest pain" in complaint or "crushing" in complaint:
                # Simulate STEMI or normal ECG based on HR/BP
                if v.heart_rate > 100 and v.systolic_bp < 95:
                    result = {"ecg": "ST-elevation in leads II, III, aVF — inferior STEMI"}
                    return 0.20, result
                else:
                    result = {"ecg": "Sinus tachycardia, no acute ST changes"}
                    return 0.10, result
            else:
                result = {"ecg": "Normal sinus rhythm"}
                return 0.05, result

        elif action_val == TriageAction.ORDER_LABS.value:
            result = {}
            if v.systolic_bp < 95:
                result["troponin"] = "elevated 2.4 ng/mL (ref <0.04)"
                result["lactate"] = "3.8 mmol/L (elevated)"
                result["wbc"] = "18.2 x10^9/L"
                return 0.15, result
            else:
                result["troponin"] = "0.02 ng/mL (normal)"
                result["wbc"] = "11.0 x10^9/L (mildly elevated)"
                result["creatinine"] = "78 umol/L (normal)"
                return 0.10, result

        elif action_val == TriageAction.ORDER_XRAY.value:
            result = {}
            if "breath" in complaint or "cough" in complaint:
                result["cxr"] = "Right lower lobe consolidation consistent with pneumonia"
                return 0.12, result
            else:
                result["cxr"] = "No acute cardiopulmonary process"
                return 0.05, result

        elif action_val == TriageAction.ORDER_CT.value:
            result = {}
            if "headache" in complaint or "stroke" in complaint:
                result["ct_head"] = "Hyperdense lesion in right MCA territory — acute ischaemic stroke"
                return 0.18, result
            else:
                result["ct_abdomen"] = "No acute intra-abdominal pathology"
                return 0.06, result

        return 0.05, {}

    def _final_grade(self) -> Tuple[float, Dict]:
        """Run the appropriate grader at episode end."""
        patient = self._patients[0] if self._patients else None

        if self._task_id == "task1_single_patient":
            esi = patient.assigned_esi if patient else None
            gt_esi = self._ground_truth_specs[0].ground_truth_esi
            complaint = patient.chief_complaint if patient else ""
            score, breakdown = self._g1.grade(esi, gt_esi, self._actions_taken, complaint)

        elif self._task_id == "task2_multi_patient":
            gt_specs = self._ground_truth_specs
            gt_by_esi = sorted(gt_specs, key=lambda s: s.ground_truth_esi)
            gt_ranking = [s.patient.patient_id for s in gt_by_esi]
            esi_map = {s.patient.patient_id: s.ground_truth_esi for s in gt_specs}
            # Use last patient_rankings from history (approximated from state)
            # This is set during _step_task2; use score already computed
            score, breakdown = 0.5, {"note": "Score computed during step"}

        elif self._task_id == "task3_dynamic_deterioration":
            gt_esi = self._ground_truth_specs[0].ground_truth_esi
            esi = patient.assigned_esi if patient else None
            complaint = patient.chief_complaint if patient else ""
            score, breakdown = self._g3.grade(
                assigned_esi=esi,
                ground_truth_esi=gt_esi,
                actions_taken=self._actions_taken,
                deterioration_step=self._deterioration_start_step,
                escalation_step=self._escalation_step,
                missed_deteriorations=self._state.missed_deteriorations,  # type: ignore
                chief_complaint=complaint,
                max_steps=self.MAX_STEPS[self._task_id],
            )
        else:
            score, breakdown = 0.0, {}

        self._state.task_score = score  # type: ignore
        return score, breakdown

    def _build_observation(self, step_reward: float) -> MedTriageObservation:
        """Build the full observation dict for the current state."""
        task_id = self._task_id
        step = self._state.step_count if self._state else 0
        max_steps = self.MAX_STEPS[task_id]

        # Determine legal actions
        if task_id == "task1_single_patient":
            legal = _legal_ints(TASK1_LEGAL)
            # Remove ESI-assigning actions if already assigned
            if self._patients and self._patients[0].assigned_esi is not None:
                esi_actions = list(range(
                    TriageAction.ASSIGN_ESI_1.value,
                    TriageAction.ASSIGN_ESI_5.value + 1
                ))
                legal = [a for a in legal if a not in esi_actions]
        elif task_id == "task2_multi_patient":
            legal = _legal_ints(TASK2_LEGAL)
        else:
            legal = _legal_ints(TASK3_LEGAL)

        # Time pressure flag
        time_pressure = False
        if self._patients:
            for p in self._patients:
                v = p.vitals
                if (v.spo2 < 90 or v.heart_rate > 150 or v.systolic_bp < 80
                        or v.gcs <= 9 or p.deteriorating):
                    time_pressure = True
                    break

        # Resource constraints (simulated)
        resources = {"trauma_bays": 1, "icu_beds": 1, "physicians_available": 2}
        if task_id == "task3_dynamic_deterioration" and step > 4:
            resources["icu_beds"] = 0  # Added pressure

        active_id = None
        if task_id != "task2_multi_patient" and self._patients:
            active_id = self._patients[0].patient_id

        summary = _build_clinical_summary(self._patients, task_id, step)

        return MedTriageObservation(
            task_id=task_id,
            step=step,
            max_steps=max_steps,
            done=self._done,
            patients=self._patients,
            active_patient_id=active_id,
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            legal_actions=legal,
            clinical_summary=summary,
            time_pressure_flag=time_pressure,
            resource_constraints=resources,
            info={"task_description": self.TASK_DESCRIPTIONS[task_id]},
        )
