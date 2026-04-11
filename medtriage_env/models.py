"""
MedTriageEnv - Emergency Department Triage Simulator
Models: Action, Observation, State — OpenEnv spec compliant.

Every field is typed and documented so agents always know what they're seeing.
Pydantic v2 used to match OpenEnv's openenv.core.env_server.types pattern.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ESILevel(IntEnum):
    """Emergency Severity Index — standard ED triage scale."""
    RESUSCITATION = 1   # Immediate life-saving intervention
    EMERGENT      = 2   # High risk, severe pain, confused
    URGENT        = 3   # Stable but needs multiple resources
    LESS_URGENT   = 4   # Stable, one resource needed
    NON_URGENT    = 5   # Stable, no resources needed


class TriageAction(IntEnum):
    """Actions the agent can take during a triage step."""
    # Priority / ESI assignment
    ASSIGN_ESI_1 = 1
    ASSIGN_ESI_2 = 2
    ASSIGN_ESI_3 = 3
    ASSIGN_ESI_4 = 4
    ASSIGN_ESI_5 = 5
    # Diagnostic orders
    ORDER_ECG        = 6
    ORDER_LABS       = 7
    ORDER_XRAY       = 8
    ORDER_CT         = 9
    # Care escalation
    CALL_PHYSICIAN   = 10
    ACTIVATE_TRAUMA  = 11
    TRANSFER_ICU     = 12
    # Supportive
    ADMINISTER_O2    = 13
    IV_ACCESS        = 14
    PAIN_MANAGEMENT  = 15
    # Flow control
    DISCHARGE        = 16
    REASSESS         = 17
    NOOP             = 18   # No action (penalised if overused)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class MedTriageAction(BaseModel):
    """
    Action sent by the agent to the environment.

    For single-patient tasks: set `action` and optionally `target_patient_id`.
    For multi-patient tasks: set `patient_rankings` (list of patient IDs,
    most urgent first) in addition to the primary action.
    """

    action: TriageAction = Field(
        ...,
        description=(
            "Primary triage action chosen from the TriageAction enum. "
            "Actions 1-5 assign an ESI level. Actions 6-18 order diagnostics, "
            "escalate care, or manage flow."
        ),
    )
    target_patient_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the patient this action targets. Required for multi-patient "
            "tasks; ignored for single-patient episodes."
        ),
    )
    patient_rankings: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ordered list of patient IDs from most urgent to least urgent. "
            "Required for Task 2 (multi-patient prioritisation). "
            "Example: ['P003', 'P001', 'P005', 'P002', 'P004']"
        ),
    )
    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Agent's free-text clinical reasoning. Not used in grading but "
            "logged for analysis and improves interpretability."
        ),
    )

    model_config = ConfigDict(use_enum_values=False)


# ---------------------------------------------------------------------------
# Vital signs sub-model
# ---------------------------------------------------------------------------

class VitalSigns(BaseModel):
    """
    Standard emergency department vital signs.
    Reference ranges (adult) are provided in field descriptions.
    """

    heart_rate: int = Field(
        ...,
        ge=0, le=300,
        description="Heart rate in beats per minute. Normal: 60–100 bpm.",
    )
    systolic_bp: int = Field(
        ...,
        ge=0, le=300,
        description="Systolic blood pressure in mmHg. Normal: 90–140 mmHg.",
    )
    diastolic_bp: int = Field(
        ...,
        ge=0, le=200,
        description="Diastolic blood pressure in mmHg. Normal: 60–90 mmHg.",
    )
    respiratory_rate: int = Field(
        ...,
        ge=0, le=60,
        description="Respiratory rate in breaths per minute. Normal: 12–20.",
    )
    spo2: int = Field(
        ...,
        ge=0, le=100,
        description=(
            "Oxygen saturation (pulse oximetry) as a percentage. "
            "Normal: ≥95%. Below 90% is hypoxic emergency."
        ),
    )
    temperature: float = Field(
        ...,
        ge=30.0, le=45.0,
        description=(
            "Body temperature in Celsius. "
            "Normal: 36.1–37.2°C. Fever: ≥38.0°C. Hypothermia: <35°C."
        ),
    )
    gcs: int = Field(
        ...,
        ge=3, le=15,
        description=(
            "Glasgow Coma Scale (neurological status). "
            "15 = fully alert. ≤8 = severe impairment, intubation threshold."
        ),
    )
    pain_score: int = Field(
        ...,
        ge=0, le=10,
        description="Numeric pain score (0 = none, 10 = worst imaginable).",
    )


# ---------------------------------------------------------------------------
# Patient sub-model
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    """Full record for one patient present in the ED."""

    patient_id: str = Field(..., description="Unique patient identifier e.g. 'P001'.")
    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    sex: str = Field(..., description="Biological sex: 'M' or 'F'.")
    chief_complaint: str = Field(
        ...,
        description=(
            "Primary reason for ED visit in plain English. "
            "Example: 'crushing chest pain radiating to left arm'."
        ),
    )
    vitals: VitalSigns = Field(..., description="Current vital signs.")
    pmh: List[str] = Field(
        default_factory=list,
        description=(
            "Past medical history list. "
            "Example: ['hypertension', 'type-2 diabetes', 'COPD']."
        ),
    )
    medications: List[str] = Field(
        default_factory=list,
        description="Current medications. Example: ['metformin 500mg', 'lisinopril 10mg'].",
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="Known drug/food allergies. Example: ['penicillin'].",
    )
    arrival_mode: str = Field(
        default="walk-in",
        description="How patient arrived: 'walk-in', 'ambulance', 'helicopter', 'police'.",
    )
    time_in_ed_minutes: int = Field(
        default=0,
        ge=0,
        description="Minutes the patient has already been waiting in the ED.",
    )
    # Dynamic fields — updated each step in Task 3
    deteriorating: bool = Field(
        default=False,
        description=(
            "True if patient vitals are worsening this step. "
            "Triggers penalty if agent does not escalate within 2 steps."
        ),
    )
    assigned_esi: Optional[int] = Field(
        default=None,
        description="ESI level assigned by the agent this episode (1–5), if any.",
    )
    test_results: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Results of ordered diagnostic tests. "
            "Populated after ORDER_* actions. "
            "Example: {'ecg': 'ST-elevation in leads II, III, aVF', 'troponin': 'elevated 2.4'}."
        ),
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class MedTriageObservation(BaseModel):
    """
    Full observation returned after reset() and each step().

    The agent sees all patients, current rewards, available actions,
    and a natural-language clinical summary to guide reasoning.
    """

    # Episode metadata
    task_id: str = Field(
        ...,
        description=(
            "Active task: 'task1_single_patient', 'task2_multi_patient', "
            "or 'task3_dynamic_deterioration'."
        ),
    )
    step: int = Field(..., ge=0, description="Current step number within the episode (0-indexed).")
    max_steps: int = Field(..., description="Maximum steps allowed in this episode.")
    done: bool = Field(default=False, description="True if the episode has ended.")

    # Patients
    patients: List[PatientRecord] = Field(
        ...,
        description="All patients currently in the ED. Single-element list for Task 1.",
    )
    active_patient_id: Optional[str] = Field(
        default=None,
        description=(
            "For Task 1 and Task 3: the patient the agent should focus on. "
            "None for Task 2 (agent must assess all patients independently)."
        ),
    )

    # Per-step reward values
    step_reward: float = Field(
        default=0.0,
        description=(
            "Reward earned this step (−1.0 to +1.0). "
            "Partial credit for correct triage direction; penalties for missed deterioration."
        ),
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of step_reward over the entire episode so far.",
    )

    # Action space
    legal_actions: List[int] = Field(
        ...,
        description=(
            "List of valid TriageAction integer values at this step. "
            "Attempting an illegal action returns a small penalty and no state change."
        ),
    )

    # Context
    clinical_summary: str = Field(
        ...,
        description=(
            "Plain-English narrative summary of the current clinical situation. "
            "Written in ED nursing/physician shorthand. "
            "Example: '58M c/o substernal CP 10/10, diaphoretic, HR 110, BP 88/60 — "
            "appears acutely ill.'"
        ),
    )
    time_pressure_flag: bool = Field(
        default=False,
        description=(
            "True when one or more patients are in immediate danger and require "
            "urgent action within the next 1–2 steps to avoid a bad outcome."
        ),
    )
    resource_constraints: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Current ED resource availability. "
            "Example: {'trauma_bays': 1, 'icu_beds': 0, 'physicians_available': 2}."
        ),
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary information for debugging and logging.",
    )


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class MedTriageState(BaseModel):
    """
    Episode-level metadata returned by state().
    Tracks progress across all tasks.
    """

    episode_id: str = Field(..., description="Unique UUID for this episode.")
    step_count: int = Field(default=0, description="Steps taken so far.")
    task_id: str = Field(..., description="Which task is active.")
    task_description: str = Field(
        ...,
        description="Human-readable description of the current task objective.",
    )
    num_patients: int = Field(
        default=1,
        description="Number of patients in this episode.",
    )
    # Grader scores (updated at episode end)
    task_score: Optional[float] = Field(
        default=None,
        description=(
            "Final task score kept strictly inside the configured score range. "
            "None until the episode is complete."
        ),
    )
    # Tracking flags
    missed_deteriorations: int = Field(
        default=0,
        description="Number of deterioration events the agent failed to respond to in time.",
    )
    correct_esi_assignments: int = Field(
        default=0,
        description="Running count of ESI assignments matching ground truth.",
    )
    total_esi_assignments: int = Field(
        default=0,
        description="Total ESI assignments made this episode.",
    )
    # Config echo
    seed: Optional[int] = Field(
        default=None,
        description="Random seed used to generate this episode (for reproducibility).",
    )


# ---------------------------------------------------------------------------
# Step result wrapper (mirrors OpenEnv's StepResult)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Wrapper returned by step() and reset() — mirrors OpenEnv's StepResult."""

    observation: MedTriageObservation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)
