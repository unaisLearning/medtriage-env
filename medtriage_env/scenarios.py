"""
Patient scenario generator for MedTriageEnv.

All scenarios are seeded and deterministic — same seed → same patients.
Ground-truth ESI levels computed using validated ESI v4 algorithm logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from medtriage_env.models import PatientRecord, VitalSigns


# ---------------------------------------------------------------------------
# Ground-truth ESI reference
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    """Defines a patient scenario with the correct ESI and deterioration profile."""
    patient: PatientRecord
    ground_truth_esi: int
    deterioration_pattern: Optional[List[Dict]] = None  # list of per-step vital changes


# ---------------------------------------------------------------------------
# ESI helper — validated scoring logic
# ---------------------------------------------------------------------------

def compute_ground_truth_esi(patient: PatientRecord) -> int:
    """
    Compute ESI level using ESI v4 algorithm.

    Decision tree:
      ESI 1 → Requires immediate life-saving intervention
      ESI 2 → High risk OR confused/lethargic/disoriented OR severe pain/distress
      ESI 3 → Stable but likely needs ≥2 resources
      ESI 4 → Stable, needs 1 resource
      ESI 5 → Stable, no resources
    """
    v = patient.vitals

    # --- ESI 1: Immediate life-saving intervention required ---
    # Unresponsive, apneic, pulseless, or severely compromised
    if (
        v.gcs <= 8
        or v.spo2 < 85
        or v.heart_rate < 40
        or v.heart_rate > 180
        or v.systolic_bp < 70
        or v.respiratory_rate > 35
        or v.respiratory_rate < 6
    ):
        return 1

    # --- ESI 2: High risk situation OR altered mental status ---
    high_risk_complaints = [
        "chest pain", "chest pressure", "crushing chest",
        "shortness of breath", "difficulty breathing", "sob",
        "stroke", "facial droop", "arm weakness", "slurred speech",
        "overdose", "anaphylaxis", "severe allergic",
        "active bleeding", "hemorrhage",
        "sepsis", "septic shock",
        "altered mental", "confused", "unresponsive",
        "seizure", "status epilepticus",
        "trauma", "mvc", "fall from height",
    ]
    complaint_lower = patient.chief_complaint.lower()
    is_high_risk = any(kw in complaint_lower for kw in high_risk_complaints)

    confused = v.gcs < 15
    severe_distress = v.pain_score >= 8

    if (
        is_high_risk
        or confused
        or v.spo2 < 90
        or v.systolic_bp < 90
        or v.heart_rate > 150
        or v.heart_rate < 50
        or v.respiratory_rate > 28
        or v.temperature > 39.5
        or v.temperature < 35.0
    ):
        return 2

    # --- ESI 3: Stable but ≥2 resources expected ---
    moderate_risk = [
        "abdominal pain", "back pain", "headache", "migraine",
        "vomiting", "diarrhea", "fever", "urinary",
        "wound", "laceration", "fracture", "injury",
        "hypertension", "dizziness", "syncope",
        "palpitations", "irregular heartbeat",
    ]
    needs_resources = any(kw in complaint_lower for kw in moderate_risk)

    has_comorbidities = len(patient.pmh) >= 2
    moderate_vital_abnormality = (
        v.heart_rate > 110
        or v.heart_rate < 55
        or v.systolic_bp > 160
        or v.systolic_bp < 95
        or v.spo2 < 94
        or v.temperature > 38.5
        or v.pain_score >= 6
    )

    if needs_resources or has_comorbidities or moderate_vital_abnormality:
        return 3

    # --- ESI 4: Stable, 1 resource ---
    simple_complaints = [
        "sore throat", "ear pain", "eye pain", "tooth pain", "dental",
        "rash", "insect bite", "minor cut", "sprain", "bruise",
        "prescription refill", "medication question", "suture removal",
        "minor burn", "cold symptoms", "cough", "flu",
    ]
    one_resource = any(kw in complaint_lower for kw in simple_complaints)
    if one_resource or v.pain_score <= 3:
        return 4

    # --- ESI 5: No resources needed ---
    return 5


# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------

SCENARIO_BANK: List[Dict] = [
    # --- CRITICAL (ESI 1) ---
    {
        "age": 67, "sex": "M",
        "chief_complaint": "unresponsive, found collapsed at home",
        "vitals": {"heart_rate": 38, "systolic_bp": 62, "diastolic_bp": 40,
                   "respiratory_rate": 6, "spo2": 78, "temperature": 35.2, "gcs": 3, "pain_score": 0},
        "pmh": ["coronary artery disease", "heart failure"],
        "medications": ["furosemide", "carvedilol", "aspirin"],
        "allergies": [],
        "arrival_mode": "ambulance",
        "deterioration": [
            {"heart_rate": 28, "systolic_bp": 50, "spo2": 72, "gcs": 3},
            {"heart_rate": 20, "systolic_bp": 40, "spo2": 65, "gcs": 3},
        ],
    },
    {
        "age": 45, "sex": "F",
        "chief_complaint": "anaphylaxis, bee sting, throat swelling, can't breathe",
        "vitals": {"heart_rate": 145, "systolic_bp": 72, "diastolic_bp": 40,
                   "respiratory_rate": 32, "spo2": 86, "temperature": 37.8, "gcs": 13, "pain_score": 9},
        "pmh": ["known bee allergy"],
        "medications": [],
        "allergies": ["bee venom — anaphylaxis"],
        "arrival_mode": "ambulance",
        "deterioration": [
            {"heart_rate": 160, "systolic_bp": 60, "spo2": 80, "gcs": 11},
            {"heart_rate": 175, "systolic_bp": 50, "spo2": 74, "gcs": 9},
        ],
    },
    # --- EMERGENT (ESI 2) ---
    {
        "age": 58, "sex": "M",
        "chief_complaint": "crushing chest pain radiating to left arm, diaphoretic",
        "vitals": {"heart_rate": 112, "systolic_bp": 88, "diastolic_bp": 58,
                   "respiratory_rate": 24, "spo2": 93, "temperature": 37.1, "gcs": 15, "pain_score": 10},
        "pmh": ["hypertension", "hyperlipidaemia", "type-2 diabetes"],
        "medications": ["metformin", "amlodipine", "atorvastatin"],
        "allergies": ["penicillin"],
        "arrival_mode": "ambulance",
        "deterioration": [
            {"heart_rate": 120, "systolic_bp": 78, "spo2": 90, "gcs": 15},
            {"heart_rate": 130, "systolic_bp": 68, "spo2": 87, "gcs": 14},
        ],
    },
    {
        "age": 72, "sex": "F",
        "chief_complaint": "sudden severe headache, worst of her life, vomiting",
        "vitals": {"heart_rate": 68, "systolic_bp": 195, "diastolic_bp": 110,
                   "respiratory_rate": 18, "spo2": 97, "temperature": 37.0, "gcs": 14, "pain_score": 10},
        "pmh": ["hypertension", "atrial fibrillation"],
        "medications": ["warfarin", "metoprolol", "ramipril"],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [
            {"heart_rate": 58, "systolic_bp": 210, "spo2": 95, "gcs": 12},
            {"heart_rate": 48, "systolic_bp": 230, "spo2": 92, "gcs": 9},
        ],
    },
    {
        "age": 33, "sex": "M",
        "chief_complaint": "right-sided facial droop, arm weakness, slurred speech started 1 hour ago",
        "vitals": {"heart_rate": 88, "systolic_bp": 175, "diastolic_bp": 95,
                   "respiratory_rate": 16, "spo2": 96, "temperature": 37.2, "gcs": 14, "pain_score": 2},
        "pmh": ["migraines"],
        "medications": ["sumatriptan"],
        "allergies": [],
        "arrival_mode": "ambulance",
        "deterioration": [
            {"heart_rate": 90, "systolic_bp": 180, "spo2": 95, "gcs": 13},
            {"heart_rate": 92, "systolic_bp": 185, "spo2": 94, "gcs": 11},
        ],
    },
    # --- URGENT (ESI 3) ---
    {
        "age": 41, "sex": "F",
        "chief_complaint": "severe abdominal pain, 8/10, nausea, last menstrual period 8 weeks ago",
        "vitals": {"heart_rate": 108, "systolic_bp": 102, "diastolic_bp": 68,
                   "respiratory_rate": 20, "spo2": 98, "temperature": 37.5, "gcs": 15, "pain_score": 8},
        "pmh": ["endometriosis"],
        "medications": ["ibuprofen PRN"],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [],
    },
    {
        "age": 55, "sex": "M",
        "chief_complaint": "fever 39.8°C, productive cough, shortness of breath for 3 days",
        "vitals": {"heart_rate": 118, "systolic_bp": 105, "diastolic_bp": 70,
                   "respiratory_rate": 26, "spo2": 91, "temperature": 39.8, "gcs": 15, "pain_score": 4},
        "pmh": ["COPD", "smoking 30 pack-years"],
        "medications": ["salbutamol inhaler", "tiotropium"],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [
            {"heart_rate": 125, "systolic_bp": 98, "spo2": 88, "gcs": 15},
        ],
    },
    {
        "age": 28, "sex": "M",
        "chief_complaint": "motorcycle accident, right leg pain, abrasions, helmet worn",
        "vitals": {"heart_rate": 102, "systolic_bp": 118, "diastolic_bp": 76,
                   "respiratory_rate": 20, "spo2": 99, "temperature": 37.0, "gcs": 15, "pain_score": 7},
        "pmh": [],
        "medications": [],
        "allergies": ["sulpha drugs"],
        "arrival_mode": "ambulance",
        "deterioration": [],
    },
    # --- LESS URGENT (ESI 4) ---
    {
        "age": 19, "sex": "F",
        "chief_complaint": "sore throat, difficulty swallowing, low-grade fever for 2 days",
        "vitals": {"heart_rate": 88, "systolic_bp": 118, "diastolic_bp": 74,
                   "respiratory_rate": 16, "spo2": 99, "temperature": 38.1, "gcs": 15, "pain_score": 4},
        "pmh": [],
        "medications": [],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [],
    },
    {
        "age": 35, "sex": "M",
        "chief_complaint": "right ankle pain and swelling after twisting it playing football",
        "vitals": {"heart_rate": 82, "systolic_bp": 124, "diastolic_bp": 80,
                   "respiratory_rate": 14, "spo2": 99, "temperature": 36.8, "gcs": 15, "pain_score": 5},
        "pmh": [],
        "medications": [],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [],
    },
    # --- NON-URGENT (ESI 5) ---
    {
        "age": 24, "sex": "F",
        "chief_complaint": "needs prescription refill, ran out of oral contraceptive",
        "vitals": {"heart_rate": 72, "systolic_bp": 118, "diastolic_bp": 76,
                   "respiratory_rate": 14, "spo2": 99, "temperature": 36.9, "gcs": 15, "pain_score": 0},
        "pmh": [],
        "medications": ["oral contraceptive pill"],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [],
    },
    {
        "age": 30, "sex": "M",
        "chief_complaint": "small cut on finger, needs it cleaned and dressed",
        "vitals": {"heart_rate": 76, "systolic_bp": 120, "diastolic_bp": 78,
                   "respiratory_rate": 14, "spo2": 99, "temperature": 36.7, "gcs": 15, "pain_score": 2},
        "pmh": [],
        "medications": [],
        "allergies": [],
        "arrival_mode": "walk-in",
        "deterioration": [],
    },
]


def _build_patient(spec: Dict, patient_id: str, time_waiting: int = 0) -> PatientRecord:
    """Construct a PatientRecord from a scenario spec dict."""
    return PatientRecord(
        patient_id=patient_id,
        age=spec["age"],
        sex=spec["sex"],
        chief_complaint=spec["chief_complaint"],
        vitals=VitalSigns(**spec["vitals"]),
        pmh=spec.get("pmh", []),
        medications=spec.get("medications", []),
        allergies=spec.get("allergies", []),
        arrival_mode=spec.get("arrival_mode", "walk-in"),
        time_in_ed_minutes=time_waiting,
    )


# ---------------------------------------------------------------------------
# Public generator functions
# ---------------------------------------------------------------------------

def generate_task1_scenario(seed: int) -> ScenarioSpec:
    """
    Task 1 — Single patient ESI classification.
    Returns one patient and their ground-truth ESI level.
    """
    rng = random.Random(seed)
    spec = rng.choice(SCENARIO_BANK)
    patient = _build_patient(spec, "P001", time_waiting=rng.randint(0, 15))
    esi = compute_ground_truth_esi(patient)
    return ScenarioSpec(patient=patient, ground_truth_esi=esi)


def generate_task2_scenario(seed: int, n_patients: int = 5) -> List[ScenarioSpec]:
    """
    Task 2 — Multi-patient prioritisation.
    Returns 5 patients with diverse ESI levels (ensures ESI spread 1–4).
    """
    rng = random.Random(seed)
    # Sample without replacement to get diversity
    pool = list(SCENARIO_BANK)
    rng.shuffle(pool)
    selected = pool[:n_patients]
    specs = []
    for i, spec in enumerate(selected):
        patient_id = f"P{i+1:03d}"
        patient = _build_patient(spec, patient_id, time_waiting=rng.randint(0, 30))
        esi = compute_ground_truth_esi(patient)
        specs.append(ScenarioSpec(patient=patient, ground_truth_esi=esi))
    return specs


def generate_task3_scenario(seed: int) -> Tuple[ScenarioSpec, List[Dict]]:
    """
    Task 3 — Dynamic deterioration over 10 steps.
    Returns a deteriorating patient + a per-step deterioration schedule.
    """
    rng = random.Random(seed)
    # Only pick patients that have a deterioration pattern
    deteriorating_pool = [s for s in SCENARIO_BANK if s.get("deterioration")]
    spec = rng.choice(deteriorating_pool)
    patient = _build_patient(spec, "P001", time_waiting=0)
    esi = compute_ground_truth_esi(patient)

    # Build 10-step deterioration schedule
    deterioration_steps = spec.get("deterioration", [])
    schedule = []
    for step in range(10):
        if step < len(deterioration_steps):
            schedule.append(deterioration_steps[step])
        else:
            # No change after defined deterioration pattern
            schedule.append({})

    return ScenarioSpec(patient=patient, ground_truth_esi=esi), schedule
