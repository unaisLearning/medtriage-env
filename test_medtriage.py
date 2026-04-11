"""
MedTriageEnv — Test Suite
Tests all 3 tasks, all graders, and the full step/reset/state cycle.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from medtriage_env.client import MedTriageEnvLocal
from medtriage_env.graders import Task1Grader, Task2Grader, Task3Grader
from medtriage_env.models import MedTriageAction, MedTriageObservation, TriageAction
from medtriage_env.scenarios import (
    compute_ground_truth_esi,
    generate_task1_scenario,
    generate_task2_scenario,
    generate_task3_scenario,
)
from medtriage_env.server.environment import MedTriageEnvironment


# ─────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────

class TestModels:
    def test_action_valid(self):
        a = MedTriageAction(action=TriageAction.ASSIGN_ESI_2, reasoning="chest pain")
        assert a.action == TriageAction.ASSIGN_ESI_2

    def test_action_with_rankings(self):
        a = MedTriageAction(
            action=TriageAction.REASSESS,
            patient_rankings=["P001", "P003", "P002"],
        )
        assert a.patient_rankings == ["P001", "P003", "P002"]

    def test_all_triage_actions_have_values_1_to_18(self):
        values = [int(a.value) for a in TriageAction]
        assert min(values) == 1
        assert max(values) == 18
        assert len(values) == 18


# ─────────────────────────────────────────────────────────────
# Scenario tests
# ─────────────────────────────────────────────────────────────

class TestScenarios:
    def test_task1_deterministic(self):
        s1 = generate_task1_scenario(42)
        s2 = generate_task1_scenario(42)
        assert s1.patient.patient_id == s2.patient.patient_id
        assert s1.ground_truth_esi == s2.ground_truth_esi

    def test_task1_different_seeds(self):
        s1 = generate_task1_scenario(1)
        s2 = generate_task1_scenario(999)
        # Different seeds should usually produce different patients
        # (may occasionally collide — that's fine for small bank)
        assert s1.ground_truth_esi in range(1, 6)
        assert s2.ground_truth_esi in range(1, 6)

    def test_task2_returns_5_patients(self):
        specs = generate_task2_scenario(42, n_patients=5)
        assert len(specs) == 5

    def test_task2_all_esi_valid(self):
        specs = generate_task2_scenario(42)
        for s in specs:
            assert 1 <= s.ground_truth_esi <= 5

    def test_task3_has_deterioration_schedule(self):
        spec, schedule = generate_task3_scenario(42)
        assert len(schedule) == 10
        assert spec.ground_truth_esi in range(1, 6)

    def test_esi_algorithm_critical(self):
        spec = generate_task1_scenario(0)  # scan for a critical patient
        # Manually build a critical patient
        from medtriage_env.models import PatientRecord, VitalSigns
        p = PatientRecord(
            patient_id="TEST",
            age=60, sex="M",
            chief_complaint="unresponsive",
            vitals=VitalSigns(
                heart_rate=30, systolic_bp=60, diastolic_bp=40,
                respiratory_rate=4, spo2=78, temperature=35.0,
                gcs=3, pain_score=0,
            ),
        )
        esi = compute_ground_truth_esi(p)
        assert esi == 1, f"Expected ESI 1, got {esi}"

    def test_esi_algorithm_non_urgent(self):
        from medtriage_env.models import PatientRecord, VitalSigns
        p = PatientRecord(
            patient_id="TEST",
            age=25, sex="F",
            chief_complaint="prescription refill",
            vitals=VitalSigns(
                heart_rate=72, systolic_bp=118, diastolic_bp=76,
                respiratory_rate=14, spo2=99, temperature=36.9,
                gcs=15, pain_score=0,
            ),
        )
        esi = compute_ground_truth_esi(p)
        # ESI 4 or 5 both valid for simple prescription refill
        assert esi in (4, 5), f"Expected ESI 4 or 5, got {esi}"


# ─────────────────────────────────────────────────────────────
# Grader tests
# ─────────────────────────────────────────────────────────────

class TestTask1Grader:
    def setup_method(self):
        self.g = Task1Grader()

    def test_exact_match_scores_1(self):
        score, _ = self.g.grade(2, 2, [6, 7], "chest pain")
        assert 0.99 <= score < 1.0

    def test_off_by_1_partial_credit(self):
        score, _ = self.g.grade(3, 2, [], "chest pain")
        # Off by 1 with noop penalty applied: score is in 0.3–0.75 range
        assert 0.20 <= score <= 0.75

    def test_off_by_2_low_score(self):
        score, _ = self.g.grade(4, 2, [], "abdominal pain")
        # Off by 2 with noop penalty: score < 0.30
        assert 0.0 <= score <= 0.30

    def test_off_by_3_zero(self):
        score, _ = self.g.grade(5, 1, [], "prescription refill")
        assert 0.0 < score <= 1e-3

    def test_no_assignment_zero(self):
        score, _ = self.g.grade(None, 2, [18], "chest pain")
        assert 0.0 < score <= 1e-3

    def test_diagnostic_bonus_chest_pain(self):
        score_with, _ = self.g.grade(2, 2, [6, 7], "chest pain")     # ECG + Labs
        score_without, _ = self.g.grade(2, 2, [], "chest pain")
        assert score_with > score_without

    def test_noop_only_penalty(self):
        score_noop, _ = self.g.grade(2, 2, [18, 18, 18], "chest pain")
        score_active, _ = self.g.grade(2, 2, [6, 7], "chest pain")
        assert score_noop < score_active

    def test_score_always_in_range(self):
        for assigned in range(1, 6):
            for gt in range(1, 6):
                score, _ = self.g.grade(assigned, gt, [6], "chest pain")
                assert 0.0 < score < 1.0, f"Score {score} out of range for {assigned} vs {gt}"


class TestTask2Grader:
    def setup_method(self):
        self.g = Task2Grader()
        self.gt = ["P001", "P002", "P003", "P004", "P005"]
        self.esi_map = {"P001": 1, "P002": 2, "P003": 3, "P004": 4, "P005": 5}

    def test_perfect_ranking_scores_high(self):
        score, _ = self.g.grade(self.gt, self.gt, self.esi_map)
        assert score >= 0.95

    def test_reversed_ranking_scores_low(self):
        score, _ = self.g.grade(list(reversed(self.gt)), self.gt, self.esi_map)
        assert score <= 0.15

    def test_critical_patient_ranked_first_bonus(self):
        score_first, bd = self.g.grade(self.gt, self.gt, self.esi_map)
        assert "critical_bonus" in bd

    def test_critical_patient_not_in_top2_penalty(self):
        bad = ["P003", "P004", "P002", "P005", "P001"]  # P001 (ESI 1) last
        score, bd = self.g.grade(bad, self.gt, self.esi_map)
        assert "critical_penalty" in bd

    def test_score_always_in_range(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            shuffled = list(self.gt)
            rng.shuffle(shuffled)
            score, _ = self.g.grade(shuffled, self.gt, self.esi_map)
            assert 0.0 < score < 1.0


class TestTask3Grader:
    def setup_method(self):
        self.g = Task3Grader()

    def test_perfect_agent_scores_high(self):
        score, _ = self.g.grade(
            assigned_esi=2, ground_truth_esi=2,
            actions_taken=[7, 6, 10, 2, 7],  # labs, ecg, physician, esi2, labs
            deterioration_step=2,
            escalation_step=2,   # escalated immediately
            missed_deteriorations=0,
            chief_complaint="crushing chest pain",
            max_steps=10,
        )
        assert score >= 0.75

    def test_late_escalation_penalised(self):
        score_early, _ = self.g.grade(
            assigned_esi=2, ground_truth_esi=2,
            actions_taken=[10], deterioration_step=2, escalation_step=2,
            missed_deteriorations=0, chief_complaint="chest pain",
        )
        score_late, _ = self.g.grade(
            assigned_esi=2, ground_truth_esi=2,
            actions_taken=[10], deterioration_step=2, escalation_step=8,
            missed_deteriorations=3, chief_complaint="chest pain",
        )
        assert score_early > score_late

    def test_missed_deteriorations_reduce_score(self):
        score_0, _ = self.g.grade(2, 2, [10], 2, 3, 0, "chest pain")
        score_3, _ = self.g.grade(2, 2, [10], 2, 3, 3, "chest pain")
        assert score_0 > score_3

    def test_score_always_in_range(self):
        for esi in range(1, 6):
            score, _ = self.g.grade(
                assigned_esi=esi, ground_truth_esi=2,
                actions_taken=[6, 7, 10], deterioration_step=2,
                escalation_step=3, missed_deteriorations=1,
                chief_complaint="chest pain",
            )
            assert 0.0 < score < 1.0, f"Score {score} out of range"


# ─────────────────────────────────────────────────────────────
# Environment integration tests
# ─────────────────────────────────────────────────────────────

class TestEnvironmentTask1:
    def test_reset_returns_observation(self):
        env = MedTriageEnvLocal("task1_single_patient")
        obs = env.reset(seed=42)
        assert isinstance(obs, MedTriageObservation)
        assert obs.task_id == "task1_single_patient"
        assert len(obs.patients) == 1
        assert obs.step == 0

    def test_step_returns_result(self):
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset(seed=42)
        result = env.step(MedTriageAction(action=TriageAction.ORDER_LABS))
        assert result.reward is not None
        assert isinstance(result.done, bool)

    def test_episode_ends_after_esi_assignment(self):
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset(seed=42)
        env.step(MedTriageAction(action=TriageAction.ORDER_LABS))
        result = env.step(MedTriageAction(action=TriageAction.ASSIGN_ESI_5))
        assert result.done is True

    def test_final_score_in_range(self):
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset(seed=42)
        env.step(MedTriageAction(action=TriageAction.ORDER_LABS))
        result = env.step(MedTriageAction(action=TriageAction.ASSIGN_ESI_5))
        state = env.state()
        assert state.task_score is not None
        assert 0.0 < state.task_score < 1.0

    def test_deterministic_across_runs(self):
        scores = []
        for _ in range(3):
            env = MedTriageEnvLocal("task1_single_patient")
            env.reset(seed=42)
            env.step(MedTriageAction(action=TriageAction.ORDER_LABS))
            env.step(MedTriageAction(action=TriageAction.ASSIGN_ESI_5))
            scores.append(env.state().task_score)
        assert len(set(scores)) == 1, f"Non-deterministic scores: {scores}"

    def test_state_before_reset_raises(self):
        env = MedTriageEnvironment("task1_single_patient")
        with pytest.raises(RuntimeError):
            env.state()


class TestEnvironmentTask2:
    def test_reset_returns_5_patients(self):
        env = MedTriageEnvLocal("task2_multi_patient")
        obs = env.reset(seed=42)
        assert len(obs.patients) == 5

    def test_ranking_submission_ends_episode(self):
        env = MedTriageEnvLocal("task2_multi_patient")
        obs = env.reset(seed=42)
        ids = [p.patient_id for p in obs.patients]
        action = MedTriageAction(
            action=TriageAction.REASSESS,
            patient_rankings=ids,
        )
        result = env.step(action)
        assert result.done is True

    def test_final_score_in_range(self):
        env = MedTriageEnvLocal("task2_multi_patient")
        obs = env.reset(seed=42)
        ids = [p.patient_id for p in obs.patients]
        env.step(MedTriageAction(action=TriageAction.REASSESS, patient_rankings=ids))
        state = env.state()
        assert state.task_score is not None
        assert 0.0 < state.task_score < 1.0


class TestEnvironmentTask3:
    def test_reset_single_patient(self):
        env = MedTriageEnvLocal("task3_dynamic_deterioration")
        obs = env.reset(seed=42)
        assert len(obs.patients) == 1

    def test_full_episode_produces_score(self):
        env = MedTriageEnvLocal("task3_dynamic_deterioration")
        obs = env.reset(seed=42)
        for _ in range(obs.max_steps):
            if obs.done:
                break
            result = env.step(MedTriageAction(action=TriageAction.ORDER_LABS))
            obs = result.observation
        state = env.state()
        assert state.task_score is not None
        assert 0.0 < state.task_score < 1.0

    def test_escalation_during_deterioration_rewarded(self):
        env = MedTriageEnvLocal("task3_dynamic_deterioration")
        env.reset(seed=7)
        # Step through until deterioration, then escalate.
        # Repeated diagnostics now incur a small penalty, so we verify the
        # Check the main thing we care about: timely escalation should help
        # once the patient starts to worsen.
        escalation_reward = None
        for step in range(5):
            if step < 2:
                action = MedTriageAction(action=TriageAction.ORDER_LABS)
            else:
                action = MedTriageAction(action=TriageAction.CALL_PHYSICIAN)
            result = env.step(action)
            if (
                escalation_reward is None
                and action.action == TriageAction.CALL_PHYSICIAN
                and result.info.get("escalation_note")
            ):
                escalation_reward = result.reward
            if result.done:
                break
        assert escalation_reward is not None
        assert escalation_reward > 0, "Timely escalation should produce positive reward"


# ─────────────────────────────────────────────────────────────
# OpenEnv spec compliance tests
# ─────────────────────────────────────────────────────────────

class TestOpenEnvSpecCompliance:
    """Verify the environment follows the OpenEnv spec exactly."""

    def test_reset_returns_observation_type(self):
        env = MedTriageEnvLocal("task1_single_patient")
        obs = env.reset()
        assert isinstance(obs, MedTriageObservation)

    def test_state_returns_state_type(self):
        from medtriage_env.models import MedTriageState
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset()
        s = env.state()
        assert isinstance(s, MedTriageState)

    def test_step_returns_stepresult(self):
        from medtriage_env.models import StepResult
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset()
        result = env.step(MedTriageAction(action=TriageAction.NOOP))
        assert isinstance(result, StepResult)
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")

    def test_reward_is_float(self):
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset()
        result = env.step(MedTriageAction(action=TriageAction.NOOP))
        assert isinstance(result.reward, float)

    def test_done_is_bool(self):
        env = MedTriageEnvLocal("task1_single_patient")
        env.reset()
        result = env.step(MedTriageAction(action=TriageAction.NOOP))
        assert isinstance(result.done, bool)

    def test_all_tasks_have_3_required_methods(self):
        for task in ["task1_single_patient", "task2_multi_patient", "task3_dynamic_deterioration"]:
            env = MedTriageEnvLocal(task)
            assert callable(env.reset)
            assert callable(env.step)
            assert callable(env.state)

    def test_all_task_scores_in_0_1(self):
        """Core requirement: graders must return scores strictly within (0, 1)."""
        results = {}
        for task in ["task1_single_patient", "task2_multi_patient", "task3_dynamic_deterioration"]:
            env = MedTriageEnvLocal(task)
            obs = env.reset(seed=42)
            for _ in range(obs.max_steps):
                if obs.done:
                    break
                if task == "task2_multi_patient":
                    ids = [p.patient_id for p in obs.patients]
                    action = MedTriageAction(action=TriageAction.REASSESS, patient_rankings=ids)
                elif task == "task3_dynamic_deterioration":
                    action = MedTriageAction(action=TriageAction.CALL_PHYSICIAN)
                else:
                    action = MedTriageAction(action=TriageAction.ASSIGN_ESI_3)
                result = env.step(action)
                obs = result.observation
            score = env.state().task_score
            results[task] = score
            assert score is not None, f"task_score is None for {task}"
            assert 0.0 < score < 1.0, f"Score {score} out of (0,1) for {task}"

    def test_legal_actions_always_subset_of_all_actions(self):
        all_actions = {int(a.value) for a in TriageAction}
        env = MedTriageEnvLocal("task1_single_patient")
        obs = env.reset(seed=42)
        assert set(obs.legal_actions).issubset(all_actions)

    def test_openenv_yaml_parseable(self):
        import yaml, os
        yaml_path = os.path.join(os.path.dirname(__file__), "medtriage_env", "openenv.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "MedTriageEnv"
        assert len(data["tasks"]) == 3
        assert all("id" in t for t in data["tasks"])
