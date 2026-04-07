"""
MedTriageEnv — HTTP Client

Mirrors the OpenEnv HTTPEnvClient pattern.
Agents import this and call reset() / step() / state() — never touching HTTP directly.

Usage:
    from medtriage_env.client import MedTriageEnv
    from medtriage_env.models import MedTriageAction, TriageAction

    env = MedTriageEnv(base_url="http://localhost:8000")
    obs = env.reset(seed=42, task_id="task1_single_patient")
    result = env.step(MedTriageAction(action=TriageAction.ASSIGN_ESI_2))
    state = env.state()
    env.close()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from medtriage_env.models import (
    MedTriageAction,
    MedTriageObservation,
    MedTriageState,
    StepResult,
)


class MedTriageEnv:
    """
    HTTP client for MedTriageEnv.
    Wraps the FastAPI server with a clean Python interface.
    """

    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        task_id: str = "task1_single_patient",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        self.timeout = timeout
        self._session = requests.Session()
        self._last_obs: Optional[MedTriageObservation] = None

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> MedTriageObservation:
        """Reset the environment. Returns initial observation."""
        payload: Dict[str, Any] = {
            "task_id": task_id or self.task_id,
        }
        if seed is not None:
            payload["seed"] = seed

        resp = self._post("/reset", payload)
        obs = MedTriageObservation(**resp)
        self._last_obs = obs
        return obs

    def step(self, action: MedTriageAction) -> StepResult:
        """Execute one action. Returns StepResult(observation, reward, done, info)."""
        action_val = (
            int(action.action.value)
            if hasattr(action.action, "value")
            else int(action.action)
        )
        payload: Dict[str, Any] = {
            "action": action_val,
            "task_id": self.task_id,
        }
        if action.target_patient_id is not None:
            payload["target_patient_id"] = action.target_patient_id
        if action.patient_rankings is not None:
            payload["patient_rankings"] = action.patient_rankings
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning

        resp = self._post("/step", payload)
        obs = MedTriageObservation(**resp["observation"])
        result = StepResult(
            observation=obs,
            reward=resp.get("reward", 0.0),
            done=resp.get("done", False),
            info=resp.get("info", {}),
        )
        self._last_obs = obs
        return result

    def state(self) -> MedTriageState:
        """Return current episode state and metadata."""
        resp = self._get("/state", params={"task_id": self.task_id})
        return MedTriageState(**resp)

    def close(self) -> None:
        """Clean up HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, str]:
        return self._get("/health")

    def list_tasks(self) -> List[Dict]:
        return self._get("/tasks")["tasks"]

    def list_actions(self) -> List[Dict]:
        return self._get("/actions")["actions"]

    def wait_for_ready(self, max_retries: int = 30, delay: float = 1.0) -> bool:
        """Poll /health until server is up. Useful after docker start."""
        for attempt in range(max_retries):
            try:
                resp = self._get("/health")
                if resp.get("status") == "ok":
                    return True
            except Exception:
                pass
            time.sleep(delay)
        return False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MedTriageEnv":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to MedTriageEnv at {self.base_url}. "
                "Is the server running? Start it with: uvicorn medtriage_env.server.app:app"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Server returned error {resp.status_code}: {resp.text}"
            ) from e

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to MedTriageEnv at {self.base_url}."
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Server returned error {resp.status_code}: {resp.text}"
            ) from e


# ---------------------------------------------------------------------------
# Embedded local client (no HTTP server needed — for inference.py)
# ---------------------------------------------------------------------------

class MedTriageEnvLocal:
    """
    Direct in-process client — no HTTP overhead.
    Used in inference.py so it runs fast within the 20-min limit.
    Exposes the same interface as MedTriageEnv.
    """

    def __init__(self, task_id: str = "task1_single_patient"):
        from medtriage_env.server.environment import MedTriageEnvironment
        self._env = MedTriageEnvironment(task_id=task_id)
        self.task_id = task_id

    def reset(self, seed: Optional[int] = None, task_id: Optional[str] = None) -> MedTriageObservation:
        if task_id and task_id != self.task_id:
            from medtriage_env.server.environment import MedTriageEnvironment
            self._env = MedTriageEnvironment(task_id=task_id)
            self.task_id = task_id
        return self._env.reset(seed=seed)

    def step(self, action: MedTriageAction) -> StepResult:
        return self._env.step(action)

    def state(self) -> MedTriageState:
        return self._env.state()

    def close(self) -> None:
        pass

    def __enter__(self) -> "MedTriageEnvLocal":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
