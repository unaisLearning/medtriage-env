"""
MedTriageEnv — FastAPI Server
Exposes reset(), step(), state() over HTTP following the OpenEnv spec.

Endpoints:
  POST /reset        → MedTriageObservation
  POST /step         → StepResult
  GET  /state        → MedTriageState
  GET  /health       → {"status": "ok"}
  GET  /tasks        → list of available tasks
  GET  /             → web UI (Gradio-style status page)

Task selection via MEDTRIAGE_TASK env var or query param.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Path hack for both in-repo and standalone container use
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")

from medtriage_env.models import (
    MedTriageAction,
    MedTriageObservation,
    MedTriageState,
    StepResult,
    TriageAction,
)
from medtriage_env.server.environment import MedTriageEnvironment

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedTriageEnv",
    description=(
        "Emergency Department Triage Simulator — OpenEnv compatible. "
        "The agent triages patients, orders diagnostics, and responds "
        "to deterioration. 3 tasks: easy, medium, hard."
    ),
    version="1.0.0",
)

# Active task from environment variable (default: task1)
DEFAULT_TASK = os.getenv("MEDTRIAGE_TASK", "task1_single_patient")

VALID_TASKS = [
    "task1_single_patient",
    "task2_multi_patient",
    "task3_dynamic_deterioration",
]

# Global environment instance (single-session; fine for evaluation)
_env: Optional[MedTriageEnvironment] = None


def get_env(task_id: Optional[str] = None) -> MedTriageEnvironment:
    global _env
    task = task_id or DEFAULT_TASK
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Valid tasks: {VALID_TASKS}",
        )
    if _env is None or _env._task_id != task:
        _env = MedTriageEnvironment(task_id=task)
    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: int
    target_patient_id: Optional[str] = None
    patient_rankings: Optional[List[str]] = None
    reasoning: Optional[str] = None
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check for the app."""
    return {"status": "ok", "environment": "MedTriageEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "task1_single_patient",
                "difficulty": "easy",
                "max_steps": 5,
                "description": (
                    "Assess a single ED patient and assign the correct ESI triage level (1–5). "
                    "Order appropriate diagnostics. Grader uses partial credit by ESI distance."
                ),
            },
            {
                "id": "task2_multi_patient",
                "difficulty": "medium",
                "max_steps": 8,
                "description": (
                    "Five patients arrive simultaneously. Rank them from most urgent to least urgent. "
                    "Grader uses Kendall's tau-b rank correlation with the final score kept inside the task range."
                ),
            },
            {
                "id": "task3_dynamic_deterioration",
                "difficulty": "hard",
                "max_steps": 10,
                "description": (
                    "Manage a single patient whose vitals deteriorate over 10 steps. "
                    "Order diagnostics, escalate care, and reassess. "
                    "Grader weights ESI accuracy (30%) + escalation timeliness (35%) "
                    "+ diagnostic coverage (20%) + efficiency (15%)."
                ),
            },
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """
    Reset the environment and return the initial observation.
    Optionally specify task_id and seed for reproducibility.
    """
    env = get_env(request.task_id)
    obs = env.reset(seed=request.seed)
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """
    Execute one action and return observation, reward, done, info.
    action: integer value from TriageAction enum (1–18).
    """
    env = get_env(request.task_id)

    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first.",
        )

    # Validate action
    valid_action_values = [int(a.value) for a in TriageAction]
    if request.action not in valid_action_values:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid action {request.action}. "
                f"Valid values: {valid_action_values}"
            ),
        )

    # Build typed action
    action = MedTriageAction(
        action=TriageAction(request.action),
        target_patient_id=request.target_patient_id,
        patient_rankings=request.patient_rankings,
        reasoning=request.reasoning,
    )

    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state(task_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """Return current episode state and metadata."""
    env = get_env(task_id)
    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first.",
        )
    return env.state().model_dump()


@app.get("/actions")
def list_actions() -> Dict[str, Any]:
    """Return all valid action IDs and their names."""
    return {
        "actions": [
            {"id": int(a.value), "name": a.name}
            for a in TriageAction
        ]
    }


# ---------------------------------------------------------------------------
# Web UI — renders in HF Spaces browser
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    """Simple status + documentation page shown in HF Space."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MedTriageEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8f9fa; color: #212529; padding: 2rem; }
  .container { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: #6c757d; margin-bottom: 2rem; }
  .card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
          padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;
              color: #495057; text-transform: uppercase; letter-spacing: 0.05em; }
  .badge { display: inline-block; font-size: 0.75rem; font-weight: 500;
           padding: 0.2rem 0.6rem; border-radius: 4px; margin-right: 0.5rem; }
  .easy   { background: #d1fae5; color: #065f46; }
  .medium { background: #fef3c7; color: #92400e; }
  .hard   { background: #fee2e2; color: #991b1b; }
  .endpoint { font-family: monospace; background: #f1f3f5;
              padding: 0.2rem 0.5rem; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th { text-align: left; padding: 0.5rem; border-bottom: 2px solid #dee2e6;
       color: #6c757d; font-weight: 500; }
  td { padding: 0.5rem; border-bottom: 1px solid #f1f3f5; }
  .tag { font-family: monospace; font-size: 0.82rem; color: #495057; }
  a { color: #2563eb; text-decoration: none; }
</style>
</head>
<body>
<div class="container">
  <h1>MedTriageEnv</h1>
  <p class="subtitle">Emergency Department Triage Simulator &mdash; OpenEnv compatible RL environment</p>

  <div class="card">
    <h2>Status</h2>
    <p>Server is <strong style="color:#065f46">running</strong>. 3 tasks available.</p>
  </div>

  <div class="card">
    <h2>Tasks</h2>
    <table>
      <tr>
        <th>Task ID</th><th>Difficulty</th><th>Max Steps</th><th>Description</th>
      </tr>
      <tr>
        <td class="tag">task1_single_patient</td>
        <td><span class="badge easy">Easy</span></td>
        <td>5</td>
        <td>Assign ESI level to one patient. Order diagnostics.</td>
      </tr>
      <tr>
        <td class="tag">task2_multi_patient</td>
        <td><span class="badge medium">Medium</span></td>
        <td>8</td>
        <td>Rank 5 simultaneous patients by urgency.</td>
      </tr>
      <tr>
        <td class="tag">task3_dynamic_deterioration</td>
        <td><span class="badge hard">Hard</span></td>
        <td>10</td>
        <td>Manage a deteriorating patient over 10 steps.</td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h2>API Endpoints</h2>
    <table>
      <tr><th>Method</th><th>Path</th><th>Description</th></tr>
      <tr><td>GET</td><td><span class="endpoint">/health</span></td><td>Health check</td></tr>
      <tr><td>POST</td><td><span class="endpoint">/reset</span></td><td>Start new episode</td></tr>
      <tr><td>POST</td><td><span class="endpoint">/step</span></td><td>Execute action</td></tr>
      <tr><td>GET</td><td><span class="endpoint">/state</span></td><td>Episode metadata</td></tr>
      <tr><td>GET</td><td><span class="endpoint">/tasks</span></td><td>List all tasks</td></tr>
      <tr><td>GET</td><td><span class="endpoint">/actions</span></td><td>List all actions</td></tr>
      <tr><td>GET</td><td><span class="endpoint">/docs</span></td><td>Interactive API docs (Swagger)</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Quick start</h2>
    <pre style="background:#f1f3f5;padding:1rem;border-radius:6px;overflow-x:auto;font-size:0.85rem">
# Reset environment (Task 1)
curl -X POST /reset -H "Content-Type: application/json" \\
  -d '{"task_id": "task1_single_patient", "seed": 42}'

# Assign ESI level 2 (emergent)
curl -X POST /step -H "Content-Type: application/json" \\
  -d '{"action": 2, "reasoning": "Chest pain with hemodynamic instability"}'

# Get state
curl /state
    </pre>
  </div>

  <div class="card">
    <h2>Links</h2>
    <p>
      <a href="/docs">Swagger UI</a> &nbsp;&bull;&nbsp;
      <a href="/tasks">Tasks JSON</a> &nbsp;&bull;&nbsp;
      <a href="/actions">Actions JSON</a> &nbsp;&bull;&nbsp;
      <a href="/health">Health</a>
    </p>
  </div>
</div>
</body>
</html>
"""


def main() -> None:
    import uvicorn

    uvicorn.run(
        "medtriage_env.server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
