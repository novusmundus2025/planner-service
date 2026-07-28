# MundusX Planner Service

LangGraph-based planning service for the MundusX control-plane layer.

The Rust control plane remains the source of truth for auth, node registry, job state,
policy, scheduling, and persistence. This service owns AI workflow planning for complex
requests and returns a portable job graph that the Rust scheduler can execute.

## Responsibilities

- Classify request shape and required capabilities.
- Split complex work into parallel responsibility chunks.
- Create explicit reducer and final synthesizer steps for multi-node jobs.
- Return scheduler requirements such as role, modality, context, and tool needs.
- Fall back to deterministic planning when LangGraph is unavailable or disabled.

The LangGraph workflow has separate classification, decomposition, and graph
validation stages. The returned graph is portable: the Rust control plane
remains responsible for persistence and selects a contributor for every ready
step.

## API

### `POST /v1/plan`

Request:

```json
{
  "request_id": "job-123",
  "prompt": "Analyze this codebase and propose fixes.",
  "model": "optional-model-name",
  "classification": {},
  "available_capability_summary": {},
  "policy": {}
}
```

Response:

```json
{
  "request_id": "job-123",
  "planner_provider": "langgraph",
  "planner_status": "planned",
  "plan": {
    "summary": "Plan generated for coding request.",
    "steps": []
  },
  "graph": {
    "nodes": [],
    "edges": []
  },
  "scheduling_requirements": {}
}
```

If LangGraph is unavailable or times out, the service returns a deterministic fallback
plan with `planner_provider` set to `deterministic`.

## Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn mundusx_planner_service.api:app --reload --port 8091
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Railway Deployment

Railpack reads the repository-root `railpack.json`. It starts Uvicorn on
`0.0.0.0` and Railway's injected `PORT`, with `8091` as the local fallback.
The root `requirements.txt` instructs Railpack's pip provider to install the
local project and all dependencies declared in `pyproject.toml`.

## Control Plane Integration

The Rust control plane should call this service only for requests that benefit from
multi-step planning. If the call fails, times out, or returns `planner_status=degraded`,
Rust should use its built-in fallback planner and record planner metadata on the job.
