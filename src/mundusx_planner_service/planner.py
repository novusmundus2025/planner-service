from __future__ import annotations

import os
from collections.abc import Iterable

from .models import (
    NodeRole,
    PlanRequest,
    PlanResponse,
    PlannerProvider,
    PlannerStatus,
    PlanStep,
    TaskType,
)


def _contains_any(value: str, needles: Iterable[str]) -> bool:
    return any(needle in value for needle in needles)


def classify_task(prompt: str) -> TaskType:
    normalized = prompt.lower()
    if _contains_any(
        normalized,
        ["code", "bug", "refactor", "compile", "test", "java", "rust", "python", "typescript"],
    ):
        return TaskType.CODING
    if _contains_any(normalized, ["pdf", "document", "transcript", "summarize", "contract"]):
        return TaskType.DOCUMENT
    if _contains_any(normalized, ["classify", "predict", "infer", "extract"]):
        return TaskType.INFERENCE
    return TaskType.CHAT


def infer_requirements(request: PlanRequest, task_type: TaskType) -> dict[str, object]:
    prompt = request.prompt.lower()
    requires_vision = _contains_any(
        prompt,
        ["image", "photo", "screenshot", "diagram", "ocr", "video", "frame", "vision"],
    )
    requires_embeddings = _contains_any(
        prompt,
        ["embedding", "embeddings", "vector", "semantic search", "similarity", "rag"],
    )
    requires_tools = _contains_any(
        prompt,
        ["tool", "tools", "function call", "browser", "terminal", "shell", "execute"],
    )

    role = {
        TaskType.CHAT: NodeRole.CHAT,
        TaskType.CODING: NodeRole.CODING,
        TaskType.DOCUMENT: NodeRole.BATCH,
        TaskType.INFERENCE: NodeRole.BATCH,
    }[task_type]
    preferred_roles = {role}
    if requires_vision:
        preferred_roles.add(NodeRole.VISION)
    if requires_embeddings:
        preferred_roles.add(NodeRole.EMBEDDING)
    if requires_tools:
        preferred_roles.add(NodeRole.TOOL_USE)

    desired_context_tokens = max(2048, len(request.prompt) // 4)
    if len(request.prompt) > 16_000:
        desired_context_tokens = max(desired_context_tokens, 8192)
    elif len(request.prompt) > 4_000:
        desired_context_tokens = max(desired_context_tokens, 4096)

    return {
        "task_type": task_type.value,
        "model": request.model,
        "desired_context_tokens": desired_context_tokens,
        "requires_vision": requires_vision,
        "requires_embeddings": requires_embeddings,
        "requires_tools": requires_tools,
        "preferred_roles": sorted(role.value for role in preferred_roles),
    }


def deterministic_plan(request: PlanRequest, degraded_reason: str | None = None) -> PlanResponse:
    task_type = classify_task(request.prompt)
    requirements = infer_requirements(request, task_type)
    complex_request = len(request.prompt) > 2000 or task_type in {TaskType.CODING, TaskType.DOCUMENT}

    if complex_request:
        steps = [
            PlanStep(
                id="scope",
                name="Scope request",
                responsibility="Identify the required work, constraints, and expected output.",
                required_output="A concise scope summary and acceptance checklist.",
                reason="Complex requests need a bounded scope before distributed execution.",
                preferred_roles=[NodeRole.BATCH],
            ),
            PlanStep(
                id="execute",
                name="Execute work",
                responsibility="Perform the primary implementation or analysis task.",
                required_output="Task result with enough detail for verification.",
                reason="The main task can be assigned to the best matching worker node.",
                depends_on=["scope"],
                preferred_roles=[NodeRole(requirements["preferred_roles"][0])],
            ),
            PlanStep(
                id="reduce",
                name="Reduce final answer",
                responsibility="Merge outputs, remove duplication, and produce final response.",
                required_output="Final answer ready for the requesting client.",
                reason="Reducer step keeps the final response coherent after chunked work.",
                depends_on=["execute"],
                preferred_roles=[NodeRole.REDUCER],
            ),
        ]
    else:
        steps = [
            PlanStep(
                id="execute",
                name="Execute request",
                responsibility="Answer the request directly.",
                required_output="Final answer ready for the requesting client.",
                reason="Simple requests do not need graph decomposition.",
                preferred_roles=[NodeRole(requirements["preferred_roles"][0])],
            )
        ]

    graph = {
        "nodes": [step.to_dict() for step in steps],
        "edges": [
            {"from": dependency, "to": step.id}
            for step in steps
            for dependency in step.depends_on
        ],
    }
    status = PlannerStatus.DEGRADED if degraded_reason else PlannerStatus.PLANNED
    provider = PlannerProvider.DETERMINISTIC
    return PlanResponse(
        request_id=request.request_id,
        planner_provider=provider,
        planner_status=status,
        plan={
            "summary": f"{provider.value} plan generated for {task_type.value} request.",
            "steps": [step.to_dict() for step in steps],
        },
        graph=graph,
        scheduling_requirements=requirements,
        degraded_reason=degraded_reason,
    )


def plan_request(request: PlanRequest) -> PlanResponse:
    engine = os.getenv("MUNDUSX_PLANNER_ENGINE", "auto").strip().lower()
    if engine == "deterministic":
        return deterministic_plan(request)

    try:
        from .workflow import langgraph_plan

        return langgraph_plan(request)
    except Exception as exc:
        return deterministic_plan(request, degraded_reason=f"langgraph_unavailable: {exc}")
