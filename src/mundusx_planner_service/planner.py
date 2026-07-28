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


def should_decompose(request: PlanRequest, task_type: TaskType) -> bool:
    normalized = request.prompt.lower()
    return (
        len(request.prompt) > 2000
        or task_type in {TaskType.CODING, TaskType.DOCUMENT}
        or _contains_any(
            normalized,
            [
                "detailed history",
                "comprehensive",
                "in depth",
                "in-depth",
                "covering its",
                "compare and contrast",
                "multiple perspectives",
            ],
        )
    )


def build_plan(
    request: PlanRequest,
    task_type: TaskType,
    requirements: dict[str, object],
    provider: PlannerProvider,
    degraded_reason: str | None = None,
) -> PlanResponse:
    complex_request = should_decompose(request, task_type)

    if complex_request:
        execution_role = NodeRole(requirements["preferred_roles"][0])
        steps = [
            PlanStep(
                id="scope",
                name="Scope request",
                responsibility="Identify the required work, constraints, and expected output.",
                required_output="A concise scope summary and acceptance checklist.",
                reason="Complex requests need a bounded scope before distributed execution.",
                preferred_roles=[NodeRole.BATCH],
                required_role=NodeRole.BATCH,
                fallback_roles=[execution_role],
            ),
            PlanStep(
                id="chunk-foundations",
                name="Analyze foundations",
                responsibility="chunk_analysis",
                required_output="Evidence-backed findings about origins, foundations, and context.",
                reason="A bounded responsibility can run independently on a contributor.",
                depends_on=["scope"],
                preferred_roles=[NodeRole.CHUNK_ANALYSIS, execution_role],
                required_role=NodeRole.CHUNK_ANALYSIS,
                fallback_roles=[NodeRole.BATCH, execution_role],
            ),
            PlanStep(
                id="chunk-developments",
                name="Analyze major developments",
                responsibility="chunk_analysis",
                required_output="Evidence-backed findings about major developments and turning points.",
                reason="Independent analysis enables parallel contributor execution.",
                depends_on=["scope"],
                preferred_roles=[NodeRole.CHUNK_ANALYSIS, execution_role],
                required_role=NodeRole.CHUNK_ANALYSIS,
                fallback_roles=[NodeRole.BATCH, execution_role],
            ),
            PlanStep(
                id="chunk-impact",
                name="Analyze outcomes and current impact",
                responsibility="chunk_analysis",
                required_output="Evidence-backed findings about outcomes, implications, and current state.",
                reason="A separate responsibility improves coverage before reduction.",
                depends_on=["scope"],
                preferred_roles=[NodeRole.CHUNK_ANALYSIS, execution_role],
                required_role=NodeRole.CHUNK_ANALYSIS,
                fallback_roles=[NodeRole.BATCH, execution_role],
            ),
            PlanStep(
                id="reduce",
                name="Reduce partial results",
                responsibility="reduce",
                required_output="A deduplicated, ordered set of accepted findings.",
                reason="Reduction controls context size and removes conflicts before synthesis.",
                depends_on=["chunk-foundations", "chunk-developments", "chunk-impact"],
                preferred_roles=[NodeRole.REDUCER],
                required_role=NodeRole.REDUCER,
                unavailable_timeout_seconds=60,
                on_unavailable="preserve_chunks_and_degrade",
            ),
            PlanStep(
                id="synthesize",
                name="Synthesize final answer",
                responsibility="synthesize",
                required_output="One coherent final answer that satisfies the original request.",
                reason="Synthesis converts accepted reduced findings into the client response.",
                depends_on=["reduce"],
                preferred_roles=[NodeRole.SYNTHESIZER],
                required_role=NodeRole.SYNTHESIZER,
                unavailable_timeout_seconds=60,
                on_unavailable="preserve_reduction_and_degrade",
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
                required_role=NodeRole(requirements["preferred_roles"][0]),
            )
        ]

    graph = {
        "nodes": [step.to_dict() for step in steps],
        "edges": [
            {"from": dependency, "to": step.id}
            for step in steps
            for dependency in step.depends_on
        ],
        "execution_policy": {
            "role_matching": "required",
            "preserve_completed_outputs": True,
            "unavailable_status": "degraded",
            "retryable": True,
        },
    }
    status = PlannerStatus.DEGRADED if degraded_reason else PlannerStatus.PLANNED
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


def deterministic_plan(request: PlanRequest, degraded_reason: str | None = None) -> PlanResponse:
    task_type = classify_task(request.prompt)
    requirements = infer_requirements(request, task_type)
    return build_plan(
        request,
        task_type,
        requirements,
        PlannerProvider.DETERMINISTIC,
        degraded_reason,
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
