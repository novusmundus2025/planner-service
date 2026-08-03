from __future__ import annotations

import os
from collections.abc import Iterable

from .models import (
    CapacityClass,
    NodeRole,
    PlanRequest,
    PlanResponse,
    PlannerProvider,
    PlannerStatus,
    PlanStep,
    TaskType,
    ValidationLevel,
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
        or task_type is TaskType.DOCUMENT
        or (
            task_type is TaskType.CODING
            and _contains_any(
                normalized,
                [
                    "refactor",
                    "add tests",
                    "multiple files",
                    "entire repository",
                    "new feature",
                    "implement",
                ],
            )
        )
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


def output_budget_profile(request: PlanRequest, task_type: TaskType) -> dict[str, int]:
    normalized = request.prompt.lower()
    long_form = _contains_any(
        normalized,
        [
            "detailed history",
            "comprehensive",
            "in detail",
            "in-depth",
            "deep dive",
            "full history",
            "report",
            "timeline",
            "write an article",
        ],
    )
    concise = _contains_any(
        normalized,
        ["briefly", "concise", "short answer", "one paragraph", "summarize briefly"],
    )

    if task_type is TaskType.CODING:
        final = 3072
    elif long_form:
        final = 3072
    elif concise:
        final = 512
    else:
        final = 1536

    return {
        "scope": 384,
        "chunk": min(1024, max(512, final // 4)),
        "reduce": min(3072, max(1024, final // 2)),
        "synthesize": final,
        "direct": final,
    }


def build_plan(
    request: PlanRequest,
    task_type: TaskType,
    requirements: dict[str, object],
    provider: PlannerProvider,
    degraded_reason: str | None = None,
) -> PlanResponse:
    complex_request = should_decompose(request, task_type)
    coding_request = task_type == TaskType.CODING
    budgets = output_budget_profile(request, task_type)
    coding = task_type is TaskType.CODING

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
                recommended_max_tokens=budgets["scope"],
                minimum_max_tokens=256,
                expected_artifact_types=["structured_data"],
                minimum_capacity_class=CapacityClass.MICRO,
                recommended_capacity_class=CapacityClass.STANDARD,
                context_budget_tokens=4096,
                validation_level=ValidationLevel.STRUCTURAL,
                allowed_parallelism=1,
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
                recommended_max_tokens=budgets["chunk"],
                minimum_max_tokens=384,
                expected_artifact_types=["code", "patch"] if coding_request else ["text"],
                artifact_targets=["implementation"] if coding_request else [],
                minimum_capacity_class=CapacityClass.STANDARD,
                recommended_capacity_class=CapacityClass.PERFORMANCE,
                context_budget_tokens=8192,
                expected_artifact_bytes=262144,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository"] if coding else [],
                requires_repository=coding,
                validation_level=ValidationLevel.SYNTAX if coding else ValidationLevel.STRUCTURAL,
                allowed_parallelism=3,
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
                recommended_max_tokens=budgets["chunk"],
                minimum_max_tokens=384,
                expected_artifact_types=["code", "patch"] if coding_request else ["text"],
                artifact_targets=["tests"] if coding_request else [],
                minimum_capacity_class=CapacityClass.STANDARD,
                recommended_capacity_class=CapacityClass.PERFORMANCE,
                context_budget_tokens=8192,
                expected_artifact_bytes=262144,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository"] if coding else [],
                requires_repository=coding,
                validation_level=ValidationLevel.SYNTAX if coding else ValidationLevel.STRUCTURAL,
                allowed_parallelism=3,
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
                recommended_max_tokens=budgets["chunk"],
                minimum_max_tokens=384,
                expected_artifact_types=["code", "patch", "command"] if coding_request else ["text"],
                artifact_targets=["integration", "documentation"] if coding_request else [],
                minimum_capacity_class=CapacityClass.STANDARD,
                recommended_capacity_class=CapacityClass.PERFORMANCE,
                context_budget_tokens=8192,
                expected_artifact_bytes=262144,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository", "test"] if coding else [],
                requires_repository=coding,
                requires_tests=coding,
                validation_level=ValidationLevel.TEST if coding else ValidationLevel.STRUCTURAL,
                allowed_parallelism=3,
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
                recommended_max_tokens=budgets["reduce"],
                minimum_max_tokens=768,
                unavailable_timeout_seconds=60,
                on_unavailable="preserve_chunks_and_degrade",
                expected_artifact_types=["code", "patch", "command"] if coding_request else ["text"],
                max_input_artifacts=20,
                minimum_capacity_class=CapacityClass.PERFORMANCE,
                recommended_capacity_class=CapacityClass.HEAVY,
                context_budget_tokens=16384,
                expected_artifact_count=3,
                expected_artifact_bytes=1048576,
                model_quality_floor="strong",
                validation_level=ValidationLevel.STRUCTURAL,
                reducer_credibility="high",
                synthesizer_credibility="high",
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
                recommended_max_tokens=budgets["synthesize"],
                minimum_max_tokens=1024,
                unavailable_timeout_seconds=60,
                on_unavailable="preserve_reduction_and_degrade",
                expected_artifact_types=["code", "patch", "command", "test_report"] if coding_request else ["text"],
                max_input_artifacts=20,
                minimum_capacity_class=CapacityClass.HEAVY,
                recommended_capacity_class=CapacityClass.SYNTHESIS,
                context_budget_tokens=32768,
                expected_artifact_count=4,
                expected_artifact_bytes=2097152,
                model_quality_floor="strong",
                validation_level=ValidationLevel.STRUCTURAL,
                reducer_credibility="high",
                synthesizer_credibility="high",
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
                recommended_max_tokens=budgets["direct"],
                minimum_max_tokens=min(512, budgets["direct"]),
                expected_artifact_types=["code", "patch", "command"] if coding_request else ["text"],
                minimum_capacity_class=CapacityClass.MICRO,
                recommended_capacity_class=(
                    CapacityClass.STANDARD if coding else CapacityClass.MICRO
                ),
                context_budget_tokens=int(requirements["desired_context_tokens"]),
                expected_artifact_bytes=262144 if coding else 65536,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository"] if coding else [],
                requires_repository=coding,
                validation_level=ValidationLevel.SYNTAX if coding else ValidationLevel.STRUCTURAL,
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
            "result_protocol": "artifact_manifest_v1",
            "artifact_batch_max_items": 20,
            "artifact_batch_max_bytes": 65536,
            "conflict_policy": "report_without_silent_overwrite",
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
