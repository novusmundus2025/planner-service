from __future__ import annotations

import os
from collections.abc import Iterable

from .models import (
    CapacityClass,
    NodeRole,
    PlannerProvider,
    PlannerStatus,
    PlanRequest,
    PlanResponse,
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
    coding_deliverable_groups = (
        ["implement", "create", "build", "generate", "refactor", "new feature"],
        ["test", "tests", "testing", "regression"],
        ["documentation", "docs", "readme", "markdown"],
        ["review", "audit", "security"],
    )
    requested_coding_deliverables = sum(
        _contains_any(normalized, group) for group in coding_deliverable_groups
    )
    return (
        len(request.prompt) > 2000
        or task_type is TaskType.DOCUMENT
        or (
            task_type is TaskType.CODING
            and (
                requested_coding_deliverables >= 2
                or _contains_any(
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


def _live_parallel_width(request: PlanRequest) -> int:
    summary = request.available_capability_summary
    slots = summary.get("eligible_parallel_slots", summary.get("eligible_nodes", 1))
    try:
        return max(1, min(6, int(slots)))
    except (TypeError, ValueError):
        return 1


def _strongest_live_capacity(request: PlanRequest) -> CapacityClass:
    counts = request.available_capability_summary.get("capacity_class_counts", {})
    ordered = [
        CapacityClass.SERVER,
        CapacityClass.SYNTHESIS,
        CapacityClass.HEAVY,
        CapacityClass.PERFORMANCE,
        CapacityClass.STANDARD,
        CapacityClass.MICRO,
    ]
    if not isinstance(counts, dict) or not counts:
        return CapacityClass.HEAVY
    for capacity in ordered:
        try:
            if int(counts.get(capacity.value, 0)) > 0:
                return capacity
        except (TypeError, ValueError):
            continue
    return CapacityClass.MICRO


def _live_max_context(request: PlanRequest, default: int = 8192) -> int:
    value = request.available_capability_summary.get("max_context_tokens", 0)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(2048, parsed) if parsed > 0 else default


def _capacity_at_least(value: CapacityClass, threshold: CapacityClass) -> bool:
    order = {
        CapacityClass.MICRO: 0,
        CapacityClass.STANDARD: 1,
        CapacityClass.PERFORMANCE: 2,
        CapacityClass.HEAVY: 3,
        CapacityClass.SYNTHESIS: 4,
        CapacityClass.SERVER: 5,
    }
    return order[value] >= order[threshold]


def _prompt_requires_repository(prompt: str) -> bool:
    return _contains_any(
        prompt.lower(),
        [
            "repository",
            " repo ",
            "codebase",
            "workspace",
            "existing project",
            "project files",
        ],
    )


def _live_tool_available(request: PlanRequest, tool: str) -> bool:
    counts = request.available_capability_summary.get("tool_counts", {})
    if not isinstance(counts, dict):
        return False
    try:
        return int(counts.get(tool, 0)) > 0
    except (TypeError, ValueError):
        return False


def _requested_workstreams(request: PlanRequest, task_type: TaskType) -> list[tuple[str, str, str]]:
    """Return prompt-specific responsibilities, not presentation section templates."""
    prompt = request.prompt.lower()
    streams: list[tuple[str, str, str]] = []

    if task_type is TaskType.CODING:
        streams.append(
            (
                "implementation",
                "Implement the requested behavior",
                "Complete runnable source or a repository-ready patch satisfying the core request.",
            )
        )
        if _contains_any(prompt, ["test", "tests", "testing", "regression"]):
            streams.append(
                (
                    "tests",
                    "Design and implement verification",
                    "Focused automated tests and the commands or evidence needed "
                    "to verify behavior.",
                )
            )
        if _contains_any(prompt, ["documentation", "docs", "readme", "markdown"]):
            streams.append(
                (
                    "documentation",
                    "Document the delivered interface",
                    "User-facing Markdown documentation aligned with the implementation.",
                )
            )
        if _contains_any(prompt, ["review", "code review", "audit"]):
            streams.append(
                (
                    "review",
                    "Review correctness and maintainability",
                    "Specific findings, risks, and fixes grounded in the delivered implementation.",
                )
            )
        if _contains_any(prompt, ["security", "threat", "vulnerability"]):
            streams.append(
                (
                    "security",
                    "Assess security properties",
                    "Threat-focused findings and concrete mitigations for the "
                    "requested implementation.",
                )
            )
        return streams

    if _contains_any(prompt, ["compare", "versus", " vs "]):
        streams.extend(
            [
                (
                    "criteria",
                    "Define decision criteria",
                    "Criteria tied directly to the user's decision.",
                ),
                (
                    "evidence",
                    "Compare the alternatives",
                    "Evidence for each alternative against the criteria.",
                ),
                (
                    "tradeoffs",
                    "Evaluate tradeoffs",
                    "A balanced recommendation with material disadvantages.",
                ),
            ]
        )
    elif _contains_any(prompt, ["history", "timeline", "to date"]):
        streams.extend(
            [
                (
                    "origins",
                    "Establish origins and context",
                    "Verified origins and initial context.",
                ),
                (
                    "developments",
                    "Trace material developments",
                    "Major turning points supported by evidence.",
                ),
                (
                    "current-state",
                    "Assess the current state",
                    "Current position and material implications.",
                ),
            ]
        )
    elif task_type is TaskType.DOCUMENT:
        streams.extend(
            [
                (
                    "extract",
                    "Extract the material content",
                    "Accurate facts and requirements from the document.",
                ),
                (
                    "analyze",
                    "Analyze the requested issues",
                    "Findings tied to the source material.",
                ),
                (
                    "verify",
                    "Check completeness and conflicts",
                    "Missing, conflicting, or uncertain points.",
                ),
            ]
        )
    else:
        streams.extend(
            [
                ("answer", "Develop the core answer", "A direct answer grounded in the request."),
                (
                    "verify",
                    "Verify facts and assumptions",
                    "Corrections, evidence, and uncertainty boundaries.",
                ),
                (
                    "implications",
                    "Assess implications and tradeoffs",
                    "Material consequences and tradeoffs.",
                ),
            ]
        )
    return streams


def _build_adaptive_steps(
    request: PlanRequest,
    task_type: TaskType,
    requirements: dict[str, object],
    budgets: dict[str, int],
) -> list[PlanStep]:
    execution_role = NodeRole(requirements["preferred_roles"][0])
    coding = task_type is TaskType.CODING
    width = _live_parallel_width(request)
    strongest_capacity = _strongest_live_capacity(request)
    work_capacity = (
        CapacityClass.STANDARD
        if _capacity_at_least(strongest_capacity, CapacityClass.STANDARD)
        else CapacityClass.MICRO
    )
    reducer_capacity = (
        CapacityClass.PERFORMANCE
        if _capacity_at_least(strongest_capacity, CapacityClass.PERFORMANCE)
        else work_capacity
    )
    synthesis_capacity = (
        CapacityClass.HEAVY
        if _capacity_at_least(strongest_capacity, CapacityClass.HEAVY)
        else reducer_capacity
    )
    max_context = _live_max_context(request)
    scope_context = min(4096, max_context)
    work_context = min(8192, max_context)
    reducer_context = min(16384, max_context)
    synthesis_context = min(32768, max_context)
    repository_required = coding and _prompt_requires_repository(request.prompt)
    workstreams = _requested_workstreams(request, task_type)

    # A one-slot cluster gains nothing from artificial fan-out. The scheduler can
    # reconsider a new request as soon as more capacity joins.
    if width <= 1:
        return [
            PlanStep(
                id="execute",
                name="Execute request",
                responsibility="Answer the complete request",
                required_output="One complete final answer satisfying every requested deliverable.",
                reason="The live cluster exposes one eligible parallel slot.",
                preferred_roles=[execution_role],
                required_role=execution_role,
                recommended_max_tokens=budgets["direct"],
                minimum_max_tokens=min(512, budgets["direct"]),
                expected_artifact_types=["code", "patch", "command"] if coding else ["text"],
                minimum_capacity_class=CapacityClass.MICRO,
                recommended_capacity_class=CapacityClass.STANDARD,
                context_budget_tokens=int(requirements["desired_context_tokens"]),
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository"] if repository_required else [],
                requires_repository=repository_required,
                validation_level=ValidationLevel.SYNTAX if coding else ValidationLevel.STRUCTURAL,
            )
        ]

    steps = [
        PlanStep(
            id="scope",
            name="Scope request",
            responsibility=(
                "Identify required deliverables, constraints, and acceptance conditions."
            ),
            required_output="A concise scope contract shared by downstream work.",
            reason="A shared scope prevents independent workers from diverging.",
            preferred_roles=[NodeRole.BATCH],
            required_role=NodeRole.BATCH,
            fallback_roles=[execution_role],
            recommended_max_tokens=budgets["scope"],
            minimum_max_tokens=256,
            expected_artifact_types=["structured_data"],
            minimum_capacity_class=CapacityClass.MICRO,
            recommended_capacity_class=CapacityClass.STANDARD,
            context_budget_tokens=scope_context,
            validation_level=ValidationLevel.STRUCTURAL,
        )
    ]

    selected = workstreams
    for stream_id, name, required_output in selected:
        is_tests = stream_id == "tests"
        dependencies = ["scope"]
        if stream_id in {"review", "security"} and any(
            existing.id == "work-implementation" for existing in steps
        ):
            dependencies = ["work-implementation"]
        steps.append(
            PlanStep(
                id=f"work-{stream_id}",
                name=name,
                responsibility=stream_id,
                required_output=required_output,
                reason="The planner derived this responsibility from the requested deliverables.",
                depends_on=dependencies,
                preferred_roles=[NodeRole.CHUNK_ANALYSIS, execution_role],
                required_role=NodeRole.CHUNK_ANALYSIS,
                fallback_roles=[NodeRole.BATCH, execution_role],
                recommended_max_tokens=budgets["chunk"],
                minimum_max_tokens=384,
                expected_artifact_types=["code", "patch", "test_report"] if coding else ["text"],
                artifact_targets=[stream_id] if coding else [],
                minimum_capacity_class=work_capacity,
                recommended_capacity_class=CapacityClass.PERFORMANCE,
                context_budget_tokens=work_context,
                expected_artifact_bytes=262144,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=(
                    (["repository"] if repository_required else [])
                    + (["test"] if is_tests and _live_tool_available(request, "test") else [])
                ),
                requires_repository=repository_required,
                requires_tests=is_tests,
                validation_level=(
                    ValidationLevel.TEST
                    if is_tests
                    else ValidationLevel.SYNTAX
                    if coding
                    else ValidationLevel.STRUCTURAL
                ),
                allowed_parallelism=width,
            )
        )

    dependencies = [step.id for step in steps if step.id.startswith("work-")]
    steps.append(
        PlanStep(
            id="reduce",
            name="Validate and reduce partial results",
            responsibility="reduce",
            required_output="A deduplicated, conflict-checked set of accepted results.",
            reason="Independent results require validation before final synthesis.",
            depends_on=dependencies,
            preferred_roles=[NodeRole.REDUCER],
            required_role=NodeRole.REDUCER,
            recommended_max_tokens=budgets["reduce"],
            minimum_max_tokens=768,
            unavailable_timeout_seconds=60,
            on_unavailable="preserve_chunks_and_degrade",
            expected_artifact_types=["code", "patch", "command", "test_report"]
            if coding
            else ["text"],
            max_input_artifacts=20,
            minimum_capacity_class=reducer_capacity,
            recommended_capacity_class=CapacityClass.HEAVY,
            context_budget_tokens=reducer_context,
            expected_artifact_count=len(dependencies),
            expected_artifact_bytes=1048576,
            model_quality_floor="strong",
            validation_level=ValidationLevel.STRUCTURAL,
            reducer_credibility="high",
            synthesizer_credibility="high",
        )
    )
    steps.append(
        PlanStep(
            id="synthesize",
            name="Synthesize final answer",
            responsibility="synthesize",
            required_output="One coherent final answer satisfying the original request.",
            reason="Synthesis converts validated results into the client response.",
            depends_on=["reduce"],
            preferred_roles=[NodeRole.SYNTHESIZER],
            required_role=NodeRole.SYNTHESIZER,
            recommended_max_tokens=budgets["synthesize"],
            minimum_max_tokens=1024,
            unavailable_timeout_seconds=60,
            on_unavailable="preserve_reduction_and_degrade",
            expected_artifact_types=["code", "patch", "command", "test_report"]
            if coding
            else ["text"],
            max_input_artifacts=20,
            minimum_capacity_class=synthesis_capacity,
            recommended_capacity_class=CapacityClass.SYNTHESIS,
            context_budget_tokens=synthesis_context,
            expected_artifact_count=len(dependencies) + 1,
            expected_artifact_bytes=2097152,
            model_quality_floor="strong",
            validation_level=ValidationLevel.STRUCTURAL,
            reducer_credibility="high",
            synthesizer_credibility="high",
        )
    )
    return steps


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
        steps = _build_adaptive_steps(request, task_type, requirements, budgets)
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
                expected_artifact_types=["code", "patch", "command"]
                if coding_request
                else ["text"],
                minimum_capacity_class=CapacityClass.MICRO,
                recommended_capacity_class=(
                    CapacityClass.STANDARD if coding else CapacityClass.MICRO
                ),
                context_budget_tokens=int(requirements["desired_context_tokens"]),
                expected_artifact_bytes=262144 if coding else 65536,
                model_quality_floor="coding" if coding else "baseline",
                required_tools=["repository"]
                if coding and _prompt_requires_repository(request.prompt)
                else [],
                requires_repository=coding and _prompt_requires_repository(request.prompt),
                validation_level=ValidationLevel.SYNTAX if coding else ValidationLevel.STRUCTURAL,
            )
        ]

    graph = {
        "nodes": [step.to_dict() for step in steps],
        "edges": [
            {"from": dependency, "to": step.id} for step in steps for dependency in step.depends_on
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
