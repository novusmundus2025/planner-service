from __future__ import annotations

from typing import TypedDict

from .models import PlannerProvider, PlanRequest, PlanResponse, TaskType
from .planner import build_plan, classify_task, infer_requirements


class PlannerState(TypedDict, total=False):
    request: PlanRequest
    task_type: TaskType
    requirements: dict[str, object]
    response: PlanResponse


def _build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PlannerState)

    def classify(state: PlannerState) -> PlannerState:
        request = state["request"]
        task_type = classify_task(request.prompt)
        return {
            "request": request,
            "task_type": task_type,
            "requirements": infer_requirements(request, task_type),
        }

    def decompose(state: PlannerState) -> PlannerState:
        response = build_plan(
            state["request"],
            state["task_type"],
            state["requirements"],
            PlannerProvider.LANGGRAPH,
        )
        return {**state, "response": response}

    def validate(state: PlannerState) -> PlannerState:
        response = state["response"]
        node_ids = {node["id"] for node in response.graph["nodes"]}
        for edge in response.graph["edges"]:
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                raise ValueError("planner graph contains an edge to an unknown node")
        return state

    graph.add_node("classify", classify)
    graph.add_node("decompose", decompose)
    graph.add_node("validate", validate)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "decompose")
    graph.add_edge("decompose", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def langgraph_plan(request: PlanRequest) -> PlanResponse:
    app = _build_graph()
    result = app.invoke({"request": request})
    return result["response"]
