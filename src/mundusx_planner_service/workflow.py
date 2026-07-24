from __future__ import annotations

from typing import TypedDict

from .models import PlanRequest, PlanResponse, PlannerProvider, PlannerStatus
from .planner import deterministic_plan


class PlannerState(TypedDict, total=False):
    request: PlanRequest
    response: PlanResponse


def _build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PlannerState)

    def classify_and_plan(state: PlannerState) -> PlannerState:
        request = state["request"]
        response = deterministic_plan(request)
        return {
            "request": request,
            "response": PlanResponse(
                request_id=response.request_id,
                planner_provider=PlannerProvider.LANGGRAPH,
                planner_status=PlannerStatus.PLANNED,
                plan={
                    **response.plan,
                    "summary": response.plan["summary"].replace(
                        "deterministic", "langgraph"
                    ),
                },
                graph=response.graph,
                scheduling_requirements=response.scheduling_requirements,
            ),
        }

    graph.add_node("classify_and_plan", classify_and_plan)
    graph.set_entry_point("classify_and_plan")
    graph.add_edge("classify_and_plan", END)
    return graph.compile()


def langgraph_plan(request: PlanRequest) -> PlanResponse:
    app = _build_graph()
    result = app.invoke({"request": request})
    return result["response"]
