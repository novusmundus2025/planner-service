from __future__ import annotations

from typing import Any, TypedDict


class MediaState(TypedDict, total=False):
    request: dict[str, Any]
    operation: str
    profile_id: str
    response: dict[str, Any]


def media_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Plan explicit media work; the queue owns live node ranking and leases."""
    from langgraph.graph import END, StateGraph

    def classify(state: MediaState) -> MediaState:
        item = state["request"]
        kind = item.get("kind")
        if kind not in ("image", "video"):
            raise ValueError("kind must be image or video")
        reference = item.get("has_reference", False)
        if not isinstance(reference, bool) or (kind == "image" and reference):
            raise ValueError("Reference PNG applies only to video")
        operation = "text_to_image" if kind == "image" else (
            "image_to_video" if reference else "text_to_video"
        )
        return {**state, "operation": operation}

    def capability(state: MediaState) -> MediaState:
        item = state["request"]
        seconds = item.get("duration_seconds", 3)
        if state["operation"] == "text_to_image":
            if "duration_seconds" in item:
                raise ValueError("duration_seconds applies only to video")
            profile = "qwen-image-fp8-832x480-v1"
        else:
            if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 10:
                raise ValueError("Video duration must be an integer from 1 to 10")
            prefix = "wan22-i2v-lightning-14b" if state["operation"] == "image_to_video" else "wan22-14b"
            version = "v3" if state["operation"] == "image_to_video" else "v1"
            profile = f"{prefix}-480p-{seconds * 16 + 1}f-{version}"
        return {**state, "profile_id": profile}

    def route(state: MediaState) -> MediaState:
        return {**state, "response": {
            "planner_provider": "langgraph", "execution_route": "media_queue",
            "operation": state["operation"], "profile_id": state["profile_id"],
            "asynchronous": True, "assignment_policy": "eligible_profile_then_ranked_fair",
            "requires_verified_profile": True, "node_selection_owner": "media_scheduler",
        }}

    graph = StateGraph(MediaState)
    graph.add_node("classify_media", classify)
    graph.add_node("require_capability", capability)
    graph.add_node("route_queue", route)
    graph.set_entry_point("classify_media")
    graph.add_edge("classify_media", "require_capability")
    graph.add_edge("require_capability", "route_queue")
    graph.add_edge("route_queue", END)
    return graph.compile().invoke({"request": request})["response"]
