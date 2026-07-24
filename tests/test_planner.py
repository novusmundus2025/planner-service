import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundusx_planner_service.models import PlanRequest  # noqa: E402
from mundusx_planner_service.planner import deterministic_plan, plan_request  # noqa: E402


class PlannerTests(unittest.TestCase):
    def test_simple_chat_gets_single_execute_node(self):
        response = deterministic_plan(
            PlanRequest(request_id="req-1", prompt="Hello, what is MundusX?")
        )

        self.assertEqual(response.planner_provider, "deterministic")
        self.assertEqual(response.planner_status, "planned")
        self.assertEqual(len(response.graph["nodes"]), 1)
        self.assertEqual(response.graph["nodes"][0]["id"], "execute")
        self.assertIn("chat", response.scheduling_requirements["preferred_roles"])

    def test_coding_request_gets_scope_execute_reduce_graph(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-2",
                prompt="Refactor this Rust code and add tests for the scheduler.",
                model="qwen",
            )
        )

        self.assertEqual([node["id"] for node in response.graph["nodes"]], ["scope", "execute", "reduce"])
        self.assertEqual(
            response.graph["edges"],
            [{"from": "scope", "to": "execute"}, {"from": "execute", "to": "reduce"}],
        )
        self.assertEqual(response.scheduling_requirements["model"], "qwen")
        self.assertIn("coding", response.scheduling_requirements["preferred_roles"])

    def test_modality_requirements_are_inferred(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-3",
                prompt="Use OCR on this screenshot, create embeddings, then run a terminal tool.",
            )
        )

        self.assertTrue(response.scheduling_requirements["requires_vision"])
        self.assertTrue(response.scheduling_requirements["requires_embeddings"])
        self.assertTrue(response.scheduling_requirements["requires_tools"])
        self.assertIn("vision", response.scheduling_requirements["preferred_roles"])
        self.assertIn("embedding", response.scheduling_requirements["preferred_roles"])
        self.assertIn("tool_use", response.scheduling_requirements["preferred_roles"])

    def test_plan_request_can_force_deterministic_engine(self):
        previous = os.environ.get("MUNDUSX_PLANNER_ENGINE")
        os.environ["MUNDUSX_PLANNER_ENGINE"] = "deterministic"
        try:
            response = plan_request(PlanRequest(request_id="req-4", prompt="Hello"))
        finally:
            if previous is None:
                os.environ.pop("MUNDUSX_PLANNER_ENGINE", None)
            else:
                os.environ["MUNDUSX_PLANNER_ENGINE"] = previous

        self.assertEqual(response.planner_provider, "deterministic")
        self.assertEqual(response.planner_status, "planned")


if __name__ == "__main__":
    unittest.main()
