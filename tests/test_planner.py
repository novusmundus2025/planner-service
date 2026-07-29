import os
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundusx_planner_service.models import PlanRequest  # noqa: E402
from mundusx_planner_service.planner import deterministic_plan, plan_request  # noqa: E402
from mundusx_planner_service.workflow import langgraph_plan  # noqa: E402


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

    def test_coding_request_gets_parallel_chunks_reduce_and_synthesis(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-2",
                prompt="Refactor this Rust code and add tests for the scheduler.",
                model="qwen",
            )
        )

        self.assertEqual(
            [node["id"] for node in response.graph["nodes"]],
            [
                "scope",
                "chunk-foundations",
                "chunk-developments",
                "chunk-impact",
                "reduce",
                "synthesize",
            ],
        )
        synth = response.graph["nodes"][-1]
        self.assertEqual(synth["preferred_roles"], ["synthesizer"])
        self.assertEqual(synth["required_role"], "synthesizer")
        self.assertEqual(synth["fallback_roles"], [])
        self.assertEqual(synth["on_unavailable"], "preserve_reduction_and_degrade")
        self.assertEqual(synth["depends_on"], ["reduce"])
        self.assertGreaterEqual(synth["recommended_max_tokens"], 2048)
        self.assertEqual(synth["minimum_max_tokens"], 1024)
        reducer = response.graph["nodes"][-2]
        self.assertEqual(reducer["required_role"], "reducer")
        self.assertGreaterEqual(reducer["recommended_max_tokens"], 1024)
        self.assertEqual(reducer["on_unavailable"], "preserve_chunks_and_degrade")
        self.assertTrue(response.graph["execution_policy"]["preserve_completed_outputs"])
        self.assertEqual(response.scheduling_requirements["model"], "qwen")
        self.assertIn("coding", response.scheduling_requirements["preferred_roles"])

    def test_detailed_history_is_decomposed_even_when_prompt_is_short(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-history",
                prompt="Give me a detailed history of the European Union.",
            )
        )

        self.assertEqual(response.graph["nodes"][-2]["id"], "reduce")
        self.assertEqual(response.graph["nodes"][-1]["id"], "synthesize")
        self.assertEqual(
            response.graph["nodes"][-1]["recommended_max_tokens"],
            3072,
        )
        self.assertGreater(
            response.graph["nodes"][-1]["recommended_max_tokens"],
            response.graph["nodes"][1]["recommended_max_tokens"],
        )
        self.assertEqual(
            response.graph["nodes"][1]["preferred_roles"][0],
            "chunk_analysis",
        )

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

    @unittest.skipUnless(find_spec("langgraph"), "langgraph dependency is not installed")
    def test_langgraph_workflow_runs_multistage_plan(self):
        response = langgraph_plan(
            PlanRequest(
                request_id="req-langgraph",
                prompt="Give me a detailed history of the European Union.",
            )
        )

        self.assertEqual(response.planner_provider, "langgraph")
        self.assertEqual(response.planner_status, "planned")
        self.assertEqual(response.graph["nodes"][-1]["id"], "synthesize")


if __name__ == "__main__":
    unittest.main()
