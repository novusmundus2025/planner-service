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
                available_capability_summary={"eligible_nodes": 3, "eligible_parallel_slots": 3},
            )
        )

        self.assertEqual(
            [node["id"] for node in response.graph["nodes"]],
            [
                "scope",
                "work-implementation",
                "work-tests",
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
        self.assertEqual(reducer["minimum_capacity_class"], "performance")
        self.assertEqual(reducer["recommended_capacity_class"], "heavy")
        self.assertEqual(reducer["reducer_credibility"], "high")
        self.assertEqual(synth["minimum_capacity_class"], "heavy")
        self.assertEqual(synth["recommended_capacity_class"], "synthesis")
        self.assertEqual(synth["synthesizer_credibility"], "high")
        coding_step = response.graph["nodes"][1]
        self.assertFalse(coding_step["requires_repository"])
        self.assertEqual(coding_step["validation_level"], "syntax")
        self.assertGreaterEqual(coding_step["context_budget_tokens"], 8192)
        self.assertTrue(response.graph["execution_policy"]["preserve_completed_outputs"])
        self.assertEqual(
            response.graph["execution_policy"]["result_protocol"], "artifact_manifest_v1"
        )
        self.assertEqual(reducer["max_input_artifacts"], 20)
        self.assertIn("patch", synth["expected_artifact_types"])
        self.assertEqual(response.graph["nodes"][1]["artifact_targets"], ["implementation"])
        self.assertEqual(response.graph["nodes"][2]["artifact_targets"], ["tests"])
        self.assertEqual(response.scheduling_requirements["model"], "qwen")
        self.assertIn("coding", response.scheduling_requirements["preferred_roles"])

    def test_detailed_history_is_decomposed_even_when_prompt_is_short(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-history",
                prompt="Give me a detailed history of the European Union.",
                available_capability_summary={"eligible_nodes": 3, "eligible_parallel_slots": 3},
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

    def test_live_capacity_bounds_dynamic_workstreams(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-crud",
                prompt=(
                    "Implement a Node.js CRUD API with tests, Markdown documentation, "
                    "and a code review."
                ),
                available_capability_summary={"eligible_nodes": 2, "eligible_parallel_slots": 2},
            )
        )

        self.assertEqual(
            [node["id"] for node in response.graph["nodes"]],
            [
                "scope",
                "work-implementation",
                "work-tests",
                "work-documentation",
                "work-review",
                "reduce",
                "synthesize",
            ],
        )
        self.assertNotIn("chunk-foundations", [node["id"] for node in response.graph["nodes"]])
        self.assertEqual(response.graph["nodes"][1]["allowed_parallelism"], 2)
        self.assertEqual(response.graph["nodes"][4]["depends_on"], ["work-implementation"])

    def test_micro_cluster_gets_feasible_standalone_generation_stages(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-micro-crud",
                prompt=(
                    "Create a Node.js Express CRUD API with Markdown documentation "
                    "and a code review."
                ),
                available_capability_summary={
                    "eligible_nodes": 2,
                    "eligible_parallel_slots": 2,
                    "max_context_tokens": 4096,
                    "capacity_class_counts": {"micro": 2},
                    "tool_counts": {},
                },
            )
        )

        implementation = response.graph["nodes"][1]
        reducer = response.graph["nodes"][-2]
        synthesis = response.graph["nodes"][-1]
        self.assertEqual(implementation["minimum_capacity_class"], "micro")
        self.assertEqual(implementation["context_budget_tokens"], 4096)
        self.assertFalse(implementation["requires_repository"])
        self.assertEqual(implementation["required_tools"], [])
        self.assertEqual(reducer["minimum_capacity_class"], "micro")
        self.assertEqual(synthesis["minimum_capacity_class"], "micro")

    def test_one_live_slot_avoids_artificial_fanout(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-one-slot",
                prompt="Implement a Node.js CRUD API with tests and documentation.",
                available_capability_summary={"eligible_nodes": 1, "eligible_parallel_slots": 1},
            )
        )

        self.assertEqual([node["id"] for node in response.graph["nodes"]], ["execute"])

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

    def test_bounded_single_file_coding_step_fits_standard_capacity(self):
        response = deterministic_plan(
            PlanRequest(request_id="req-bounded", prompt="Fix this code bug")
        )

        step = response.graph["nodes"][0]
        self.assertEqual(step["minimum_capacity_class"], "micro")
        self.assertEqual(step["recommended_capacity_class"], "standard")
        self.assertFalse(step["requires_repository"])
        self.assertEqual(step["model_quality_floor"], "coding")
        self.assertLessEqual(step["expected_artifact_bytes"], 262144)

    def test_repository_bound_request_requires_repository_tool(self):
        response = deterministic_plan(
            PlanRequest(
                request_id="req-repository",
                prompt="Refactor the scheduler in this repository and add tests.",
                available_capability_summary={
                    "eligible_nodes": 2,
                    "eligible_parallel_slots": 2,
                    "tool_counts": {"repository": 1, "test": 1},
                },
            )
        )

        implementation = response.graph["nodes"][1]
        tests = response.graph["nodes"][2]
        self.assertTrue(implementation["requires_repository"])
        self.assertIn("repository", implementation["required_tools"])
        self.assertIn("test", tests["required_tools"])

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
