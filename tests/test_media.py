import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mundusx_planner_service.media import media_plan


class MediaTests(unittest.TestCase):
    def test_image_routes_to_async_queue_without_vision_llm_requirement(self):
        result = media_plan({"kind": "image"})
        self.assertEqual(result["execution_route"], "media_queue")
        self.assertEqual(result["operation"], "text_to_image")
        self.assertTrue(result["asynchronous"])
        self.assertEqual(result["node_selection_owner"], "media_scheduler")

    def test_t2v_and_i2v_choose_exact_verified_duration_profile(self):
        for seconds in range(1, 11):
            for reference in (False, True):
                result = media_plan({"kind": "video", "duration_seconds": seconds,
                                     "has_reference": reference})
                self.assertIn(f"-{seconds * 16 + 1}f-", result["profile_id"])
                self.assertEqual(result["operation"], "image_to_video" if reference else "text_to_video")

    def test_invalid_media_does_not_plan_llm_fallback(self):
        for item in ({"kind": "chat"}, {"kind": "image", "has_reference": True},
                     {"kind": "video", "duration_seconds": 11},
                     {"kind": "video", "duration_seconds": True},
                     {"kind": "image", "duration_seconds": 3}):
            with self.assertRaises(ValueError):
                media_plan(item)
