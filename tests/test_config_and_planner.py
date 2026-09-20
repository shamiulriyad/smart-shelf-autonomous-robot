import copy
import json
import unittest

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)
from config.loader import ConfigError, DEFAULT_CONFIG_PATH, _build, load_config
from navigation.path_planner import PathPlanner


def raw_config():
    with open(DEFAULT_CONFIG_PATH) as f:
        return json.load(f)


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(cfg.products.get_by_marker(102).shelf_id, "C")
        self.assertEqual(cfg.shelves.get("C").marker_id, 203)

    def test_product_and_shelf_markers_are_classified_separately(self):
        cfg = load_config()
        self.assertEqual(cfg.classify("ARUCO_ORIGINAL", 102).kind, "product")
        self.assertEqual(cfg.classify("ARUCO_ORIGINAL", 203).kind, "shelf")
        self.assertIsNone(cfg.classify("ARUCO_ORIGINAL", 500))
        self.assertIsNone(cfg.classify("DICT_4X4_50", 102))       # a different dictionary is not our marker

    def test_marker_sizes_come_from_config(self):
        cfg = load_config()
        self.assertEqual(cfg.marker_types["product"].size_cm, 5.0)
        self.assertEqual(cfg.marker_types["shelf"].size_cm, 18.7)

    def _assert_invalid(self, mutate):
        raw = copy.deepcopy(raw_config())
        mutate(raw)
        with self.assertRaises(ConfigError):
            _build(raw)

    def test_rejects_product_id_outside_range(self):
        self._assert_invalid(lambda r: r["products"].update({"250": {"name": "X", "shelf": "A"}}))

    def test_rejects_unknown_shelf(self):
        self._assert_invalid(lambda r: r["products"]["101"].update({"shelf": "Z"}))

    def test_rejects_overlapping_ranges(self):
        self._assert_invalid(lambda r: r["marker_types"]["shelf"].update({"id_range": [150, 299]}))

    def test_rejects_duplicate_shelf_marker(self):
        self._assert_invalid(lambda r: r["shelves"]["B"].update({"marker_id": 201}))

    def test_rejects_blocked_approach_point(self):
        self._assert_invalid(lambda r: r["shelves"]["A"].update({"approach": {"x": 2.0, "y": 4.0}}))

    def test_rejects_unknown_dictionary(self):
        self._assert_invalid(lambda r: r["marker_types"]["shelf"].update({"dictionary": "NOPE"}))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.planner = PathPlanner(self.cfg.room_map)

    def test_paths_reach_every_shelf_and_stay_on_free_space(self):
        for shelf in self.cfg.shelves.all():
            path = self.planner.plan(self.cfg.room_map.robot_start, shelf.approach)
            self.assertIsNotNone(path, shelf.shelf_id)
            self.assertEqual(path[-1], shelf.approach)
            for (x0, y0), (x1, y1) in zip(path, path[1:]):
                for t in range(11):
                    point = (x0 + (x1 - x0) * t / 10, y0 + (y1 - y0) * t / 10)
                    self.assertTrue(self.cfg.room_map.is_free(point), (shelf.shelf_id, point))

    def test_blocked_goal_returns_none(self):
        shelf = self.cfg.shelves.get("A")
        self.assertIsNone(self.planner.plan(self.cfg.room_map.robot_start, (shelf.x, shelf.y)))


if __name__ == "__main__":
    unittest.main()
