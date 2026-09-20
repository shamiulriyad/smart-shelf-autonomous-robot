import logging
import unittest

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)
from aruco.perception import Perception
from aruco.pose_estimator import PoseEstimator
from camera.camera_source import SimulatedWorldCamera, build_world_markers
from config.loader import load_config
from navigation.path_planner import PathPlanner
from robot.robot_controller import SimulatedRobot
from robot.state_machine import PickAndPlaceStateMachine, State

CFG = load_config()
SHELF_C = CFG.shelves.get("C")
S = State


def make(pose):
    """World camera + perception bound to a pose [x, y, heading]."""
    camera = SimulatedWorldCamera(build_world_markers(CFG, 102))
    camera.bind(lambda: tuple(pose))
    return camera, Perception(camera, CFG, PoseEstimator(None), display=False)


def ids(camera):
    return sorted(m.marker_id for m in camera.visible_markers())


class PositionAwareCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_shelf_203_is_not_visible_from_the_start_position(self):
        camera, vision = make([5.0, 0.5, 90.0])                     # robot start; Shelf C is at (8.0, 4.0)
        self.assertNotIn(203, ids(camera))
        self.assertIsNone(vision.look().find("shelf", 203))         # and the real detector agrees
        self.assertIsNotNone(vision.look().find("product", 102))    # the product to pick IS in front of the robot

    def test_203_becomes_visible_only_when_close_enough_and_facing_the_shelf(self):
        approach = SHELF_C.approach                                  # (8.0, 3.0)
        cases = [
            ([5.0, 0.5, 45.0], False),                               # far away
            ([8.0, 1.0, 90.0], False),                               # facing it but 2.75 m from the shelf face: out of range
            ([*approach, 45.0], False),                              # at the approach point, looking 45 deg off
            ([*approach, 180.0], False),                             # at the approach point, looking away
            ([*approach, 90.0], True),                               # at the approach point, facing the shelf
            ([8.0, 3.45, 90.0], True),                               # at the placement point
        ]
        for pose, visible in cases:
            camera, vision = make(list(pose))
            self.assertEqual(203 in ids(camera), visible, pose)
            self.assertEqual(vision.look().find("shelf", 203) is not None, visible, pose)

    def test_marker_is_not_visible_from_behind_its_face(self):
        camera, _ = make([8.0, 5.0, 270.0])                          # behind the shelf, looking at it
        self.assertNotIn(203, ids(camera))

    def test_apparent_size_and_position_follow_distance_and_bearing(self):
        _, near = make([8.0, 3.45, 90.0])
        _, far = make([8.0, 3.0, 90.0])
        near_det, far_det = near.look().find("shelf", 203), far.look().find("shelf", 203)
        self.assertGreater(near_det.side_px, far_det.side_px * 2)    # 0.30 m vs 0.75 m
        _, left = make([8.0, 3.0, 75.0])                             # turned right of the shelf -> marker on the left
        self.assertLess(left.look().find("shelf", 203).image_offset[0], -0.2)

    def test_full_run_sees_203_only_from_near_the_shelf(self):
        camera = SimulatedWorldCamera(build_world_markers(CFG, 102))
        vision = Perception(camera, CFG, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, CFG.room_map.robot_start, CFG.standoff_cm)
        camera.bind(lambda: (*robot.current_position, robot.current_heading))
        marker_x, marker_y = SHELF_C.x, SHELF_C.y - CFG.room_map.shelf_depth_m / 2
        sightings, real_read = [], camera.read

        def spy():
            if 203 in ids(camera):
                sightings.append((robot.position, robot.holding))
            return real_read()

        camera.read = spy
        machine = PickAndPlaceStateMachine(robot, CFG.products, CFG.shelves, PathPlanner(CFG.room_map),
                                           calibrated=False)
        result = machine.run(102)
        self.assertTrue(result.success, result.failure_reason)
        self.assertTrue(sightings)
        for (x, y), holding in sightings:
            self.assertLess(((x - marker_x) ** 2 + (y - marker_y) ** 2) ** 0.5, 1.5)   # detection range
            self.assertGreater(y, 2.9)                               # never seen from the start area
            self.assertEqual(holding, 102)                           # always while still carrying the product
        h = result.history
        self.assertLess(h.index(S.NAVIGATE_TO_SHELF), h.index(S.DETECT_SHELF))
        self.assertEqual(robot.shelf_contents, {"C": [102]})
        self.assertIsNone(robot.holding)


if __name__ == "__main__":
    unittest.main()
