import logging
import unittest

import numpy as np

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)
from aruco.calibration import CameraCalibration
from aruco.perception import Perception
from aruco.pose_estimator import PoseEstimator
from camera.camera_source import (ScriptStep, SimulatedWorldCamera, SyntheticCameraSource,
                                  build_world_markers)
from config.loader import load_config
from navigation.path_planner import PathPlanner
from robot.robot_controller import SimulatedRobot
from robot.state_machine import MachineSettings, PickAndPlaceStateMachine, PlacementPreconditionError, State

D = "ARUCO_ORIGINAL"
BLANK = []
PRODUCT_B = [(D, 102)]
SHELF_C = [(D, 203)]
BOTH = PRODUCT_B + SHELF_C


def run(script, product_id=102, settings=None, calibration=None):
    cfg = load_config()
    vision = Perception(SyntheticCameraSource([ScriptStep(m, n) for m, n in script]), cfg,
                        PoseEstimator(calibration), display=False)
    robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
    machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                       calibrated=calibration is not None, settings=settings)
    return machine.run(product_id), robot


FAST = MachineSettings(search_max_frames=30, recovery_scan_frames=3, max_recoveries=2)
S = State


class StateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def test_happy_path(self):
        result, robot = run([(BLANK, 3), (BOTH, 1)])
        self.assertTrue(result.success)
        self.assertEqual(result.history, [
            S.SEARCH_PRODUCT, S.DETECT_PRODUCT, S.VERIFY_PRODUCT, S.PICK_PRODUCT, S.GET_TARGET_SHELF,
            S.NAVIGATE_TO_SHELF, S.DETECT_SHELF, S.ALIGN_WITH_SHELF, S.APPROACH_SHELF,
            S.MOVE_TO_PLACEMENT_POSITION, S.PLACE_PRODUCT, S.VERIFY_PLACEMENT, S.DONE])
        self.assertEqual(robot.shelf_contents, {"C": [102]})
        self.assertIsNone(robot.holding)
        self.assertEqual(robot.position, (8.0, 3.45))     # shelf C placement point: 30 cm in front of the shelf face

    def test_shelf_detection_alone_does_not_place_the_product(self):
        # Stop the run right after DETECT_SHELF: the product must still be in the gripper.
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        seen = []

        def on_state(state):
            seen.append((state, robot.holding, dict(robot.shelf_contents)))
            if state == S.MOVE_TO_PLACEMENT_POSITION:
                raise KeyboardInterrupt

        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False, on_state=on_state)
        with self.assertRaises(KeyboardInterrupt):
            machine.run(102)
        by_state = {state: (holding, contents) for state, holding, contents in seen}
        for state in (S.PICK_PRODUCT, S.GET_TARGET_SHELF, S.NAVIGATE_TO_SHELF, S.DETECT_SHELF,
                      S.ALIGN_WITH_SHELF, S.APPROACH_SHELF, S.MOVE_TO_PLACEMENT_POSITION):
            if state != S.PICK_PRODUCT:      # on_state fires on entry, before PICK runs
                self.assertEqual(by_state[state], (102, {}), state.name)
        self.assertEqual(machine.carried_product.marker_id, 102)
        self.assertIsNone(machine.placed_on)

    def test_robot_actually_travels_carrying_the_product_before_shelf_detection(self):
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        snapshots = {}

        def on_state(state):        # fires on entry to each state
            snapshots[state] = (robot.position, robot.holding)

        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False, on_state=on_state)
        self.assertTrue(machine.run(102).success)
        start = cfg.room_map.robot_start
        self.assertEqual(snapshots[S.NAVIGATE_TO_SHELF], (start, 102))
        pos, holding = snapshots[S.DETECT_SHELF]                    # position on entry == after navigation
        self.assertEqual(holding, 102)                              # still carried while travelling
        self.assertEqual(pos, cfg.shelves.get("C").approach)
        self.assertNotEqual(pos, start)
        self.assertEqual(snapshots[S.ALIGN_WITH_SHELF][0], pos)     # alignment does not move the robot
        self.assertEqual(snapshots[S.PLACE_PRODUCT][0], machine.placement_position)
        self.assertNotEqual(machine.placement_position, pos)        # final approach moved it closer
        self.assertEqual(machine.target_shelf.shelf_id, "C")

    def test_shelf_search_is_refused_until_robot_is_near_the_shelf(self):
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        machine._shelf = machine.target_shelf = cfg.shelves.get("C")
        self.assertEqual(machine._detect_shelf(), S.NAVIGATE_TO_SHELF)      # robot still at its start, marker in view
        robot.move_to(*machine._shelf.approach)
        self.assertEqual(machine._detect_shelf(), S.ALIGN_WITH_SHELF)

    def test_injected_203_while_far_away_still_needs_navigation_alignment_and_final_approach(self):
        # The script camera ignores position, so it "sees" 203 from the very start, while the robot is at (5.0, 0.5).
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        self.assertEqual(robot.position, (5.0, 0.5))
        placed_at = []
        real_place = robot.place_product
        robot.place_product = lambda shelf_id: (placed_at.append(robot.position), real_place(shelf_id))[1]
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        result = machine.run(102)
        self.assertTrue(result.success)
        h = result.history
        self.assertLess(h.index(S.NAVIGATE_TO_SHELF), h.index(S.DETECT_SHELF))    # travelled before "finding" the shelf
        self.assertLess(h.index(S.ALIGN_WITH_SHELF), h.index(S.APPROACH_SHELF))
        self.assertLess(h.index(S.APPROACH_SHELF), h.index(S.MOVE_TO_PLACEMENT_POSITION))
        self.assertLess(h.index(S.MOVE_TO_PLACEMENT_POSITION), h.index(S.PLACE_PRODUCT))
        self.assertEqual(placed_at, [machine.placement_position])                 # placed only at the placement point
        self.assertNotEqual(placed_at[0], (5.0, 0.5))

    def test_place_is_refused_with_a_clear_error_unless_every_precondition_holds(self):
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        product, shelf = cfg.products.get_by_marker(102), cfg.shelves.get("C")
        robot.pick_product(102)
        robot.move_to(*shelf.approach)
        machine._product, machine._shelf, machine.target_shelf, machine.carried_product = product, shelf, shelf, product
        machine.shelf_detected_marker = 203                                        # 203 seen: necessary, not sufficient
        with self.assertRaises(PlacementPreconditionError) as ctx:
            machine._place_product()
        message = str(ctx.exception)
        self.assertIn("alignment succeeded", message)
        self.assertIn("measured distance to the shelf marker", message)
        self.assertIn("final placement position", message)
        self.assertIn("final approach movement completed", message)
        self.assertEqual(robot.holding, 102)                                       # nothing was released
        self.assertEqual(robot.shelf_contents, {})
        machine.shelf_detected_marker = 201                                        # some other shelf's marker
        with self.assertRaises(PlacementPreconditionError) as ctx:
            machine._place_product()
        self.assertIn("detected shelf marker is the target shelf marker", str(ctx.exception))
        machine.shelf_detected_marker, machine.carried_product = 203, None         # nothing carried
        with self.assertRaises(PlacementPreconditionError) as ctx:
            machine._place_product()
        self.assertIn("a product is being carried", str(ctx.exception))

    def test_seeing_the_shelf_marker_is_not_reaching_it(self):
        """The gap the camera measures has to be travelled; measuring it alone places nothing."""
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        product, shelf = cfg.products.get_by_marker(102), cfg.shelves.get("C")
        robot.pick_product(102)
        robot.move_to(*shelf.approach)
        machine._product, machine._shelf, machine.target_shelf, machine.carried_product = product, shelf, shelf, product
        machine.shelf_detected_marker = 203
        machine.alignment_complete = True
        robot._aligned = True                        # aligned, marker verified, gap NOT yet closed
        self.assertFalse(machine.approach_complete)
        with self.assertRaises(PlacementPreconditionError) as ctx:
            machine._place_product()
        self.assertIn("measured distance to the shelf marker", str(ctx.exception))
        self.assertEqual(robot.holding, 102)

        before = robot.position
        self.assertEqual(machine._approach_shelf(), S.MOVE_TO_PLACEMENT_POSITION)
        self.assertTrue(machine.approach_complete)
        self.assertIsNotNone(machine.measured_distance_cm)       # a distance was actually read off the camera
        self.assertGreater(robot.position[1], before[1])         # and the robot drove some of it
        self.assertEqual(robot.position, machine._placement_position())

    def test_approach_stops_at_the_measured_standoff_on_a_position_aware_camera(self):
        """With a camera that reacts to the robot's pose, the approach ends on the measurement."""
        cfg = load_config()
        camera = SimulatedWorldCamera(build_world_markers(cfg, 102))
        vision = Perception(camera, cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        camera.bind(lambda: (*robot.current_position, robot.current_heading))
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        distances = []
        result = machine.run(102)
        self.assertTrue(result.success, result.failure_reason)
        self.assertIn(S.APPROACH_SHELF, result.history)
        # The reading that ended the approach is inside the accepted band, not a map assumption.
        self.assertLessEqual(machine.measured_distance_cm,
                             cfg.standoff_cm + machine.cfg.approach_tolerance_cm)
        self.assertEqual(robot.shelf_contents, {"C": [102]})

    def test_carried_product_is_cleared_only_after_verified_placement(self):
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        machine = PickAndPlaceStateMachine(robot, cfg.products, cfg.shelves, PathPlanner(cfg.room_map),
                                           calibrated=False)
        self.assertTrue(machine.run(102).success)
        self.assertIsNone(machine.carried_product)
        self.assertEqual(machine.placed_on, "C")

    def test_robot_refuses_to_place_without_final_approach(self):
        cfg = load_config()
        vision = Perception(SyntheticCameraSource([ScriptStep(BOTH, 1)]), cfg, PoseEstimator(None), display=False)
        robot = SimulatedRobot(vision, cfg.room_map.robot_start, cfg.standoff_cm)
        robot.pick_product(102)
        self.assertFalse(robot.move_to_placement_position(8.0, 3.45))     # not aligned yet
        robot.align_with_shelf(robot.detect_target_shelf(203))            # 203 injected while the robot is far away
        self.assertFalse(robot.place_product("C"))                        # aligned, but no final approach
        self.assertFalse(robot.move_to_placement_position(8.0, 3.45))     # "final approach" may not cross the room
        self.assertEqual(robot.position, cfg.room_map.robot_start)        # ...and the robot did not teleport
        self.assertFalse(robot.place_product("C"))
        self.assertEqual(robot.holding, 102)
        robot.move_to(*cfg.shelves.get("C").approach)                     # real navigation voids the alignment
        self.assertFalse(robot.move_to_placement_position(8.0, 3.45))
        robot.align_with_shelf(robot.detect_target_shelf(203))
        self.assertTrue(robot.move_to_placement_position(8.0, 3.45))
        self.assertTrue(robot.place_product("C"))
        self.assertEqual(robot.shelf_contents, {"C": [102]})

    def test_shelf_lost_during_alignment_recovers_and_completes(self):
        # Product frames for search+detect+verify (5), then shelf frames for scan+3 stable (4), then gone during ALIGN_WITH_SHELF.
        result, robot = run([(PRODUCT_B, 5), (SHELF_C, 4), (BLANK, 40), (BOTH, 1)])
        self.assertTrue(result.success)
        self.assertIn(S.RECOVERY, result.history)
        i = result.history.index(S.RECOVERY)
        self.assertEqual(result.history[i + 1], S.DETECT_SHELF)
        self.assertEqual(result.history[-5:], [S.APPROACH_SHELF, S.MOVE_TO_PLACEMENT_POSITION,
                                               S.PLACE_PRODUCT, S.VERIFY_PLACEMENT, S.DONE])
        self.assertEqual(robot.shelf_contents, {"C": [102]})

    def test_lost_product_marker_triggers_recovery_then_completes(self):
        # Visible briefly, gone long enough to count as lost, then back.
        result, robot = run([(BLANK, 2), (PRODUCT_B, 2), (BLANK, 40), (BOTH, 1)])
        self.assertTrue(result.success)
        self.assertIn(S.RECOVERY, result.history)
        self.assertEqual(robot.shelf_contents, {"C": [102]})

    def test_shelf_never_found_ends_in_failed_after_bounded_recoveries(self):
        result, robot = run([(PRODUCT_B, 1)], settings=FAST)
        self.assertEqual(result.final_state, S.FAILED)
        self.assertEqual(result.history.count(S.RECOVERY), FAST.max_recoveries + 1)
        self.assertEqual(robot.shelf_contents, {})
        self.assertEqual(robot.holding, 102)                # still holding: nothing was dropped anywhere

    def test_wrong_shelf_marker_is_never_used(self):
        result, robot = run([(PRODUCT_B + [(D, 201)], 1)], settings=FAST)    # shelf A visible, C wanted
        self.assertFalse(result.success)
        self.assertEqual(robot.shelf_contents, {})

    def test_product_never_found_fails_without_crashing(self):
        result, _ = run([(BLANK, 1)], settings=FAST)
        self.assertEqual(result.final_state, S.FAILED)
        self.assertNotIn(S.PICK_PRODUCT, result.history)

    def test_unknown_product_fails_immediately(self):
        result, _ = run([(BLANK, 1)], product_id=999)
        self.assertEqual(result.history, [S.FAILED])
        self.assertIn("not in the product database", result.failure_reason)

    def test_calibrated_camera_gives_metric_shelf_pose(self):
        # Synthetic frames draw the 18.7 cm shelf marker 140 px wide; with f=600 px that is ~80 cm away.
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        calibration = CameraCalibration(K, np.zeros((5, 1)), (640, 480), 0.2, 15, (7, 6), 2.5)
        result, robot = run([(BOTH, 1)], calibration=calibration)
        self.assertTrue(result.success)
        pose = robot.last_observation.find("shelf", 203).pose
        self.assertAlmostEqual(pose.z_cm, 18.7 * 600 / 140, delta=4.0)
        self.assertLess(abs(pose.yaw_deg), 5.0)

    def test_other_products_in_view_are_ignored(self):
        result, robot = run([([(D, 101)] + BOTH, 1)])
        self.assertTrue(result.success)
        self.assertEqual(robot.shelf_contents, {"C": [102]})


if __name__ == "__main__":
    unittest.main()
