"""The real-webcam workflow: `python main.py --pick 102` with a person aiming the camera.

These tests drive the same code path as a live run (MachineSettings.for_live_camera, an ordinary
CameraSource, no --synthetic) and feed it a scripted stand-in for the webcam, so the physical test can
be reproduced without a camera: show marker 102, walk to the wall (a long stretch of frames with
nothing in view), then show marker 203.

The rule under test throughout: detecting a marker is never enough to place. Only PLACE_PRODUCT places,
and only after picking, navigating, detecting the *target* shelf marker, aligning and the final approach.
"""
import logging
import unittest

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)
from aruco.perception import Perception
from aruco.pose_estimator import PoseEstimator
from camera.camera_source import ScriptStep, SyntheticCameraSource
from config.loader import load_config
from navigation.path_planner import PathPlanner
from robot.robot_controller import SimulatedRobot
from robot.state_machine import MachineSettings, PickAndPlaceStateMachine, State

D = "ARUCO_ORIGINAL"
BLANK = []
PRODUCT_102 = [(D, 102)]
SHELF_203 = [(D, 203)]          # Shelf C: the target shelf of product 102
SHELF_204 = [(D, 204)]          # Shelf D: a real shelf marker, but the wrong one
S = State

# The operator walking the last stretch: the same marker, rendered larger because the camera is nearer.
# At this scale the 18.7 cm shelf marker reads ~36 cm away, inside the 30+10 cm stand-off band.
CLOSE = 2.4

# Frame budgets instead of wall-clock ones, so the tests are deterministic and fast. Everything else
# (operator_driven, no robot sweep, generous missed-frame tolerance) matches a live run.
LIVE = dict(search_timeout_s=None, confirm_timeout_s=None, search_max_frames=4000, max_recoveries=1)
SHORT = dict(LIVE, search_max_frames=40)

# What the old, simulated-sweep budget was: this is what a live run used to get.
SIMULATED_BUDGET = dict(operator_driven=False, search_max_frames=300, search_timeout_s=None,
                        confirm_max_frames=300, confirm_timeout_s=None, max_missed_frames=15,
                        recovery_scan_frames=15, max_recoveries=3)


class LiveRun:
    """One scripted live-camera run, recording every placement and where it happened."""

    def __init__(self, script, product_id=102, prepare=None, on_state=None, **settings):
        self.cfg = load_config()
        self.camera = SyntheticCameraSource([ScriptStep(*step) for step in script])
        self.frames_read = 0
        camera_read = self.camera.read

        def counting_read():
            self.frames_read += 1
            return camera_read()

        self.camera.read = counting_read
        vision = Perception(self.camera, self.cfg, PoseEstimator(None), display=False)
        self.robot = SimulatedRobot(vision, self.cfg.room_map.robot_start, self.cfg.standoff_cm)
        self.placements = []
        real_place = self.robot.place_product

        def spy(shelf_id):
            self.placements.append((shelf_id, self.robot.position, self.robot.holding))
            return real_place(shelf_id)

        self.robot.place_product = spy
        self.machine = PickAndPlaceStateMachine(
            self.robot, self.cfg.products, self.cfg.shelves, PathPlanner(self.cfg.room_map),
            calibrated=False, on_state=on_state,
            settings=MachineSettings.for_live_camera(**dict(LIVE, **settings)))
        if prepare is not None:
            prepare(self.robot, self.machine)
        self.result = self.machine.run(product_id)


class LiveCameraWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)

    def assert_nothing_placed(self, run):
        self.assertEqual(run.placements, [], "place_product() was called")
        self.assertEqual(run.robot.shelf_contents, {})
        self.assertIsNone(run.machine.placed_on)
        self.assertNotIn(S.PLACE_PRODUCT, run.result.history)

    # ---------------------------------------------------------------- TEST 1
    def test_product_detected_but_no_shelf_marker_ever_seen_never_places(self):
        run = LiveRun([(PRODUCT_102, 6), (BLANK, 1)], **SHORT)
        self.assertFalse(run.result.success)
        self.assertEqual(run.result.final_state, S.FAILED)
        self.assert_nothing_placed(run)
        self.assertEqual(run.robot.holding, 102)        # still carried, not dropped anywhere
        self.assertEqual(run.machine.carried_product.marker_id, 102)

    # ---------------------------------------------------------------- TEST 2
    def test_wrong_shelf_marker_204_never_places(self):
        run = LiveRun([(PRODUCT_102, 6), (SHELF_204, 1)], **SHORT)
        self.assertFalse(run.result.success)
        self.assert_nothing_placed(run)
        self.assertIsNone(run.machine.shelf_detected_marker)
        self.assertEqual(run.robot.holding, 102)

    def test_product_marker_102_shown_again_is_not_mistaken_for_the_shelf(self):
        # The operator brings 102 back into view while walking: still a product marker, still no placement.
        run = LiveRun([(PRODUCT_102, 6), (SHELF_204 + PRODUCT_102, 1)], **SHORT)
        self.assertFalse(run.result.success)
        self.assert_nothing_placed(run)

    # ---------------------------------------------------------------- TEST 3
    def test_shelf_203_detected_but_alignment_not_complete_never_places(self):
        # 203 is seen and verified, then leaves the view before ALIGN_WITH_SHELF can use it.
        run = LiveRun([(PRODUCT_102, 6), (SHELF_203, 4), (BLANK, 1)], **SHORT)
        self.assertFalse(run.result.success)
        self.assertIn(S.DETECT_SHELF, run.result.history)
        self.assertIn(S.ALIGN_WITH_SHELF, run.result.history)
        self.assertFalse(run.machine.alignment_complete)
        self.assert_nothing_placed(run)

    # ---------------------------------------------------------------- TEST 4
    def test_aligned_but_final_placement_position_not_reached_never_places(self):
        def block_final_approach(robot, machine):
            robot.move_to_placement_position = lambda x, y: False

        run = LiveRun([(PRODUCT_102, 6), (SHELF_203, 5), (SHELF_203, 1, CLOSE)],
                      prepare=block_final_approach, **SHORT)
        self.assertFalse(run.result.success)
        self.assertIn(S.MOVE_TO_PLACEMENT_POSITION, run.result.history)
        self.assertTrue(run.machine.alignment_complete)          # alignment did succeed
        self.assertFalse(run.machine.at_final_placement_position)
        self.assert_nothing_placed(run)
        self.assertEqual(run.robot.holding, 102)

    # ---------------------------------------------------------------- TEST 5
    def test_full_live_sequence_places_only_at_the_end(self):
        run = LiveRun([(PRODUCT_102, 6), (SHELF_203, 12), (SHELF_203, 1, CLOSE)], **SHORT)
        self.assertTrue(run.result.success, run.result.failure_reason)
        self.assertEqual(run.result.history, [
            S.SEARCH_PRODUCT, S.DETECT_PRODUCT, S.VERIFY_PRODUCT, S.PICK_PRODUCT, S.GET_TARGET_SHELF,
            S.NAVIGATE_TO_SHELF, S.DETECT_SHELF, S.ALIGN_WITH_SHELF, S.APPROACH_SHELF,
            S.MOVE_TO_PLACEMENT_POSITION, S.PLACE_PRODUCT, S.VERIFY_PLACEMENT, S.DONE])
        self.assertEqual(run.placements, [("C", run.machine.placement_position, 102)])
        self.assertEqual(run.robot.shelf_contents, {"C": [102]})
        self.assertIsNone(run.robot.holding)
        self.assertEqual(run.machine.shelf_detected_marker, 203)

    # ---------------------------------------------------------------- TEST 6
    def test_long_walk_to_the_shelf_keeps_searching_and_places_only_after_203_appears(self):
        """The physical test: 102 shown, removed, a long walk with nothing in view, then 203.

        This is what the live budgets exist for. With the simulated-sweep budget (300 frames per
        search) the walk exhausted the recoveries and the run ended in FAILED.
        """
        walk = 1500                                   # ~50 s of webcam frames with no marker in view
        run = LiveRun([(PRODUCT_102, 6), (BLANK, walk), (SHELF_203, 10), (SHELF_203, 1, CLOSE)], **LIVE)
        self.assertTrue(run.result.success, run.result.failure_reason)
        self.assertNotIn(S.RECOVERY, run.result.history)             # the walk is normal, not a failure
        self.assertEqual(run.placements, [("C", run.machine.placement_position, 102)])
        self.assertGreater(run.frames_read, walk)                    # it really waited out the whole walk
        self.assertEqual(run.result.history.count(S.PLACE_PRODUCT), 1)

    def test_no_placement_while_the_shelf_marker_has_not_been_seen_yet(self):
        """Same script, stopped the moment the shelf search starts: the product is still in the gripper."""
        states = []
        parts = []

        class Stop(Exception):
            pass

        def watch(state):
            states.append(state)
            if state == S.DETECT_SHELF:
                raise Stop()

        with self.assertRaises(Stop):
            LiveRun([(PRODUCT_102, 6), (BLANK, 1500), (SHELF_203, 1, CLOSE)],
                    prepare=lambda robot, machine: parts.append((robot, machine)),
                    on_state=watch, **LIVE)
        robot, machine = parts[0]
        self.assertEqual(states[-1], S.DETECT_SHELF)
        self.assertNotIn(S.PLACE_PRODUCT, states)
        self.assertEqual(robot.holding, 102)                   # picked and still carried
        self.assertEqual(robot.shelf_contents, {})             # nothing placed anywhere
        self.assertEqual(machine.carried_product.marker_id, 102)
        self.assertEqual(machine.target_shelf.shelf_id, "C")
        self.assertIsNone(machine.shelf_detected_marker)       # 203 has not been seen
        self.assertFalse(machine.alignment_complete)
        self.assertFalse(machine.at_final_placement_position)
        self.assertIsNone(machine.placed_on)

    # ---------------------------------------------------------------- TEST 7
    def test_shelf_203_seen_from_across_the_room_is_never_close_enough_to_place(self):
        """The bug the operator hit: 203 comes into view and the product goes down on the spot.

        Here 203 is visible and verified for the whole run, but always at ~86 cm. The stand-off is
        30 cm, so the approach never completes and nothing is ever placed.
        """
        run = LiveRun([(PRODUCT_102, 6), (SHELF_203, 1)], **SHORT)
        self.assertFalse(run.result.success)
        self.assertEqual(run.machine.shelf_detected_marker, 203)     # the shelf WAS found...
        self.assertTrue(run.machine.alignment_complete)              # ...and aligned with...
        self.assertIn(S.APPROACH_SHELF, run.result.history)
        self.assertFalse(run.machine.approach_complete)              # ...but never reached
        self.assertGreater(run.machine.measured_distance_cm, run.robot.standoff_cm)
        self.assert_nothing_placed(run)
        self.assertEqual(run.robot.holding, 102)                     # still in the gripper

    def test_walking_the_last_stretch_is_what_completes_the_approach(self):
        """Nothing changes until the camera itself reads a shorter distance."""
        readings = []
        run = LiveRun([(PRODUCT_102, 6), (SHELF_203, 12), (SHELF_203, 1, CLOSE)],
                      on_state=lambda state: readings.append(None), **SHORT)
        self.assertTrue(run.result.success, run.result.failure_reason)
        self.assertTrue(run.machine.approach_complete)
        self.assertLessEqual(run.machine.measured_distance_cm,
                             run.robot.standoff_cm + run.machine.cfg.approach_tolerance_cm)
        self.assertFalse(run.machine.distance_is_metric)             # uncalibrated: an estimate, and it says so
        self.assertEqual(run.placements, [("C", run.machine.placement_position, 102)])

    # ---------------------------------------------------------------- live-mode behaviour
    def test_the_simulated_sweep_is_not_run_when_the_operator_aims_the_camera(self):
        """A person turning the laptop is the search; faking a robot sweep would invent headings."""
        headings = {}

        def run_with(**settings):
            def watch(state):
                headings.setdefault(state, robot_box[0].current_heading)
            robot_box = []
            run = LiveRun([(PRODUCT_102, 6), (BLANK, 200), (SHELF_203, 1, CLOSE)],
                          prepare=lambda robot, machine: robot_box.append(robot),
                          on_state=watch, **settings)
            return run, dict(headings)

        headings = {}
        _, live = run_with(**LIVE)
        self.assertAlmostEqual(live[S.DETECT_SHELF], live[S.ALIGN_WITH_SHELF])   # 200 frames, no turning

        headings = {}
        _, swept = run_with(**dict(SIMULATED_BUDGET, search_max_frames=4000, max_recoveries=1))
        self.assertNotAlmostEqual(swept[S.DETECT_SHELF], swept[S.ALIGN_WITH_SHELF])

    def test_live_budget_survives_a_walk_that_the_simulated_budget_fails(self):
        script = [(PRODUCT_102, 6), (BLANK, 1500), (SHELF_203, 1, CLOSE)]
        old = LiveRun(script, **SIMULATED_BUDGET)
        self.assertFalse(old.result.success)                         # the bug this suite pins down
        self.assertEqual(old.result.final_state, S.FAILED)
        self.assertEqual(old.placements, [])
        self.assertTrue(LiveRun(script, **LIVE).result.success)      # the live budget handles it


if __name__ == "__main__":
    unittest.main()
