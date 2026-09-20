"""Pick-and-place state machine.

Drives a Robot using a ProductManager/ShelfManager/PathPlanner. A lost marker or timeout sends the
machine to RECOVERY (bounded retries) instead of crashing; unrecoverable problems (unknown product,
no path) end in FAILED.
"""
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterator, List, Optional, Tuple

from aruco.detector import Detection
from navigation.path_planner import PathPlanner
from products.product_manager import Product, ProductManager
from robot.robot_controller import Robot
from shelves.shelf_manager import Shelf, ShelfManager

log = logging.getLogger("warehouse")


class State(Enum):
    SEARCH_PRODUCT = auto()
    DETECT_PRODUCT = auto()
    VERIFY_PRODUCT = auto()
    PICK_PRODUCT = auto()
    GET_TARGET_SHELF = auto()
    NAVIGATE_TO_SHELF = auto()
    DETECT_SHELF = auto()
    ALIGN_WITH_SHELF = auto()
    APPROACH_SHELF = auto()
    MOVE_TO_PLACEMENT_POSITION = auto()
    PLACE_PRODUCT = auto()
    VERIFY_PLACEMENT = auto()
    RECOVERY = auto()
    DONE = auto()
    FAILED = auto()


# Where to resume after RECOVERY. Once the product is in the gripper we never go back to searching for it.
_RESUME = {
    State.SEARCH_PRODUCT: State.SEARCH_PRODUCT,
    State.DETECT_PRODUCT: State.SEARCH_PRODUCT,
    State.VERIFY_PRODUCT: State.SEARCH_PRODUCT,
    State.PICK_PRODUCT: State.SEARCH_PRODUCT,
    State.NAVIGATE_TO_SHELF: State.NAVIGATE_TO_SHELF,
    State.DETECT_SHELF: State.DETECT_SHELF,
    State.ALIGN_WITH_SHELF: State.DETECT_SHELF,
    State.APPROACH_SHELF: State.DETECT_SHELF,                 # the measured gap is only valid for a marker in view
    State.MOVE_TO_PLACEMENT_POSITION: State.DETECT_SHELF,     # alignment is void once we move: re-find and re-align
    State.PLACE_PRODUCT: State.DETECT_SHELF,
    State.VERIFY_PLACEMENT: State.DETECT_SHELF,
}


@dataclass
class MachineSettings:
    stable_frames: int = 3            # consecutive frames a marker must be seen to count as detected
    max_missed_frames: int = 15       # consecutive frames without the marker before it counts as lost
    search_max_frames: int = 300      # frames to look for a marker before giving up (RECOVERY)
    search_timeout_s: Optional[float] = None   # wall-clock budget for one search; None = frames only
    confirm_max_frames: Optional[int] = None   # frames to stabilise a found marker (default: search_max_frames)
    confirm_timeout_s: Optional[float] = None  # wall-clock budget for stabilising (None = frames only)
    progress_every_s: float = 5.0     # how often a long search reports that it is still waiting
    recovery_scan_frames: int = 15    # frames the simulated recovery sweep lasts
    max_recoveries: int = 3           # total RECOVERY visits per run before FAILED
    operator_driven: bool = False     # a person aims the camera (live webcam), the simulated robot does not
    approach_tolerance_cm: float = 10.0   # how much further than the stand-off a measured distance may read

    @classmethod
    def for_live_camera(cls, **overrides) -> "MachineSettings":
        """Budgets for a real webcam held by a person.

        The defaults above are sized for the simulated sweep, where the robot turns 2 deg per frame and
        a marker either is or is not reachable within ~300 frames (~10 s). With a live webcam the same
        300 frames mean "the operator has 10 seconds to carry the laptop to the wall", which is why a
        real run used to end in RECOVERY -> FAILED. Here the search waits on the wall clock instead,
        reports progress, and never fakes a robot sweep the operator is actually performing by hand.
        """
        live = dict(max_missed_frames=45, search_max_frames=10 ** 7, search_timeout_s=180.0,
                    confirm_max_frames=900, confirm_timeout_s=30.0, recovery_scan_frames=0,
                    operator_driven=True)
        live.update(overrides)
        return cls(**live)

    def budget_description(self) -> str:
        return "{:.0f} s".format(self.search_timeout_s) if self.search_timeout_s is not None \
            else "{} frames".format(self.search_max_frames)


@dataclass
class RunResult:
    success: bool
    final_state: State
    history: List[State]
    failure_reason: Optional[str] = None

    def sequence(self) -> str:
        return " -> ".join(s.name for s in self.history)


class PlacementPreconditionError(RuntimeError):
    """place_product() was about to run while a required condition was false. This is a logic bug,
    never a normal runtime failure, so it is raised instead of being routed through RECOVERY."""


class PickAndPlaceStateMachine:
    ARRIVAL_TOLERANCE_M = 0.1     # how far short of the approach point still counts as "reached the shelf area"
    POSITION_TOLERANCE_M = 0.02   # how close to placement_position counts as "reached it"
    APPROACH_STEP_M = 0.10        # one creep step of the measured approach
    ODOMETRY_ARRIVAL_M = 0.01     # map distance to the placement point that counts as "drove the whole way"

    # Defaults so the state handlers can be exercised before/without run(); run() re-initialises all of them.
    _product: Optional[Product] = None
    _shelf: Optional[Shelf] = None
    target_shelf: Optional[Shelf] = None
    placement_position: Optional[Tuple[float, float]] = None
    carried_product: Optional[Product] = None

    def __init__(self, robot: Robot, products: ProductManager, shelves: ShelfManager,
                 planner: PathPlanner, calibrated: bool, settings: Optional[MachineSettings] = None,
                 on_state: Optional[Callable[[State], None]] = None):
        self.robot = robot
        self.products = products
        self.shelves = shelves
        self.planner = planner
        self.calibrated = calibrated
        self.cfg = settings or MachineSettings()
        self._on_state = on_state
        self._handlers = {
            State.SEARCH_PRODUCT: self._search_product,
            State.DETECT_PRODUCT: self._detect_product,
            State.VERIFY_PRODUCT: self._verify_product,
            State.PICK_PRODUCT: self._pick_product,
            State.GET_TARGET_SHELF: self._get_target_shelf,
            State.NAVIGATE_TO_SHELF: self._navigate_to_shelf,
            State.DETECT_SHELF: self._detect_shelf,
            State.ALIGN_WITH_SHELF: self._align_with_shelf,
            State.APPROACH_SHELF: self._approach_shelf,
            State.MOVE_TO_PLACEMENT_POSITION: self._move_to_placement_position,
            State.PLACE_PRODUCT: self._place_product,
            State.VERIFY_PLACEMENT: self._verify_placement,
            State.RECOVERY: self._recovery,
        }
        self._reset_placement_flags()

    def _reset_placement_flags(self) -> None:
        """Evidence that the robot may place. Each flag is earned by its own state and voided whenever
        the robot moves away again (NAVIGATE_TO_SHELF / DETECT_SHELF clear them)."""
        self.shelf_detected_marker: Optional[int] = None      # set by DETECT_SHELF after verifying the marker
        self.alignment_complete = False                       # set by ALIGN_WITH_SHELF
        self.approach_complete = False                        # set by APPROACH_SHELF once the gap is closed
        self.measured_distance_cm: Optional[float] = None     # last camera->shelf-marker distance, cm
        self.distance_is_metric = False                       # True only when it came from a calibrated pose
        self.at_final_placement_position = False              # set by MOVE_TO_PLACEMENT_POSITION

    # ------------------------------------------------------------------ run loop
    def run(self, product_marker_id: int) -> RunResult:
        self._product: Optional[Product] = self.products.get_by_marker(product_marker_id)
        self._shelf: Optional[Shelf] = None
        self.target_shelf: Optional[Shelf] = None          # set by GET_TARGET_SHELF from the config
        self.placement_position: Optional[Tuple[float, float]] = None   # set by MOVE_TO_PLACEMENT_POSITION
        self._product_det: Optional[Detection] = None
        self._shelf_det: Optional[Detection] = None
        self.carried_product: Optional[Product] = None     # set by PICK_PRODUCT, cleared only by a verified placement
        self.placed_on: Optional[str] = None               # shelf ID the product was placed on
        self._reason: Optional[str] = None
        self._recoveries = 0
        self._resume_to = State.SEARCH_PRODUCT
        self._hinted = set()
        self._reset_placement_flags()

        history: List[State] = []
        if self._product is None:
            return self._finish(State.FAILED, history,
                                "marker {} is not in the product database".format(product_marker_id))

        state = State.SEARCH_PRODUCT
        while state not in (State.DONE, State.FAILED):
            history.append(state)
            log.info("\nROBOT STATE: %s", state.name)
            if self._on_state:
                self._on_state(state)
            failed_state = state
            state = self._handlers[state]()
            if state == State.RECOVERY:
                # Fallback for a state without an explicit entry: never go back to hunting for a product
                # that is already in the gripper, or the robot would try to pick a second one.
                self._resume_to = _RESUME.get(
                    failed_state,
                    State.NAVIGATE_TO_SHELF if self.carried_product is not None else State.SEARCH_PRODUCT)
                log.info("  Problem in %s: %s", failed_state.name, self._reason)
        return self._finish(state, history, self._reason if state == State.FAILED else None)

    def _finish(self, state: State, history: List[State], reason: Optional[str]) -> RunResult:
        history.append(state)
        if self._on_state:
            self._on_state(state)
        if state == State.DONE:
            log.info("\nROBOT STATE: DONE\n  %s successfully placed on Shelf %s", self._product.name, self.placed_on)
        result = RunResult(state == State.DONE, state, history, reason)
        log.info("\nROBOT STATE SEQUENCE\n  %s", result.sequence())
        if reason:
            log.info("FAILED: %s", reason)
        return result

    def _fail(self, reason: str, recoverable: bool = True) -> State:
        self._reason = reason
        return State.RECOVERY if recoverable else State.FAILED

    # ------------------------------------------------------------------ helpers
    def _budget(self, max_frames: int, timeout_s: Optional[float],
                note: Optional[Callable[[float], str]] = None) -> Iterator[int]:
        """One yield per frame, while both the frame count and the wall clock allow it.

        `note(elapsed_s)` is logged every progress_every_s seconds so that a human driving the camera
        can see the machine is still waiting for a marker rather than silently doing something else.
        """
        start = last_note = time.monotonic()
        for index in range(max_frames):
            now = time.monotonic()
            if timeout_s is not None and now - start >= timeout_s:
                return
            if note is not None and now - last_note >= self.cfg.progress_every_s:
                last_note = now
                log.info("%s", note(now - start))
            yield index

    def _confirm(self, find: Callable[[], Optional[Detection]], stable: int) -> Optional[Detection]:
        """Latest detection once `find` succeeded on `stable` consecutive frames; None if the marker
        is lost (max_missed_frames consecutive misses, or a flickering marker never stabilises)."""
        seen = missed = 0
        latest = None
        max_frames = self.cfg.confirm_max_frames if self.cfg.confirm_max_frames is not None \
            else self.cfg.search_max_frames
        for _ in self._budget(max_frames, self.cfg.confirm_timeout_s):
            det = find()
            if det is not None:
                seen, missed, latest = seen + 1, 0, det
                if seen >= stable:
                    return latest
            else:
                seen, missed = 0, missed + 1
                if missed > self.cfg.max_missed_frames:
                    return None
        return None

    def _scan_for(self, find: Callable[[], Optional[Detection]], kind: str, wanted_id: int,
                  note: Optional[Callable[[float], str]] = None) -> bool:
        """Look until `find` sees the marker once, within the search budget.

        The simulated robot sweeps to bring the marker into view; when the operator carries the camera
        (operator_driven) there is nothing to sweep, so the machine just keeps looking at the frames
        the operator supplies.
        """
        for _ in self._budget(self.cfg.search_max_frames, self.cfg.search_timeout_s, note):
            if not self.cfg.operator_driven:
                self.robot.search_step()
            if find() is not None:
                return True
            self._hint_other_markers(kind, wanted_id)
        return False

    def _hint_other_markers(self, kind: str, wanted_id: int) -> None:
        """Tell the operator when any other known marker is in view (once per marker).

        Product markers are reported too: seeing the product marker 102 again while the shelf marker is
        being searched for must not look like progress, and must never be mistaken for the shelf.
        """
        observation = self.robot.last_observation
        for det in (observation.detections if observation else []):
            if det.kind is None or (det.kind == kind and det.marker_id == wanted_id):
                continue
            if (det.kind, det.marker_id) in self._hinted:
                continue
            self._hinted.add((det.kind, det.marker_id))
            carrying = "" if self.carried_product is None else \
                "; still carrying {}, nothing is placed until {} is seen".format(self._carrying(), wanted_id)
            log.info("  Ignoring %s marker %d in view; waiting for %s marker %d%s",
                     det.kind, det.marker_id, kind, wanted_id, carrying)

    def _pos(self) -> str:
        return "({:.2f}, {:.2f})".format(*self.robot.current_position)

    def _carrying(self) -> str:
        p = self.carried_product
        return "{} ({})".format(p.name, p.marker_id) if p is not None else "nothing"

    def _near_shelf(self) -> bool:
        """True once the robot is at (or closer than) the shelf approach point."""
        s = self._shelf
        reach = math.hypot(s.x - s.approach[0], s.y - s.approach[1]) + self.ARRIVAL_TOLERANCE_M
        return math.hypot(s.x - self.robot.position[0], s.y - self.robot.position[1]) <= reach

    def _describe_pose(self, det: Detection) -> List[str]:
        if det.pose is not None:
            p = det.pose
            return [
                "  Relative Position: x={:+.1f} cm (right), y={:+.1f} cm (down), z={:.1f} cm (forward); "
                "distance {:.1f} cm; bearing {:+.1f} deg".format(p.x_cm, p.y_cm, p.z_cm, p.distance_cm, p.bearing_deg),
                "  Relative Rotation: yaw {:+.1f} deg (0 = marker faces camera), rvec = [{:+.3f}, {:+.3f}, {:+.3f}] rad".format(
                    p.yaw_deg, *p.rvec),
            ]
        return [
            "  Relative Position: UNAVAILABLE (camera not calibrated)",
            "  Relative Rotation: UNAVAILABLE (camera not calibrated)",
            "  Image-space only: offset x={:+.2f} y={:+.2f} (-1..1), {:.0f} px wide, ~{:.0f} cm "
            "(rough guess, not a measurement)".format(det.image_offset[0], det.image_offset[1],
                                                       det.side_px, det.approx_distance_cm or float("nan")),
        ]

    # ------------------------------------------------------------------ states
    def _search_product(self) -> State:
        target = self._product.marker_id
        log.info("  Searching for product marker %d (%s)", target, self._product.name)
        if self.cfg.operator_driven:
            log.info("  Hold product marker %d in front of the camera (waiting up to %s)",
                     target, self.cfg.budget_description())
        note = lambda s: "  ... still looking for product marker {} ({}) after {:.0f} s".format(
            target, self._product.name, s)
        if self._scan_for(lambda: self.robot.detect_product(target), "product", target, note):
            return State.DETECT_PRODUCT
        return self._fail("product marker {} not found within {}".format(target, self.cfg.budget_description()))

    def _detect_product(self) -> State:
        target = self._product.marker_id
        det = self._confirm(lambda: self.robot.detect_product(target), self.cfg.stable_frames)
        if det is None:
            return self._fail("product marker {} lost while confirming".format(target))
        self._product_det = det
        log.info("PRODUCT DETECTED\n  Marker ID: %d\n  Product: %s\n  Target Shelf: %s",
                 det.marker_id, self._product.name, self._product.shelf_id)
        return State.VERIFY_PRODUCT

    def _verify_product(self) -> State:
        det = self._product_det
        # Identity check: the marker must belong to the product marker type and match the request.
        if det.kind != "product" or self.products.get_by_marker(det.marker_id) is not self._product:
            return self._fail("marker {} does not verify as {}".format(det.marker_id, self._product.name))
        fresh = self._confirm(lambda: self.robot.detect_product(self._product.marker_id), 1)
        if fresh is None:
            return self._fail("product marker {} no longer visible at verification".format(det.marker_id))
        log.info("  Verified: marker %d is %s", det.marker_id, self._product.name)
        return State.PICK_PRODUCT

    def _pick_product(self) -> State:
        log.info("  Picking %s (%d)\n  Robot position: %s", self._product.name, self._product.marker_id, self._pos())
        if not self.robot.pick_product(self._product.marker_id):
            return self._fail("pick failed")
        self.carried_product = self._product
        log.info("  Carrying: %s (%d)", self._product.name, self._product.marker_id)
        return State.GET_TARGET_SHELF

    def _get_target_shelf(self) -> State:
        shelf = self.shelves.get(self._product.shelf_id)
        if shelf is None:
            return self._fail("shelf '{}' is not in the shelf database".format(self._product.shelf_id),
                              recoverable=False)
        self._shelf = self.target_shelf = shelf
        log.info("  Target shelf: %s\n  Shelf marker: %d\n  Shelf map position: (%.2f, %.2f)",
                 shelf.shelf_id, shelf.marker_id, shelf.x, shelf.y)
        return State.NAVIGATE_TO_SHELF

    def _navigate_to_shelf(self) -> State:
        self._reset_placement_flags()
        log.info("NAVIGATING TO SHELF %s\n  Carrying %s (%d)\n  Target Shelf: %s\n  Target Shelf Marker: %d",
                 self._shelf.shelf_id, self.carried_product.name, self.carried_product.marker_id,
                 self._shelf.shelf_id, self._shelf.marker_id)
        log.info("  Current position: %s\n  Target approach: (%.2f, %.2f)\n  Carrying: %s (%d)",
                 self._pos(), *self._shelf.approach, self.carried_product.name, self.carried_product.marker_id)
        path = self.planner.plan(self.robot.position, self._shelf.approach)
        if path is None:
            return self._fail("no path to shelf {} approach point".format(self._shelf.shelf_id), recoverable=False)
        log.info("  Path: %s\n  Moving:", " -> ".join("({:.2f}, {:.2f})".format(x, y) for x, y in path))
        for x, y in path[1:]:
            before = self._pos()
            if not self.robot.move_to(x, y):
                return self._fail("movement to ({:.2f}, {:.2f}) failed".format(x, y))
            log.info("  %s -> %s", before, self._pos())
        if not self._near_shelf():
            return self._fail("navigation ended at {}, not near shelf {}".format(self._pos(), self._shelf.shelf_id))
        log.info("  Navigation complete\n  Robot position: %s", self._pos())
        return State.DETECT_SHELF

    def _detect_shelf(self) -> State:
        self._reset_placement_flags()
        marker = self._shelf.marker_id
        if not self._near_shelf():
            # Marker 203 only means "found my shelf" once the robot has actually travelled there.
            log.info("  Robot at %s is not near Shelf %s yet; navigating first", self._pos(), self._shelf.shelf_id)
            return State.NAVIGATE_TO_SHELF
        log.info("  Searching for target shelf marker %d... (robot %s, heading %.0f deg)",
                 marker, self._pos(), self.robot.current_heading)
        if self.cfg.operator_driven:
            log.info("  Move the camera to Shelf %s and show marker %d (waiting up to %s).\n"
                     "  Still carrying %s: NOTHING is placed until marker %d is actually seen.",
                     self._shelf.shelf_id, marker, self.cfg.budget_description(), self._carrying(), marker)
        note = lambda s: ("  ... still searching for Shelf {} marker {} after {:.0f} s; carrying {}, "
                          "not placed".format(self._shelf.shelf_id, marker, s, self._carrying()))
        find = lambda: self.robot.detect_target_shelf(marker)
        if not self._scan_for(find, "shelf", marker, note):
            return self._fail("shelf marker {} not found within {}".format(marker, self.cfg.budget_description()))
        det = self._confirm(find, self.cfg.stable_frames)
        if det is None:
            return self._fail("shelf marker {} lost while confirming".format(marker))
        # Identity check: the marker must be a shelf marker AND belong to the shelf this product targets.
        if det.kind != "shelf" or self.shelves.get_by_marker(det.marker_id) != self.target_shelf:
            return self._fail("marker {} is not the target shelf {}".format(det.marker_id, self._shelf.shelf_id))
        self._shelf_det = det
        self.shelf_detected_marker = det.marker_id
        # Detection only confirms the target shelf was found; the product is placed later, after
        # ALIGN_WITH_SHELF and MOVE_TO_PLACEMENT_POSITION.
        if self.cfg.operator_driven:
            log.info("SHELF MARKER %d DETECTED\n  Marker %d came into the camera's view", det.marker_id, det.marker_id)
        else:
            log.info("  Marker %d came into view after sweeping to heading %.0f deg",
                     det.marker_id, self.robot.current_heading)
        log.info("  Verified: marker %d is the target Shelf %s", det.marker_id, self._shelf.shelf_id)
        log.info("  Shelf %s marker %d detected\nTARGET SHELF DETECTED\n  Shelf: %s\n  Marker ID: %d\n%s",
                 self._shelf.shelf_id, det.marker_id, self._shelf.shelf_id, det.marker_id,
                 "\n".join(self._describe_pose(det)))
        return State.ALIGN_WITH_SHELF

    def _align_with_shelf(self) -> State:
        det = self._confirm(lambda: self.robot.detect_target_shelf(self._shelf.marker_id), 1)
        if det is None:
            return self._fail("shelf marker {} lost before alignment".format(self._shelf.marker_id))
        log.info("  Robot position: %s\n  Shelf position: (%.2f, %.2f)\n  Aligning with Shelf %s marker %d...",
                 self._pos(), self._shelf.x, self._shelf.y, self._shelf.shelf_id, det.marker_id)
        result = self.robot.align_with_shelf(det)
        if not result.success:
            return self._fail("alignment failed")
        if result.metric:
            log.info("  Relative pose used for alignment: %s", result.summary)
        else:
            log.info("  Simulated alignment: %s", result.summary)
            log.info("  NOTE: alignment was NOT metric (camera uncalibrated); run --calibrate for real distances/angles")
        self.alignment_complete = True
        log.info("  Alignment successful (heading now %.1f deg)", self.robot.current_heading)
        # Facing the shelf is not standing in front of it: APPROACH_SHELF measures and closes the gap.
        return State.APPROACH_SHELF

    def _measure_shelf_distance(self, det: Detection) -> Optional[Tuple[float, bool]]:
        """(distance from the camera to the shelf marker in cm, is_metric), or None if unmeasurable.

        A calibrated camera gives the real forward distance out of solvePnP. Uncalibrated, the only
        number available is the apparent-size estimate; it is used so the robot still has to close a
        real gap, but it is never reported as a measurement.
        """
        if det.pose is not None:
            return det.pose.z_cm, True
        if det.approx_distance_cm:
            return float(det.approx_distance_cm), False
        return None

    def _distance_source(self) -> str:
        return "metric, calibrated camera" if self.distance_is_metric \
            else "apparent-size estimate, camera UNCALIBRATED"

    def _approach_shelf(self) -> State:
        """Close the measured camera-to-shelf-marker gap down to the stand-off.

        This is what makes "found the shelf" different from "standing in front of it". Seeing marker
        203 across the room is a sighting, not an arrival: the distance is measured here, and the
        robot has to travel it before anything may be placed.

        Two ways to travel it, because there are two kinds of robot:
          * the simulated robot drives itself, creeping forward and re-measuring after every step,
            capped at the placement point the map allows (it may not drive through the shelf);
          * when a person carries the camera (operator_driven) there is nothing to drive, so the
            machine keeps measuring and reporting how much closer they still have to go. Only the
            measurement can end the state -- waiting is not arriving.
        """
        marker = self._shelf.marker_id
        standoff = self.robot.standoff_cm
        limit = standoff + self.cfg.approach_tolerance_cm
        find = lambda: self.robot.detect_target_shelf(marker)
        log.info("APPROACHING SHELF %s\n  Measuring the distance to marker %d before any placement\n"
                 "  Stand-off: %.0f cm (accepted up to %.0f cm)\n  Carrying: %s",
                 self._shelf.shelf_id, marker, standoff, limit, self._carrying())
        if self.cfg.operator_driven:
            log.info("  Carry the camera toward Shelf %s. NOTHING is placed until marker %d actually "
                     "measures %.0f cm or closer (waiting up to %s).",
                     self._shelf.shelf_id, marker, limit, self.cfg.budget_description())
        reported = [None]        # last distance written to the log, so a live run does not print every frame
        note = lambda elapsed: "  ... {:.0f} s into the approach: marker {} at {}, need {:.0f} cm".format(
            elapsed, marker, "no measurement yet" if reported[0] is None else "{:.0f} cm".format(reported[0]), limit)

        for _ in self._budget(self.cfg.search_max_frames, self.cfg.search_timeout_s, note):
            det = self._confirm(find, 1)
            if det is None:
                return self._fail("shelf marker {} lost during the approach; nothing placed".format(marker))
            measured = self._measure_shelf_distance(det)
            if measured is None:
                return self._fail("cannot measure the distance to shelf marker {}".format(marker))
            distance, self.distance_is_metric = measured
            self.measured_distance_cm = distance
            remaining_cm = distance - standoff

            if distance <= limit:
                self.approach_complete = True
                log.info("  Measured distance to Shelf %s marker %d: %.0f cm (%s)",
                         self._shelf.shelf_id, marker, distance, self._distance_source())
                log.info("  STAND-OFF REACHED: %.0f cm measured, %.0f cm required. Approach complete.",
                         distance, standoff)
                return State.MOVE_TO_PLACEMENT_POSITION

            if self.cfg.operator_driven:
                # The operator is the drive train. Report the gap when it has meaningfully changed.
                if reported[0] is None or abs(reported[0] - distance) >= 5.0:
                    reported[0] = distance
                    log.info("  Marker %d measured at %.0f cm (%s): move %.0f cm CLOSER to reach the "
                             "%.0f cm stand-off; still carrying %s, not placed",
                             marker, distance, self._distance_source(), remaining_cm, standoff, self._carrying())
                continue

            # Simulated robot: drive the gap it just measured, but never past the point the map allows.
            x, y = self.placement_position = self._placement_position()
            room_left = math.hypot(x - self.robot.position[0], y - self.robot.position[1])
            if room_left <= self.ODOMETRY_ARRIVAL_M:
                self.approach_complete = True
                log.info("  Drove the full %.2f m to the placement point %s; the camera still reads %.0f cm "
                         "(%s), which the map does not support. Approach complete by odometry.",
                         math.hypot(x - self._shelf.approach[0], y - self._shelf.approach[1]), self._pos(),
                         distance, self._distance_source())
                return State.MOVE_TO_PLACEMENT_POSITION
            step = min(self.APPROACH_STEP_M, remaining_cm / 100.0, room_left)
            log.info("  Marker %d measured at %.0f cm (%s): %.0f cm to close, creeping %.2f m toward "
                     "(%.2f, %.2f) from %s",
                     marker, distance, self._distance_source(), remaining_cm, step, x, y, self._pos())
            if not self.robot.creep_toward(x, y, step):
                return self._fail("approach drive toward shelf {} failed".format(self._shelf.shelf_id))

        return self._fail("did not get within {:.0f} cm of shelf marker {} within {} (last reading {}); "
                          "still carrying {}".format(
                              limit, marker, self.cfg.budget_description(),
                              "none" if self.measured_distance_cm is None
                              else "{:.0f} cm".format(self.measured_distance_cm), self._carrying()))

    def _placement_position(self) -> Tuple[float, float]:
        """Final stand-off point: from the approach point straight toward the shelf, stopping
        `standoff` in front of the shelf's front face."""
        ax, ay = self._shelf.approach
        dx, dy = self._shelf.x - ax, self._shelf.y - ay
        centre_gap = math.hypot(dx, dy)
        if centre_gap == 0:
            return ax, ay
        travel = max(0.0, centre_gap - self.planner.map.shelf_depth_m / 2 - self.robot.standoff_cm / 100.0)
        return ax + dx / centre_gap * travel, ay + dy / centre_gap * travel

    def _move_to_placement_position(self) -> State:
        x, y = self.placement_position = self._placement_position()
        log.info("  Moving from %s\n  to final placement position (%.2f, %.2f)", self._pos(), x, y)
        if not self.robot.move_to_placement_position(x, y):
            return self._fail("final approach to ({:.2f}, {:.2f}) failed".format(x, y))
        self.at_final_placement_position = self._at_placement_position()
        log.info("  Placement position reached\n  Robot position: %s", self._pos())
        return State.PLACE_PRODUCT

    def _at_placement_position(self) -> bool:
        return self.placement_position is not None and \
            math.hypot(self.robot.position[0] - self.placement_position[0],
                       self.robot.position[1] - self.placement_position[1]) <= self.POSITION_TOLERANCE_M

    def _check_placement_preconditions(self) -> None:
        """Every condition must hold before place_product(); marker 203 being visible is only one of them.
        Deliberately explicit (not `assert`, which python -O strips)."""
        expected = self.shelves.get(self._product.shelf_id)
        checks = [
            ("a product is being carried", self.carried_product is not None),
            ("the carried product is the requested product",
             self.carried_product == self._product and self.robot.holding == self._product.marker_id),
            ("target shelf is the product's configured shelf", expected is not None and self.target_shelf == expected),
            ("shelf detection succeeded", self.shelf_detected_marker is not None),
            ("detected shelf marker is the target shelf marker",
             expected is not None and self.shelf_detected_marker == expected.marker_id),
            ("robot reached the shelf approach area", self._shelf is not None and self._near_shelf()),
            ("alignment succeeded", self.alignment_complete),
            ("the measured distance to the shelf marker was closed to the stand-off", self.approach_complete),
            ("a shelf distance was actually measured", self.measured_distance_cm is not None),
            ("robot is at the final placement position", self._at_placement_position()),
            ("final approach movement completed", self.at_final_placement_position),
        ]
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise PlacementPreconditionError(
                "refusing to place: {} (robot at {}, placement position {}, last measured shelf distance {})".format(
                    "; ".join("NOT " + name for name in failed), self._pos(), self.placement_position,
                    "none" if self.measured_distance_cm is None
                    else "{:.0f} cm".format(self.measured_distance_cm)))
        log.info("  Placement preconditions satisfied (%d/%d)", len(checks), len(checks))

    def _place_product(self) -> State:
        self._check_placement_preconditions()
        log.info("  Placing %s (%d)\n  Target Shelf: %s", self.carried_product.name,
                 self.carried_product.marker_id, self._shelf.shelf_id)
        if not self.robot.place_product(self._shelf.shelf_id):
            return self._fail("place failed")
        self.placed_on = self._shelf.shelf_id
        return State.VERIFY_PLACEMENT

    def _verify_placement(self) -> State:
        # Simulation: checks the simulated gripper state only. A real robot needs a sensor
        # (gripper load, shelf-side camera, ...) here; the webcam cannot confirm a placement.
        marker = self._product.marker_id
        if self.robot.holding is not None:
            return self._fail("robot still holding product {}".format(self.robot.holding))
        if self.placed_on != self._shelf.shelf_id:
            return self._fail("product {} was not recorded on shelf {}".format(marker, self._shelf.shelf_id))
        log.info("  Product %d placed on Shelf %s\n  Gripper empty\n  Placement verified (simulated)",
                 marker, self.placed_on)
        self.carried_product = None
        return State.DONE

    def _recovery(self) -> State:
        self.robot.stop()
        self._recoveries += 1
        if self._recoveries > self.cfg.max_recoveries:
            self._reason = "gave up after {} recovery attempts (last: {})".format(self.cfg.max_recoveries, self._reason)
            return State.FAILED
        if self.cfg.operator_driven:
            log.info("  Recovery attempt %d/%d: bring the marker back into the camera's view. "
                     "Resuming at %s (carrying %s)",
                     self._recoveries, self.cfg.max_recoveries, self._resume_to.name, self._carrying())
        else:
            log.info("  Recovery attempt %d/%d: sweeping to re-acquire marker, then resuming at %s",
                     self._recoveries, self.cfg.max_recoveries, self._resume_to.name)
        for _ in range(self.cfg.recovery_scan_frames):
            if not self.cfg.operator_driven:      # nothing to sweep when a person aims the camera
                self.robot.search_step()
            self.robot.look()
        return self._resume_to
