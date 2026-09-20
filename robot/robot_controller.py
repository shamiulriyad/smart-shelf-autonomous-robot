"""Robot abstraction + simulation.

`Robot` is the interface the state machine drives. `SimulatedRobot` only logs and updates a
simulated (x, y) position; it does not move anything and does not localise itself from the
camera. A real robot subclass (motors, gripper, Pi camera) implements the same methods later.
"""
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from aruco.detector import Detection
from aruco.perception import Observation
from camera.camera_source import DEFAULT_FOV_DEG

log = logging.getLogger("warehouse")


class Vision(Protocol):
    def look(self) -> Observation: ...
    def set_status(self, text: str) -> None: ...


@dataclass
class AlignmentResult:
    success: bool
    metric: bool                 # True only if the correction was computed from a calibrated pose
    commands: List[str] = field(default_factory=list)
    summary: str = ""            # relative position/orientation to the shelf marker (or the simulated stand-in)


class Robot(ABC):
    standoff_cm: float = 30.0    # distance to keep from the shelf face when placing

    def __init__(self, vision: Vision):
        self._vision = vision
        self.last_observation: Optional[Observation] = None

    # --- perception (the camera is mounted on the robot) ---
    def look(self) -> Observation:
        self.last_observation = self._vision.look()
        return self.last_observation

    def detect_product(self, marker_id: int) -> Optional[Detection]:
        return self.look().find("product", marker_id)

    def detect_target_shelf(self, marker_id: int) -> Optional[Detection]:
        return self.look().find("shelf", marker_id)

    # --- actions ---
    @property
    @abstractmethod
    def position(self) -> Tuple[float, float]:
        """Current (x, y) on the room map, metres. Simulated here; odometry/localisation on a real robot."""

    @property
    def current_position(self) -> Tuple[float, float]:
        return self.position

    @property
    @abstractmethod
    def current_heading(self) -> float:
        """Current heading in degrees on the room map."""

    @property
    @abstractmethod
    def holding(self) -> Optional[int]:
        """Marker ID of the product currently held, or None."""

    @abstractmethod
    def search_step(self) -> None:
        """One incremental step of the search motion (e.g. rotate in place)."""

    @abstractmethod
    def pick_product(self, product_id: int) -> bool: ...

    @abstractmethod
    def move_to(self, x: float, y: float) -> bool: ...

    @abstractmethod
    def creep_toward(self, x: float, y: float, distance_m: float) -> bool:
        """Advance at most `distance_m` along the straight line to (x, y), stopping short of it.

        Used by APPROACH_SHELF to close the gap the camera measured to the shelf marker, one small
        step at a time. Unlike move_to(), this does NOT void the alignment: it is the aligned final
        approach itself, and the state machine re-measures the marker after every step.
        """

    @abstractmethod
    def move_to_placement_position(self, x: float, y: float) -> bool:
        """Short final approach from the shelf approach point to the placement point, after alignment."""

    @abstractmethod
    def align_with_shelf(self, shelf_detection: Detection) -> AlignmentResult: ...

    @abstractmethod
    def place_product(self, shelf_id: str) -> bool: ...

    @abstractmethod
    def stop(self) -> None: ...


class SimulatedRobot(Robot):
    MAX_FINAL_APPROACH_M = 0.75      # the final approach is a short creep, never a way to cross the room

    def __init__(self, vision: Vision, start: Tuple[float, float], standoff_cm: float,
                 step_delay_s: float = 0.0, camera_fov_deg: float = DEFAULT_FOV_DEG):
        super().__init__(vision)
        self.x, self.y = start
        self.heading_deg = 90.0
        self.standoff_cm = standoff_cm
        self._fov_deg = camera_fov_deg
        self._delay = step_delay_s
        self._holding: Optional[int] = None
        self._aligned = False
        self._at_placement = False
        self.shelf_contents: Dict[str, List[int]] = {}

    @property
    def position(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def current_heading(self) -> float:
        return self.heading_deg

    @property
    def holding(self) -> Optional[int]:
        return self._holding

    def search_step(self) -> None:
        self.heading_deg = (self.heading_deg + 2.0) % 360.0     # slow simulated sweep
        self._aligned = False                                   # turning voids any earlier alignment
        self._at_placement = False

    def pick_product(self, product_id: int) -> bool:
        if self._holding is not None:
            log.info("[ROBOT-SIM] cannot pick %d: already holding %d", product_id, self._holding)
            return False
        log.info("[ROBOT-SIM] picking product %d (gripper closes)", product_id)
        self._pause()
        self._holding = product_id
        self._aligned = False
        self._at_placement = False
        return True

    def move_to(self, x: float, y: float) -> bool:
        distance = math.hypot(x - self.x, y - self.y)
        self.heading_deg = math.degrees(math.atan2(y - self.y, x - self.x)) if distance else self.heading_deg
        log.info("[ROBOT-SIM] move_to (%.2f, %.2f)  %.2f m, heading %.0f deg", x, y, distance, self.heading_deg)
        self._pause()
        self.x, self.y = x, y
        self._aligned = False
        self._at_placement = False
        return True

    def creep_toward(self, x: float, y: float, distance_m: float) -> bool:
        if distance_m <= 0:
            return True
        if not self._aligned:
            log.info("[ROBOT-SIM] cannot creep toward (%.2f, %.2f): not aligned with a shelf", x, y)
            return False
        remaining = math.hypot(x - self.x, y - self.y)
        if remaining == 0:
            return True
        self.heading_deg = math.degrees(math.atan2(y - self.y, x - self.x)) % 360.0
        if distance_m >= remaining:                     # land exactly on the point, no float drift
            self.x, self.y = x, y
        else:
            self.x += distance_m / remaining * (x - self.x)
            self.y += distance_m / remaining * (y - self.y)
        log.info("[ROBOT-SIM] approach: crept %.2f m toward (%.2f, %.2f) -> (%.2f, %.2f), heading %.0f deg",
                 min(distance_m, remaining), x, y, self.x, self.y, self.heading_deg)
        self._pause()
        self._at_placement = False      # the final approach still has to be commanded explicitly
        return True

    def move_to_placement_position(self, x: float, y: float) -> bool:
        if not self._aligned:
            log.info("[ROBOT-SIM] cannot approach placement position: not aligned with a shelf")
            return False
        distance = math.hypot(x - self.x, y - self.y)
        if distance > self.MAX_FINAL_APPROACH_M:
            log.info("[ROBOT-SIM] cannot approach placement position: %.2f m away (max %.2f m); navigate to the shelf first",
                     distance, self.MAX_FINAL_APPROACH_M)
            return False
        log.info("[ROBOT-SIM] final approach to (%.2f, %.2f)  %.2f m (slow, straight line)", x, y, distance)
        self._pause()
        self.x, self.y = x, y
        self._at_placement = True
        return True

    def align_with_shelf(self, shelf_detection: Detection) -> AlignmentResult:
        pose = shelf_detection.pose
        if pose is not None:
            commands = [
                "rotate {:+.1f} deg to centre the shelf marker".format(pose.bearing_deg),
                "drive {:+.1f} cm to reach the {:.0f} cm stand-off".format(pose.z_cm - self.standoff_cm, self.standoff_cm),
                "shift around the shelf to remove the {:+.1f} deg viewing angle".format(pose.yaw_deg),
            ]
            summary = "x={:+.1f} cm, distance {:.1f} cm, bearing {:+.1f} deg, yaw {:+.1f} deg".format(
                pose.x_cm, pose.distance_cm, pose.bearing_deg, pose.yaw_deg)
            result = AlignmentResult(True, True, commands, summary)
        else:
            offset_x = shelf_detection.image_offset[0]
            side = "right" if offset_x > 0 else "left"
            commands = [
                "coarse only: shelf marker is {:.0f}% of half-width to the {} of image centre".format(abs(offset_x) * 100, side),
                "marker appears {:.0f} px wide; no metric distance/angle without camera calibration".format(shelf_detection.side_px),
            ]
            summary = ("heading corrected from the image centre only; no metric offset. The distance to "
                       "the shelf is measured separately during the approach, not assumed here.")
            result = AlignmentResult(True, False, commands, summary)
        for command in commands:
            log.info("[ROBOT-SIM] align: %s", command)
        # Turn to face the marker: bearing from the calibrated pose, else from its image offset.
        if pose is not None:
            turn = -pose.bearing_deg
        else:
            turn = -math.degrees(math.atan(shelf_detection.image_offset[0] * math.tan(math.radians(self._fov_deg / 2))))
        self.heading_deg = (self.heading_deg + turn) % 360.0
        log.info("[ROBOT-SIM] align: turned %+.1f deg, heading now %.1f deg", turn, self.heading_deg)
        self._pause()
        self._aligned = True
        return result

    def place_product(self, shelf_id: str) -> bool:
        if self._holding is None or not self._aligned or not self._at_placement:
            log.info("[ROBOT-SIM] cannot place: holding=%s aligned=%s at_placement=%s",
                     self._holding, self._aligned, self._at_placement)
            return False
        log.info("[ROBOT-SIM] placing product %d on Shelf %s (gripper opens)", self._holding, shelf_id)
        self._pause()
        self.shelf_contents.setdefault(shelf_id, []).append(self._holding)
        self._holding = None
        return True

    def stop(self) -> None:
        log.info("[ROBOT-SIM] stop")

    def _pause(self) -> None:
        if self._delay:
            time.sleep(self._delay)
