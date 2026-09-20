"""Camera abstraction. Everything downstream only calls read()/release(), so a Raspberry Pi
camera can be added later as another CameraSource subclass without touching the rest."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from cv2 import aruco

from aruco.detector import SUPPORTED_DICTIONARIES


class CameraError(Exception):
    pass


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Next BGR frame, or None if no frame is available."""

    def release(self) -> None:
        pass


class WebcamSource(CameraSource):
    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError("Cannot open camera {}".format(index))
        # Calibration is only valid at the resolution it was captured at, so allow pinning it.
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


@dataclass
class ScriptStep:
    markers: Sequence[Tuple[str, int]]   # (dictionary name, marker id) to draw; empty = blank scene
    frames: int
    scale: float = 1.0                   # >1 draws the markers larger, i.e. the camera is closer to them


class SyntheticCameraSource(CameraSource):
    """Renders frames containing real ArUco markers on a script. Used by tests and by --synthetic
    so the whole pipeline (detector included) can run without a webcam.

    The last step repeats forever unless loop_last=False, in which case read() then returns None.
    """
    _CELL = 200          # pixels per marker slot
    _MARKER = 140        # marker side, leaving a white quiet zone inside the slot

    def __init__(self, script: List[ScriptStep], frame_size: Tuple[int, int] = (640, 480),
                 loop_last: bool = True):
        self._script = script
        self._size = frame_size
        self._loop_last = loop_last
        self._step = 0
        self._used = 0

    def read(self) -> Optional[np.ndarray]:
        if self._step >= len(self._script):
            return None
        step = self._script[self._step]
        frame = self._render(step.markers, step.scale)
        self._used += 1
        if self._used >= step.frames:
            if self._step < len(self._script) - 1 or not self._loop_last:
                self._step += 1
                self._used = 0
            else:
                self._used = 0
        return frame

    def _render(self, markers: Sequence[Tuple[str, int]], scale: float = 1.0) -> np.ndarray:
        """Draw each marker in its own slot. `scale` stands in for the camera being closer: the
        marker covers more pixels, which is exactly what the distance estimate reads."""
        width, height = self._size
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        cell = max(1, int(round(self._CELL * scale)))
        side = max(1, int(round(self._MARKER * scale)))
        pad = (cell - side) // 2
        y0 = (height - cell) // 2
        for slot, (dict_name, marker_id) in enumerate(markers):
            dictionary = aruco.getPredefinedDictionary(SUPPORTED_DICTIONARIES[dict_name])
            tile = np.full((cell, cell), 255, dtype=np.uint8)               # white quiet zone
            tile[pad:pad + side, pad:pad + side] = aruco.generateImageMarker(dictionary, marker_id, side)
            x0 = slot * cell
            # A tile bigger than the frame is clipped, like a marker held close to a real camera.
            left, top = max(0, x0), max(0, y0)
            right, bottom = min(width, x0 + cell), min(height, y0 + cell)
            if right <= left or bottom <= top:
                continue
            frame[top:bottom, left:right] = cv2.cvtColor(
                tile[top - y0:bottom - y0, left - x0:right - x0], cv2.COLOR_GRAY2BGR)
        return frame


DEFAULT_FOV_DEG = 60.0       # horizontal field of view of the simulated (and assumed) robot camera


@dataclass(frozen=True)
class WorldMarker:
    """A marker fixed in the simulated room (map metres)."""
    dictionary: str
    marker_id: int
    x: float
    y: float
    size_cm: float           # physical side of the black square
    facing_deg: float        # map direction the marker face points to (a shelf face points at the aisle)


class SimulatedWorldCamera(CameraSource):
    """Renders what a robot camera would see from the robot's simulated pose.

    A marker is visible only if it is inside the field of view, inside the detection range, faces the
    robot, and its black square fits in the image. Apparent size follows the pinhole model, so distance
    and bearing come out of the pixels the same way they would from a real camera. Nothing here
    depends on frame counts or on the state machine: move the robot and the view changes.

    Simplification: markers are drawn frontal (no perspective skew), so yaw always reads as ~0.
    """
    MIN_SIDE_PX = 30         # smaller markers are not reliably detected

    def __init__(self, markers: Sequence[WorldMarker], frame_size: Tuple[int, int] = (640, 480),
                 fov_deg: float = DEFAULT_FOV_DEG, max_range_m: float = 1.5, min_range_m: float = 0.15,
                 max_view_angle_deg: float = 75.0):
        self._markers = list(markers)
        self._size = frame_size
        self._fov = fov_deg
        self._max_range = max_range_m
        self._min_range = min_range_m
        self._max_view_angle = max_view_angle_deg
        self._focal_px = (frame_size[0] / 2) / math.tan(math.radians(fov_deg / 2))
        self._pose: Optional[Callable[[], Tuple[float, float, float]]] = None

    def bind(self, pose_provider: Callable[[], Tuple[float, float, float]]) -> None:
        """pose_provider() -> (x_m, y_m, heading_deg) of the robot this camera is mounted on."""
        self._pose = pose_provider

    def _project(self, marker: WorldMarker, pose: Tuple[float, float, float]):
        """(centre_x_px, side_px) if `marker` is visible from `pose`, else None."""
        x, y, heading = pose
        dx, dy = marker.x - x, marker.y - y
        dist = math.hypot(dx, dy)
        if not self._min_range <= dist <= self._max_range:
            return None
        bearing = math.degrees(math.atan2(dy, dx))
        rel = (bearing - heading + 180.0) % 360.0 - 180.0            # +CCW: marker to the robot's left
        if abs(rel) >= self._fov / 2:
            return None
        to_robot = (bearing + 180.0 - marker.facing_deg + 180.0) % 360.0 - 180.0
        if abs(to_robot) > self._max_view_angle:                     # seen from behind / too obliquely
            return None
        side = marker.size_cm / 100.0 * self._focal_px / (dist * math.cos(math.radians(rel)))
        centre_x = self._size[0] / 2 - self._focal_px * math.tan(math.radians(rel))
        # The black square itself must fit in the image (the white quiet zone may be cut off at the edge).
        if side < self.MIN_SIDE_PX or side > self._size[1] or centre_x - side / 2 < 0 \
                or centre_x + side / 2 > self._size[0]:
            return None
        return centre_x, side

    def visible_markers(self) -> List[WorldMarker]:
        if self._pose is None:
            raise CameraError("SimulatedWorldCamera is not bound to a robot pose")
        pose = self._pose()
        return [m for m in self._markers if self._project(m, pose) is not None]

    def read(self) -> Optional[np.ndarray]:
        if self._pose is None:
            raise CameraError("SimulatedWorldCamera is not bound to a robot pose")
        pose = self._pose()
        width, height = self._size
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        seen = []
        for marker in self._markers:
            projection = self._project(marker, pose)
            if projection is not None:
                seen.append((math.hypot(marker.x - pose[0], marker.y - pose[1]), marker, projection))
        for _, marker, (centre_x, side) in sorted(seen, key=lambda s: -s[0]):     # far first
            side_px = int(round(side))
            tile_px = int(round(side * 1.5))
            pad = (tile_px - side_px) // 2
            dictionary = aruco.getPredefinedDictionary(SUPPORTED_DICTIONARIES[marker.dictionary])
            tile = np.full((tile_px, tile_px), 255, dtype=np.uint8)
            tile[pad:pad + side_px, pad:pad + side_px] = aruco.generateImageMarker(dictionary, marker.marker_id, side_px)
            x0 = int(round(centre_x - tile_px / 2))
            y0 = (height - tile_px) // 2
            left, top = max(0, x0), max(0, y0)                       # clip the quiet zone to the frame
            right, bottom = min(width, x0 + tile_px), min(height, y0 + tile_px)
            frame[top:bottom, left:right] = cv2.cvtColor(tile[top - y0:bottom - y0, left - x0:right - x0],
                                                         cv2.COLOR_GRAY2BGR)
        return frame


PICKUP_DISTANCE_M = 0.3      # the product to pick sits this far in front of the robot's start pose (heading 90 deg)


def build_world_markers(config, product_marker_id: int) -> List[WorldMarker]:
    """Simulated room: one marker on the front face of every shelf, plus the product to pick.

    Positions come from the config (shelf x/y, map shelf depth, robot start). Shelf faces point toward
    -y, the aisle side where each shelf's approach point is. `config` is a config.loader.WarehouseConfig.
    """
    product_t, shelf_t = config.marker_types["product"], config.marker_types["shelf"]
    start_x, start_y = config.room_map.robot_start
    markers = [WorldMarker(product_t.dictionary, product_marker_id, start_x, start_y + PICKUP_DISTANCE_M,
                           product_t.size_cm, 270.0)]
    for shelf in config.shelves.all():
        markers.append(WorldMarker(shelf_t.dictionary, shelf.marker_id, shelf.x,
                                   shelf.y - config.room_map.shelf_depth_m / 2, shelf_t.size_cm, 270.0))
    return markers
