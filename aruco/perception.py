"""Perception facade: camera frame -> classified detections (+ pose when calibrated) -> overlay.

This is the only object the robot/state-machine layers use to "see". They never touch OpenCV.
"""
import logging
from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import cv2
import numpy as np

from aruco.detector import Detection, MarkerDetector
from aruco.pose_estimator import PoseEstimator
from camera.camera_source import CameraError, CameraSource
from config.loader import WarehouseConfig

log = logging.getLogger("warehouse")

MAX_CONSECUTIVE_READ_FAILURES = 30


class UserAbort(Exception):
    """Raised when the operator presses Q/ESC in the preview window."""


@dataclass
class Observation:
    frame_index: int
    detections: List[Detection]

    def find(self, kind: str, marker_id: int) -> Optional[Detection]:
        for det in self.detections:
            if det.kind == kind and det.marker_id == marker_id:
                return det
        return None

    def of_kind(self, kind: str) -> List[Detection]:
        return [d for d in self.detections if d.kind == kind]


class Perception:
    def __init__(self, camera: CameraSource, config: WarehouseConfig, pose_estimator: PoseEstimator,
                 display: bool = True, labeler: Optional[Callable[[Detection], str]] = None):
        self._camera = camera
        self._config = config
        self._pose = pose_estimator
        self._detector = MarkerDetector(config.dictionaries())
        self._display = display
        self._labeler = labeler
        self._status = ""
        self._frame_index = 0
        self._read_failures = 0

    @property
    def calibrated(self) -> bool:
        return self._pose.calibrated

    def set_status(self, text: str) -> None:
        self._status = text

    def look(self) -> Observation:
        frame = self._camera.read()
        if frame is None:
            self._read_failures += 1
            if self._read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                raise CameraError("Camera stopped delivering frames.")
            return Observation(self._frame_index, [])
        self._read_failures = 0
        self._frame_index += 1

        height, width = frame.shape[:2]
        detections = []
        for det in self._detector.detect(frame):
            marker_type = self._config.classify(det.dictionary, det.marker_id)
            if marker_type is not None:
                det = replace(
                    det, kind=marker_type.kind, marker_size_cm=marker_type.size_cm,
                    approx_distance_cm=self._pose.approx_distance_cm(det, marker_type.size_cm, width),
                    pose=self._pose.estimate(det, marker_type.size_cm, (width, height)))
            detections.append(det)

        if self._display:
            self._show(frame, detections)
        return Observation(self._frame_index, detections)

    def close(self) -> None:
        self._camera.release()
        if self._display:
            cv2.destroyAllWindows()

    def _show(self, frame: np.ndarray, detections: List[Detection]) -> None:
        for det in detections:
            colour = {"product": (0, 200, 255), "shelf": (0, 255, 0)}.get(det.kind, (128, 128, 128))
            cv2.polylines(frame, [det.corners.astype(np.int32)], True, colour, 2)
            label = self._labeler(det) if self._labeler else "{} {}".format(det.kind or "?", det.marker_id)
            x, y = int(det.center[0]), int(det.center[1])
            cv2.putText(frame, label, (x - 60, y - int(det.side_px / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if det.pose is not None:
                self._pose.draw_axes(frame, det)
                text = "x={:+.1f} z={:.1f} cm yaw={:+.0f} deg".format(det.pose.x_cm, det.pose.z_cm, det.pose.yaw_deg)
                cv2.putText(frame, text, (x - 60, y + int(det.side_px / 2) + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
        banner = "CALIBRATED" if self.calibrated else "UNCALIBRATED - no metric pose"
        cv2.putText(frame, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.calibrated else (0, 0, 255), 2)
        cv2.putText(frame, self._status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("warehouse robot prototype", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            raise UserAbort()
