"""Marker pose relative to the camera, from solvePnP.

Replaces the removed cv2.aruco.estimatePoseSingleMarkers / drawAxis (gone in OpenCV 5, and
deprecated in 4.7+). Pose is only produced when a camera calibration is loaded; without one,
no metric pose is reported at all.

Camera frame convention (OpenCV): x right, y down, z forward (out of the lens). Units = cm.
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from aruco.calibration import CameraCalibration
from aruco.detector import Detection

log = logging.getLogger("warehouse")


@dataclass
class Pose:
    rvec: np.ndarray   # (3,) Rodrigues rotation, marker -> camera
    tvec: np.ndarray   # (3,) marker centre in camera frame, cm

    @property
    def x_cm(self) -> float:      # +right of the camera axis
        return float(self.tvec[0])

    @property
    def y_cm(self) -> float:      # +below the camera axis
        return float(self.tvec[1])

    @property
    def z_cm(self) -> float:      # forward distance along the camera axis
        return float(self.tvec[2])

    @property
    def distance_cm(self) -> float:
        return float(np.linalg.norm(self.tvec))

    @property
    def bearing_deg(self) -> float:
        """Horizontal angle to the marker; positive = marker is to the right of the camera axis."""
        return math.degrees(math.atan2(self.x_cm, self.z_cm))

    @property
    def yaw_deg(self) -> float:
        """Rotation of the marker about the vertical axis relative to the camera.

        0 = marker faces the camera squarely. Non-zero = the robot is viewing it at an angle,
        i.e. the amount the robot would have to move around to face it head-on.
        """
        normal = cv2.Rodrigues(self.rvec)[0][:, 2]      # marker +z axis, points out of the marker face
        return math.degrees(math.atan2(normal[0], -normal[2]))


class PoseEstimator:
    def __init__(self, calibration: Optional[CameraCalibration]):
        self.calibration = calibration
        self._warned_size = False

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def estimate(self, det: Detection, marker_size_cm: float,
                 frame_size: Tuple[int, int]) -> Optional[Pose]:
        """Pose of one marker, or None when the camera is uncalibrated or the solve fails."""
        if self.calibration is None:
            return None
        if tuple(self.calibration.image_size) != tuple(frame_size):
            # Intrinsics depend on resolution; using them at another resolution gives wrong numbers.
            if not self._warned_size:
                log.warning("Frame size %s != calibration size %s: pose disabled. Recalibrate at this "
                            "resolution or set the camera resolution to match.",
                            tuple(frame_size), tuple(self.calibration.image_size))
                self._warned_size = True
            return None
        half = marker_size_cm / 2.0
        # Marker corners in marker coordinates, same order as the detector (TL, TR, BR, BL).
        object_points = np.array([[-half, half, 0], [half, half, 0],
                                  [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(object_points, det.corners.astype(np.float32),
                                      self.calibration.camera_matrix, self.calibration.dist_coeffs,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or tvec[2] <= 0:
            return None
        return Pose(rvec.reshape(3), tvec.reshape(3))

    def approx_distance_cm(self, det: Detection, marker_size_cm: float, frame_width: int) -> float:
        """Rough distance assuming a frontal marker; focal length ~ frame width when uncalibrated.

        This is the same heuristic the original detector used. It is only a hint for humans.
        """
        focal_px = self.calibration.camera_matrix[0, 0] if self.calibration else frame_width
        return marker_size_cm * focal_px / det.side_px

    def draw_axes(self, frame: np.ndarray, det: Detection) -> None:
        if self.calibration is not None and det.pose is not None and det.marker_size_cm:
            cv2.drawFrameAxes(frame, self.calibration.camera_matrix, self.calibration.dist_coeffs,
                              det.pose.rvec, det.pose.tvec, det.marker_size_cm * 0.5)
