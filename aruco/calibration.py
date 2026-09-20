"""Camera calibration from a chessboard: capture, compute, save/load.

Pose estimation is only trusted when a calibration file produced here is loaded.
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("warehouse")

DEFAULT_CALIBRATION_PATH = "config/camera_calibration.json"


class CalibrationError(Exception):
    pass


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]          # (width, height) the calibration is valid for
    rms_error_px: float
    num_images: int
    board_inner_corners: Tuple[int, int]
    square_size_cm: float

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "camera_matrix": self.camera_matrix.tolist(),
                "dist_coeffs": np.asarray(self.dist_coeffs).ravel().tolist(),
                "image_size": list(self.image_size),
                "rms_reprojection_error_px": self.rms_error_px,
                "num_images": self.num_images,
                "board_inner_corners": list(self.board_inner_corners),
                "square_size_cm": self.square_size_cm,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CameraCalibration":
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(
                camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
                image_size=tuple(d["image_size"]),
                rms_error_px=float(d["rms_reprojection_error_px"]),
                num_images=int(d["num_images"]),
                board_inner_corners=tuple(d["board_inner_corners"]),
                square_size_cm=float(d["square_size_cm"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise CalibrationError("Cannot read calibration file {}: {}".format(path, e))


def load_if_available(path: str) -> Optional[CameraCalibration]:
    """Return the saved calibration, or None (=> camera is treated as uncalibrated)."""
    try:
        return CameraCalibration.load(path)
    except CalibrationError as e:
        if os.path.exists(path):
            log.warning("%s -- continuing UNCALIBRATED.", e)
        return None


_CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_chessboard(gray: np.ndarray, board_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Sub-pixel refined inner corners, or None if the full board is not visible."""
    found, corners = cv2.findChessboardCorners(
        gray, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CORNER_CRITERIA)


def calibrate_from_corners(corner_sets: Sequence[np.ndarray], image_size: Tuple[int, int],
                           board_size: Tuple[int, int], square_size_cm: float,
                           min_images: int = 10) -> CameraCalibration:
    if len(corner_sets) < min_images:
        raise CalibrationError("Need at least {} usable board views, got {}.".format(min_images, len(corner_sets)))
    # Board corners in board coordinates (cm); z=0 because the board is flat.
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size_cm
    rms, matrix, dist, _, _ = cv2.calibrateCamera(
        [objp] * len(corner_sets), list(corner_sets), image_size, None, None)
    return CameraCalibration(matrix, dist, tuple(image_size), float(rms), len(corner_sets),
                             tuple(board_size), square_size_cm)


def calibrate_from_frames(frames: Sequence[np.ndarray], board_size: Tuple[int, int],
                          square_size_cm: float, min_images: int = 10) -> CameraCalibration:
    corner_sets: List[np.ndarray] = []
    image_size = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        size = gray.shape[::-1]
        if image_size is not None and size != image_size:
            raise CalibrationError("All calibration frames must have the same resolution.")
        image_size = size
        corners = find_chessboard(gray, board_size)
        if corners is not None:
            corner_sets.append(corners)
    if image_size is None:
        raise CalibrationError("No frames given.")
    return calibrate_from_corners(corner_sets, image_size, board_size, square_size_cm, min_images)


def describe_quality(rms_px: float) -> str:
    if rms_px < 0.5:
        return "good"
    if rms_px < 1.0:
        return "acceptable"
    return "POOR - recapture with the board flat, sharp, and at more varied angles/distances"


def run_interactive_capture(camera, board_size: Tuple[int, int], square_size_cm: float,
                            output_path: str, target_images: int = 20, min_images: int = 10
                            ) -> Optional[CameraCalibration]:
    """Live webcam capture. SPACE = grab a view, ENTER = finish and calibrate, Q/ESC = cancel."""
    window = "calibration"
    corner_sets: List[np.ndarray] = []
    image_size = None
    print("Calibration: show the chessboard ({}x{} inner corners, {} cm squares).".format(
        board_size[0], board_size[1], square_size_cm))
    print("Vary distance and tilt between captures. SPACE=capture  ENTER=finish  Q/ESC=cancel")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                raise CalibrationError("Camera returned no frame.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            corners = find_chessboard(gray, board_size)

            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, board_size, corners, True)
            status = "board {}  captured {}/{} (min {})".format(
                "FOUND" if corners is not None else "not found", len(corner_sets), target_images, min_images)
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if corners is not None else (0, 0, 255), 2)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("Calibration cancelled.")
                return None
            if key == 32 and corners is not None:
                # Reject near-duplicate views: they add no information and bias the result.
                if corner_sets and np.mean(np.linalg.norm(corners - corner_sets[-1], axis=2)) < 15:
                    print("  view too similar to the previous one - move/tilt the board")
                    continue
                corner_sets.append(corners)
                print("  captured {}/{}".format(len(corner_sets), target_images))
            if key in (13, 10) or len(corner_sets) >= target_images:
                if len(corner_sets) < min_images:
                    print("  need at least {} captures (have {})".format(min_images, len(corner_sets)))
                    continue
                break
    finally:
        cv2.destroyWindow(window)

    calibration = calibrate_from_corners(corner_sets, image_size, board_size, square_size_cm, min_images)
    calibration.save(output_path)
    print("Saved {} | image size {}x{} | RMS reprojection error {:.3f} px ({})".format(
        output_path, image_size[0], image_size[1], calibration.rms_error_px,
        describe_quality(calibration.rms_error_px)))
    return calibration
