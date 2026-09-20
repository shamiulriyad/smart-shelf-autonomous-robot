import math
import os
import tempfile
import unittest

import cv2
import numpy as np
from cv2 import aruco

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)
from aruco.calibration import CalibrationError, CameraCalibration, calibrate_from_frames, load_if_available
from aruco.detector import MarkerDetector
from aruco.pose_estimator import PoseEstimator

K_TRUE = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
SIZE = (640, 480)


def rotation(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))[0]


def project_tile(tile, px_to_cm, rvec, tvec, background):
    """Render a flat image lying in the z=0 plane as seen by a camera with intrinsics K_TRUE."""
    h, w = tile.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    plane = np.float32([[*px_to_cm(x, y), 0] for x, y in src])
    dst, _ = cv2.projectPoints(plane, rvec, tvec, K_TRUE, None)
    H = cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32))
    return cv2.warpPerspective(tile, H, SIZE, borderValue=background)


class PoseTests(unittest.TestCase):
    def render_marker(self, tvec, yaw_rad, size_cm=18.7, marker_id=203):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        s, pad = 200, 40
        tile = np.full((s + 2 * pad,) * 2, 255, np.uint8)
        tile[pad:pad + s, pad:pad + s] = aruco.generateImageMarker(dictionary, marker_id, s)
        # Marker coordinates: x right, y up, centred. Tile pixel y grows downward.
        to_cm = lambda x, y: ((x - pad - s / 2) / s * size_cm, -(y - pad - s / 2) / s * size_cm)
        R = rotation(0, yaw_rad, 0) @ rotation(math.pi, 0, 0)     # faces the camera, then turned by yaw
        rvec, _ = cv2.Rodrigues(R)
        frame = project_tile(tile, to_cm, rvec, np.array(tvec, dtype=np.float64), background=90)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def test_recovers_known_pose(self):
        tvec, yaw = (10.0, -5.0, 110.0), math.radians(20)
        det = MarkerDetector(["ARUCO_ORIGINAL"]).detect(self.render_marker(tvec, yaw))
        self.assertEqual([d.marker_id for d in det], [203])
        calibration = CameraCalibration(K_TRUE, np.zeros((5, 1)), SIZE, 0.1, 10, (7, 6), 2.5)
        pose = PoseEstimator(calibration).estimate(det[0], 18.7, SIZE)
        np.testing.assert_allclose(pose.tvec, tvec, rtol=0.03, atol=1.5)
        self.assertAlmostEqual(pose.yaw_deg, -20.0, delta=3.0)

    def test_uncalibrated_gives_no_pose(self):
        det = MarkerDetector().detect(self.render_marker((0, 0, 110.0), 0.0))[0]
        self.assertIsNone(PoseEstimator(None).estimate(det, 18.7, SIZE))

    def test_pose_disabled_when_resolution_differs_from_calibration(self):
        det = MarkerDetector().detect(self.render_marker((0, 0, 110.0), 0.0))[0]
        calibration = CameraCalibration(K_TRUE, np.zeros((5, 1)), (1280, 720), 0.1, 10, (7, 6), 2.5)
        self.assertIsNone(PoseEstimator(calibration).estimate(det, 18.7, SIZE))


class CalibrationTests(unittest.TestCase):
    def board_frames(self, count=16, seed=1):
        cols, rows, sq_px, sq_cm = 7, 6, 60, 2.5      # 7x6 INNER corners => 8x7 squares
        w, h = (cols + 1) * sq_px, (rows + 1) * sq_px
        tile = np.full((h + 2 * sq_px, w + 2 * sq_px), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    tile[sq_px + r * sq_px:sq_px + (r + 1) * sq_px, sq_px + c * sq_px:sq_px + (c + 1) * sq_px] = 0
        to_cm = lambda x, y: ((x - tile.shape[1] / 2) / sq_px * sq_cm, (y - tile.shape[0] / 2) / sq_px * sq_cm)
        rng = np.random.default_rng(seed)
        frames = []
        for _ in range(count):
            rvec, _ = cv2.Rodrigues(rotation(rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.4, 0.4)))
            tvec = np.array([rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(40, 60)])
            frames.append(cv2.cvtColor(project_tile(tile, to_cm, rvec, tvec, background=200), cv2.COLOR_GRAY2BGR))
        return frames

    def test_recovers_intrinsics_from_synthetic_views(self):
        calibration = calibrate_from_frames(self.board_frames(), (7, 6), 2.5)
        self.assertAlmostEqual(calibration.camera_matrix[0, 0], 600.0, delta=30.0)
        self.assertAlmostEqual(calibration.camera_matrix[0, 2], 320.0, delta=15.0)
        self.assertLess(calibration.rms_error_px, 1.0)

    def test_too_few_views_is_an_error(self):
        with self.assertRaises(CalibrationError):
            calibrate_from_frames(self.board_frames(4), (7, 6), 2.5, min_images=10)

    def test_save_load_roundtrip_and_missing_file(self):
        calibration = calibrate_from_frames(self.board_frames(), (7, 6), 2.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal.json")
            self.assertIsNone(load_if_available(path))
            calibration.save(path)
            loaded = CameraCalibration.load(path)
        np.testing.assert_allclose(loaded.camera_matrix, calibration.camera_matrix)
        self.assertEqual(loaded.image_size, calibration.image_size)


if __name__ == "__main__":
    unittest.main()
