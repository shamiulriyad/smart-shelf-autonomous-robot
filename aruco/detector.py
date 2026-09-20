"""Detection layer: wraps the existing MarkerDetectionSystem and returns structured results.

The detection itself (multi-dictionary ArucoDetector, centroids, (dictionary, id) keying) is
reused unchanged from marker_detection_system.py. This module only reshapes its output.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from marker_detection_system import MarkerDetectionSystem

if TYPE_CHECKING:
    from aruco.pose_estimator import Pose

SUPPORTED_DICTIONARIES = MarkerDetectionSystem.SUPPORTED_DICTIONARIES


@dataclass
class Detection:
    marker_id: int
    dictionary: str
    corners: np.ndarray                     # (4, 2) pixels: top-left, top-right, bottom-right, bottom-left
    center: Tuple[float, float]             # pixels
    side_px: float                          # mean edge length in pixels
    image_offset: Tuple[float, float]       # centre relative to image centre, -1..1 (x right, y down)
    # Filled in by Perception once the marker is recognised as a product/shelf marker:
    kind: Optional[str] = None
    marker_size_cm: Optional[float] = None
    approx_distance_cm: Optional[float] = None   # rough frontal-view estimate, NOT a calibrated measurement
    pose: Optional["Pose"] = None                # only set when the camera is calibrated


class MarkerDetector:
    def __init__(self, dictionaries: Optional[Iterable[str]] = None):
        self._mds = MarkerDetectionSystem()
        if dictionaries is not None:
            wanted = set(dictionaries)
            unknown = wanted - set(SUPPORTED_DICTIONARIES)
            if unknown:
                raise ValueError("Unsupported dictionaries: {}".format(sorted(unknown)))
            # Only run the dictionaries the config actually uses (faster, fewer false positives).
            self._mds.detectors = {n: d for n, d in self._mds.detectors.items() if n in wanted}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        detections = []
        for (dict_name, marker_id), info in self._mds.detect_markers(frame).items():
            corners = np.asarray(info['Corners'], dtype=np.float32)
            cx, cy = info['Centroid']
            side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
            detections.append(Detection(
                marker_id=marker_id,
                dictionary=dict_name,
                corners=corners,
                center=(float(cx), float(cy)),
                side_px=side_px,
                image_offset=((cx - width / 2) / (width / 2), (cy - height / 2) / (height / 2)),
            ))
        return detections
