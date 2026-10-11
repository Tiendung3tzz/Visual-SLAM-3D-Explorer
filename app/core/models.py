from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class TrackingStatus:
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RELOCALIZING = "RELOCALIZING"


@dataclass
class FrameResult:
    """All visualization data associated with one processed frame.

    ``camera_pose`` is a camera-to-world 4x4 transform when it exists. A
    missing pose is intentional: a LOST frame must not grow the trajectory.
    """

    frame_id: int
    timestamp: float = 0.0
    rgb_frame: Optional[np.ndarray] = None
    predicted_depth: Optional[np.ndarray] = None
    tracking_status: str = TrackingStatus.INITIALIZING
    tracking_method: str = "offline artifact"
    camera_pose: Optional[np.ndarray] = None
    keypoints: list[tuple[float, float]] = field(default_factory=list)
    matches: int = 0
    inliers: int = 0
    new_landmarks: Optional[np.ndarray] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_pose(self) -> bool:
        return self.camera_pose is not None and np.isfinite(self.camera_pose).all()

    @property
    def inlier_ratio(self) -> float:
        if self.matches <= 0:
            return 0.0
        return float(self.inliers / self.matches)

