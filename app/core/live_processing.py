from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .models import FrameResult, TrackingStatus


@dataclass
class LiveConfig:
    orb_features: int = 1200
    min_matches: int = 24
    min_inliers: int = 12
    map_stride: int = 8


class LiveProcessor:
    """Small CPU-friendly ORB/essential-matrix visual-odometry processor.

    The notebook remains the V3 benchmark source. This processor is the
    interactive live path: it performs real work on the selected video and
    deliberately emits no pose when geometry is rejected.
    """

    def __init__(self, config: Optional[LiveConfig] = None) -> None:
        self.config = config or LiveConfig()
        self.orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_keypoints: list[cv2.KeyPoint] = []
        self.previous_descriptors: Optional[np.ndarray] = None
        self.pose = np.eye(4, dtype=np.float64)
        self.frame_id = -1
        self.map_points = np.empty((0, 3), dtype=np.float32)
        self.map_colors = np.empty((0, 3), dtype=np.uint8)

    def reset(self) -> None:
        self.__init__(self.config)

    def process(self, frame_bgr: np.ndarray, timestamp: float = 0.0) -> FrameResult:
        self.frame_id += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        points = [keypoint.pt for keypoint in keypoints]
        result = FrameResult(
            frame_id=self.frame_id,
            timestamp=timestamp,
            rgb_frame=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            tracking_status=TrackingStatus.INITIALIZING,
            tracking_method="ORB + Essential",
            keypoints=points,
            diagnostics={"orb_keypoints": len(keypoints), "map_points": len(self.map_points)},
        )

        if self.previous_gray is None:
            self._remember(gray, keypoints, descriptors)
            return result

        if descriptors is None or self.previous_descriptors is None:
            result.tracking_status = TrackingStatus.LOST
            result.diagnostics["lost_reason"] = "no_descriptors"
            self._remember(gray, keypoints, descriptors)
            return result

        matches = self._ratio_matches(self.previous_descriptors, descriptors)
        result.matches = len(matches)
        if len(matches) < self.config.min_matches:
            result.tracking_status = TrackingStatus.LOST
            result.diagnostics["lost_reason"] = "too_few_correspondences"
            self._remember(gray, keypoints, descriptors)
            return result

        prev_xy = np.float32([self.previous_keypoints[m.queryIdx].pt for m in matches])
        curr_xy = np.float32([keypoints[m.trainIdx].pt for m in matches])
        h, w = gray.shape[:2]
        focal = float(max(w, h))
        principal = (w / 2.0, h / 2.0)
        essential, mask = cv2.findEssentialMat(
            prev_xy, curr_xy, focal=focal, pp=principal, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if essential is None or mask is None:
            result.tracking_status = TrackingStatus.LOST
            result.diagnostics["lost_reason"] = "essential_matrix_failed"
            self._remember(gray, keypoints, descriptors)
            return result

        inliers, rotation, translation, pose_mask = cv2.recoverPose(
            essential, prev_xy, curr_xy, focal=focal, pp=principal, mask=mask
        )
        result.inliers = int(inliers)
        if inliers < self.config.min_inliers:
            result.tracking_status = TrackingStatus.LOST
            result.diagnostics["lost_reason"] = "recover_pose_rejected"
            self._remember(gray, keypoints, descriptors)
            return result

        relative = np.eye(4, dtype=np.float64)
        relative[:3, :3] = rotation
        relative[:3, 3] = translation.ravel()
        self.pose = self.pose @ np.linalg.inv(relative)
        result.camera_pose = self.pose.copy()
        result.tracking_status = TrackingStatus.TRACKING
        result.diagnostics.update(
            {
                "pnp_inliers": int(inliers),
                "inlier_ratio": result.inlier_ratio,
                "map_points": len(self.map_points),
            }
        )
        self._add_sparse_points(prev_xy, curr_xy, pose_mask, frame_bgr)
        self._remember(gray, keypoints, descriptors)
        result.new_landmarks = self.map_points.copy()
        result.diagnostics["map_points"] = len(self.map_points)
        return result

    def _ratio_matches(self, previous: np.ndarray, current: np.ndarray) -> list[cv2.DMatch]:
        pairs = self.matcher.knnMatch(previous, current, k=2)
        return [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.75 * second.distance]

    def _remember(self, gray: np.ndarray, keypoints: list[cv2.KeyPoint], descriptors: Optional[np.ndarray]) -> None:
        self.previous_gray = gray
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

    def _add_sparse_points(self, prev_xy: np.ndarray, curr_xy: np.ndarray, mask: np.ndarray, frame_bgr: np.ndarray) -> None:
        good = mask.ravel().astype(bool)
        if good.sum() == 0:
            return
        # Use a unit-baseline two-view reconstruction for an inspectable sparse map.
        h, w = frame_bgr.shape[:2]
        focal = float(max(w, h))
        k = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        p0 = k @ np.hstack((np.eye(3), np.zeros((3, 1))))
        p1 = k @ np.hstack((np.eye(3), np.array([[1.0], [0.0], [0.0]])))
        xyz4 = cv2.triangulatePoints(p0, p1, prev_xy[good].T, curr_xy[good].T)
        xyz = (xyz4[:3] / np.maximum(np.abs(xyz4[3:4]), 1e-6)).T
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite][:: self.config.map_stride]
        pix = np.clip(np.round(curr_xy[good][finite][:: self.config.map_stride]).astype(int), [0, 0], [w - 1, h - 1])
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[pix[:, 1], pix[:, 0]]
        if len(xyz):
            self.map_points = np.vstack((self.map_points, xyz.astype(np.float32))) [-25000:]
            self.map_colors = np.vstack((self.map_colors, rgb.astype(np.uint8))) [-25000:]

