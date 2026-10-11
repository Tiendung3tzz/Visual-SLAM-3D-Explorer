from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .models import FrameResult, TrackingStatus
from .ply import load_ply


@dataclass
class ArtifactBundle:
    root: Path
    frames: list[FrameResult]
    sparse_points: np.ndarray
    sparse_colors: np.ndarray
    dense_points: np.ndarray
    dense_colors: np.ndarray
    sensor_points: np.ndarray
    sensor_colors: np.ndarray
    tracking_summary: dict
    diagnostics_summary: dict
    rgbd_metrics: dict
    depth_preview: Optional[Path] = None
    map_is_temporal: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def tracked_count(self) -> int:
        return sum(frame.tracking_status == TrackingStatus.TRACKING for frame in self.frames)


def load_artifact_bundle(root: str | Path) -> ArtifactBundle:
    root = Path(root)
    trajectory_path = root / "trajectory.csv"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Replay artifact missing: {trajectory_path}")

    rows = list(csv.DictReader(trajectory_path.open(encoding="utf-8", newline="")))
    frames = [_frame_from_row(row) for row in rows]
    tracking_summary = _read_json(root / "tracking.json")
    diagnostics_summary = _read_json(root / "tracking_diagnostics.json")
    rgbd_metrics = _read_json(root / "rgbd_trajectory_metrics.json")

    sparse_points, sparse_colors = _load_optional(root / "sparse_map.ply")
    dense_points, dense_colors = _load_optional(root / "predicted_depth_single_frame_3d.ply")
    sensor_points, sensor_colors = _load_optional(root / "tum_sensor_single_frame_3d.ply")

    preview = _find_depth_preview(root)
    return ArtifactBundle(
        root=root,
        frames=frames,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
        dense_points=dense_points,
        dense_colors=dense_colors,
        sensor_points=sensor_points,
        sensor_colors=sensor_colors,
        tracking_summary=tracking_summary,
        diagnostics_summary=diagnostics_summary,
        rgbd_metrics=rgbd_metrics,
        depth_preview=preview,
    )


def _frame_from_row(row: dict[str, str]) -> FrameResult:
    status_raw = (row.get("status") or "").lower()
    if status_raw in {"tracked", "init"}:
        status = TrackingStatus.TRACKING
    elif status_raw in {"lost", "not_estimated"}:
        status = TrackingStatus.LOST if status_raw == "lost" else TrackingStatus.INITIALIZING
    else:
        status = TrackingStatus.INITIALIZING

    values = [_float_or_nan(row.get(key, "")) for key in ("x_units", "y_units", "z_units")]
    pose = None
    if all(math.isfinite(value) for value in values):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = values
    raw_status = status_raw or "unknown"
    inliers = int(float(row.get("inliers") or 0))
    return FrameResult(
        frame_id=int(row.get("frame", 0)),
        timestamp=_float_or_nan(row.get("time_s", "0")),
        tracking_status=status,
        tracking_method="V3 artifact replay",
        camera_pose=pose,
        inliers=inliers,
        matches=inliers,
        diagnostics={"raw_status": raw_status, "source_idx": row.get("source_idx", "")},
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    try:
        return load_ply(path)
    except (OSError, ValueError):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)


def _find_depth_preview(root: Path) -> Optional[Path]:
    candidates = [
        root.parent / "assets" / "screenshots" / "notebook_outputs" / "cell_19_output_00.png",
        root.parent / "assets" / "screenshots" / "notebook_outputs" / "cell_21_output_00.png",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _float_or_nan(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return float("nan")


def cumulative_trajectory(frames: Iterable[FrameResult], upto_frame: int) -> np.ndarray:
    points = [frame.camera_pose[:3, 3] for frame in frames if frame.frame_id <= upto_frame and frame.has_pose]
    return np.asarray(points, dtype=np.float32).reshape((-1, 3)) if points else np.empty((0, 3), dtype=np.float32)

