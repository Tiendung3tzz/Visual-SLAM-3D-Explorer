from pathlib import Path

import numpy as np

from app.core.artifacts import load_artifact_bundle
from app.core.models import TrackingStatus


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_artifacts_load_and_preserve_real_counts() -> None:
    bundle = load_artifact_bundle(ROOT / "visual_slam_upgrade")

    assert bundle.frame_count == 1362
    assert bundle.tracking_summary["tracked"] == 281
    assert bundle.tracking_summary["map_points"] == 9285
    assert bundle.diagnostics_summary["keyframes"] == 33
    assert bundle.rgbd_metrics["compared_frames"] == 400
    assert bundle.sparse_points.shape == (9285, 3)
    assert bundle.dense_points.shape[1] == 3
    assert bundle.depth_preview is not None
    assert bundle.depth_preview.name == "cell_19_output_00.png"

    assert bundle.frames[0].tracking_status == TrackingStatus.TRACKING
    assert bundle.frames[1].camera_pose is None
    assert bundle.frames[4].tracking_status == TrackingStatus.TRACKING
    assert bundle.frames[-1].tracking_status == TrackingStatus.LOST


def test_missing_pose_is_not_interpreted_as_a_zero_pose() -> None:
    bundle = load_artifact_bundle(ROOT / "visual_slam_upgrade")
    missing = bundle.frames[1]
    assert missing.camera_pose is None
    assert not missing.has_pose
    assert np.all(np.isfinite(bundle.frames[0].camera_pose))

