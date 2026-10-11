import cv2
import numpy as np

from app.core.live_processing import LiveConfig, LiveProcessor
from app.core.depth_estimation import backproject_relative_depth
from app.core.models import TrackingStatus


def test_live_processor_emits_initializing_then_rejects_blank_geometry() -> None:
    processor = LiveProcessor(LiveConfig(orb_features=300, min_matches=8, min_inliers=5))
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    first = processor.process(blank)
    second = processor.process(blank)

    assert first.tracking_status == TrackingStatus.INITIALIZING
    assert second.tracking_status == TrackingStatus.LOST
    assert second.camera_pose is None


def test_live_processor_keeps_real_frame_rgb_data() -> None:
    processor = LiveProcessor(LiveConfig(orb_features=300))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (100, 90), (255, 255, 255), 2)
    result = processor.process(frame)
    assert result.rgb_frame is not None
    assert result.rgb_frame.shape == (120, 160, 3)


def test_relative_depth_builds_a_single_frame_colored_cloud() -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    depth = np.linspace(0.0, 1.0, 24 * 32, dtype=np.float32).reshape(24, 32)

    points, colors = backproject_relative_depth(rgb, depth, stride=4)

    assert points.shape[1] == 3
    assert colors.shape == points.shape
    assert len(points) > 0
    assert np.all(colors[:, 0] == 255)

