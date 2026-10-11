from pathlib import Path

from app.core.models import TrackingStatus
from app.core.replay import ReplaySession


def test_replay_trajectory_is_capped_at_requested_frame() -> None:
    root = Path(__file__).resolve().parents[1] / "visual_slam_upgrade"
    replay = ReplaySession.from_folder(root)

    early = replay.trajectory_at(4)
    late = replay.trajectory_at(20)
    assert len(early) > 0
    assert len(late) >= len(early)
    assert replay.result_at(1357).tracking_status == TrackingStatus.LOST

