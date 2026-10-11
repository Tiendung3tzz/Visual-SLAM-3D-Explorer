from __future__ import annotations

from pathlib import Path

import cv2

from .artifacts import ArtifactBundle, cumulative_trajectory, load_artifact_bundle
from .models import FrameResult


class ReplaySession:
    """Frame-synchronous access to notebook artifacts."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.frame_id = 0
        self.video: cv2.VideoCapture | None = None

    @classmethod
    def from_folder(cls, folder: str | Path) -> "ReplaySession":
        return cls(load_artifact_bundle(folder))

    def attach_video(self, path: str | Path) -> None:
        if self.video is not None:
            self.video.release()
        self.video = cv2.VideoCapture(str(path))
        if not self.video.isOpened():
            self.video.release()
            self.video = None
            raise OSError(f"Could not open video: {path}")

    def result_at(self, frame_id: int) -> FrameResult:
        frame_id = max(0, min(frame_id, len(self.bundle.frames) - 1))
        self.frame_id = frame_id
        result = self.bundle.frames[frame_id]
        if self.video is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = self.video.read()
            if ok:
                result.rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return result

    def trajectory_at(self, frame_id: int):
        return cumulative_trajectory(self.bundle.frames, frame_id)

