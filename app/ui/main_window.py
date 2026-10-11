from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("PySide6 is required to run the desktop UI. Install requirements.txt.") from exc

from app.core.artifacts import ArtifactBundle
from app.core.depth_estimation import backproject_relative_depth, depth_to_rgb
from app.core.live_processing import LiveProcessor
from app.core.models import FrameResult, TrackingStatus
from app.core.replay import ReplaySession
from app.ui.widgets import COLORS, DiagnosticsPanel, ImagePanel, PointCloudView, TrajectoryPanel, VideoPanel


class _DepthSignals(QObject):
    ready = Signal(int, object, object, float)
    failed = Signal(int, str)


class _DepthTask(QRunnable):
    def __init__(self, holder: dict, frame_id: int, rgb: np.ndarray, signals: _DepthSignals) -> None:
        super().__init__()
        self.holder = holder
        self.frame_id = frame_id
        self.rgb = rgb.copy()
        self.signals = signals

    def run(self) -> None:
        try:
            if self.holder["estimator"] is None:
                from app.core.depth_estimation import DepthEstimator

                self.holder["estimator"] = DepthEstimator()
            depth, elapsed_ms = self.holder["estimator"].infer(self.rgb)
            self.signals.ready.emit(self.frame_id, depth, depth_to_rgb(depth), elapsed_ms)
        except Exception as exc:  # optional dependency/model errors stay in the UI
            self.signals.failed.emit(self.frame_id, str(exc))


class DepthInferenceService:
    def __init__(self) -> None:
        self.holder = {"estimator": None}
        self.busy = False
        self.signals = _DepthSignals()
        self.signals.ready.connect(self._finished)
        self.signals.failed.connect(self._failed)
        self.last_error = ""
        self._on_ready = None
        self._on_failed = None

    def submit(self, frame_id: int, rgb: np.ndarray) -> bool:
        if self.busy:
            return False
        self.busy = True
        try:
            QThreadPool.globalInstance().start(_DepthTask(self.holder, frame_id, rgb, self.signals))
            return True
        except Exception as exc:
            self.busy = False
            self.last_error = str(exc)
            if self._on_failed:
                self._on_failed(frame_id, self.last_error)
            return False

    def _finished(self, frame_id: int, depth: np.ndarray, image: np.ndarray, elapsed_ms: float) -> None:
        self.busy = False
        if self._on_ready:
            self._on_ready(frame_id, depth, image, elapsed_ms)

    def _failed(self, frame_id: int, message: str) -> None:
        self.busy = False
        self.last_error = message
        if self._on_failed:
            self._on_failed(frame_id, message)


class MainWindow(QMainWindow):
    def __init__(self, artifact_dir: Optional[str | Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("Visual SLAM Dashboard")
        self.resize(1440, 960)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parents[2] / "visual_slam_upgrade"
        self.replay: Optional[ReplaySession] = None
        self.live_capture: Optional[cv2.VideoCapture] = None
        self.live_processor: Optional[LiveProcessor] = None
        self.live_history: list[FrameResult] = []
        self.live_tick_times: list[float] = []
        self.depth_service = DepthInferenceService()
        self.depth_preview_rgb: Optional[np.ndarray] = None
        self.live_dense_points = np.empty((0, 3), dtype=np.float32)
        self.live_dense_colors = np.empty((0, 3), dtype=np.uint8)
        self.depth_preview_frame = -1
        self.depth_time_ms: Optional[float] = None
        self.mode = "replay"
        self.map_mode = "Sparse Map"
        self.depth_enabled = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build_ui()
        self._apply_theme()
        self.depth_service._on_ready = self._depth_ready
        self.depth_service._on_failed = self._depth_failed
        self._load_replay(self.artifact_dir)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("◈  Visual SLAM Dashboard")
        title.setObjectName("appTitle")
        subtitle = QLabel("Replayable V3 baseline · CPU-compatible live path")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()
        self.source_label = QLabel("NO SOURCE")
        self.source_label.setObjectName("sourceBadge")
        header.addWidget(self.source_label)
        outer.addLayout(header)

        toolbar = QHBoxLayout()
        self.open_video_button = QPushButton("Open video")
        self.open_video_button.clicked.connect(self._open_video)
        toolbar.addWidget(self.open_video_button)
        self.open_replay_button = QPushButton("Replay folder")
        self.open_replay_button.clicked.connect(self._open_replay_folder)
        toolbar.addWidget(self.open_replay_button)
        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Replay Mode", "Live Processing"])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        toolbar.addWidget(self.mode_combo)
        toolbar.addWidget(QLabel("Map"))
        self.map_combo = QComboBox()
        self.map_combo.addItems(["Sparse Map", "Dense Point Cloud · single frame", "Sensor Depth · single frame"])
        self.map_combo.currentTextChanged.connect(self._map_changed)
        toolbar.addWidget(self.map_combo)
        self.depth_check = QCheckBox("AI Depth")
        self.depth_check.setChecked(True)
        self.depth_check.toggled.connect(self._depth_toggled)
        toolbar.addWidget(self.depth_check)
        self.feature_check = QCheckBox("ORB features")
        self.feature_check.setChecked(True)
        self.feature_check.toggled.connect(self._features_toggled)
        toolbar.addWidget(self.feature_check)
        toolbar.addStretch()
        self.play_button = QPushButton("Start")
        self.play_button.clicked.connect(self._toggle_play)
        toolbar.addWidget(self.play_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        toolbar.addWidget(self.stop_button)
        toolbar.addWidget(QLabel("Speed"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "2.0x"])
        self.speed_combo.setCurrentIndex(1)
        toolbar.addWidget(self.speed_combo)
        outer.addLayout(toolbar)

        content = QSplitter(Qt.Horizontal)
        left = QSplitter(Qt.Vertical)
        self.video_panel = VideoPanel()
        self.depth_panel = ImagePanel("AI Depth · relative / single frame")
        left.addWidget(self.video_panel)
        left.addWidget(self.depth_panel)
        left.setSizes([480, 260])
        right = QSplitter(Qt.Vertical)
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_title = QHBoxLayout()
        map_title.addWidget(QLabel("3D MAP"))
        map_title.addStretch()
        self.landmark_check = QCheckBox("landmarks")
        self.landmark_check.setChecked(True)
        self.landmark_check.toggled.connect(self._map_flags_changed)
        map_title.addWidget(self.landmark_check)
        self.trajectory_check = QCheckBox("trajectory")
        self.trajectory_check.setChecked(True)
        self.trajectory_check.toggled.connect(self._map_flags_changed)
        map_title.addWidget(self.trajectory_check)
        self.keyframe_check = QCheckBox("keyframes")
        self.keyframe_check.setChecked(True)
        self.keyframe_check.toggled.connect(self._map_flags_changed)
        map_title.addWidget(self.keyframe_check)
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self.map_view.reset_view())
        map_title.addWidget(reset)
        map_layout.addLayout(map_title)
        self.map_view = PointCloudView()
        map_layout.addWidget(self.map_view, 1)
        right.addWidget(map_container)
        self.trajectory_panel = TrajectoryPanel()
        right.addWidget(self.trajectory_panel)
        right.setSizes([500, 230])
        content.addWidget(left)
        content.addWidget(right)
        content.setSizes([650, 760])
        outer.addWidget(content, 1)

        self.diagnostics = DiagnosticsPanel()
        outer.addWidget(self.diagnostics)

        bottom = QHBoxLayout()
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.valueChanged.connect(self._seek_changed)
        bottom.addWidget(self.seek, 1)
        self.frame_label = QLabel("Frame -- / --")
        bottom.addWidget(self.frame_label)
        outer.addLayout(bottom)
        self.setCentralWidget(root)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0d1117; color: #e6edf3; font-family: Segoe UI; }
            QFrame#panel { background: #151b24; border: 1px solid #2c3948; border-radius: 8px; }
            QLabel#appTitle { color: #e6edf3; font-size: 20px; font-weight: 700; }
            QLabel#subtitle, QLabel#mutedLabel { color: #7e8da0; }
            QLabel#sourceBadge, QLabel#statusBadge { color: #27c2b3; background: #123934; border-radius: 9px; padding: 4px 9px; font-weight: 700; }
            QLabel#statusBadge[status="LOST"] { color: #ff6b6b; background: #3e2026; }
            QLabel#statusBadge[status="INITIALIZING"] { color: #f3b562; background: #3d2d18; }
            QLabel#panelTitle { color: #e6edf3; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
            QLabel#imageCanvas { background: #090c10; border-radius: 6px; color: #7e8da0; }
            QLabel#metricName { color: #7e8da0; font-size: 10px; }
            QLabel#metricValue { color: #e6edf3; font-size: 15px; font-weight: 700; }
            QPushButton, QComboBox, QCheckBox { background: #1b2430; border: 1px solid #2c3948; border-radius: 5px; padding: 6px 9px; color: #e6edf3; }
            QPushButton:hover, QComboBox:hover { border-color: #27c2b3; }
            QSlider::groove:horizontal { height: 4px; background: #2c3948; }
            QSlider::handle:horizontal { width: 12px; margin: -5px 0; background: #27c2b3; border-radius: 6px; }
            """
        )

    def _load_replay(self, folder: Path) -> None:
        try:
            self.replay = ReplaySession.from_folder(folder)
        except (OSError, ValueError, FileNotFoundError) as exc:
            self.source_label.setText("NO REPLAY ARTIFACT")
            self.statusBar().showMessage(str(exc))
            return
        self.mode = "replay"
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.blockSignals(False)
        self.source_label.setText("REPLAY · V3 ARTIFACT")
        self.seek.setRange(0, max(0, self.replay.bundle.frame_count - 1))
        self._render_replay(0)

    def _open_replay_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select replay artifact folder", str(self.artifact_dir))
        if folder:
            self._load_replay(Path(folder))

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open RGB video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self.statusBar().showMessage(f"Could not open video: {path}")
            return
        if self.live_capture is not None:
            self.live_capture.release()
        self.live_capture = capture
        self.live_processor = LiveProcessor()
        self.live_history = []
        self.live_tick_times = []
        self.depth_preview_rgb = None
        self.live_dense_points = np.empty((0, 3), dtype=np.float32)
        self.live_dense_colors = np.empty((0, 3), dtype=np.uint8)
        self.depth_preview_frame = -1
        self.depth_time_ms = None
        self.mode = "live"
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.blockSignals(False)
        self.source_label.setText("LIVE · ORB / ESSENTIAL")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.seek.setRange(0, max(0, frame_count - 1))
        self._stop()
        self.statusBar().showMessage("Live Processing: poses are emitted only after geometry passes inlier checks")

    def _mode_changed(self, index: int) -> None:
        self.mode = "live" if index == 1 else "replay"
        if self.mode == "replay" and self.replay:
            self.source_label.setText("REPLAY · V3 ARTIFACT")
            self._render_replay(self.seek.value())
        elif self.mode == "live" and self.live_capture:
            self.source_label.setText("LIVE · ORB / ESSENTIAL")

    def _map_changed(self, value: str) -> None:
        self.map_mode = value
        self._render_current()

    def _depth_toggled(self, enabled: bool) -> None:
        self.depth_enabled = enabled
        self._render_current()

    def _features_toggled(self, enabled: bool) -> None:
        self.video_panel.show_features = enabled
        self._render_current()

    def _map_flags_changed(self) -> None:
        self.map_view.show_landmarks = self.landmark_check.isChecked()
        self.map_view.show_trajectory = self.trajectory_check.isChecked()
        self.map_view.show_keyframes = self.keyframe_check.isChecked()
        self.map_view.update()

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Resume")
            return
        speed = float(self.speed_combo.currentText().rstrip("x"))
        self.timer.start(max(8, int(33 / speed)))
        self.play_button.setText("Pause")

    def _stop(self) -> None:
        self.timer.stop()
        self.play_button.setText("Start")
        if self.mode == "live" and self.live_capture:
            self.live_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.live_processor = LiveProcessor()
            self.live_history.clear()
            self.live_tick_times.clear()
            self.depth_preview_rgb = None
            self.live_dense_points = np.empty((0, 3), dtype=np.float32)
            self.live_dense_colors = np.empty((0, 3), dtype=np.uint8)
            self.depth_preview_frame = -1
            self.depth_time_ms = None

    def _tick(self) -> None:
        if self.mode == "replay":
            if not self.replay or self.replay.bundle.frame_count == 0:
                return
            next_frame = self.seek.value() + 1
            if next_frame >= self.replay.bundle.frame_count:
                self.timer.stop()
                self.play_button.setText("Start")
                return
            self.seek.blockSignals(True)
            self.seek.setValue(next_frame)
            self.seek.blockSignals(False)
            self._render_replay(next_frame)
            return
        if not self.live_capture or not self.live_processor:
            return
        ok, frame = self.live_capture.read()
        if not ok:
            self.timer.stop()
            self.play_button.setText("Start")
            return
        fps = self.live_capture.get(cv2.CAP_PROP_FPS) or 30.0
        result = self.live_processor.process(frame, self.live_processor.frame_id / fps)
        self.live_history.append(result)
        self.live_tick_times.append(time.perf_counter())
        frame_index = max(0, int(self.live_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.seek.blockSignals(True)
        self.seek.setValue(frame_index)
        self.seek.blockSignals(False)
        self._render_result(result, self.live_processor.map_points, self.live_processor.map_colors)

    def _seek_changed(self, value: int) -> None:
        if self.mode == "replay" and self.replay and not self.timer.isActive():
            self._render_replay(value)

    def _render_replay(self, frame_id: int) -> None:
        if not self.replay:
            return
        result = self.replay.result_at(frame_id)
        bundle = self.replay.bundle
        if result.rgb_frame is None:
            rgb_preview = bundle.root.parent / "assets" / "screenshots" / "notebook_outputs" / "cell_08_output_01.png"
            self.video_panel.set_file(rgb_preview if rgb_preview.exists() else None, "Offline notebook artifact preview · select a video for frame-synchronous RGB")
        self._render_result(result, *self._replay_cloud(bundle), replay=True)

    def _replay_cloud(self, bundle: ArtifactBundle) -> tuple[np.ndarray, np.ndarray]:
        if self.map_mode.startswith("Dense Point Cloud"):
            return bundle.dense_points, bundle.dense_colors
        if self.map_mode.startswith("Sensor Depth"):
            return bundle.sensor_points, bundle.sensor_colors
        return bundle.sparse_points, bundle.sparse_colors

    def _render_current(self) -> None:
        if self.mode == "replay" and self.replay:
            self._render_replay(self.seek.value())
        elif self.mode == "live" and self.live_history and self.live_processor:
            self._render_result(self.live_history[-1], self.live_processor.map_points, self.live_processor.map_colors)

    def _render_result(self, result: FrameResult, points: np.ndarray, colors: np.ndarray, replay: bool = False) -> None:
        map_message = None
        if result.rgb_frame is not None:
            self.video_panel.set_rgb(
                result.rgb_frame,
                result.keypoints,
                f"Frame {result.frame_id} · {result.tracking_method} · {'LIVE' if not replay else 'REPLAY'}",
            )
        elif replay and self.replay and self.replay.bundle.depth_preview:
            self.video_panel.meta.setText("Offline artifact preview · no RGB video attached")

        if replay and self.replay:
            bundle = self.replay.bundle
            if self.map_mode.startswith("Dense Point Cloud"):
                map_message = "Single-frame AI depth artifact · arbitrary display units"
            elif self.map_mode.startswith("Sensor Depth"):
                map_message = "Single-frame TUM sensor depth artifact"
            if self.depth_enabled:
                self.depth_panel.set_file(bundle.depth_preview, "Notebook output · Depth Anything V2 relative depth · single frame")
            else:
                self.depth_panel.image.setText("AI Depth disabled")
                self.depth_panel.meta.setText("Toggle AI Depth to show the stored relative-depth artifact")
            trajectory = self.replay.trajectory_at(result.frame_id)
            keyframes = self._keyframe_positions(bundle, result.frame_id)
            metrics = self._replay_metrics(result, bundle)
        else:
            if self.map_mode.startswith("Dense Point Cloud"):
                points, colors = self.live_dense_points, self.live_dense_colors
                map_message = (
                    f"AI depth unavailable: {self.depth_service.last_error[:120]}"
                    if self.depth_service.last_error
                    else "AI depth cloud is waiting for inference"
                ) if not len(points) else "Single-frame AI depth · arbitrary display units"
            elif self.map_mode.startswith("Sensor Depth"):
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
                map_message = "Sensor depth is unavailable for a generic MP4 video"
            else:
                map_message = "Live sparse landmarks · ORB geometry"
            trajectory = np.asarray([r.camera_pose[:3, 3] for r in self.live_history if r.has_pose], dtype=np.float32).reshape((-1, 3))
            keyframes = np.empty((0, 3), dtype=np.float32)
            metrics = self._live_metrics(result)
            if self.depth_enabled and result.rgb_frame is not None:
                if not self.depth_service.submit(result.frame_id, result.rgb_frame) and self.depth_service.last_error:
                    self.depth_panel.image.setText("AI Depth unavailable")
                    self.depth_panel.meta.setText(self.depth_service.last_error[:180])
            if self.depth_enabled and self.depth_preview_rgb is not None:
                self.depth_panel.set_rgb(self.depth_preview_rgb, f"Frame {self.depth_preview_frame} · {self.depth_time_ms:.0f} ms · async AI depth")
            elif not self.depth_enabled:
                self.depth_panel.image.setText("AI Depth disabled")
                self.depth_panel.meta.setText("Live depth inference is optional and disabled")
            elif not self.depth_service.last_error:
                self.depth_panel.image.setText("AI Depth waiting for optional model")
                self.depth_panel.meta.setText("Inference runs asynchronously; the last result is retained")

        camera_position = result.camera_pose[:3, 3] if result.has_pose else None
        self.map_view.set_scene(points, colors, trajectory, camera_position, keyframes, map_message)
        self._map_flags_changed()
        self.trajectory_panel.set_trajectory(trajectory)
        self.diagnostics.update_metrics(metrics, result.tracking_status)
        total = self.replay.bundle.frame_count if replay and self.replay else self.seek.maximum() + 1
        self.frame_label.setText(f"Frame {result.frame_id:04d} / {max(0, total - 1):04d}")

    def _keyframe_positions(self, bundle: ArtifactBundle, frame_id: int) -> np.ndarray:
        ids = set(int(value) for value in bundle.tracking_summary.get("keyframes", []))
        return np.asarray([frame.camera_pose[:3, 3] for frame in bundle.frames if frame.frame_id in ids and frame.frame_id <= frame_id and frame.has_pose], dtype=np.float32).reshape((-1, 3))

    def _replay_metrics(self, result: FrameResult, bundle: ArtifactBundle) -> dict:
        summary = bundle.diagnostics_summary
        coverage = float(summary.get("monocular_tracking_coverage", 0.0)) * 100
        return {
            "frame_id": result.frame_id,
            "timestamp": f"{result.timestamp:.3f}s" if math.isfinite(result.timestamp) else "--",
            "orb_keypoints": "--",
            "matches": result.matches or "--",
            "inliers": result.inliers or "--",
            "inlier_ratio": f"{result.inlier_ratio:.1%}" if result.matches else "--",
            "map_points": summary.get("map_points", "--"),
            "keyframes": summary.get("keyframes", "--"),
            "coverage": f"{coverage:.2f}%",
            "lost_frames": sum(frame.tracking_status == TrackingStatus.LOST for frame in bundle.frames),
            "longest_streak": summary.get("longest_tracked_streak", "--"),
            "tracking_fps": "unmeasured",
            "depth_time": "offline",
            "tracking_method": result.tracking_method,
            "ate_rmse": f"{bundle.rgbd_metrics.get('ate_rmse_m', 0):.4f} m" if bundle.rgbd_metrics else "--",
            "rpe_translation": "--",
            "rpe_rotation": "--",
        }

    def _live_metrics(self, result: FrameResult) -> dict:
        tracked = sum(item.tracking_status == TrackingStatus.TRACKING for item in self.live_history)
        lost = sum(item.tracking_status == TrackingStatus.LOST for item in self.live_history)
        streak = run = 0
        for item in self.live_history:
            run = run + 1 if item.tracking_status == TrackingStatus.TRACKING else 0
            streak = max(streak, run)
        return {
            "frame_id": result.frame_id,
            "timestamp": f"{result.timestamp:.3f}s",
            "orb_keypoints": result.diagnostics.get("orb_keypoints", "--"),
            "matches": result.matches,
            "inliers": result.inliers,
            "inlier_ratio": f"{result.inlier_ratio:.1%}",
            "map_points": result.diagnostics.get("map_points", "--"),
            "keyframes": "n/a",
            "coverage": f"{100 * tracked / max(1, len(self.live_history)):.2f}%",
            "lost_frames": lost,
            "longest_streak": streak,
            "depth_time": f"{self.depth_time_ms:.0f} ms" if self.depth_time_ms is not None else "waiting",
            "tracking_fps": self._live_fps(),
            "tracking_method": result.tracking_method,
            "ate_rmse": "--",
            "rpe_translation": "--",
            "rpe_rotation": "--",
        }

    def _live_fps(self) -> str:
        if len(self.live_tick_times) < 2:
            return "--"
        elapsed = self.live_tick_times[-1] - self.live_tick_times[0]
        return f"{(len(self.live_tick_times) - 1) / elapsed:.1f}" if elapsed > 0 else "--"

    def _depth_ready(self, frame_id: int, depth: np.ndarray, image: np.ndarray, elapsed_ms: float) -> None:
        self.depth_preview_frame = frame_id
        self.depth_preview_rgb = image
        self.depth_time_ms = elapsed_ms
        matching = next((item for item in reversed(self.live_history) if item.frame_id == frame_id), None)
        if matching is not None and matching.rgb_frame is not None:
            self.live_dense_points, self.live_dense_colors = backproject_relative_depth(
                matching.rgb_frame, depth
            )
        if self.mode == "live" and self.live_history:
            self._render_result(self.live_history[-1], self.live_processor.map_points, self.live_processor.map_colors)

    def _depth_failed(self, frame_id: int, message: str) -> None:
        self.depth_panel.image.setText("AI Depth unavailable")
        self.depth_panel.meta.setText(f"Frame {frame_id}: {message[:180]}")

