from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QLabel,
        QProgressBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only without GUI extras
    raise ImportError("PySide6 is required to run the desktop UI. Install requirements.txt.") from exc


COLORS = {
    "background": QColor("#0d1117"),
    "panel": QColor("#151b24"),
    "panel_alt": QColor("#1b2430"),
    "border": QColor("#2c3948"),
    "muted": QColor("#7e8da0"),
    "text": QColor("#e6edf3"),
    "teal": QColor("#27c2b3"),
    "blue": QColor("#59a6ff"),
    "orange": QColor("#f3b562"),
    "red": QColor("#ff6b6b"),
}


def rgb_to_pixmap(rgb: np.ndarray, max_size: Optional[tuple[int, int]] = None) -> QPixmap:
    if rgb.ndim != 3:
        raise ValueError("Expected an HxWx3 image")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image)
    if max_size:
        pixmap = pixmap.scaled(*max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap


class Panel(QFrame):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFrameShape(QFrame.StyledPanel)
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("panelTitle")


class ImagePanel(Panel):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.title_label)
        self.image = QLabel("Waiting for frame data")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(180)
        self.image.setObjectName("imageCanvas")
        layout.addWidget(self.image, 1)
        self.meta = QLabel("No frame selected")
        self.meta.setObjectName("mutedLabel")
        layout.addWidget(self.meta)

    def set_pixmap(self, pixmap: QPixmap, meta: str = "") -> None:
        self.image.setPixmap(pixmap)
        self.meta.setText(meta)

    def set_rgb(self, rgb: np.ndarray, meta: str = "") -> None:
        self.set_pixmap(rgb_to_pixmap(rgb, (900, 440)), meta)

    def set_file(self, path: Optional[Path], meta: str = "") -> None:
        if path and path.exists():
            pixmap = QPixmap(str(path)).scaled(900, 440, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.set_pixmap(pixmap, meta)
        else:
            self.image.setText("No depth preview available")
            self.meta.setText(meta)


class VideoPanel(ImagePanel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("RGB Video", parent)
        self.show_features = True

    def set_rgb(self, rgb: np.ndarray, keypoints: list[tuple[float, float]], meta: str = "") -> None:
        pixmap = rgb_to_pixmap(rgb, (900, 440))
        if self.show_features and keypoints:
            painter = QPainter(pixmap)
            painter.setPen(QPen(COLORS["teal"], 2))
            sx = pixmap.width() / rgb.shape[1]
            sy = pixmap.height() / rgb.shape[0]
            for x, y in keypoints[:: max(1, len(keypoints) // 800)]:
                painter.drawEllipse(QPointF(x * sx, y * sy), 1.5, 1.5)
            painter.end()
        self.set_pixmap(pixmap, meta)


class PointCloudView(QFrame):
    """Lightweight interactive point-cloud viewport with orbit/pan/zoom."""

    view_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.trajectory = np.empty((0, 3), dtype=np.float32)
        self.keyframes = np.empty((0, 3), dtype=np.float32)
        self.camera_position: Optional[np.ndarray] = None
        self.pitch = -0.35
        self.yaw = 0.65
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=np.float32)
        self.last_mouse = QPoint()
        self.drag_button = Qt.NoButton
        self.show_landmarks = True
        self.show_trajectory = True
        self.show_keyframes = True
        self.empty_message = "No 3D map loaded"

    def set_scene(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        trajectory: np.ndarray,
        camera_position: Optional[np.ndarray],
        keyframes: np.ndarray,
        empty_message: Optional[str] = None,
    ) -> None:
        if len(points) > 16000:
            indices = np.linspace(0, len(points) - 1, 16000, dtype=int)
            points, colors = points[indices], colors[indices]
        self.points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        self.colors = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
        self.trajectory = np.asarray(trajectory, dtype=np.float32).reshape((-1, 3))
        self.keyframes = np.asarray(keyframes, dtype=np.float32).reshape((-1, 3))
        self.camera_position = None if camera_position is None else np.asarray(camera_position, dtype=np.float32)
        if empty_message is not None:
            self.empty_message = empty_message
        self.update()

    def reset_view(self) -> None:
        self.pitch, self.yaw, self.zoom = -0.35, 0.65, 1.0
        self.pan[:] = 0
        self.update()

    def mousePressEvent(self, event) -> None:
        self.last_mouse = event.position().toPoint()
        self.drag_button = event.button()

    def mouseMoveEvent(self, event) -> None:
        if self.drag_button == Qt.NoButton:
            return
        delta = event.position().toPoint() - self.last_mouse
        self.last_mouse = event.position().toPoint()
        if self.drag_button == Qt.LeftButton:
            self.yaw += delta.x() * 0.012
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.012, -1.5, 1.5))
        else:
            self.pan += np.array([delta.x(), -delta.y()], dtype=np.float32) / max(self.height(), 1) * 3
        self.view_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.drag_button = Qt.NoButton

    def wheelEvent(self, event) -> None:
        self.zoom *= 1.15 if event.angleDelta().y() > 0 else 0.87
        self.zoom = float(np.clip(self.zoom, 0.12, 8.0))
        self.view_changed.emit()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), COLORS["background"])
        if not len(self.points) and not len(self.trajectory):
            painter.setPen(COLORS["muted"])
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_message)
            painter.end()
            return

        all_points = [arr for arr in (self.points, self.trajectory, self.keyframes) if len(arr)]
        combined = np.vstack(all_points)
        center = np.nanmedian(combined, axis=0)
        spread = float(np.nanpercentile(np.linalg.norm(combined - center, axis=1), 90)) or 1.0
        projected = self._project(self.points, center, spread)
        if self.show_landmarks:
            for (x, y, depth), color in zip(projected, self.colors):
                if depth > 0:
                    painter.setPen(QPen(QColor(int(color[0]), int(color[1]), int(color[2])), 2))
                    painter.drawPoint(QPointF(x, y))

        if self.show_trajectory and len(self.trajectory) > 1:
            painter.setPen(QPen(COLORS["red"], 2))
            path = self._project(self.trajectory, center, spread)
            for first, second in zip(path[:-1], path[1:]):
                painter.drawLine(QPointF(first[0], first[1]), QPointF(second[0], second[1]))
        if self.show_keyframes and len(self.keyframes):
            painter.setPen(QPen(COLORS["orange"], 6))
            for x, y, _ in self._project(self.keyframes, center, spread):
                painter.drawPoint(QPointF(x, y))
        if self.camera_position is not None:
            painter.setPen(QPen(COLORS["teal"], 8))
            x, y, _ = self._project(self.camera_position.reshape(1, 3), center, spread)[0]
            painter.drawPoint(QPointF(x, y))

        painter.setPen(QPen(COLORS["border"], 1))
        painter.drawText(12, 22, "LMB orbit   RMB pan   wheel zoom")
        painter.setPen(QPen(COLORS["muted"], 1))
        painter.drawText(12, self.height() - 12, "X red  ·  Y green  ·  Z blue")
        painter.end()

    def _project(self, points: np.ndarray, center: np.ndarray, spread: float) -> np.ndarray:
        if not len(points):
            return np.empty((0, 3), dtype=np.float32)
        points = (points - center) / spread
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotated = points @ (rx @ ry).T
        scale = min(self.width(), self.height()) * 0.40 * self.zoom
        x = self.width() / 2 + rotated[:, 0] * scale + self.pan[0]
        y = self.height() / 2 - rotated[:, 1] * scale + self.pan[1]
        return np.column_stack((x, y, rotated[:, 2]))


class TrajectoryPanel(Panel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Estimated Trajectory", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.title_label)
        self.canvas = TrajectoryCanvas()
        layout.addWidget(self.canvas, 1)

    def set_trajectory(self, trajectory: np.ndarray) -> None:
        self.canvas.trajectory = trajectory
        self.canvas.update()


class TrajectoryCanvas(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.trajectory = np.empty((0, 3), dtype=np.float32)
        self.setMinimumHeight(180)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), COLORS["background"])
        painter.setPen(QPen(COLORS["border"], 1))
        painter.drawRect(self.rect().adjusted(8, 8, -8, -8))
        if len(self.trajectory) < 2:
            painter.setPen(COLORS["muted"])
            painter.drawText(self.rect(), Qt.AlignCenter, "Trajectory unavailable")
            painter.end()
            return
        xy = self.trajectory[:, [0, 2]]
        lo, hi = np.nanmin(xy, axis=0), np.nanmax(xy, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        pad = 24
        width, height = max(self.width() - 2 * pad, 1), max(self.height() - 2 * pad, 1)
        points = np.column_stack((pad + (xy[:, 0] - lo[0]) / span[0] * width, pad + height - (xy[:, 1] - lo[1]) / span[1] * height))
        painter.setPen(QPen(COLORS["teal"], 3))
        for first, second in zip(points[:-1], points[1:]):
            painter.drawLine(QPointF(*first), QPointF(*second))
        painter.setPen(QPen(COLORS["teal"], 10))
        painter.drawPoint(QPointF(*points[-1]))
        painter.end()


class DiagnosticsPanel(Panel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Tracking Diagnostics", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        header = QGridLayout()
        header.addWidget(self.title_label, 0, 0)
        self.status = QLabel("● INITIALIZING")
        self.status.setObjectName("statusBadge")
        header.addWidget(self.status, 0, 1, alignment=Qt.AlignRight)
        layout.addLayout(header)
        grid = QGridLayout()
        self.values: dict[str, QLabel] = {}
        fields = [
            ("Frame", "frame_id"),
            ("Timestamp", "timestamp"),
            ("ORB keypoints", "orb_keypoints"),
            ("Matches", "matches"),
            ("Inliers", "inliers"),
            ("Inlier ratio", "inlier_ratio"),
            ("Map points", "map_points"),
            ("Keyframes", "keyframes"),
            ("Coverage", "coverage"),
            ("Lost frames", "lost_frames"),
            ("Longest streak", "longest_streak"),
            ("Tracking FPS", "tracking_fps"),
            ("Depth time", "depth_time"),
            ("Tracking method", "tracking_method"),
            ("ATE RMSE", "ate_rmse"),
            ("RPE translation", "rpe_translation"),
            ("RPE rotation", "rpe_rotation"),
        ]
        for index, (label, key) in enumerate(fields):
            row, col = divmod(index, 6)
            box = QVBoxLayout()
            name = QLabel(label)
            name.setObjectName("metricName")
            value = QLabel("--")
            value.setObjectName("metricValue")
            box.addWidget(name)
            box.addWidget(value)
            wrapper = QWidget()
            wrapper.setLayout(box)
            grid.addWidget(wrapper, row, col)
            self.values[key] = value
        layout.addLayout(grid)

    def update_metrics(self, metrics: dict, status: str) -> None:
        self.status.setText(f"● {status}")
        self.status.setProperty("status", status)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        for key, value in metrics.items():
            if key in self.values:
                self.values[key].setText(str(value))

