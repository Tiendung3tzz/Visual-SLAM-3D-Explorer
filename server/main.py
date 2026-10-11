from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.artifacts import ArtifactBundle, cumulative_trajectory, load_artifact_bundle
from app.core.depth_estimation import backproject_relative_depth, depth_to_rgb
from app.core.live_processing import LiveConfig, LiveProcessor
from app.core.models import FrameResult, TrackingStatus


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.getenv("SLAM_ARTIFACT_DIR", ROOT / "visual_slam_upgrade"))
UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", ROOT / "runtime_uploads"))
UPLOAD_ROOT.mkdir(exist_ok=True)

cors_origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()]
cors_regex = os.getenv("CORS_ORIGIN_REGEX") or r"https://.*\.vercel\.app"

app = FastAPI(title="Visual SLAM Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")

_bundle: ArtifactBundle | None = None
_uploads: dict[str, Path] = {}


def bundle() -> ArtifactBundle:
    global _bundle
    if _bundle is None:
        _bundle = load_artifact_bundle(ARTIFACT_ROOT)
    return _bundle


def _sample_cloud(points: np.ndarray, colors: np.ndarray, limit: int = 12000) -> dict[str, list]:
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        points, colors = points[indices], colors[indices]
    return {"points": points.astype(float).round(5).tolist(), "colors": colors.astype(int).tolist()}


def _pose_translation(result: FrameResult) -> list[float] | None:
    return result.camera_pose[:3, 3].astype(float).tolist() if result.has_pose else None


def _frame_payload(frame_id: int) -> dict[str, Any]:
    data = bundle()
    frame_id = max(0, min(int(frame_id), data.frame_count - 1))
    result = data.frames[frame_id]
    trajectory = cumulative_trajectory(data.frames, frame_id)
    summary = data.diagnostics_summary
    return {
        "frame_id": result.frame_id,
        "timestamp": result.timestamp,
        "tracking_status": result.tracking_status,
        "tracking_method": result.tracking_method,
        "camera_position": _pose_translation(result),
        "trajectory": trajectory.round(5).tolist(),
        "matches": result.matches,
        "inliers": result.inliers,
        "inlier_ratio": result.inlier_ratio,
        "diagnostics": {
            "map_points": summary.get("map_points", 0),
            "keyframes": summary.get("keyframes", 0),
            "coverage": float(summary.get("monocular_tracking_coverage", 0)) * 100,
            "lost_frames": sum(item.tracking_status == TrackingStatus.LOST for item in data.frames),
            "longest_streak": summary.get("longest_tracked_streak", 0),
            "tracking_fps": "unmeasured",
            "depth_time": "offline",
            "ate_rmse": data.rgbd_metrics.get("ate_rmse_m", "--"),
        },
    }


def _cloud_for_mode(mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    data = bundle()
    normalized = mode.lower()
    if normalized.startswith("dense"):
        return data.dense_points, data.dense_colors, "Single-frame AI depth artifact · arbitrary display units"
    if normalized.startswith("sensor"):
        return data.sensor_points, data.sensor_colors, "Single-frame TUM sensor depth artifact"
    return data.sparse_points, data.sparse_colors, "Sparse V3 landmarks"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    data = bundle()
    return {
        "frame_count": data.frame_count,
        "tracked_frames": data.tracking_summary.get("tracked", 0),
        "coverage": float(data.diagnostics_summary.get("monocular_tracking_coverage", 0)) * 100,
        "longest_streak": data.diagnostics_summary.get("longest_tracked_streak", 0),
        "map_points": data.diagnostics_summary.get("map_points", len(data.sparse_points)),
        "keyframes": data.diagnostics_summary.get("keyframes", 0),
        "ate_rmse_m": data.rgbd_metrics.get("ate_rmse_m"),
        "rgbd_tracked_poses": data.rgbd_metrics.get("tracking_poses"),
        "depth_preview_url": "/assets/screenshots/notebook_outputs/cell_19_output_00.png",
        "rgb_preview_url": "/assets/screenshots/notebook_outputs/cell_08_output_01.png",
        "map_temporal": False,
    }


@app.get("/api/replay/frame/{frame_id}")
def replay_frame(frame_id: int) -> dict[str, Any]:
    return _frame_payload(frame_id)


@app.get("/api/pointcloud/{mode}")
def pointcloud(mode: str) -> dict[str, Any]:
    points, colors, label = _cloud_for_mode(mode)
    payload = _sample_cloud(points, colors)
    payload["mode"] = mode
    payload["label"] = label
    payload["available"] = bool(len(points))
    return payload


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    video_id = uuid.uuid4().hex
    path = UPLOAD_ROOT / f"{video_id}{suffix}"
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable video")
    metadata = {
        "video_id": video_id,
        "filename": file.filename,
        "fps": capture.get(cv2.CAP_PROP_FPS) or 30.0,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    capture.release()
    _uploads[video_id] = path
    return metadata


@app.websocket("/ws/live/{video_id}")
async def live_stream(websocket: WebSocket, video_id: str) -> None:
    await websocket.accept()
    path = _uploads.get(video_id)
    if path is None or not path.exists():
        await websocket.send_json({"type": "error", "message": "Uploaded video was not found"})
        await websocket.close(code=1008)
        return
    capture = cv2.VideoCapture(str(path))
    processor = LiveProcessor(LiveConfig(orb_features=1200))
    history: list[FrameResult] = []
    live_depth_enabled = os.getenv("SLAM_LIVE_DEPTH", "0") == "1"
    depth_estimator = None
    dense_points = np.empty((0, 3), dtype=np.float32)
    dense_colors = np.empty((0, 3), dtype=np.uint8)
    depth_time_ms: float | None = None
    depth_error = ""
    depth_jpeg: str | None = None
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while True:
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok:
                await websocket.send_json({"type": "complete"})
                break
            result = await asyncio.to_thread(processor.process, frame, processor.frame_id / fps)
            history.append(result)
            if live_depth_enabled and result.frame_id % 12 == 0:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    depth_estimator, depth, depth_time_ms = await asyncio.to_thread(_infer_depth, depth_estimator, rgb)
                    dense_points, dense_colors = backproject_relative_depth(rgb, depth)
                    depth_rgb = depth_to_rgb(depth)
                    depth_ok, depth_buffer = cv2.imencode(".jpg", cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                    depth_jpeg = base64.b64encode(depth_buffer.tobytes()).decode("ascii") if depth_ok else None
                    depth_error = ""
                except Exception as exc:
                    depth_error = str(exc)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            poses = np.asarray([item.camera_pose[:3, 3] for item in history if item.has_pose], dtype=np.float32).reshape((-1, 3))
            keypoints = result.keypoints[:: max(1, len(result.keypoints) // 800)]
            payload = {
                "type": "frame",
                "frame_id": result.frame_id,
                "timestamp": result.timestamp,
                "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "tracking_status": result.tracking_status,
                "tracking_method": result.tracking_method,
                "camera_position": _pose_translation(result),
                "keypoints": [[round(x, 2), round(y, 2)] for x, y in keypoints],
                "trajectory": poses.round(5).tolist(),
                "map_points": processor.map_points[:: max(1, len(processor.map_points) // 2500)].round(5).tolist(),
                "map_colors": processor.map_colors[:: max(1, len(processor.map_colors) // 2500)].tolist(),
                "matches": result.matches,
                "inliers": result.inliers,
                "inlier_ratio": result.inlier_ratio,
                "diagnostics": {
                    "orb_keypoints": result.diagnostics.get("orb_keypoints", 0),
                    "map_points": len(processor.map_points),
                    "coverage": 100 * sum(item.tracking_status == TrackingStatus.TRACKING for item in history) / len(history),
                    "lost_frames": sum(item.tracking_status == TrackingStatus.LOST for item in history),
                    "longest_streak": _longest_streak(history),
                    "tracking_fps": fps,
                    "depth_time": "unavailable",
                    "sensor_depth": "unavailable for generic MP4",
                },
                "dense_available": False,
                "sensor_available": False,
            }
            payload["dense_points"] = dense_points.round(5).tolist()
            payload["dense_colors"] = dense_colors.tolist()
            payload["dense_available"] = bool(len(dense_points))
            payload["diagnostics"]["depth_time"] = depth_time_ms if depth_time_ms is not None else (depth_error or "unavailable")
            payload["depth_error"] = depth_error
            payload["depth_jpeg"] = depth_jpeg
            payload["depth_time"] = depth_time_ms
            await websocket.send_json(payload)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        capture.release()


def _longest_streak(history: list[FrameResult]) -> int:
    longest = run = 0
    for item in history:
        run = run + 1 if item.tracking_status == TrackingStatus.TRACKING else 0
        longest = max(longest, run)
    return longest


def _infer_depth(estimator: Any, rgb: np.ndarray) -> tuple[Any, np.ndarray, float]:
    if estimator is None:
        from app.core.depth_estimation import DepthEstimator

        estimator = DepthEstimator()
    depth, elapsed_ms = estimator.infer(rgb)
    return estimator, depth, elapsed_ms


# Keep the catch-all frontend mount last so it never shadows /api or /ws.
FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

