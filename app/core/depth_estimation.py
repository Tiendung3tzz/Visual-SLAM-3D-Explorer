from __future__ import annotations

import time
from typing import Any

import numpy as np


class DepthEstimator:
    """Lazy Depth Anything V2 adapter.

    Importing this module does not import Torch or download a model. The
    dashboard can therefore run in CPU-only Replay Mode without AI extras.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    def infer(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        from PIL import Image

        started = time.perf_counter()
        with self.torch.inference_mode():
            inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            depth = self.model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return depth.detach().cpu().numpy().astype(np.float32), elapsed_ms


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    import cv2

    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [2, 98])
    normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def backproject_relative_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    stride: int = 8,
    max_points: int = 30000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a single-frame colored cloud from relative depth.

    The output uses arbitrary display units, matching the notebook's
    Depth-Anything visualization. It must not be treated as metric or fused
    across frames without scale alignment.
    """

    finite = np.isfinite(depth)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [2, 98])
    normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
    z = 5.0 - normalized * 4.5
    height, width = depth.shape[:2]
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    zz = z[::stride, ::stride]
    valid = np.isfinite(zz) & (zz > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    fx = fy = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    xyz = np.column_stack(
        ((uu[valid] - cx) * zz[valid] / fx, (vv[valid] - cy) * zz[valid] / fy, zz[valid])
    ).astype(np.float32)
    colors = rgb[::stride, ::stride][valid].astype(np.uint8)
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=int)
        xyz, colors = xyz[indices], colors[indices]
    return xyz, colors

