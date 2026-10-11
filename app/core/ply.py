from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np


_TYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertex positions and colors from common PLY variants.

    The notebook exports binary little-endian PLY files, but ASCII support is
    useful when a user selects a point cloud from another pipeline.
    """

    path = Path(path)
    raw = path.read_bytes()
    marker = b"end_header"
    header_end = raw.find(marker)
    if header_end < 0:
        raise ValueError(f"PLY header not found: {path}")
    data_offset = raw.find(b"\n", header_end) + 1
    header = raw[:data_offset].decode("ascii", errors="replace")
    fmt_match = re.search(r"format\s+(\S+)", header)
    count_match = re.search(r"element\s+vertex\s+(\d+)", header)
    if not fmt_match or not count_match:
        raise ValueError(f"Unsupported PLY header: {path}")
    fmt = fmt_match.group(1)
    count = int(count_match.group(1))
    properties = re.findall(r"^property\s+(\S+)\s+(\S+)\s*$", header, re.MULTILINE)
    names = [name for _, name in properties]

    if fmt == "ascii":
        rows = np.loadtxt(path, dtype=float, skiprows=header.count("\n"), ndmin=2)
        if rows.shape[0] != count:
            rows = rows[:count]
        xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
        colors = _colors_from_columns(rows, names)
        return xyz, colors

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")

    dtype_fields = []
    for type_name, name in properties:
        if type_name not in _TYPE_MAP:
            raise ValueError(f"Unsupported PLY property type {type_name!r}")
        dtype_fields.append((name, "<" + _TYPE_MAP[type_name]))
    dtype = np.dtype(dtype_fields)
    vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=data_offset)
    try:
        xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"PLY does not contain x/y/z vertex properties: {path}") from exc
    colors = np.full((len(vertices), 3), 180, dtype=np.uint8)
    color_names = ("red", "green", "blue")
    if all(name in vertices.dtype.names for name in color_names):
        colors = np.column_stack([vertices[name] for name in color_names]).clip(0, 255).astype(np.uint8)
    return xyz, colors


def _colors_from_columns(rows: np.ndarray, names: list[str]) -> np.ndarray:
    colors = np.full((len(rows), 3), 180, dtype=np.uint8)
    color_names = ("red", "green", "blue")
    if all(name in names for name in color_names):
        colors = rows[:, [names.index(name) for name in color_names]].clip(0, 255).astype(np.uint8)
    return colors

