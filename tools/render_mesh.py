# SPDX-License-Identifier: MIT
"""Render a mesh to PNG (**visual checks travel as images**).

No OpenGL and no external renderer: a plain point-splatting rasterizer with a
z-buffer, needing only numpy and trimesh (and no GPU). It exists so a human can
see whether the shape came out and whether there are holes in it.

Example:

    python tools/render_mesh.py mesh.ply out.png --views 4 --size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Accumulate face normals onto vertices and normalize."""
    normals = np.zeros_like(verts)
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(normals, faces[:, i], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    """Rotation matrix: about Y, then about X."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ ry


def _render_one(
    verts: np.ndarray, normals: np.ndarray, size: int, yaw: float, pitch: float, splat: int
) -> np.ndarray:
    """Draw one view: orthographic projection, z-buffer, flat Lambert shading."""
    rot = _rotation(yaw, pitch)
    p = verts @ rot.T
    n = normals @ rot.T

    margin = 0.08
    scale = (size * (1.0 - 2.0 * margin)) / 2.0
    px = np.clip((p[:, 0] * scale + size / 2.0).astype(np.int32), 0, size - 1)
    py = np.clip((-p[:, 1] * scale + size / 2.0).astype(np.int32), 0, size - 1)
    depth = p[:, 2]

    light = np.array([0.4, 0.6, 1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.0, 1.0) * 0.75 + 0.25

    zbuf = np.full((size, size), -np.inf, dtype=np.float64)
    img = np.zeros((size, size), dtype=np.float64)
    # Draw back to front so nearer points win (last write to a pixel is fine).
    order = np.argsort(depth)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            ys = np.clip(py[order] + dy, 0, size - 1)
            xs = np.clip(px[order] + dx, 0, size - 1)
            zs = depth[order]
            keep = zs > zbuf[ys, xs]
            zbuf[ys[keep], xs[keep]] = zs[keep]
            img[ys[keep], xs[keep]] = shade[order][keep]
    return img


def render(
    mesh_path: Path,
    out_path: Path,
    size: int = 512,
    views: int = 4,
    splat: int = 1,
    rotx: float = 0.0,
    largest_only: bool = False,
) -> None:
    """Draw the mesh from several viewpoints into a single PNG strip.

    Args:
        rotx: Degrees to rotate about the X axis before drawing, because models
            disagree about which axis points up.
        largest_only: If true, draw **only the largest connected component**.
            Use it to tell whether floating debris is real geometry or just
            splatting noise.
    """
    mesh = trimesh.load(mesh_path, process=False)
    if largest_only:
        parts = mesh.split(only_watertight=False)
        if len(parts):
            mesh = max(parts, key=lambda p: len(p.faces))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if rotx:
        verts = verts @ _rotation(0.0, np.deg2rad(rotx)).T
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = (verts.min(0) + verts.max(0)) / 2.0
    verts = verts - center
    radius = np.abs(verts).max()
    verts = verts / max(radius, 1e-12)
    normals = _vertex_normals(verts, faces)

    angles = [(i * 2.0 * np.pi / views, np.deg2rad(15.0)) for i in range(views)]
    tiles = [_render_one(verts, normals, size, yaw, pitch, splat) for yaw, pitch in angles]
    strip = np.concatenate(tiles, axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a mesh to PNG")
    parser.add_argument("mesh")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--splat", type=int, default=1)
    parser.add_argument("--rotx", type=float, default=0.0)
    parser.add_argument(
        "--largest-only", action="store_true", help="draw only the largest connected component"
    )
    args = parser.parse_args()
    render(
        Path(args.mesh),
        Path(args.out),
        args.size,
        args.views,
        args.splat,
        args.rotx,
        args.largest_only,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
