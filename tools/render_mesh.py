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
    verts: np.ndarray,
    normals: np.ndarray,
    size: int,
    yaw: float,
    pitch: float,
    splat: int,
    two_sided: bool = False,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Draw one view: orthographic projection, z-buffer, flat Lambert shading.

    With `colors` (per-vertex RGB in 0..1) the shade multiplies the vertex's own
    colour and the view comes back as RGB; without it, greyscale as before.
    """
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
    # **Two-sided lighting shows the shape when the winding does not agree.**
    # TRELLIS.2's meshes come out with about half their faces wound the other
    # way (49.2% measured at 512), which renders as salt-and-pepper speckle and
    # hides exactly the detail a visual check is for. Taking the magnitude
    # lights both sides, so what is on screen is the geometry rather than the
    # orientation.
    facing = np.abs(n @ light) if two_sided else np.clip(n @ light, 0.0, 1.0)
    shade = np.clip(facing, 0.0, 1.0) * 0.75 + 0.25

    zbuf = np.full((size, size), -np.inf, dtype=np.float64)
    value = shade[:, None] * colors if colors is not None else shade[:, None]
    img = np.zeros((size, size, value.shape[1]), dtype=np.float64)
    # Draw back to front so nearer points win (last write to a pixel is fine).
    order = np.argsort(depth)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            ys = np.clip(py[order] + dy, 0, size - 1)
            xs = np.clip(px[order] + dx, 0, size - 1)
            zs = depth[order]
            keep = zs > zbuf[ys, xs]
            zbuf[ys[keep], xs[keep]] = zs[keep]
            img[ys[keep], xs[keep]] = value[order][keep]
    return img


def render(
    mesh_path: Path,
    out_path: Path,
    size: int = 512,
    views: int = 4,
    splat: int = 1,
    rotx: float = 0.0,
    largest_only: bool = False,
    two_sided: bool = False,
    color: bool = False,
) -> None:
    """Draw the mesh from several viewpoints into a single PNG strip.

    Args:
        rotx: Degrees to rotate about the X axis before drawing, because models
            disagree about which axis points up.
        largest_only: If true, draw **only the largest connected component**.
            Use it to tell whether floating debris is real geometry or just
            splatting noise.
        color: If true, use the mesh's vertex colours instead of grey. **It
            raises when the mesh has none** rather than returning a grey image
            that looks like a texture stage that produced nothing.
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

    colors = None
    if color:
        raw = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
        if raw is None or len(raw) != len(verts):
            raise ValueError(f"{mesh_path} carries no vertex colours")
        colors = np.asarray(raw, dtype=np.float64)[:, :3] / 255.0

    angles = [(i * 2.0 * np.pi / views, np.deg2rad(15.0)) for i in range(views)]
    tiles = [
        _render_one(verts, normals, size, yaw, pitch, splat, two_sided, colors)
        for yaw, pitch in angles
    ]
    strip = np.concatenate(tiles, axis=1)
    Image.fromarray((strip.squeeze() * 255).astype(np.uint8)).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a mesh to PNG")
    parser.add_argument("mesh")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--splat", type=int, default=1)
    parser.add_argument("--rotx", type=float, default=0.0)
    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="light both faces of a triangle (for meshes whose winding is inconsistent)",
    )
    parser.add_argument(
        "--largest-only", action="store_true", help="draw only the largest connected component"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="shade the mesh's own vertex colours (fails if it has none)",
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
        args.two_sided,
        args.color,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
