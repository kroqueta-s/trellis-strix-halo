# SPDX-License-Identifier: MIT
"""Verify the rasterizer stand-in (**agree on known shapes before using it for real**).

Upstream TRELLIS's post-processing removes interior shells by the visibility
ratio per face. A wrong ratio either **leaves the shell in place** or **removes
faces that were needed**, so it is checked here.

What is checked:

- Viewing an outward-facing box from all around, **every face is visible at some
  point**.
- A **small box hidden inside** another box is **visible from no viewpoint at
  all** (visibility ratio 0).
- Near faces occlude far ones (the z-buffer works).

Run it with this repository's virtual environment, since torch is required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import raster  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _box(center: tuple[float, float, float], size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and outward-facing faces of an axis-aligned box."""
    c = np.array(center, dtype=np.float32)
    h = size / 2.0
    verts = (
        np.array(
            [
                [-1, -1, -1],
                [+1, -1, -1],
                [+1, +1, -1],
                [-1, +1, -1],
                [-1, -1, +1],
                [+1, -1, +1],
                [+1, +1, +1],
                [-1, +1, +1],
            ],
            dtype=np.float32,
        )
        * h
        + c
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -z
            [4, 5, 6],
            [4, 6, 7],  # +z
            [0, 1, 5],
            [0, 5, 4],  # -y
            [3, 7, 6],
            [3, 6, 2],  # +y
            [0, 4, 7],
            [0, 7, 3],  # -x
            [1, 2, 6],
            [1, 6, 5],  # +x
        ],
        dtype=np.int64,
    )
    return verts, faces


def _perspective(fov_deg: float, near: float, far: float) -> torch.Tensor:
    """Square-FOV projection matrix, matching `utils3d.torch.perspective_from_fov_xy`."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = torch.zeros(4, 4, device=DEVICE, dtype=torch.float32)
    m[0, 0] = f
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """View matrix."""
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.cross(right, forward, dim=0)
    m = torch.eye(4, device=DEVICE, dtype=torch.float32)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -torch.stack([right @ eye, true_up @ eye, -(forward @ eye)])
    m[2, 3] = forward @ eye
    return m


def _visibility(
    verts: np.ndarray, faces: np.ndarray, views: int = 60, res: int = 256
) -> np.ndarray:
    """Return the visibility ratio per face, seen from viewpoints on a sphere."""
    v = torch.tensor(verts, device=DEVICE, dtype=torch.float32)[None]
    f = torch.tensor(faces, device=DEVICE, dtype=torch.int64)
    projection = _perspective(40.0, 1.0, 10.0)
    ctx = raster.RastContext()
    counts = torch.zeros(faces.shape[0], device=DEVICE, dtype=torch.int32)
    up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    center = torch.zeros(3, device=DEVICE)
    for i in range(views):
        # Spread roughly uniformly using the golden angle.
        yaw = 2.0 * math.pi * i * 0.618033988749895
        pitch = math.asin(2.0 * (i + 0.5) / views - 1.0)
        eye = (
            torch.tensor(
                [
                    math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                ],
                device=DEVICE,
            )
            * 3.0
        )
        if torch.allclose(torch.nn.functional.normalize(eye, dim=0).abs(), up.abs(), atol=1e-3):
            continue  # straight up or down degenerates the view matrix
        buffers = raster.rasterize_triangle_faces(
            ctx, v, f, res, res, view=_look_at(eye, center, up), projection=projection
        )
        seen = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        counts[torch.unique(seen).long()] += 1
    return (counts.float() / views).cpu().numpy()


def test_outer_box_is_fully_visible() -> None:
    """Every face of an outward-facing box is visible from somewhere."""
    verts, faces = _box((0.0, 0.0, 0.0), 1.0)
    vis = _visibility(verts, faces)
    assert (vis > 0).all(), f"some faces are never visible: {vis}"


def test_hidden_box_is_never_visible() -> None:
    """**A box hidden inside a box is visible from no viewpoint** (visibility ratio 0).

    Upstream treats a ratio of zero as an interior face and wires it to the
    min-cut source. If it does not reach zero, the mechanism that removes
    interior shells never fires.
    """
    outer_v, outer_f = _box((0.0, 0.0, 0.0), 1.0)
    inner_v, inner_f = _box((0.0, 0.0, 0.0), 0.4)
    verts = np.concatenate([outer_v, inner_v], axis=0)
    faces = np.concatenate([outer_f, inner_f + len(outer_v)], axis=0)
    vis = _visibility(verts, faces)
    outer_vis, inner_vis = vis[: len(outer_f)], vis[len(outer_f) :]
    assert (outer_vis > 0).all(), f"some faces of the outer box are never visible: {outer_vis}"
    assert (inner_vis == 0).all(), f"the hidden box was visible: {inner_vis}"


def test_z_buffer_prefers_the_near_face() -> None:
    """A near face occludes a far one (the front face wins the pixel)."""
    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],  # far
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],  # near
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    v = torch.tensor(verts, device=DEVICE)[None]
    f = torch.tensor(faces, device=DEVICE)
    eye = torch.tensor([0.0, 0.0, 5.0], device=DEVICE)
    view = _look_at(
        eye, torch.zeros(3, device=DEVICE), torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )
    buffers = raster.rasterize_triangle_faces(
        raster.RastContext(), v, f, 128, 128, view=view, projection=_perspective(40.0, 1.0, 10.0)
    )
    seen = torch.unique(buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1).tolist()
    assert seen == [1], f"only the near face should be visible: {seen}"


def test_background_is_zero() -> None:
    """Empty pixels get `face_id` 0 (background)."""
    verts, faces = _box((0.0, 0.0, 0.0), 0.2)
    v = torch.tensor(verts, device=DEVICE)[None]
    f = torch.tensor(faces, device=DEVICE)
    eye = torch.tensor([0.0, 0.0, 3.0], device=DEVICE)
    view = _look_at(
        eye, torch.zeros(3, device=DEVICE), torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )
    buffers = raster.rasterize_triangle_faces(
        raster.RastContext(), v, f, 128, 128, view=view, projection=_perspective(40.0, 1.0, 10.0)
    )
    mask = buffers["mask"][0]
    assert mask.sum() > 0, "nothing was drawn"
    assert (buffers["face_id"][0][mask < 0.5] == 0).all(), "background is not 0"


def main() -> int:
    """Run every test."""
    print(f"device: {DEVICE}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
