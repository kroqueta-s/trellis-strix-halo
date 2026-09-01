# SPDX-License-Identifier: MIT
"""Replace the `utils3d.torch` rasterizer with **a z-buffer written in torch**.

Upstream TRELLIS's post-processing (`_fill_holes` in
`trellis/utils/postprocessing_utils.py`) **rasterizes from many viewpoints to
get a visibility ratio per face** and min-cuts away the faces with a ratio of
zero. That is upstream's way of removing shells sealed inside the model.
**It never cuts by size.**

Only that rasterization depends on `nvdiffrast` (CUDA only), which is
unavailable here. Replacing it means **upstream's `postprocess_mesh` runs
without a single line being modified.**

## The approach: **fill triangles exactly, do not approximate**

The first implementation scattered sample points over each triangle and pushed
them into a z-buffer. `tests/test_raster.py` caught it **breaking down on large
triangles**: geometry behind leaked through the gaps between samples, so **a box
hidden inside a box counted as "visible"**. Since a zero visibility ratio is what
identifies interior faces, upstream's mechanism stops working when that is wrong.

So a pixel centre is tested against the triangle **exactly, by the sign of the
edge functions**. Triangles are bucketed by the size of their screen-space
bounding box, and each bucket evaluates a `K x K` grid in one go. **Faces on this
machine are mostly sub-pixel** (a million faces from a resolution-256 grid, drawn
at 1024^2), so most buckets are `K=1`: a single pixel test.

Depth is interpolated linearly in screen-space barycentrics, which is correct
because NDC z is linear in screen space.
"""

from __future__ import annotations

from typing import Any

import torch

# Cap on triangle-times-pixel tests evaluated at once. This sets the VRAM peak.
TILE_BUDGET = 16_000_000

# Bits used to pack the depth. The low 32 bits hold the face index.
_DEPTH_BITS = 21
_FACE_BITS = 32
_EMPTY = (1 << 62) - 1


class RastContext:
    """Stand-in for `utils3d.torch.RastContext`. **It holds no state.**

    Upstream carries an nvdiffrast GL or CUDA context; this implementation is
    pure torch and has nothing to hold. Arguments are accepted and discarded, so
    that callers need no change.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.backend = kwargs.get("backend", "torch")


def _rasterize(
    screen: torch.Tensor, depth: torch.Tensor, keep: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Fill screen-space triangles into a z-buffer and return the packed key per pixel.

    Args:
        screen: `[F, 3, 2]` screen coordinates, in pixels.
        depth: `[F, 3]` NDC z (-1 near, 1 far).
        keep: `[F]` booleans; false faces are discarded (in front of the near
            plane, for instance).
        width: Width.
        height: Height.

    Returns:
        `[H*W]` of `int64`, depth in the high bits and face index in the low
        bits. Empty pixels hold `_EMPTY`.
    """
    device = screen.device
    buffer = torch.full((height * width,), _EMPTY, dtype=torch.int64, device=device)
    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if index.numel() == 0:
        return buffer

    tri = screen[index]  # [f, 3, 2]
    lo = tri.amin(dim=1)
    hi = tri.amax(dim=1)
    x0 = lo[:, 0].floor().clamp(0, width - 1).long()
    y0 = lo[:, 1].floor().clamp(0, height - 1).long()
    x1 = hi[:, 0].ceil().clamp(0, width - 1).long()
    y1 = hi[:, 1].ceil().clamp(0, height - 1).long()
    span = torch.maximum(x1 - x0, y1 - y0) + 1  # [f]

    depth_scale = float((1 << _DEPTH_BITS) - 1)
    # Bucket by bounding-box size rounded up to a power of two. **Mostly 1 pixel.**
    bucket = torch.pow(2, torch.ceil(torch.log2(span.float().clamp(min=1.0)))).long()
    for size in torch.unique(bucket).tolist():
        sel = torch.nonzero(bucket == size, as_tuple=False).squeeze(1)
        per_chunk = max(1, TILE_BUDGET // (size * size))
        for start in range(0, sel.numel(), per_chunk):
            part = sel[start : start + per_chunk]
            faces_here = index[part]
            v = tri[part]  # [n, 3, 2]
            grid = torch.arange(size, device=device)
            n_here = part.numel()
            shape = (n_here, size, size)
            px = (x0[part][:, None, None] + grid[None, :, None]).expand(shape)
            py = (y0[part][:, None, None] + grid[None, None, :]).expand(shape)
            sx = px.to(v.dtype) + 0.5
            sy = py.to(v.dtype) + 0.5

            ax, ay = v[:, 0, 0][:, None, None], v[:, 0, 1][:, None, None]
            bx, by = v[:, 1, 0][:, None, None], v[:, 1, 1][:, None, None]
            cx, cy = v[:, 2, 0][:, None, None], v[:, 2, 1][:, None, None]
            # Edge functions: inside when all three share the sign of the area.
            w0 = (cx - bx) * (sy - by) - (cy - by) * (sx - bx)
            w1 = (ax - cx) * (sy - cy) - (ay - cy) * (sx - cx)
            w2 = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)
            area = w0 + w1 + w2
            sign = torch.where(area >= 0, 1.0, -1.0)
            inside = (w0 * sign >= 0) & (w1 * sign >= 0) & (w2 * sign >= 0) & (area.abs() > 1e-12)
            inside &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not bool(inside.any()):
                continue

            safe_area = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
            z = depth[faces_here]  # [n, 3]
            zz = (
                w0 * z[:, 0][:, None, None]
                + w1 * z[:, 1][:, None, None]
                + w2 * z[:, 2][:, None, None]
            ) / safe_area
            inside &= (zz >= -1.0) & (zz <= 1.0)
            if not bool(inside.any()):
                continue

            pixel = (py * width + px)[inside]
            depth_q = (((zz.clamp(-1.0, 1.0) + 1.0) * 0.5) * depth_scale).long()[inside]
            face_idx = faces_here[:, None, None].expand(shape)[inside]
            # Depth high, face index low, so **taking the minimum wins the front face**.
            key = (depth_q << _FACE_BITS) | face_idx
            buffer.scatter_reduce_(0, pixel, key, reduce="amin")
    return buffer


def rasterize_triangle_faces(
    ctx: RastContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    attr: torch.Tensor | None = None,
    uv: torch.Tensor | None = None,
    texture: torch.Tensor | None = None,
    model: torch.Tensor | None = None,
    view: torch.Tensor | None = None,
    projection: torch.Tensor | None = None,
    antialiasing: bool | list[int] = True,
    diff_attrs: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Return the face-ID buffer (`_fill_holes` uses only `face_id` and `mask`).

    Args:
        ctx: Unused; accepted only to match upstream's signature.
        vertices: `[B, N, 3 or 4]`. **Only B = 1 is supported**, since
            `_fill_holes` calls one view at a time.
        faces: `[F, 3]`.
        width: Output width.
        height: Output height.
        view: `[4, 4]` view matrix.
        projection: `[4, 4]` projection matrix.

    Returns:
        `face_id` (`[1, H, W]`, **one-based; 0 is background**), `mask`
        (`[1, H, W]` float) and `depth` (`[1, H, W]`, 0 near and 1 far).

    Raises:
        NotImplementedError: If attribute or texture interpolation is requested,
            which never happens on this path.
    """
    if attr is not None or uv is not None or texture is not None:
        raise NotImplementedError(
            "this stand-in produces face IDs only "
            "(attribute and texture interpolation need nvdiffrast)"
        )
    if vertices.ndim != 3 or vertices.shape[0] != 1:
        raise NotImplementedError(f"only a batch of 1 is supported: {tuple(vertices.shape)}")

    device = vertices.device
    verts = vertices[0].float()
    if verts.shape[-1] == 3:
        verts = torch.cat([verts, torch.ones_like(verts[..., :1])], dim=-1)

    eye = torch.eye(4, device=device, dtype=verts.dtype)
    mvp = projection.float() if projection is not None else eye
    if view is not None:
        mvp = mvp @ view.float()
    if model is not None:
        mvp = mvp @ model.float()
    pos_clip = verts @ mvp.transpose(-1, -2)

    idx = faces.long()
    tri_clip = pos_clip[idx]  # [F, 3, 4]
    w = tri_clip[..., 3]
    # **Discard triangles that wrap in front of the near plane**, where the
    # division breaks down. `_fill_holes` views the subject from outside at
    # radius 2 with a near plane of 1, so they never actually occur.
    keep = (w > 1e-6).all(dim=1)
    inv_w = 1.0 / w.clamp(min=1e-6)
    ndc = tri_clip[..., :3] * inv_w.unsqueeze(-1)
    screen = torch.stack(
        [(ndc[..., 0] * 0.5 + 0.5) * width, (ndc[..., 1] * 0.5 + 0.5) * height], dim=-1
    )

    buffer = _rasterize(screen, ndc[..., 2], keep, width, height)

    hit = buffer != _EMPTY
    face_id = torch.zeros(height * width, dtype=torch.int64, device=device)
    face_id[hit] = (buffer[hit] & ((1 << _FACE_BITS) - 1)) + 1
    depth = torch.ones(height * width, dtype=torch.float32, device=device)
    depth[hit] = (buffer[hit] >> _FACE_BITS).float() / float((1 << _DEPTH_BITS) - 1)

    return {
        "face_id": face_id.view(1, height, width),
        "mask": hit.view(1, height, width).float(),
        "depth": depth.view(1, height, width),
    }


def install() -> None:
    """Replace the `utils3d.torch` rasterizer.

    **Call this before importing `postprocessing_utils`.** That module reaches
    for `utils3d.torch.RastContext(...)` by attribute, so replacing the module
    attributes is enough.
    """
    import utils3d.torch as u3t

    u3t.RastContext = RastContext
    u3t.rasterize_triangle_faces = rasterize_triangle_faces
