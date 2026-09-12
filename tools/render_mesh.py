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


def _sample_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Read the texture at `uv`, mixing the four texels around each sample.

    **Nearest sampling hides what this is for.** A seam shows itself as the
    neighbouring chart's colour bleeding across the cut, and bleeding only
    happens once neighbouring texels are mixed - which is what a renderer does.

    The convention is trimesh's, so that what is drawn here and what
    `visual.material.to_color` returns agree: u to the right, v up from the
    bottom of the image.
    """
    height, width = image.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (width - 1)
    y = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _covered_pixels(corners: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which pixels each triangle covers, and where inside it each one falls.

    Walks the triangles' bounding boxes and keeps the pixel centres inside the
    triangle. **A triangle smaller than a pixel keeps the pixel its centroid is
    in**: at 200,000 faces in a 512-pixel view most triangles are sub-pixel, and
    dropping them would draw the model full of holes.

    Returns the flat pixel index, the triangle index and the barycentric weights.
    """
    low = np.clip(np.floor(corners.min(axis=1)).astype(np.int64), 0, size - 1)
    high = np.clip(np.ceil(corners.max(axis=1)).astype(np.int64), 0, size)
    span = np.maximum(high - low, 0)
    counts = span[:, 0] * span[:, 1]

    total = int(counts.sum())
    triangle = np.repeat(np.arange(len(corners)), counts)
    starts = np.cumsum(counts) - counts
    offset = np.arange(total) - starts[triangle]
    width = np.maximum(span[triangle, 0], 1)
    x = low[triangle, 0] + offset % width
    y = low[triangle, 1] + offset // width

    point = np.stack([x + 0.5, y + 0.5], axis=1)
    tri = corners[triangle]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2, ep = v1 - v0, v2 - v0, point - v0
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.where(np.abs(area) < 1e-12, 1.0, area)
    beta = (ep[:, 0] * e2[:, 1] - ep[:, 1] * e2[:, 0]) / safe
    gamma = (e1[:, 0] * ep[:, 1] - e1[:, 1] * ep[:, 0]) / safe
    alpha = 1.0 - beta - gamma
    inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0) & (np.abs(area) >= 1e-12)

    pixel = (y * size + x)[inside]
    hit = triangle[inside]
    weights = np.stack([alpha, beta, gamma], axis=1)[inside]

    drawn = np.zeros(len(corners), dtype=bool)
    drawn[hit] = True
    missed = np.nonzero(~drawn)[0]
    if missed.size:
        centre = corners[missed].mean(axis=1)
        cx = np.clip(centre[:, 0].astype(np.int64), 0, size - 1)
        cy = np.clip(centre[:, 1].astype(np.int64), 0, size - 1)
        pixel = np.concatenate([pixel, cy * size + cx])
        hit = np.concatenate([hit, missed])
        weights = np.concatenate([weights, np.full((missed.size, 3), 1.0 / 3.0)])
    return pixel, hit, weights


def _render_textured(
    verts: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    image: np.ndarray,
    size: int,
    yaw: float,
    pitch: float,
    two_sided: bool = False,
    chunk: int = 1 << 17,
) -> np.ndarray:
    """Draw one view by rasterizing the triangles and sampling the texture per pixel.

    **Splatting the vertices cannot answer a texture question.** A vertex
    carries one UV, so a point-splatted view shows the atlas sampled once per
    vertex - which is vertex colours again, and exactly what a texture map is
    meant to beat. Here the UV is interpolated over the triangle and the atlas
    is read at every pixel, so a seam, a fold or a black gutter shows up.

    The triangles are taken in chunks because the pixel list is built whole.
    """
    rot = _rotation(yaw, pitch)
    p = verts @ rot.T
    n = normals @ rot.T

    margin = 0.08
    scale = (size * (1.0 - 2.0 * margin)) / 2.0
    screen = np.stack([p[:, 0] * scale + size / 2.0, -p[:, 1] * scale + size / 2.0], axis=1)

    light = np.array([0.4, 0.6, 1.0])
    light /= np.linalg.norm(light)

    img = np.zeros((size * size, 3), dtype=np.float64)
    zbuf = np.full(size * size, -np.inf, dtype=np.float64)
    for start in range(0, len(faces), chunk):
        block = faces[start : start + chunk]
        pixel, triangle, weights = _covered_pixels(screen[block], size)
        if pixel.size == 0:
            continue
        corner = block[triangle]
        depth = (p[corner, 2] * weights).sum(axis=1)
        uv = (uvs[corner] * weights[:, :, None]).sum(axis=1)
        normal = (n[corner] * weights[:, :, None]).sum(axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        facing = np.abs(normal @ light) if two_sided else np.clip(normal @ light, 0.0, 1.0)
        shade = (np.clip(facing, 0.0, 1.0) * 0.75 + 0.25)[:, None]
        colour = _sample_bilinear(image, uv) * shade

        # Nearest last: sorted by depth, the later write to a pixel is the one
        # in front, so duplicates inside this chunk settle themselves.
        order = np.argsort(depth)
        pixel, depth, colour = pixel[order], depth[order], colour[order]
        keep = depth > zbuf[pixel]
        zbuf[pixel[keep]] = depth[keep]
        img[pixel[keep]] = colour[keep]
    return img.reshape(size, size, 3)


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
    texture: bool = False,
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
        texture: If true, rasterize the triangles and sample the mesh's texture
            map at every pixel. **This is the only way to judge an atlas**: the
            splatted views draw one atlas sample per vertex, which cannot show
            a seam. It raises when the mesh carries no texture.
    """
    # `force="mesh"` because a GLB arrives as a scene, and its one mesh is what
    # is being looked at.
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
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
    if texture:
        visual = getattr(mesh, "visual", None)
        uvs = getattr(visual, "uv", None)
        picture = getattr(getattr(visual, "material", None), "baseColorTexture", None)
        if uvs is None or picture is None or len(uvs) != len(verts):
            raise ValueError(f"{mesh_path} carries no texture map")
        image = np.asarray(picture.convert("RGB"), dtype=np.float64) / 255.0
        tiles = [
            _render_textured(
                verts,
                normals,
                np.asarray(uvs, dtype=np.float64),
                faces,
                image,
                size,
                yaw,
                pitch,
                two_sided,
            )
            for yaw, pitch in angles
        ]
    else:
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
    parser.add_argument(
        "--texture",
        action="store_true",
        help="rasterize and sample the mesh's texture map per pixel (fails if it has none)",
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
        args.texture,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
