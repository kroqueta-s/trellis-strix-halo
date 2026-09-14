# SPDX-License-Identifier: MIT
"""UV unwrapping and texture baking. **A picture on the surface, not a colour per vertex.**

Vertex colours (`pipeline._apply_vertex_colors`) give one sample per vertex and
interpolate across each triangle, so their resolution *is* the mesh's. A texture
map separates the two: the colour lives in an image, and a mesh of any density
can carry detail finer than its own triangles - the warning labels and part
numbers on a model become readable rather than smeared.

**The price is that a UV atlas is bound to the vertices and faces it was built
for.** Decimation, hole closing and the manifold conversion all replace them, so
the bake has to happen at a fixed point in the chain rather than at the end
(see `pipeline._postprocess`). In particular it runs **before** `make_manifold`,
which adds patch worth 2.2-2.5x the input surface area
(`post.manifold.close.fan_area_fraction`, measured 2026-09-12): an atlas built
after it would spend about 69% of its texels on internal membrane nobody sees.

**No `nvdiffrast` and no `cumesh`.** Unwrapping is xatlas (MIT), and the bake
rasterizes in UV space, where the triangles are flat, axis-aligned and - this is
what an atlas guarantees - do not overlap. There is no depth test to do, so a
few lines of barycentric arithmetic replace a rasterizer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image

from . import config

# Atlas texels rasterized at once. **This sets the bake's VRAM peak**: a batch
# holds every texel in its triangles' bounding boxes while they are tested, and
# those boxes are wildly uneven - a triangle whose parameterization went wrong
# spans the whole atlas by itself. Counting triangles instead of texels is what
# let one bake ask for 44 GB (measured 2026-09-13). At 2048 the whole atlas is
# 4.2 M texels, so this is a couple of passes' worth, and bounded whatever the
# atlas did.
TEXEL_CHUNK = 8_000_000


@dataclass
class BakeReport:
    """What the bake did, counted rather than claimed."""

    texture_size: int = 0
    faces: int = 0
    n_charts: int = 0
    faces_in_large_charts: float = 0.0
    median_faces_per_chart: float = 0.0
    folds_moved: int = 0
    folds_own_chart: int = 0
    folded_faces: int = 0
    vertices_before: int = 0
    vertices_after: int = 0
    channels: int = 0
    texels_covered: int = 0
    texels_total: int = 0
    texels_unreached: int = 0
    texels_black: int = 0
    texels_black_after_dilate: int = 0
    fill_rounds: int = 0
    dilate_rounds: int = 0
    chart_sec: float = 0.0
    pack_sec: float = 0.0
    bake_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        out["coverage"] = round(self.texels_covered / max(self.texels_total, 1), 4)
        # **The vertex duplication is how shattered the atlas is.** A vertex is
        # copied once per chart it touches, so a large number means the average
        # chart is a handful of triangles - and an atlas of confetti carries
        # less than its texel count suggests.
        out["vertex_growth"] = round(self.vertices_after / max(self.vertices_before, 1), 2)
        return out


def _in_child(function: Any, *arguments: Any) -> Any:
    """Run `function` in a process of its own, so the heartbeat keeps beating.

    **Spawn, not fork**: this is Windows, and the parent holds a CUDA context
    that must not be inherited even where fork exists.
    """
    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
        return pool.submit(function, *arguments).result()


def unwrap(
    mesh: trimesh.Trimesh, size: int = 2048, in_process: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BakeReport]:
    """Cut the surface into charts and lay them flat, then pack them.

    **The charts are decided here, not by xatlas.** Asked to both cut and pack,
    xatlas made 50,052 charts of five triangles from 200,000 faces of this
    surface and took 31.9 s doing it (measured 2026-09-12); the causes are the
    input's roughness and its winding, and no chart option moved them. So
    `charts.project_charts` cuts by direction in space and hands xatlas charts
    that are already flat - which it places in about two seconds.

    **The vertex count still grows**: a vertex on a chart boundary needs one
    copy per chart it belongs to, because it has a different place in the
    atlas in each. The number to watch is `faces_in_large_charts`.

    **Packing runs in a child process** (`unwrap_worker`), because xatlas holds
    the GIL for its whole run and would otherwise silence the heartbeat - see
    that module. `in_process` skips the child; it is for the tests, whose
    shapes are small enough that nothing has time to notice.

    Returns the new vertices, faces and per-vertex UVs in 0..1, and which faces
    are folded: those keep their UVs but must not write the atlas (`bake`).
    """
    from . import charts as charting
    from . import unwrap_worker

    report = BakeReport(faces=int(len(mesh.faces)), vertices_before=int(len(mesh.vertices)))

    mark = time.perf_counter()
    vertices, faces, flat, chart_report = charting.project_charts(
        mesh,
        smoothing_rounds=config.CHART_SMOOTHING,
        min_faces=config.CHART_MIN_FACES,
        fold_area=config.CHART_FOLD_AREA,
    )
    report.chart_sec = round(time.perf_counter() - mark, 2)
    report.n_charts = chart_report.charts
    report.faces_in_large_charts = chart_report.faces_in_large_charts
    report.median_faces_per_chart = chart_report.median_faces_per_chart
    report.folds_moved = chart_report.folds_moved
    report.folds_own_chart = chart_report.folds_own_chart
    report.folded_faces = chart_report.folded_faces

    # **The folds are read off the projection, before packing.** A face whose
    # projected triangle turns over lands on its neighbours in the atlas; it
    # keeps its UVs and reads what is there, but writing would put its colour
    # over theirs. The packed atlas cannot tell: the packer is free to mirror
    # a whole chart (930 of 2,184 on the 200 k specimen), and a sliver's sign
    # does not survive the float32 rounding of the packing.
    folded = _turned_over(flat, faces)

    mark = time.perf_counter()
    arguments = (np.ascontiguousarray(flat, dtype=np.float32), faces, int(size))
    if in_process:
        vmapping, indices, uvs = unwrap_worker.pack(*arguments)
    else:
        vmapping, indices, uvs = _in_child(unwrap_worker.pack, *arguments)
    report.pack_sec = round(time.perf_counter() - mark, 2)

    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    # The fold mask is per face in the order handed to the packer, which xatlas
    # keeps (checked on every run, since a silent reorder would misplace it).
    if not np.array_equal(vmapping[indices], faces):
        raise RuntimeError("xatlas returned the faces in a different order than it was given")
    vertices = np.asarray(vertices, dtype=np.float64)[vmapping]
    uvs = np.asarray(uvs, dtype=np.float32)
    report.vertices_after = int(len(vertices))
    return vertices, indices, uvs, folded, report


def _turned_over(uvs: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Which faces project with a negative area: turned over in their chart."""
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    p = np.asarray(uvs, dtype=np.float64)[faces]
    signed = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
        p[:, 2, 0] - p[:, 0, 0]
    )
    return signed < 0


def _cover(
    uvs: torch.Tensor, faces: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which texel belongs to which triangle, and where inside it.

    Walks the triangles' bounding boxes in texel space and keeps the texel
    centres that fall inside. **An atlas's triangles do not overlap**, so a
    texel claimed twice is a seam artefact rather than a depth question, and the
    later write simply wins.

    Returns the flat texel index, the triangle index and the barycentric weights.
    """
    corners = uvs[faces] * size  # [F, 3, 2]
    low = torch.floor(corners.amin(dim=1)).long().clamp(0, size - 1)
    high = torch.ceil(corners.amax(dim=1)).long().clamp(0, size)
    span = (high - low).clamp(min=0)
    counts = span[:, 0] * span[:, 1]

    keep = counts > 0
    if not bool(keep.any()):
        empty = torch.zeros(0, dtype=torch.long, device=uvs.device)
        return empty, empty, torch.zeros(0, 3, device=uvs.device)

    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    counts, low, span = counts[index], low[index], span[index]

    # One entry per (triangle, texel in its box), built without a python loop:
    # the offset inside each box is the running position minus where that box
    # started.
    total = int(counts.sum())
    triangle = torch.repeat_interleave(torch.arange(index.numel(), device=uvs.device), counts)
    starts = torch.cumsum(counts, dim=0) - counts
    offset = torch.arange(total, device=uvs.device) - starts[triangle]
    width = span[triangle, 0]
    x = low[triangle, 0] + offset % width
    y = low[triangle, 1] + offset // width

    # The texel's centre, in the same units as the corners.
    point = torch.stack([x.float() + 0.5, y.float() + 0.5], dim=1)
    tri = corners[index][triangle]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2, ep = v1 - v0, v2 - v0, point - v0
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # A triangle with no area in the atlas cannot claim a texel.
    safe = torch.where(area.abs() < 1e-12, torch.ones_like(area), area)
    beta = (ep[:, 0] * e2[:, 1] - ep[:, 1] * e2[:, 0]) / safe
    gamma = (e1[:, 0] * ep[:, 1] - e1[:, 1] * ep[:, 0]) / safe
    alpha = 1.0 - beta - gamma
    inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0) & (area.abs() >= 1e-12)

    texel = (y * size + x)[inside]
    return texel, index[triangle][inside], torch.stack([alpha, beta, gamma], dim=1)[inside]


def _batches(uvs: torch.Tensor, faces: torch.Tensor, size: int, budget: int) -> list[torch.Tensor]:
    """Split `faces` so that no batch's atlas boxes hold more than `budget` texels.

    The boxes are what `_cover` allocates for, and they are wildly uneven: most
    triangles cover a handful of texels and a broken one covers the atlas. A
    batch that is a fixed number of triangles is therefore a batch of unknown
    size, which is how a bake comes to ask for tens of gigabytes. A triangle
    bigger than the budget on its own still goes in a batch of its own - there
    is nothing else to do with it - but it no longer takes its neighbours' work
    with it.
    """
    corners = uvs[faces] * size
    low = torch.floor(corners.amin(dim=1)).long().clamp(0, size - 1)
    high = torch.ceil(corners.amax(dim=1)).long().clamp(0, size)
    span = (high - low).clamp(min=0)
    counts = (span[:, 0] * span[:, 1]).to(torch.float64)
    running = torch.cumsum(counts, dim=0)
    # Which batch each triangle falls in, by how much work came before it.
    group = torch.div(running - counts, float(budget), rounding_mode="floor").long()
    group = torch.cummax(group, dim=0).values
    edges = torch.nonzero(group[1:] != group[:-1], as_tuple=False).squeeze(1) + 1
    cuts = [0, *[int(e) for e in edges], int(faces.shape[0])]
    return [faces[a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True) if b > a]


def _dilate(colour: torch.Tensor, filled: torch.Tensor, rounds: int) -> tuple[torch.Tensor, int]:
    """Spread the edge colours outward, so filtering across a seam picks up the chart.

    A texel just outside a chart is sampled whenever the renderer filters near a
    seam. Left at zero it draws a black line along every cut. **This is not
    inpainting** - it copies the nearest filled neighbour and stops when nothing
    changed, which is all a seam needs.
    """
    size = int(colour.shape[0])
    done = 0
    for _ in range(rounds):
        if bool(filled.all()):
            break
        padded = torch.nn.functional.pad(
            colour.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        )
        weight = torch.nn.functional.pad(
            filled.float().reshape(1, 1, size, size), (1, 1, 1, 1), mode="replicate"
        )
        total = torch.nn.functional.avg_pool2d(weight, 3, stride=1) * 9
        summed = torch.nn.functional.avg_pool2d(padded * weight, 3, stride=1) * 9
        neighbour = (summed / total.clamp_min(1e-6)).squeeze(0).permute(1, 2, 0)
        grew = (total.squeeze() > 0) & ~filled
        if not bool(grew.any()):
            break
        colour = torch.where(grew.unsqueeze(-1), neighbour, colour)
        filled = filled | grew
        done += 1
    return colour, done


def _material(baked: np.ndarray, image: Image.Image) -> Any:
    """The material the atlas describes: base colour alone, or the full PBR set.

    glTF puts roughness in the green channel of one texture and metallic in its
    blue, and the opacity in the base colour's alpha - **the layout is the
    format's, not a choice made here** (upstream writes the same one in
    `trellis2_texturing.py`). With only three channels baked there is nothing
    to say, and the plain colour material is what the GLB carries.
    """
    if baked.shape[2] < 6:
        return trimesh.visual.material.SimpleMaterial(image=image)
    base, metallic, roughness, alpha = (
        baked[:, :, :3],
        baked[:, :, 3:4],
        baked[:, :, 4:5],
        baked[:, :, 5:6],
    )
    return trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base, alpha], axis=2), mode="RGBA"),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=2), mode="RGB"
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        # **Opaque even though the alpha is carried**: the decoder's alpha is
        # material information, and a viewer that blended on it would make
        # holes in a mesh that is meant to be printed.
        alphaMode="OPAQUE",
        doubleSided=True,
    )


def bake(
    mesh: trimesh.Trimesh,
    query: Any,
    size: int,
    dilate: int = 4,
    in_process: bool = False,
    fill: int = 16,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Unwrap `mesh` and bake the colours `query` returns into a texture.

    Args:
        mesh: The mesh to texture. **Its vertices are replaced** by the
            unwrapped ones, which is why the result is returned rather than the
            original being changed.
        query: Takes `[N, 3]` positions on the device and returns `[N, C]` in
            0..1, the first three channels being RGB. `pipeline` passes the
            texture decoder's voxels, which carry metallic, roughness and
            alpha after the colour; whatever else arrives is written into the
            atlas the same way, and the channel count decides what material
            the GLB gets.
        size: The texture is `size` x `size`.
        dilate: How many rounds to spread the colours past the chart edges.
        fill: How many rounds to close the holes *inside* a chart - texels a
            face claimed and the decoder had no colour for. **A different job
            from `dilate`**, which only has to survive a bilinear tap across a
            seam; a hole is as wide as the latent's gap and needs as many
            rounds as it is deep.

    Returns:
        The unwrapped mesh carrying the texture, and what the bake counted.
    """
    vertices, faces, uvs, folded, report = unwrap(mesh, size, in_process=in_process)
    report.texture_size = int(size)
    report.texels_total = int(size) * int(size)

    mark = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_t = torch.as_tensor(uvs, dtype=torch.float32, device=device)
    face_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    vert_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)

    # **The channel count comes from the query, not from here.** Three is a
    # base colour; six is the decoder's full PBR vector, and the extra ones
    # ride the same trilinear sample, so they cost nothing to carry.
    colour: torch.Tensor | None = None
    filled = torch.zeros(size * size, dtype=torch.bool, device=device)
    # A folded face keeps its UVs - it reads the atlas where it lands - but
    # writes nothing, so that it does not paint over the faces it overlaps.
    writers = face_t[torch.as_tensor(~folded, device=device)]
    # **A batch is a number of texels, not a number of triangles.** `_cover`
    # walks each triangle's box in the atlas, so what it costs is the boxes and
    # not the count: one triangle whose parameterization went wrong spans the
    # whole atlas and asks for four million texels on its own. Measured
    # 2026-09-13, a handful of those on one specimen asked for 44 GB at once
    # and took the run out. Cutting the batches on the running total instead
    # bounds that without dropping a triangle or changing a texel.
    for chunk in _batches(uv_t, writers, size, TEXEL_CHUNK):
        texel, triangle, weights = _cover(uv_t, chunk, size)
        if texel.numel() == 0:
            continue
        corners = vert_t[chunk[triangle]]  # [M, 3, 3]
        point = (corners * weights.unsqueeze(-1)).sum(dim=1)
        values = query(point)
        if colour is None:
            colour = torch.zeros(size * size, values.shape[1], dtype=torch.float32, device=device)
        colour[texel] = values
        filled[texel] = True
    if colour is None:
        colour = torch.zeros(size * size, 3, dtype=torch.float32, device=device)

    channels = int(colour.shape[1])
    report.channels = channels
    report.texels_covered = int(filled.sum())
    # **A texel the decoder never saw comes back exactly zero**, the same test
    # `_apply_vertex_colors` counts vertices with. Black paint is a different
    # thing and does not land on exactly zero, so the two are separable.
    wrote_nothing = filled & (colour[:, :3].abs().sum(dim=1) == 0)
    report.texels_black = int(wrote_nothing.sum())
    # **One of those is a hole, not a black texel, so the dilation is told so.**
    # It had been written, which is exactly what the dilation steps over, and
    # the count came out identical on both sides of the pass for that reason.
    # Handing it back as unwritten is all it takes for its neighbours to fill
    # it. Measured 2026-09-13 on the reference robot: 35,992 of them, in the
    # shoulders, the hips and behind the wheels - places a ray reaches and the
    # decoder's latent does not. `texels_black` still counts them, because how
    # much of the atlas was guessed at is worth knowing.
    colour = colour.view(size, size, channels)
    # **The holes are closed first, from the inside out.** They sit in the
    # middle of a chart, so the spreading that follows would reach only their
    # rim; `fill` is sized for how deep they are, `dilate` for a bilinear tap.
    has_colour = (filled & ~wrote_nothing).view(size, size)
    colour, filled_rounds = _dilate(colour, has_colour, fill)
    report.fill_rounds = filled_rounds
    # Then the ordinary spreading past the chart edges, from everything that
    # now has a colour.
    colour, rounds = _dilate(colour, filled.view(size, size), dilate)
    report.dilate_rounds = rounds
    # The report keeps counting what the bake itself wrote, so `texels_covered`
    # and `texels_unreached` still add up.
    report.texels_unreached = report.texels_total - int(filled.sum())
    filled_grid = filled.view(size, size)
    report.texels_black_after_dilate = int(
        (filled_grid & (colour[:, :, :3].abs().sum(dim=2) == 0)).sum()
    )

    eight_bit = (colour.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(eight_bit[:, :, :3], mode="RGB")
    # **The atlas's V axis points the other way from an image's rows**, so the
    # coordinate is flipped here rather than the picture being stored upside
    # down: a texture that looks wrong when opened is one nobody can check.
    flipped = np.column_stack([uvs[:, 0], 1.0 - uvs[:, 1]])
    textured = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=flipped, material=_material(eight_bit, image)),
        process=False,
    )
    report.bake_sec = round(time.perf_counter() - mark, 2)
    return textured, report.as_dict()
