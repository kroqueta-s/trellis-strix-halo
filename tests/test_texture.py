# SPDX-License-Identifier: MIT
"""Verify the UV bake **without a GPU, weights or a model**.

The bake has one part that can be wrong quietly: the map from a texel to the
triangle under it and the barycentric weights inside it. Everything downstream
takes that on trust, and a texture that is subtly wrong looks like a texture.

**It has an exact reference.** Give the mesh vertices whose 3D positions *are*
their UV coordinates, and the position interpolated for a texel must come back
as that texel's own centre - to floating-point. Nothing about the geometry,
the model or the colours enters into it.

Run it with either virtual environment; it falls back to CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2 import texture  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _unit_quad() -> tuple[torch.Tensor, torch.Tensor]:
    """Two triangles covering the whole atlas, wound the same way."""
    uvs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=DEVICE)
    return uvs, faces


def test_a_full_quad_covers_every_texel_once() -> None:
    """**A gap is a hole in the texture and an overlap is a seam**, so count both."""
    size = 16
    uvs, faces = _unit_quad()
    texel, _triangle, _weights = texture._cover(uvs, faces, size)
    counts = torch.bincount(texel, minlength=size * size)
    assert int(counts.min()) >= 1, f"{int((counts == 0).sum())} texels uncovered"
    assert int(counts.max()) <= 2, f"a texel was claimed {int(counts.max())} times"
    # Only the shared diagonal may be claimed twice.
    assert int((counts > 1).sum()) <= size, int((counts > 1).sum())


def test_the_interpolated_position_is_the_texel_centre() -> None:
    """The exact reference: positions that are the UVs must reproduce the grid.

    This is the whole chain - bounding box, inside test, barycentric weights and
    the weighted sum - checked against arithmetic that cannot drift.
    """
    size = 32
    uvs, faces = _unit_quad()
    # z is left at zero; x and y are the UVs themselves.
    vertices = torch.cat([uvs, torch.zeros_like(uvs[:, :1])], dim=1)

    texel, triangle, weights = texture._cover(uvs, faces, size)
    corners = vertices[faces[triangle]]
    point = (corners * weights.unsqueeze(-1)).sum(dim=1)

    x = (texel % size).float()
    y = (texel // size).float()
    want = torch.stack([(x + 0.5) / size, (y + 0.5) / size], dim=1)
    error = (point[:, :2] - want).abs().max().item()
    assert error < 1e-5, f"max difference {error}"
    assert float(point[:, 2].abs().max()) < 1e-6, "z drifted away from the plane"


def test_the_weights_are_a_partition() -> None:
    """Barycentric weights sum to one and are never negative inside the triangle."""
    size = 24
    uvs, faces = _unit_quad()
    _texel, _triangle, weights = texture._cover(uvs, faces, size)
    assert float(weights.min()) >= -1e-6, float(weights.min())
    assert float((weights.sum(dim=1) - 1).abs().max()) < 1e-5


def test_a_degenerate_triangle_claims_nothing() -> None:
    """**A triangle with no area in the atlas must not divide by it.**"""
    uvs = torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=torch.float32, device=DEVICE)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=DEVICE)
    texel, _triangle, weights = texture._cover(uvs, faces, 16)
    assert texel.numel() == 0, f"a flat triangle claimed {texel.numel()} texels"
    assert not bool(torch.isnan(weights).any())


def test_a_chart_that_misses_the_atlas_is_dropped() -> None:
    """A triangle outside the square has an empty box and must not be indexed."""
    uvs = torch.tensor([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0]], dtype=torch.float32, device=DEVICE)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=DEVICE)
    texel, _triangle, _weights = texture._cover(uvs, faces, 8)
    assert texel.numel() == 0, texel.numel()


def test_dilation_spreads_outward_and_stops() -> None:
    """**Without it every seam draws a black line**; with too much it floods."""
    size = 9
    colour = torch.zeros(size, size, 3, device=DEVICE)
    filled = torch.zeros(size, size, dtype=torch.bool, device=DEVICE)
    colour[4, 4] = torch.tensor([1.0, 0.5, 0.25], device=DEVICE)
    filled[4, 4] = True

    grown, rounds = texture._dilate(colour, filled, 2)
    assert rounds == 2, rounds
    # One round reaches the eight neighbours, two reaches the 5x5 block.
    assert torch.allclose(grown[3, 3], torch.tensor([1.0, 0.5, 0.25], device=DEVICE), atol=1e-5)
    assert float(grown[2, 2].sum()) > 0, "the second round did not reach"
    assert float(grown[0, 0].sum()) == 0.0, "it spread further than it was asked to"

    # It stops early rather than looping once everything is filled.
    _full, done = texture._dilate(
        torch.ones(4, 4, 3, device=DEVICE), torch.ones(4, 4, dtype=torch.bool, device=DEVICE), 5
    )
    assert done == 0, done


def test_a_folded_face_reads_the_atlas_but_does_not_write_it() -> None:
    """A grid with one triangle turned over: the unwrap marks it, and the bake skips it."""
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    from runners.trellis2 import config

    n = 9
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [[a, b, c], [a, c, d]]
    vertices[4 * n + 4, 0] += 1.2
    vertices[4 * n + 4, 1] += 0.3
    grid = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    saved = (config.CHART_FOLD_AREA, config.CHART_MIN_FACES)
    try:
        # Out of reach, so the turned triangle stays in the chart and is folded.
        config.CHART_FOLD_AREA = 1e9
        config.CHART_MIN_FACES = 4
        size = 64
        _v, packed, uvs, folded, report = texture.unwrap(grid, size, in_process=True)
        assert report.folded_faces == 1, report.as_dict()
        assert int(folded.sum()) == 1 and len(folded) == len(packed)
        # The folded face's own texels are under other faces of its chart.
        uv_t = torch.as_tensor(uvs, device=DEVICE)
        face_t = torch.as_tensor(packed, device=DEVICE)
        theirs, _t, _w = texture._cover(uv_t, face_t[torch.as_tensor(folded, device=DEVICE)], size)
        others, _t, _w = texture._cover(uv_t, face_t[torch.as_tensor(~folded, device=DEVICE)], size)
        assert theirs.numel() > 0 and bool(torch.isin(theirs, others).all()), (theirs, others)
        # And the bake writes only the others: its coverage is theirs alone.
        _textured, baked = texture.bake(
            grid,
            lambda points: torch.ones(len(points), 3, device=points.device),
            size,
            dilate=0,
            in_process=True,
        )
        assert baked["folded_faces"] == 1, baked
        assert baked["texels_covered"] == int(others.unique().numel()), (
            baked["texels_covered"],
            int(others.unique().numel()),
        )
    finally:
        config.CHART_FOLD_AREA, config.CHART_MIN_FACES = saved


def test_the_unwrap_keeps_the_surface() -> None:
    """xatlas cuts and duplicates vertices; **it must not drop faces**."""
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices, faces, uvs, _folded, report = texture.unwrap(mesh, in_process=True)
    assert len(faces) == len(mesh.faces), (len(faces), len(mesh.faces))
    assert len(vertices) == len(uvs) >= len(mesh.vertices)
    assert report.vertices_before == len(mesh.vertices)
    assert float(np.asarray(uvs).min()) >= -1e-6 and float(np.asarray(uvs).max()) <= 1 + 1e-6


def test_the_packed_atlas_fits_and_fills() -> None:
    """Project a box into charts, pack them, and check the atlas is usable.

    **This is the path the runner takes**, end to end without a model: charts
    by direction, then xatlas placing them. Three things have to hold or the
    bake writes nonsense - the UVs are in 0..1 (what `_cover` multiplies by the
    texture size), the faces survive, and no texel is claimed by two charts.
    """
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    # Subdivided until each of the box's six sides is comfortably above
    # `CHART_MIN_FACES`, so the merging step is not what is being measured.
    mesh = trimesh.creation.box(extents=(1.0, 0.6, 0.3))
    for _ in range(3):
        mesh = mesh.subdivide()
    size = 128
    vertices, faces, uvs, _folded, report = texture.unwrap(mesh, size, in_process=True)

    assert len(faces) == len(mesh.faces), (len(faces), len(mesh.faces))
    assert len(vertices) == len(uvs), (len(vertices), len(uvs))
    # **Within a texel of the square, not exactly inside it.** The packer's
    # gutter can put a chart's outermost vertex a fraction of a texel past the
    # edge; `_cover` clamps the bounding box to the atlas, so the cost is a
    # sliver of one row. What would matter is a chart *outside* the square.
    slack = 1.5 / size
    assert float(uvs.min()) >= -slack and float(uvs.max()) <= 1 + slack, (
        float(uvs.min()),
        float(uvs.max()),
    )
    # A box has six directions, so six charts before any merging.
    assert report.n_charts >= 1, report.n_charts
    assert report.faces_in_large_charts > 0.9, report.faces_in_large_charts

    # **The charts have to fill the atlas.** Packing that leaves most of the
    # texture empty spends the resolution on nothing - the failure this whole
    # path exists to avoid, where an atlas of confetti covered 44%. A box
    # projects into six rectangles, so it should pack tightly.
    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _textured, report = texture.bake(
        surface,
        lambda points: torch.ones(len(points), 3, device=points.device),
        size,
        dilate=0,
        in_process=True,
    )
    assert report["coverage"] > 0.6, report["coverage"]
    assert report["texels_covered"] > 0, report


def test_six_channels_become_a_pbr_material() -> None:
    """Metallic, roughness and alpha come back out of the GLB material.

    **glTF decides the layout, not this code**: roughness is the green channel
    of one texture, metallic its blue, and the opacity is the base colour's
    alpha. A bake that writes them anywhere else looks right in a viewer that
    ignores them and wrong in one that does not.
    """
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    mesh = trimesh.creation.icosphere(subdivisions=2)

    def query(points: torch.Tensor) -> torch.Tensor:
        values = torch.tensor([1.0, 0.0, 0.0, 0.2, 0.6, 0.8], device=points.device)
        return values.expand(len(points), 6)

    textured, baked = texture.bake(mesh, query, 64, dilate=0, in_process=True)
    assert baked["channels"] == 6, baked
    material = textured.visual.material
    base = np.asarray(material.baseColorTexture)
    rough_metal = np.asarray(material.metallicRoughnessTexture)
    assert base.shape[2] == 4, base.shape
    assert material.alphaMode == "OPAQUE", material.alphaMode

    written = np.asarray(np.nonzero(base[:, :, 0])).T
    assert len(written), "nothing was baked"
    y, x = written[0]
    assert list(base[y, x]) == [255, 0, 0, 204], list(base[y, x])
    # Red is zero, green is roughness, blue is metallic.
    assert list(rough_metal[y, x]) == [0, 153, 51], list(rough_metal[y, x])


def test_a_texel_the_decoder_never_saw_is_a_hole_the_dilation_fills() -> None:
    """**Black from an empty query is not black paint**, and it is not left there.

    A texel the query answered with exactly zero was written, so the dilation
    used to step over it and the count came out the same on both sides. It is
    handed back as unwritten instead, and the neighbours fill it. Both numbers
    are still reported: how many there were says whether an atlas came out dark
    because the model is dark or because the colours never arrived, and how many
    are left says whether anything was near enough to fill them.
    """
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    mesh = trimesh.creation.icosphere(subdivisions=2)

    def query(points: torch.Tensor) -> torch.Tensor:
        lit = (points[:, 0] > 0).float().unsqueeze(1)
        return lit.expand(len(points), 3).contiguous()

    _textured, baked = texture.bake(mesh, query, 64, dilate=4, in_process=True)
    assert baked["texels_black"] > 0, baked
    assert baked["texels_black"] < baked["texels_covered"], baked
    # **Half this sphere has no colour at all**, so the dilation cannot reach
    # the middle of it in four rounds; what it must do is reduce the count.
    assert baked["texels_black_after_dilate"] < baked["texels_black"], baked
    # **The counting still adds up.** `texels_covered` is what the bake wrote
    # and `texels_unreached` is the rest of the atlas, whatever the dilation
    # was told afterwards.
    assert baked["texels_covered"] + baked["texels_unreached"] == baked["texels_total"], baked


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
