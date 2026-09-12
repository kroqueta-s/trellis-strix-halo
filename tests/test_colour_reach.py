# SPDX-License-Identifier: MIT
"""A colour query reaches past the eight corners, **without weights or a model**.

The carved solid's surface is a lattice's own, up to a cell from the one the
decoder drew, and a trilinear sample there finds no active voxel and returns
black - measured at 52 % of the print mesh's vertices at 1024 (2026-09-12).
The voxels are a cell away, so the query looks for them. This pins what that
search does on a handful of voxels whose answer is known: inside the cluster
the trilinear sample is untouched, a cell or two outside it the nearest voxels
answer, and far away nothing does.

Run it with either virtual environment; it falls back to CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2.pipeline import _colour_query  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 64


def _voxels() -> tuple[SimpleNamespace, SimpleNamespace]:
    """A 2x2x2 block of red voxels at (10..11)^3 and one blue voxel at (20, 20, 20)."""
    coords = [[0, x, y, z] for x in (10, 11) for y in (10, 11) for z in (10, 11)] + [
        [0, 20, 20, 20]
    ]
    feats = [[1.0, 0.0, 0.0, 0.5, 0.5, 1.0]] * 8 + [[0.0, 0.0, 1.0, 0.5, 0.5, 1.0]]
    voxels = SimpleNamespace(
        feats=torch.tensor(feats, device=DEVICE),
        coords=torch.tensor(coords, dtype=torch.int32, device=DEVICE),
        shape=torch.Size([1, 6]),
        spatial_shape=torch.Size([RESOLUTION, RESOLUTION, RESOLUTION]),
    )
    pipeline = SimpleNamespace(
        pbr_attr_layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
    )
    return voxels, pipeline


def _point(x: float, y: float, z: float) -> torch.Tensor:
    """A position in the unit cube whose voxel coordinates are (x, y, z)."""
    return (torch.tensor([[x, y, z]], device=DEVICE) / RESOLUTION) - 0.5


def test_inside_the_cluster_nothing_changes() -> None:
    voxels, pipeline = _voxels()
    plain = _colour_query(voxels, pipeline, RESOLUTION, reach=0)
    reaching = _colour_query(voxels, pipeline, RESOLUTION, reach=3)
    centre = _point(11.0, 11.0, 11.0)
    assert torch.allclose(plain(centre), reaching(centre))
    assert torch.allclose(plain(centre), torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE))
    assert (reaching.stats["searched"], reaching.stats["unreached"]) == (0, 0), reaching.stats


def test_a_point_a_cell_outside_is_coloured_by_its_neighbours() -> None:
    """Two voxels past the block the trilinear sample sees nothing; the nearest voxels answer."""
    voxels, pipeline = _voxels()
    outside = _point(13.6, 10.5, 10.5)
    plain = _colour_query(voxels, pipeline, RESOLUTION, reach=0)
    assert torch.equal(plain(outside), torch.zeros(1, 3, device=DEVICE)), plain(outside)
    assert plain.stats["unreached"] == 1
    reaching = _colour_query(voxels, pipeline, RESOLUTION, reach=3)
    assert torch.allclose(reaching(outside), torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE))
    assert (reaching.stats["searched"], reaching.stats["unreached"]) == (1, 0), reaching.stats
    # The nearest voxel centre is (11.5, 10.5, 10.5): 2.1 voxels away.
    assert abs(float(reaching.stats["distances"][0][0]) - 2.1) < 1e-6, reaching.stats
    # The other channels ride along with it.
    pbr = _colour_query(voxels, pipeline, RESOLUTION, ("base_color", "metallic", "alpha"), reach=3)
    assert torch.allclose(pbr(outside), torch.tensor([[1.0, 0.0, 0.0, 0.5, 1.0]], device=DEVICE))


def test_the_nearest_voxel_weighs_most() -> None:
    """Between red and blue, closer to blue: the answer leans blue, not an even mix."""
    voxels, pipeline = _voxels()
    reaching = _colour_query(voxels, pipeline, RESOLUTION, reach=20)
    colour = reaching(_point(18.5, 18.5, 18.5))
    assert colour[0, 2] > colour[0, 0] > 0, colour


def test_far_away_stays_black_and_is_counted() -> None:
    voxels, pipeline = _voxels()
    reaching = _colour_query(voxels, pipeline, RESOLUTION, reach=3)
    far = _point(40.0, 40.0, 40.0)
    assert torch.equal(reaching(far), torch.zeros(1, 3, device=DEVICE))
    assert (reaching.stats["searched"], reaching.stats["unreached"]) == (1, 1), reaching.stats


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
