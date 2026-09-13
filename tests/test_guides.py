# SPDX-License-Identifier: MIT
"""Verify the subdivision guides **without a GPU, weights or the vendor code**.

The guides decide where the texture decoder puts its voxels. Getting them wrong
has no error to show: the decoder still produces colours, on a grid that is not
the mesh's, and the mesh comes out black in the places the mask missed. So the
two things that must hold are checked against arithmetic rather than against a
run - the child order the upsampling uses (`x + 2y + 4z`, from
`SparseChannel2Spatial.forward`), and that growing a latent's coordinates
through the masks lands exactly on the wanted grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2.guides import SubdivisionGuides  # noqa: E402


def _grow(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The coordinates a subdivision produces, the way upstream's upsample does.

    `subidx // 2**i % 2` is the offset along axis i, taken over the children a
    parent keeps, in index order.
    """
    kept = (mask > 0).nonzero()
    parents = coords[kept[:, 0]] * 2
    child = kept[:, 1]
    offsets = torch.stack([child % 2, (child // 2) % 2, (child // 4) % 2], dim=1)
    return parents + offsets


def _as_set(coords: torch.Tensor) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in row) for row in coords}


def test_growing_a_latent_through_the_masks_lands_on_the_target() -> None:
    """**The whole point**: what comes out is the grid that was asked for."""
    torch.manual_seed(0)
    levels = 3
    target = torch.randint(0, 8, (200, 3), dtype=torch.int64).unique(dim=0)
    guides = SubdivisionGuides(target, levels)

    coords = torch.unique(target >> levels, dim=0)
    for level in range(levels):
        coords = _grow(coords, guides.mask(coords, level))
    assert _as_set(coords) == _as_set(target), (len(coords), len(target))


def test_a_target_the_latent_cannot_reach_is_counted_not_invented() -> None:
    """A voxel whose ancestor is not in the latent has nothing to subdivide."""
    levels = 2
    target = torch.tensor([[0, 0, 0], [1, 1, 1], [7, 7, 7]], dtype=torch.int64)
    guides = SubdivisionGuides(target, levels)
    # The latent holds the first ancestor only, so the far corner is out of reach.
    counted = guides.reachable(torch.tensor([[0, 0, 0]], dtype=torch.int64))
    assert counted["target_voxels"] == 3, counted
    assert counted["reachable_voxels"] == 2, counted

    coords = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    for level in range(levels):
        coords = _grow(coords, guides.mask(coords, level))
    assert _as_set(coords) == {(0, 0, 0), (1, 1, 1)}, coords


def test_the_mask_keeps_only_the_children_that_are_wanted() -> None:
    """One parent, two children: the other six must be off."""
    target = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.int64)
    guides = SubdivisionGuides(target, 1)
    mask = guides.mask(torch.tensor([[0, 0, 0]], dtype=torch.int64), 0)
    wanted = [True, False, False, False, False, True, False, False]  # 0 and 1+4
    assert [bool(v > 0) for v in mask[0]] == wanted, mask[0]


def test_a_level_count_that_does_not_match_the_decoder_is_refused() -> None:
    """The guides and the model have to agree, and silence would misplace them."""

    class Decoder:
        blocks = [0, 1, 2]  # two subdivisions

    guides = SubdivisionGuides(torch.tensor([[0, 0, 0]], dtype=torch.int64), 3)
    from runners.trellis2.guides import decode_on_grid

    try:
        decode_on_grid(Decoder(), None, guides)
    except ValueError:
        return
    raise AssertionError("a mismatched level count was accepted")


def main() -> int:
    """Run every test."""
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
