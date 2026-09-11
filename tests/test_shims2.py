# SPDX-License-Identifier: MIT
"""Verify the TRELLIS.2-only shims (**before any of TRELLIS.2 is installed**).

Run it with this repository's virtual environment. It falls back to CPU when no
GPU is present, which is what CI does.

The `o_voxel` hashmap has an exact reference: a python dictionary. Its contract
comes from `o-voxel/src/hash/hash.cu` - key `b*W*H*D + x*H*D + y*D + z`, value
the row the coordinate was inserted from, and the largest value of the value
type for a key that is absent or a coordinate outside the grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import shims  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MISS = 0xFFFFFFFF


def _hashmap(capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The two arrays exactly as `_init_hashmap` builds them upstream."""
    keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=DEVICE)
    values = torch.empty((capacity,), dtype=torch.uint32, device=DEVICE)
    return keys, values


def _unique_coords(n: int, w: int, h: int, d: int) -> torch.Tensor:
    """`[N, 4]` int32 coordinates with a zero batch column and no duplicates."""
    g = torch.Generator().manual_seed(0)
    seen: set[tuple[int, int, int]] = set()
    while len(seen) < n:
        raw = torch.stack(
            [
                torch.randint(0, w, ((n - len(seen)) * 2,), generator=g),
                torch.randint(0, h, ((n - len(seen)) * 2,), generator=g),
                torch.randint(0, d, ((n - len(seen)) * 2,), generator=g),
            ],
            dim=1,
        )
        for row in raw.tolist():
            if len(seen) < n:
                seen.add(tuple(row))
    rows = [(0, *c) for c in sorted(seen)]
    return torch.tensor(rows, dtype=torch.int32, device=DEVICE)


def _reference(coords: torch.Tensor, queries: torch.Tensor) -> list[int]:
    """What a dictionary says the answers are."""
    table: dict[tuple[int, ...], int] = {}
    for row, coord in enumerate(coords.tolist()):
        table.setdefault(tuple(coord), row)
    return [table.get(tuple(q), MISS) for q in queries.tolist()]


def _lookup(coords: torch.Tensor, queries: torch.Tensor, w: int, h: int, d: int) -> list[int]:
    keys, values = _hashmap(2 * coords.shape[0])
    shims._hashmap_insert_3d_idx_as_val(keys, values, coords, w, h, d)
    out = shims._hashmap_lookup_3d(keys, values, queries, w, h, d)
    assert out.dtype == values.dtype, f"the value dtype changed: {out.dtype}"
    return [int(v) for v in out.to(torch.int64).tolist()]


def test_hashmap_matches_a_dictionary() -> None:
    """Present, absent and out-of-grid coordinates, against the dictionary."""
    w, h, d = 37, 41, 43
    coords = _unique_coords(512, w, h, d)
    outside = torch.tensor(
        [[0, -1, 0, 0], [0, w, 0, 0], [0, 0, h, 0], [0, 0, 0, d], [0, 0, -5, 0]],
        dtype=torch.int32,
        device=DEVICE,
    )
    # Half the stored coordinates, some coordinates that were never stored, and
    # every out-of-grid case.
    absent = torch.tensor(
        [[0, x % w, (x * 7) % h, (x * 13) % d] for x in range(4096)],
        dtype=torch.int32,
        device=DEVICE,
    )
    queries = torch.cat([coords[::2], absent, outside], dim=0)

    got = _lookup(coords, queries, w, h, d)
    want = _reference(coords, queries)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert a == b, f"row {i}: got {a}, want {b} (query {queries[i].tolist()})"


def test_duplicates_keep_the_first_row() -> None:
    """A repeated coordinate resolves to the row it first appeared in."""
    w = h = d = 8
    coords = torch.tensor(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 1, 2, 3]], dtype=torch.int32, device=DEVICE
    )
    got = _lookup(coords, coords, w, h, d)
    assert got == [0, 1, 0], got


def test_a_full_grid_round_trips() -> None:
    """Every voxel of a small grid, so the linearization has no gaps to hide in."""
    w, h, d = 5, 6, 7
    rows = [[0, x, y, z] for x in range(w) for y in range(h) for z in range(d)]
    coords = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
    got = _lookup(coords, coords, w, h, d)
    assert got == list(range(len(rows))), got[:16]


def test_a_second_batch_is_a_different_key() -> None:
    """The batch column takes part in the key, as it does in the kernel."""
    w = h = d = 4
    coords = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.int32, device=DEVICE)
    got = _lookup(coords, coords, w, h, d)
    assert got == [0, 1], got


def test_too_many_keys_is_refused() -> None:
    """A hashmap that cannot hold the keys fails loudly rather than silently."""
    w = h = d = 16
    coords = _unique_coords(64, w, h, d)
    keys, values = _hashmap(16)
    try:
        shims._hashmap_insert_3d_idx_as_val(keys, values, coords, w, h, d)
    except ValueError:
        return
    raise AssertionError("a hashmap that is too small was accepted")


def test_a_miss_survives_the_callers_int_cast() -> None:
    """The caller reads the result as `.int()` and compares against `0xffffffff`.

    That is `flexible_dual_grid_to_mesh` verbatim, and it is the one place where
    the miss value has to keep its meaning across a dtype change.
    """
    w = h = d = 8
    coords = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32, device=DEVICE)
    queries = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32, device=DEVICE)
    keys, values = _hashmap(4)
    shims._hashmap_insert_3d_idx_as_val(keys, values, coords, w, h, d)
    out = shims._hashmap_lookup_3d(keys, values, queries, w, h, d).int()
    valid = out != MISS
    assert valid.tolist() == [True, False], (out.tolist(), valid.tolist())


# --------------------------------------------------------------------------------------
# grid_sample_3d - trilinear sampling of a sparse volume (the texture path)
# --------------------------------------------------------------------------------------
#
# **`F.grid_sample` is the reference for a full grid.** When every voxel is
# active, sampling a sparse volume is ordinary trilinear interpolation, and
# torch's own implementation is an independent one - not a second copy of the
# arithmetic under test. Voxel centre `i` sits at `i + 0.5` in the query units
# used here, which is `align_corners=False`.
#
# Where the grid is sparse there is nothing to compare against but the kernel's
# own rule (`FlexGEMM`'s
# `hashmap_lookup_grid_sample_3d_trilinear_neighbor_map_weight_kernel`): a
# corner that is missing or outside contributes nothing, and the weights that
# remain are renormalized. A dictionary states it directly.


def _dense_reference(feats: torch.Tensor, size: int, queries: torch.Tensor) -> torch.Tensor:
    """`F.grid_sample` over the same volume, as `[L, C]`."""
    channels = feats.shape[1]
    volume = feats.t().reshape(1, channels, size, size, size)
    # torch's last grid axis indexes the last spatial dimension, so the query's
    # (x, y, z) is handed over reversed.
    normalized = (2 * queries / size - 1).flip(-1).reshape(1, 1, 1, -1, 3)
    sampled = torch.nn.functional.grid_sample(
        volume, normalized, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled.reshape(channels, -1).t()


def _sparse_reference(
    feats: torch.Tensor, coords: torch.Tensor, size: int, queries: torch.Tensor
) -> torch.Tensor:
    """The kernel's rule, spelled out over a dictionary."""
    table = {tuple(c): row for row, c in enumerate(coords.tolist())}
    out = torch.zeros(queries.shape[0], feats.shape[1], dtype=torch.float32)
    rows = feats.float().cpu()
    for i, query in enumerate(queries.tolist()):
        base = [int(torch.floor(torch.tensor(v - 0.5))) for v in query]
        total = 0.0
        for corner in range(8):
            here = [base[axis] + ((corner >> axis) & 1) for axis in range(3)]
            if any(c < 0 or c >= size for c in here):
                continue
            row = table.get((0, *here))
            if row is None:
                continue
            weight = 1.0
            for axis in range(3):
                weight *= 1 - abs(query[axis] - here[axis] - 0.5)
            total += weight
            out[i] += weight * rows[row]
        out[i] /= max(total, 1e-12)
    return out


def _queries(n: int, low: float, high: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return (torch.rand(n, 3, generator=g) * (high - low) + low).to(DEVICE)


def test_grid_sample_matches_torch_on_a_full_grid() -> None:
    """With every voxel active this is plain trilinear interpolation."""
    size, channels = 6, 3
    coords = torch.stack(
        torch.meshgrid(*[torch.arange(size) for _ in range(3)], indexing="ij"), dim=-1
    ).reshape(-1, 3)
    coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).int().to(DEVICE)
    feats = torch.randn(coords.shape[0], channels, generator=torch.Generator().manual_seed(3)).to(
        DEVICE
    )
    # Kept clear of the border, where the reference pads and this one renormalizes.
    queries = _queries(512, 0.6, size - 0.6)

    got = shims._grid_sample_3d(
        feats, coords, torch.Size([1, channels, size, size, size]), queries.unsqueeze(0)
    )[0]
    want = _dense_reference(feats, size, queries)
    error = (got - want).abs().max().item()
    assert error < 1e-4, f"max difference {error}"


def test_grid_sample_renormalizes_around_the_holes() -> None:
    """Half the voxels removed: the surviving corners carry the whole weight."""
    size, channels = 8, 4
    coords = torch.stack(
        torch.meshgrid(*[torch.arange(size) for _ in range(3)], indexing="ij"), dim=-1
    ).reshape(-1, 3)
    keep = torch.rand(coords.shape[0], generator=torch.Generator().manual_seed(11)) > 0.5
    coords = coords[keep]
    coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).int().to(DEVICE)
    feats = torch.randn(coords.shape[0], channels, generator=torch.Generator().manual_seed(5)).to(
        DEVICE
    )
    queries = _queries(256, 0.0, float(size))

    got = shims._grid_sample_3d(
        feats, coords, torch.Size([1, channels, size, size, size]), queries.unsqueeze(0)
    )[0]
    want = _sparse_reference(feats, coords.cpu(), size, queries.cpu())
    error = (got.cpu() - want).abs().max().item()
    assert error < 1e-4, f"max difference {error}"


def test_grid_sample_returns_zero_where_nothing_is_near() -> None:
    """**A point with no active voxel around it gets zero, not a neighbour's colour.**"""
    size, channels = 8, 2
    coords = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32, device=DEVICE)
    feats = torch.ones(1, channels, device=DEVICE)
    queries = torch.tensor([[1.5, 1.5, 1.5], [6.5, 6.5, 6.5]], device=DEVICE)

    got = shims._grid_sample_3d(
        feats, coords, torch.Size([1, channels, size, size, size]), queries.unsqueeze(0)
    )[0]
    # The voxel's own centre returns the voxel exactly; far away returns nothing.
    assert torch.allclose(got[0], torch.ones(channels, device=DEVICE)), got[0].tolist()
    assert torch.all(got[1] == 0), got[1].tolist()


def test_grid_sample_refuses_the_mode_it_does_not_implement() -> None:
    """`nearest` is not on this path, and guessing at it would be worse than stopping."""
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device=DEVICE)
    feats = torch.ones(1, 1, device=DEVICE)
    try:
        shims._grid_sample_3d(
            feats,
            coords,
            torch.Size([1, 1, 2, 2, 2]),
            torch.zeros(1, 1, 3, device=DEVICE),
            "nearest",
        )
    except NotImplementedError:
        return
    raise AssertionError("nearest should not be silently answered")


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
