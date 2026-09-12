# SPDX-License-Identifier: MIT
"""Verify the frame an incoming mesh is put into, **without a GPU or weights**.

`texture_mesh` takes a mesh from anywhere and asks which way is up. Getting
that wrong has no error to report: exchanging two coordinates puts the named
axis on Z and looks right in a viewer, while quietly mirroring the model and
turning every face inside out. What it costs is measured rather than argued -
the dual grid conversion went from 5.5 s to more than five minutes on a mesh
handed in that way (2026-09-12) - but the check here is exact and needs
nothing: a tetrahedron's signed volume keeps its sign under a rotation and
changes it under a reflection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2.pipeline import _to_model_frame  # noqa: E402

# The corner of a unit cube: right-handed, and every axis distinguishable.
TETRAHEDRON = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
UP = {
    "x": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    "y": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "z": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def _signed_volume(points: np.ndarray) -> float:
    a, b, c, d = points
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def test_every_axis_ends_up_pointing_at_z() -> None:
    """Whichever way the file says is up, it is +Z afterwards."""
    for axis, pair in UP.items():
        moved = _to_model_frame(pair.copy(), axis)
        direction = moved[1] - moved[0]
        direction = direction / np.linalg.norm(direction)
        assert np.allclose(direction, [0.0, 0.0, 1.0], atol=1e-9), (axis, direction)


def test_no_axis_mirrors_the_model() -> None:
    """**A reflection is the failure with no error message.**

    It puts the axis where it belongs, so the model stands up correctly, and
    turns every face inside out on the way.
    """
    for axis in ("x", "y", "z"):
        volume = _signed_volume(_to_model_frame(TETRAHEDRON.copy(), axis))
        assert volume > 0, f"{axis} mirrored the mesh (signed volume {volume})"


def test_the_mesh_lands_in_the_unit_cube() -> None:
    """Centred on the origin and just inside the cube the encoder's grid is."""
    far = TETRAHEDRON * 80.0 + np.array([500.0, -20.0, 3.0])
    moved = _to_model_frame(far, "z")
    low, high = moved.min(axis=0), moved.max(axis=0)
    assert np.allclose((low + high) / 2.0, 0.0, atol=1e-9), (low, high)
    assert 0.9 < float((high - low).max()) <= 1.0, (high - low).max()


def test_an_axis_it_does_not_know_is_refused() -> None:
    """**Nothing may assume an axis**, so a wrong one is an error, not a guess."""
    try:
        _to_model_frame(TETRAHEDRON.copy(), "w")
    except ValueError:
        return
    raise AssertionError("an unknown up axis was accepted")


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
