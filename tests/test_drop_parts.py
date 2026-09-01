# SPDX-License-Identifier: MIT
"""Verify `drop_small_parts` (size and thinness).

A synthetic mesh is built from a body, a genuine detached part, a small crumb
and a surface flake, to confirm that **what should survive survives and what
should go is gone**.

Run it with this repository's virtual environment (trimesh is required; torch
comes in through the postprocess import)::

    .venv\\Scripts\\python.exe .\\tests\\test_drop_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = " OK " if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def build_specimen() -> tuple[trimesh.Trimesh, int]:
    """Body, part, crumb and flake. Returns (mesh, face count that should survive)."""
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # A genuine part: 20 % long, 20 % thick (an arm, say).
    limb = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    limb.apply_translation((0.8, 0.0, 0.0))
    # A small crumb: 5 % long (the size test alone already removes it).
    crumb = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    crumb.apply_translation((0.0, 0.8, 0.0))
    # A surface flake: 30 % long, 0.5 % thick (this is what used to slip through).
    flake = trimesh.creation.box(extents=(0.3, 0.3, 0.005))
    flake.apply_translation((0.0, 0.0, 0.51))
    mesh = trimesh.util.concatenate([body, limb, crumb, flake])
    return mesh, len(body.faces) + len(limb.faces)


def run(label: str, drop_small_parts) -> None:  # noqa: ANN001
    mesh, want_faces = build_specimen()
    out = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.02)
    parts = out.split(only_watertight=False)
    check(f"{label}: body and part survive as two components", len(parts) == 2, f"got {len(parts)}")
    check(f"{label}: face count matches body plus part", len(out.faces) == want_faces)

    # With the thinness test off, the flake stays (0 must mean disabled).
    out2 = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.0)
    check(
        f"{label}: min_thick_ratio=0 keeps the flake",
        len(out2.split(only_watertight=False)) == 3,
    )

    # Both at 0 does nothing at all.
    out3 = drop_small_parts(mesh, min_ratio=0.0, min_thick_ratio=0.0)
    check(f"{label}: both at 0 changes nothing", len(out3.faces) == len(mesh.faces))

    # The largest component survives even when it is itself thin.
    thin_body = trimesh.creation.box(extents=(1.0, 1.0, 0.01))
    out4 = drop_small_parts(thin_body.copy(), min_ratio=0.10, min_thick_ratio=0.02)
    check(
        f"{label}: the largest component always survives",
        len(out4.faces) == len(thin_body.faces),
    )


def main() -> int:
    from runners.trellis import postprocess as tre

    run("trellis", tre.drop_small_parts)

    total = 5
    print(f"\n{total - len(FAILURES)}/{total} passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
