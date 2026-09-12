# SPDX-License-Identifier: MIT
"""The debris pass runs a second time after the carve, and says what it dropped.

The carve and the decimation after it make parts of their own - strays the rays
could not reach, slivers the decimation pinches off - and the first debris pass
ran before either. This pins the second pass to the report, on a shape small
enough to carve on the CPU in a few seconds.

Run it with this repository's virtual environment (torch comes in through the
pipeline import)::

    .venv\\Scripts\\python.exe .\\tests\\test_postprocess_debris.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2 import config, pipeline  # noqa: E402


def test_the_second_debris_pass_is_reported() -> None:
    """A sphere with a crumb beside it comes out as one part, and the report says so."""
    sphere = trimesh.creation.icosphere(subdivisions=4)
    crumb = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    crumb.apply_translation((1.5, 0.0, 0.0))
    soup = trimesh.util.concatenate([sphere, crumb])
    saved = {k: getattr(config, k) for k in ("SHELL", "SHELL_MODE", "SHELL_GRID", "MAKE_MANIFOLD")}
    try:
        config.SHELL = True
        config.SHELL_MODE = "carve"
        config.SHELL_GRID = 48
        config.MAKE_MANIFOLD = False
        mesh, report, _textured, _bake = pipeline._postprocess(soup, None, target_faces=3000)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
    for key in (
        "shell_parts_before",
        "shell_parts_after",
        "shell_dropped_parts",
        "shell_dropped_faces",
    ):
        assert key in report, (key, sorted(report))
    assert report["shell_parts_after"] == 1, report
    assert report["shell_parts_before"] >= 1, report
    assert len(trimesh.graph.connected_components(mesh.face_adjacency)) == 1
    assert report["dropped_parts"] == 1, "the first pass drops the crumb itself"


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
