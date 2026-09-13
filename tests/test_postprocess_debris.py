# SPDX-License-Identifier: MIT
"""The debris pass runs after the carve, and only there when there is a carve.

Judging a part before the carve judges it in the wrong state: the decimation
pinches thin joints apart, and the carve that comes next would have put the
freed piece back into the solid. So when the shell is on there is one pass and
it is the one after it; when the shell is off that pass is the only one there
could be, and it runs where it always did. Both are pinned here, on a shape
small enough to carve on the CPU in a few seconds.

Run it with this repository's virtual environment (torch comes in through the
pipeline import)::

    .venv\\Scripts\\python.exe .\\tests\\test_postprocess_debris.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2 import config, pipeline  # noqa: E402


def _post_a_sphere_with_a_crumb(
    carve: bool,
) -> tuple[trimesh.Trimesh, dict[str, Any], trimesh.Trimesh | None, dict[str, Any]]:
    """Post-process a sphere with a crumb beside it, with the shell on or off."""
    sphere = trimesh.creation.icosphere(subdivisions=4)
    crumb = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    crumb.apply_translation((1.5, 0.0, 0.0))
    soup = trimesh.util.concatenate([sphere, crumb])
    names = ("SHELL", "SHELL_MODE", "SHELL_GRID", "MAKE_MANIFOLD")
    saved = {k: getattr(config, k) for k in names}
    try:
        config.SHELL = carve
        config.SHELL_MODE = "carve"
        config.SHELL_GRID = 48
        config.MAKE_MANIFOLD = False
        return pipeline._postprocess(soup, None, target_faces=3000)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def _assert_the_crumb_is_gone(mesh: trimesh.Trimesh, report: dict[str, Any]) -> None:
    """Whatever the order, one part comes out and the pass after the carve reports."""
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


def test_the_carve_leaves_one_debris_pass_and_it_is_the_one_after_it() -> None:
    """With the shell on, the crumb has only the pass after the carve to take it."""
    mesh, report, _textured, _bake = _post_a_sphere_with_a_crumb(True)
    _assert_the_crumb_is_gone(mesh, report)
    # **A report carrying both passes would mean a part had been judged before
    # the carve**, which is where the thin joints have just been pinched apart
    # and a real foot looks like debris.
    assert "dropped_parts" not in report, sorted(report)
    assert "drop_parts_sec" not in report, sorted(report)


def test_without_a_carve_the_pass_runs_where_it_always_did() -> None:
    """With the shell off nothing would take the crumb later, so the early pass stays."""
    mesh, report, _textured, _bake = _post_a_sphere_with_a_crumb(False)
    assert report["dropped_parts"] == 1, "the only pass there is drops the crumb"
    assert "shell_parts_after" not in report, sorted(report)
    assert len(trimesh.graph.connected_components(mesh.face_adjacency)) == 1


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
