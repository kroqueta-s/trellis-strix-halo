# SPDX-License-Identifier: MIT
"""The shape of what this runner reports, checked without a graphics card.

**A result with the wrong key is not an error anywhere.** hearth passes it on,
the add-on reads the key it expects, finds nothing, and shows a generation with
no settings recorded - so the one thing needed to repeat it is quietly gone.
That is why this is pinned here rather than left to a real run: a real run needs
the card, takes minutes, and nobody does it after an unrelated edit.

What is checked:

- `capabilities` answers with the contract version it was written against, and
  every setting it declares says at least its type and its default.
- **A method that is not `image_to_mesh` declares its own settings.** Without
  that a caller sends it the shape stage's, which used to be accepted and then
  dropped without a word.
- The generating result names its settings `params_used` (contract §5).
- **The mesh is renamed into place** (contract §9), so a run killed halfway
  leaves nothing that looks finished.
- The axes are reported as measured, or as `null` where they were not. **Never
  guessed**: a mesh imported on the wrong axis renders perfectly correctly, and
  the first sign of the mistake is a mirrored joint on a printed part.

The model is replaced by a stand-in, so this needs neither torch nor the weights.

Run it with this repository's virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_result_shape.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WRITES: list[Path] = []


class _Anything:
    """Stands in for a generation result: every timing it is asked for is zero."""

    def __getattr__(self, name: str) -> Any:
        return 0.0


class _Image(_Anything):
    """A picture that can only be saved, which is all the runner does with one."""

    def save(self, path: Any, *args: Any, **kwargs: Any) -> None:
        Path(str(path)).write_bytes(b"")


class _Mesh(_Anything):
    """A mesh that records where it was written."""

    def __init__(self) -> None:
        self.vertices = [0, 0, 0]
        self.faces = [0]

    def export(self, path: str, *args: Any, **kwargs: Any) -> None:
        WRITES.append(Path(path))
        Path(path).write_bytes(b"ply\n")


class _Result(_Anything):
    """What the pipeline hands back."""

    def __init__(self) -> None:
        self.mesh = _Mesh()
        self.foreground = _Image()
        self.normal = _Image()
        self.n_voxels = 0
        self.clean = {}
        self.fast_attention = True


def _progress(stage: str, message: str = "", **extra: Any) -> None:
    """Swallow progress; what it says is `test_steps.py`'s business."""


def _stub_windows_only(package: str) -> None:
    """Stand in for the modules that only exist on Windows, and for dotenv.

    **So that this can run in CI**, which is Linux. `gfxlight` imports
    `ctypes.wintypes`, which does not exist there, and neither module has
    anything to do with the shape of a result. `dotenv` is stubbed for the same
    reason: reading a `.env` cannot change what keys a result has, and it is one
    fewer thing to install.
    """
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules.setdefault("dotenv", dotenv)

    for name in ("displaykeep", "gfxlight"):
        module = types.ModuleType(f"{package}.{name}")
        module.__getattr__ = lambda attr: (lambda *a, **k: None)  # type: ignore[attr-defined]
        sys.modules[f"{package}.{name}"] = module


def _install_stubs() -> Any:
    """Replace the torch-backed modules, then import the runner.

    **The stubs go in before the import**, because the generating method imports
    its pipeline when it is called and would otherwise pull in torch.
    """
    _stub_windows_only("runners.trellis")

    pillow = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.open = lambda *a, **k: _Image()  # type: ignore[attr-defined]
    pillow.Image = image  # type: ignore[attr-defined]
    sys.modules.setdefault("PIL", pillow)
    sys.modules.setdefault("PIL.Image", image)

    pipeline = types.ModuleType("runners.trellis.pipeline")
    pipeline.generate_mesh = lambda *a, **k: _Result()  # type: ignore[attr-defined]
    pipeline.blas_backend = lambda: "stub"  # type: ignore[attr-defined]
    sys.modules["runners.trellis.pipeline"] = pipeline

    from runners.trellis import __main__ as runner

    return runner


RUNNER = _install_stubs()


def _generate() -> dict[str, Any]:
    """Run the generating method against the stand-in and return its result."""
    WRITES.clear()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        image = Path(tmp) / "in.png"
        image.write_bytes(b"")
        params = {"image_path": str(image), "out_dir": str(out_dir)}
        return RUNNER.m_image_to_mesh(params, _progress)


def test_capabilities_report_a_contract_version() -> None:
    """**A caller uses it to explain an absence**, so it has to be there."""
    caps = RUNNER.m_capabilities({}, _progress)
    assert isinstance(caps.get("contract"), int), caps
    assert caps["contract"] >= 3, caps


def test_every_declared_setting_says_its_type_and_default() -> None:
    """A setting with no type cannot be drawn, and one with no default is a guess."""
    caps = RUNNER.m_capabilities({}, _progress)
    for name, spec in caps["params"].items():
        assert isinstance(spec, dict), (name, spec)
        assert "type" in spec and "default" in spec, (name, spec)


def test_a_method_beside_image_to_mesh_declares_its_own_settings() -> None:
    """**Otherwise the caller sends the shape stage's**, and they mean nothing here."""
    caps = RUNNER.m_capabilities({}, _progress)
    able = caps["capabilities"]
    per_method = caps.get("method_params", {})
    offered = {"texture_mesh": "texture", "multi_image_to_mesh": None, "text_to_mesh": None}
    for method, flag in offered.items():
        if not able.get(flag or method, False):
            continue
        assert method in per_method, f"{method} is offered but declares no settings: {per_method}"
        for key, spec in per_method[method].items():
            assert "type" in spec and "default" in spec, (method, key, spec)


def test_the_result_names_its_settings_params_used() -> None:
    """Contract §5. Under any other key the caller finds nothing and says nothing."""
    out = _generate()
    assert "params_used" in out, sorted(out)
    assert "params" not in out, "the old key is still there, so a caller reads both"


def test_the_mesh_is_renamed_into_place_rather_than_written_to_its_name() -> None:
    """Contract §9. A run killed while writing must not leave a finished-looking file."""
    out = _generate()
    written = [p for p in WRITES if p.name.startswith("raw.ply")]
    assert written, WRITES
    assert all(p.name != "raw.ply" for p in written), f"written straight to its name: {written}"
    assert Path(out["mesh_path"]).name == "raw.ply", out["mesh_path"]


def test_the_axes_are_reported_and_never_guessed() -> None:
    """`null` where they were not measured. **A wrong axis looks perfectly correct.**"""
    out = _generate()
    assert "up_axis" in out, sorted(out)
    assert out["up_axis"] in (None, "x", "y", "z"), out["up_axis"]
    assert out.get("forward_axis") in (None, "x", "y", "z"), out.get("forward_axis")
    if out["up_axis"] is None:
        notes = RUNNER.m_capabilities({}, _progress)["notes"]
        assert "unmeasured" in notes, "unknown axes have to be said out loud somewhere"


def test_every_setting_it_accepts_is_one_it_declares() -> None:
    """**Accepted and undeclared is the same fault as declared and ignored.**

    A setting the runner reads but leaves out of `capabilities` cannot be found
    by a caller: no form draws it, no flow validates it, and the only way to
    learn it exists is to read the source. `rembg_model` was in that state -
    read on every generation, mentioned nowhere - which is how a table and the
    code behind it drift apart without either looking wrong.
    """
    caps = RUNNER.m_capabilities({}, _progress)
    accepted = set(getattr(RUNNER, "_ALLOWED", frozenset()))
    if not accepted:
        return
    undeclared = sorted(accepted - set(caps["params"]))
    assert not undeclared, f"accepted but not declared: {undeclared}"

    per_method = caps.get("method_params", {})
    for method, declared in per_method.items():
        allowed = getattr(RUNNER, f"_ALLOWED_{method.upper()}", None)
        if allowed is None:
            continue
        missing = sorted(set(allowed) - set(declared))
        assert not missing, f"{method} accepts but does not declare: {missing}"


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
