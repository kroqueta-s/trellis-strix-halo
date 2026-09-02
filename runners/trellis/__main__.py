# SPDX-License-Identifier: MIT
"""The TRELLIS runner (an implementation of the runner contract).

**This process is the only one that holds torch.** Neither hearth itself nor the
Blender add-on imports it.

The runner imports nothing from hearth, so this repository is self-contained.

Start it (hearth normally spawns it as a child process)::

    .venv\\Scripts\\python.exe -m runners.trellis
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import config, gfxlight

# **Has no effect unless it precedes torch** (setting os.environ later is
# ignored). It makes the flash and memory-efficient kernels available on
# gfx1151, measured 10-20x faster. Importing config here does not pull in torch,
# because config only reads dotenv.
if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Same rule: **only effective before torch is imported.** hipBLASLt runs the
# sparse-conv shim's skinny GEMM ~14x faster than rocBLAS (measured 2026-09-02,
# see docs/gemm_profile.md). `metrics.blas_backend` records what was used.
if config.PREFER_HIPBLASLT:
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")

NAME = "trellis"
VERSION = "image-large"


# --- Protocol (same format as hearth's rpc.py, but with no dependency on it) ---
def install_stdout_guard() -> TextIO:
    """Duplicate and hide the real stdout, **redirecting fd 1 itself to stderr**.

    **Call this first.** Upstream code prints freely, and replacing `sys.stdout`
    is not enough: **C extensions write straight to fd 1**, bypassing the Python
    side and corrupting the protocol stream. Measured 2026-09-01: `pymeshfix`
    emitted `Loading ..0%` hundreds of times directly to fd 1.

    So fd 1 is duplicated and reserved for the protocol, and **fd 1 itself is
    pointed at fd 2**. Everything that is not protocol, from Python or from
    native code, then lands on stderr.

    Returns:
        The protocol-only writer.
    """
    fd = os.dup(1)
    os.dup2(2, 1)  # **fd 1 to stderr; output from C extensions lands there too.**
    protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = sys.stderr
    return protocol


# **A lock is required because the heartbeat thread writes too.**
# The contract is one JSON object per line; interleaving breaks the reader.
_EMIT_LOCK = threading.Lock()


def emit(out: TextIO, payload: dict[str, Any]) -> None:
    """Write one message per line and always flush. **Thread-safe.**"""
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _EMIT_LOCK:
        out.write(line)
        out.flush()


# --- Methods -----------------------------------------------------------------
def m_capabilities(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Report capabilities. **Answers immediately, without loading the weights.**"""
    return {
        "name": NAME,
        "version": VERSION,
        "capabilities": {
            "image_to_mesh": True,
            "text_to_mesh": False,
            # Upstream has run_multi_image, but it is unverified here and so is
            # not advertised.
            "multi_image_to_mesh": False,
            # The texture stage needs nvdiffrast (CUDA only) and cannot run on
            # this machine.
            "texture": False,
        },
        "params": {
            "ss_steps": {"type": "int", "default": 25, "min": 1, "max": 100},
            "slat_steps": {"type": "int", "default": 25, "min": 1, "max": 100},
            "ss_guidance": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "slat_guidance": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "seed": {"type": "int", "default": 0, "min": 0},
        },
        "notes": (
            "spconv, flash_attn, kaolin and open3d are replaced by pure-torch launch-time "
            "shims (no build exists for Windows + ROCm). Attention uses fp16 flash when "
            "AOTriton is available and falls back to fp32 over chunked heads otherwise. "
            "The mesh comes back Z-up at normalized scale. No texture."
        ),
    }


def m_load(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Load the weights (measured around 14 s; the first run also fetches dinov2)."""
    from . import pipeline

    progress("load", "loading the TRELLIS weights")
    started = time.perf_counter()
    pipeline.load_pipeline(progress)
    return {"loaded": True, "elapsed_sec": round(time.perf_counter() - started, 2)}


def m_unload(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Release the weights and give the VRAM back."""
    from . import pipeline

    freed = pipeline.unload_pipeline()
    used_gb, _ = pipeline.device_memory_gb()
    return {"unloaded": freed, "vram_used_gb": round(used_gb, 2)}


_ALLOWED = frozenset({"ss_steps", "slat_steps", "ss_guidance", "slat_guidance", "seed"})


def m_image_to_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """One image to a raw mesh.

    **Background removal is done by the upstream pipeline via `rembg`.**
    **Scaling to real-world size is not done here.** Millimetres are downstream
    work (meshforge's forge).
    """
    from PIL import Image

    from . import pipeline

    image_path = Path(str(params["image_path"]))
    out_dir = Path(str(params["out_dir"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = set(params) - _ALLOWED - {"image_path", "out_dir"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)} (accepted: {sorted(_ALLOWED)})")

    progress("shape", "generating the 3D shape (several minutes)")
    result = pipeline.generate_mesh(
        Image.open(image_path),
        ss_steps=params.get("ss_steps"),
        slat_steps=params.get("slat_steps"),
        ss_guidance=params.get("ss_guidance"),
        slat_guidance=params.get("slat_guidance"),
        seed=int(params.get("seed", 0)),
        progress=progress,
    )

    progress("export", "writing the mesh")
    mesh_path = out_dir / "raw.ply"
    result.mesh.export(str(mesh_path))

    return {
        "mesh_path": str(mesh_path),
        "n_vertices": int(len(result.mesh.vertices)),
        "n_faces": int(len(result.mesh.faces)),
        "extra": {"up_axis": "z"},
        "metrics": {
            "load_sec": round(result.load_sec, 2),
            # **Never use this as a pass/fail signal.**
            "gen_sec": round(result.gen_sec, 2),
            "vram_peak_gb": round(result.vram_peak_gb, 2),
            # **Whether fast attention is in effect.** Without it generation is
            # several times slower.
            "fast_attention": result.fast_attention,
            # **Which BLAS backend served the GEMMs.** They differ by up to 14x
            # on some shapes, so a timing means nothing without this.
            "blas_backend": pipeline.blas_backend(),
            # **Breakdown of generation.** Without knowing which stage is slow
            # there is nothing to act on.
            "cond_sec": round(result.cond_sec, 2),
            "structure_sec": round(result.structure_sec, 2),
            "slat_sec": round(result.slat_sec, 2),
            "decode_sec": round(result.decode_sec, 2),
            "n_voxels": result.n_voxels,
            # **What post-processing removed.** Recorded so nothing vanishes silently.
            "clean": result.clean,
        },
        "params": {
            "ss_steps": result.ss_steps,
            "slat_steps": result.slat_steps,
            "ss_guidance": result.ss_guidance,
            "slat_guidance": result.slat_guidance,
            "seed": result.seed,
        },
    }


METHODS = {
    "capabilities": m_capabilities,
    "load": m_load,
    "unload": m_unload,
    "image_to_mesh": m_image_to_mesh,
}


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
    print(f"[{NAME}] runner started.", file=sys.stderr)

    for raw in sys.stdin:
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = int(request["id"])
            method_name = str(request["method"])
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[{NAME}] skipped an unparsable request: {exc}", file=sys.stderr)
            continue

        if method_name == "shutdown":
            emit(out, {"id": request_id, "event": "result", "result": {"bye": True}})
            break

        method = METHODS.get(method_name)
        if method is None:
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": "ValueError", "message": f"unknown method: {method_name}"},
                },
            )
            continue

        def progress(
            stage: str,
            message: str = "",
            _id: int = request_id,
            **extra: Any,
        ) -> None:
            # `extra` carries `step` and, when the length is known, `total`.
            # **Nothing estimated ever goes in here** (see steps.py).
            emit(
                out,
                {"id": _id, "event": "progress", "stage": stage, "message": message, **extra},
            )

        # **Render-loop keepalive** (gfxlight.py). Measured to change nothing
        # on the current driver; kept because it costs nothing.
        light: gfxlight.GfxLight | None = None
        if method_name == "image_to_mesh" and config.GFX_KEEPALIVE:
            light = gfxlight.GfxLight()
            light.start()
        try:
            result = method(dict(request.get("params") or {}), progress)
            if light is not None and isinstance(result.get("metrics"), dict):
                # Whether it stayed alive to the end. False means it may not
                # have taken effect.
                result["metrics"]["gfx_keepalive"] = light.is_lit()
            emit(out, {"id": request_id, "event": "result", "result": result})
        except Exception as exc:  # noqa: BLE001 - always answer, whatever happens
            import traceback

            traceback.print_exc()
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        finally:
            if light is not None:
                light.stop()

    print(f"[{NAME}] runner exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
