# SPDX-License-Identifier: MIT
"""The TRELLIS.2 runner (an implementation of the runner contract).

**This process is the only one that holds torch.** Neither hearth itself nor the
Blender add-on imports it, and this runner imports nothing from hearth.

Start it (hearth normally spawns it as a child process)::

    .venv2\\Scripts\\python.exe -m runners.trellis2

**The protocol half is the TRELLIS.1 runner's, written out again rather than
imported.** A runner is meant to be liftable into its own repository, and that
is easier to honour with one file per runner than with a shared module that
would have to move with it. The shims, the step counter and the postprocessing
*are* shared, because those are the parts where a fix must reach both.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import config

# **Has no effect unless it precedes torch** (setting os.environ later is
# ignored). It makes the flash and memory-efficient kernels available on
# gfx1151 - measured 15x faster than the math backend in both fp16 and bf16
# (2026-09-10). Importing config here does not pull in torch: it only reads
# dotenv.
if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Same rule: **only effective before torch is imported.**
if config.PREFER_HIPBLASLT:
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")

NAME = "trellis2"
VERSION = "4B"


# --- Protocol (same format as hearth's rpc.py, but with no dependency on it) ---
def install_stdout_guard() -> TextIO:
    """Duplicate and hide the real stdout, **redirecting fd 1 itself to stderr**.

    **Call this first.** Upstream code prints freely, and replacing `sys.stdout`
    is not enough: **C extensions write straight to fd 1**, bypassing the Python
    side and corrupting the protocol stream.
    """
    fd = os.dup(1)
    os.dup2(2, 1)
    protocol = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    return protocol


_EMIT_LOCK = threading.Lock()


def emit(out: TextIO, payload: dict[str, Any]) -> None:
    """Write one JSON object, one line. **Locked**: the watcher thread emits too."""
    with _EMIT_LOCK:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        out.flush()


def m_capabilities(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Report capabilities. **Answers immediately, without loading the weights.**"""
    return {
        "name": NAME,
        "version": VERSION,
        # The version of `docs/runner_contract.md` this was written against.
        "contract": 3,
        "capabilities": {
            "image_to_mesh": True,
            "text_to_mesh": False,
            "multi_image_to_mesh": False,
            # **Not implemented yet**, though unlike TRELLIS.1 it is reachable:
            # the texture stage needs UV unwrapping (xatlas can do it) and a
            # rasterizer, not CUDA specifically.
            "texture": False,
        },
        "params": {
            "resolution": {
                "type": "int",
                "default": config.RESOLUTION,
                "min": 512,
                "max": 1536,
            },
            "max_tokens": {"type": "int", "default": config.MAX_TOKENS, "min": 4096},
            "seed": {"type": "int", "default": 0, "min": 0},
        },
        "notes": (
            "spconv, flash_attn, flex_gemm's sparse convolution and o_voxel's hashmap are "
            "replaced by pure-torch launch-time shims (no build exists for Windows + ROCm); "
            "cumesh, nvdiffrast and flex_gemm's grid_sample stand in and raise if called. "
            "**The mesh is not watertight**: the decoder's own field is open (32.8% of its "
            "2x2 loops are odd at 512), the boundary loops are closed here, and what is left "
            "is the boundary non-manifold edges make. The winding is inconsistent by "
            "construction and is left that way, because correcting it costs more than the "
            "generation - see metrics.topology. Z-up, normalized scale, no texture. "
            "Asking for 1536 yields 1408: upstream applies its own token limit."
        ),
    }


def m_load(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Load the weights (measured 35-44 s; five models, one at a time on the card)."""
    from . import pipeline

    progress("load", "loading the TRELLIS.2 weights")
    started = time.perf_counter()
    pipeline.load_pipeline(progress)
    return {"loaded": True, "elapsed_sec": round(time.perf_counter() - started, 2)}


def m_unload(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Release the weights and give the VRAM back."""
    from . import pipeline

    freed = pipeline.unload_pipeline()
    used_gb, _ = pipeline.device_memory_gb()
    return {"unloaded": freed, "vram_used_gb": round(used_gb, 2)}


_ALLOWED = frozenset({"resolution", "max_tokens", "seed"})


def m_image_to_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """One image to a raw mesh.

    **Scaling to real-world size is not done here.** Millimetres are downstream
    work (meshforge's forge), and so is decimation: this returns what the model
    produced, cleaned of debris and with its holes closed.
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

    requested = params.get("resolution")
    progress("shape", "generating the shape")
    result = pipeline.generate_mesh(
        Image.open(image_path),
        resolution=int(requested) if requested else None,
        seed=int(params.get("seed", 0)),
        max_tokens=int(params["max_tokens"]) if params.get("max_tokens") else None,
        progress=progress,
    )

    progress("export", "writing the mesh")
    mesh_path = out_dir / "raw.ply"
    # **Written beside its final name, then renamed** (contract §9). A cancel
    # ends this process outright, and a run killed halfway through writing
    # fifteen million faces otherwise leaves a truncated file that looks
    # finished.
    staging = out_dir / "raw.ply.part"
    result.mesh.export(str(staging), file_type="ply")
    os.replace(staging, mesh_path)

    return {
        "mesh_path": str(mesh_path),
        "n_vertices": int(len(result.mesh.vertices)),
        "n_faces": int(len(result.mesh.faces)),
        "metrics": {
            "load_sec": round(result.load_sec, 2),
            # **Never use this as a pass/fail signal** (contract §5).
            "gen_sec": round(result.gen_sec, 2),
            "vram_peak_gb": round(result.vram_peak_gb, 2),
            # **Whether dedicated VRAM was exceeded.** Past it the card spills
            # into shared memory and everything silently slows down.
            "vram_over": result.vram_over,
            "fast_attention": result.fast_attention,
            "blas_backend": pipeline.blas_backend(),
            # **The resolution it actually ran at**, which above 1024 is chosen
            # by upstream's token limit rather than by the request.
            "resolution": result.resolution,
            "n_voxels": result.n_voxels,
            # **Breakdown of generation.** Without knowing which stage is slow
            # there is nothing to act on.
            **result.stages,
            # **What post-processing removed and added.** Nothing vanishes silently.
            "post": result.post,
            # **The state of the mesh, counted rather than claimed.**
            "topology": result.topology,
        },
        # **Up was checked; forward was not** (contract §5). A mesh imported on
        # the wrong horizontal axis renders perfectly correctly, so nobody finds
        # that mistake by looking - the first sign is a mirrored joint on a
        # printed part.
        "up_axis": "z",
        "forward_axis": None,
        "params_used": {
            "resolution": result.resolution,
            "seed": result.seed,
        },
    }


METHODS = {
    "capabilities": m_capabilities,
    "load": m_load,
    "unload": m_unload,
    "image_to_mesh": m_image_to_mesh,
}


def watch_parent(interval_sec: float = 2.0) -> None:
    """End this process if the caller that started it goes away.

    **A runner that has already taken the card is exactly the one worth
    ending**: nothing else can load a model while it holds the GPU.
    """
    watched = os.getppid()

    def gone() -> None:
        print(f"[{NAME}] parent {watched} is gone; exiting.", file=sys.stderr)
        os._exit(0)

    def watch() -> None:
        while True:
            time.sleep(interval_sec)
            if os.getppid() != watched:
                gone()

    threading.Thread(target=watch, name=f"{NAME}-parent-watch", daemon=True).start()


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
    # **Before anything is loaded.**
    watch_parent()
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

        def progress(stage: str, message: str = "", _id: int = request_id, **extra: Any) -> None:
            # `extra` carries `step` and, when the length is known, `total`.
            # **Nothing estimated ever goes in here** (contract §8).
            emit(
                out,
                {"id": _id, "event": "progress", "stage": stage, "message": message, **extra},
            )

        try:
            result = method(dict(request.get("params") or {}), progress)
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

    print(f"[{NAME}] runner exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
