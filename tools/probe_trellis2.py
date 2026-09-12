# SPDX-License-Identifier: MIT
"""Drive TRELLIS.2 to a mesh, stage by stage, **before there is a runner**.

This is the gate from the porting plan: import the upstream package with the
shims in place (G-3), then produce geometry from one image and report what it
cost (G-4). It calls upstream's public methods in order rather than `run()`,
for the same reasons the TRELLIS.1 runner does - **and because `run()` always
samples texture and finishes with `Mesh.fill_holes()`, which needs CuMesh.**

Paths and settings come from `.env`; nothing here names one.

    .venv2\\Scripts\\python.exe tools\\probe_trellis2.py --image assets\\sample.png
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(REPO_ROOT / ".env")


def _setting(key: str, default: str = "") -> str:
    return str(ENV.get(key, os.environ.get(key, default)) or default)


# **Every one of these is read at import time** - by torch for the attention
# backend, and by TRELLIS.2 for its sparse backends - so they are set before the
# first import, not after.
if _setting("TRELLIS2_FAST_ATTENTION", "on") == "on":
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
if _setting("TRELLIS2_PREFER_HIPBLASLT", "on") == "on":
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("SPCONV_ALGO", "native")

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from runners.trellis import shims  # noqa: E402

UPSTREAM = Path(_setting("TRELLIS2_REPO"))
WEIGHTS = Path(_setting("TRELLIS2_WEIGHTS_DIR"))
CONFIG_FILE = _setting("TRELLIS2_PIPELINE_CONFIG", "pipeline.local.json")
VRAM_LIMIT_GB = float(_setting("TRELLIS2_VRAM_LIMIT_GB", "0") or 0)
HEARTBEAT_SEC = float(_setting("TRELLIS2_HEARTBEAT_SEC", "10") or 10)
HEAD_CHUNK = int(_setting("TRELLIS2_ATTN_HEAD_CHUNK", "4") or 4)


class Watch:
    """Report liveness and **catch the moment dedicated VRAM is exceeded**.

    `torch.cuda.mem_get_info` counts shared memory as if it were video memory,
    so passing the 32 GB of dedicated VRAM raises nothing and silently becomes
    several times slower. The peak is printed; crossing the line is printed the
    moment it happens.
    """

    def __init__(self) -> None:
        self.peak = 0.0
        self.exceeded = False
        self.stage = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.perf_counter()
        last = started
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            self.peak = max(self.peak, used)
            now = time.perf_counter()
            if VRAM_LIMIT_GB and used > VRAM_LIMIT_GB and not self.exceeded:
                self.exceeded = True
                print(f"  ** vram_over: {used:.2f}GB > {VRAM_LIMIT_GB:.2f}GB **", flush=True)
            if now - last >= HEARTBEAT_SEC:
                last = now
                print(
                    f"  [{now - started:7.1f}s] {self.stage}: "
                    f"VRAM {used:.2f}GB (peak {self.peak:.2f}GB)",
                    flush=True,
                )
            self._stop.wait(1.0)

    def __enter__(self) -> Watch:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(r"C:\dev\trellis-strix-halo-data\probe"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-only", action="store_true", help="stop after G-3")
    parser.add_argument(
        "--cascade", type=int, default=0, help="target resolution for the cascade (1024 or 1536)"
    )
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="leave the sparse region's edge open, as upstream does",
    )
    args = parser.parse_args()

    if not UPSTREAM.is_dir():
        raise SystemExit(f"no TRELLIS.2 checkout at {UPSTREAM} (TRELLIS2_REPO)")
    # `o_voxel` lives in the checkout rather than in site-packages: its compiled
    # half cannot be built here, and `install_trellis2` replaces exactly that.
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / "o-voxel"))

    fast = shims.install(head_chunk=HEAD_CHUNK)
    shims.install_trellis2(close_mesh=not args.no_close)
    print(f"fast attention: {fast} | blas: {torch.backends.cuda.preferred_blas_library()}")

    if VRAM_LIMIT_GB:
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(VRAM_LIMIT_GB / (total / 1024**3), 1.0))

    timings: dict[str, float] = {}
    with Watch() as watch:
        watch.stage = "import"
        t0 = time.perf_counter()
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        timings["import"] = time.perf_counter() - t0

        watch.stage = "load"
        t0 = time.perf_counter()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(WEIGHTS), config_file=CONFIG_FILE)
        pipeline.cuda()
        timings["load"] = time.perf_counter() - t0
        # **The background remover comes back in the checkpoint's dtype.**
        # transformers 5 keeps fp16 weights in fp16, while upstream's BiRefNet
        # wrapper hands it a float32 tensor, so the first convolution refuses
        # ("Input type (float) and bias type (c10::Half)"). Casting the model
        # is the launch-side fix; the vendor code is not touched.
        rembg = getattr(pipeline, "rembg_model", None)
        inner = getattr(rembg, "model", None)
        if inner is not None and next(inner.parameters()).dtype != torch.float32:
            inner.float()
        # **The conditioner's layer stack moved one level down.** Upstream walks
        # `DINOv3ViTModel.layer`; transformers 5.16 keeps the 24 blocks at
        # `.model.layer` and exposes nothing by the old name. The blocks
        # themselves are unchanged - same call, same `position_embeddings`
        # keyword - so an alias is the whole fix. It shares the module list
        # rather than copying it, so nothing moves in memory.
        cond_model = getattr(getattr(pipeline, "image_cond_model", None), "model", None)
        if cond_model is not None and not hasattr(cond_model, "layer"):
            nested = getattr(getattr(cond_model, "model", None), "layer", None)
            if nested is None:
                raise RuntimeError("DINOv3: no layer stack at .layer or .model.layer")
            cond_model.layer = nested
        print(f"loaded: {sorted(pipeline.models)} | low_vram={pipeline.low_vram}")
        if args.load_only:
            print(f"G-3 only: import {timings['import']:.1f}s, load {timings['load']:.1f}s")
            return 0

        image = Image.open(args.image)
        with torch.no_grad():
            watch.stage = "preprocess"
            t0 = time.perf_counter()
            image = pipeline.preprocess_image(image)
            timings["preprocess"] = time.perf_counter() - t0

            watch.stage = "cond"
            t0 = time.perf_counter()
            cond = pipeline.get_cond([image], args.resolution)
            timings["cond"] = time.perf_counter() - t0

            torch.manual_seed(args.seed)
            watch.stage = "structure"
            t0 = time.perf_counter()
            # **The sparse structure is sampled coarser than the output.**
            # Upstream's `run()` uses 32 for its 512 pipeline and 64 for 1024.
            ss_resolution = 32 if args.resolution <= 512 else 64
            coords = pipeline.sample_sparse_structure(cond, ss_resolution, 1, {})
            timings["structure"] = time.perf_counter() - t0
            n_voxels = int(coords.shape[0])
            print(f"  active voxels: {n_voxels:,}")

            watch.stage = "shape_slat"
            t0 = time.perf_counter()
            resolution = args.resolution
            if args.cascade:
                # **The cascade samples at 512 and refines upward**, and drops
                # the target resolution in steps of 128 until the token count
                # fits, so the resolution it actually used comes back with it.
                cond_hr = pipeline.get_cond([image], 1024)
                slat, resolution = pipeline.sample_shape_slat_cascade(
                    cond,
                    cond_hr,
                    pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    args.cascade,
                    coords,
                    {},
                    args.max_tokens,
                )
                print(f"  cascade resolution: {resolution}")
            else:
                slat = pipeline.sample_shape_slat(
                    cond, pipeline.models[f"shape_slat_flow_model_{args.resolution}"], coords, {}
                )
            timings["shape_slat"] = time.perf_counter() - t0

            watch.stage = "decode"
            t0 = time.perf_counter()
            meshes, _subs = pipeline.decode_shape_slat(slat, resolution)
            timings["decode"] = time.perf_counter() - t0

    extracted = meshes[0]
    mesh = trimesh.Trimesh(
        vertices=extracted.vertices.detach().float().cpu().numpy(),
        faces=extracted.faces.detach().cpu().numpy(),
        process=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"raw_{args.cascade or args.resolution}.ply"
    mesh.export(out_file)

    print("\n--- stages ---")
    for name, seconds in timings.items():
        print(f"  {name:11s} {seconds:8.1f}s")
    print(
        f"\nvoxels {n_voxels:,} | vertices {len(mesh.vertices):,} | faces {len(mesh.faces):,} | "
        f"watertight {mesh.is_watertight} | VRAM peak {watch.peak:.2f}GB | "
        f"vram_over {watch.exceeded}\nwrote {out_file}"
    )
    return 1 if watch.exceeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
