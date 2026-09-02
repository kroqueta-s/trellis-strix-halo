# SPDX-License-Identifier: MIT
"""Profile one generation and report where the device time goes, GEMM shapes first.

Run with the repository's virtual environment, after at least one prior
generation on this machine (MIOpen's one-time tuning would otherwise dominate):

    .venv\\Scripts\\python.exe tools\\profile_gemm.py --image assets\\sample.png

The generation happens twice. The first run is unprofiled and records the honest
stage times. The second runs stage by stage under ``torch.profiler``, which
attributes device time to operators and collects every GEMM shape
(M, N, K, batch, dtype, call count). Profiling stretches wall time but not the
kernels themselves, so **speed comes from the first run and ratios and
per-shape throughput from the second**.

A fixed reference GEMM is measured before and after everything else and
reported as TFLOPS. The GPU clock idles near 700 MHz and ramps above 2.3 GHz
at the driver's discretion, so the reference doubles as a clock check: if the
two readings disagree, the run in between is suspect.

Results are written as JSON (default: under ``docs/local/profile/``, which is
not tracked) and summarised on the console.

This file is near-identical in the sibling runners, which do not share a module
because each ships as its own repository: **fix one and fix the others.**
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import config  # noqa: E402  (config reads dotenv only, no torch)

RUNNER = "trellis"

# Ops whose kernels are GEMMs. `aten::linear` and `aten::matmul` decompose into
# these, so counting the parents too would count the same kernels twice.
GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
ATTN_PREFIXES = (
    "aten::_scaled_dot_product",
    "aten::scaled_dot_product",
    "aten::_flash_attention",
    "aten::_efficient_attention",
)

# Ops that launch no kernels of their own: views and metadata, plus parents
# (`linear`, `matmul`, `to`) whose kernels are billed to the children. This
# torch build has no Kineto, so the fallback profiler still charges each call a
# few microseconds of "device" time; at 10^5 calls that invents whole seconds.
# They are totalled apart as `bookkeeping` and kept out of every ratio.
BOOKKEEPING_OPS = {
    "aten::alias",
    "aten::as_strided",
    "aten::chunk",
    "aten::contiguous",
    "aten::detach",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::expand",
    "aten::linear",
    "aten::matmul",
    "aten::narrow",
    "aten::new_empty",
    "aten::new_empty_strided",
    "aten::permute",
    "aten::reshape",
    "aten::select",
    "aten::slice",
    "aten::split",
    "aten::squeeze",
    "aten::t",
    "aten::to",
    "aten::_to_copy",
    "aten::transpose",
    "aten::unsqueeze",
    "aten::view",
}

# Environment variables worth keeping next to a measurement (see docs/local).
ENV_PREFIXES = ("TORCH_", "ROCBLAS_", "HIPBLASLT_", "MIOPEN_", "PYTORCH_", "HSA_")


def _class_of(op: str) -> str:
    if op in GEMM_OPS:
        return "gemm"
    if op.startswith(ATTN_PREFIXES):
        return "attention"
    if "conv" in op:
        return "conv"
    if op in BOOKKEEPING_OPS:
        return "bookkeeping"
    return "other"


def _tensor_shapes(shapes: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Keep only real tensor shapes, so profiler rows and dispatch rows can meet."""
    return tuple(tuple(s) for s in shapes if s)


def _gemm_mnkb(op: str, shapes: tuple[tuple[int, ...], ...]) -> tuple[int, int, int, int] | None:
    """Return (M, N, K, batch) for one GEMM call, or None if the shapes are unexpected."""
    try:
        if op in ("aten::mm", "aten::addmm"):
            a, b = shapes[-2], shapes[-1]
            return a[0], b[1], a[1], 1
        if op in ("aten::bmm", "aten::baddbmm"):
            a, b = shapes[-2], shapes[-1]
            return a[1], b[2], a[2], a[0]
    except (IndexError, ValueError):
        return None
    return None


class _GemmDtypeRecorder:
    """Record the dtype of every GEMM call, keyed the same way the profiler keys rows.

    The profiler reports shapes but not dtypes, so this rides along as a
    ``TorchDispatchMode`` during the profiled run. Only GEMM ops are recorded;
    everything else passes straight through.
    """

    def __init__(self) -> None:
        self.dtypes: dict[tuple[str, tuple[tuple[int, ...], ...]], str] = {}
        self._mode: Any = None

    def __enter__(self) -> _GemmDtypeRecorder:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        recorder = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(
                self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
            ) -> Any:
                name = func.name() if hasattr(func, "name") else str(func)
                if name in GEMM_OPS:
                    shapes = tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))
                    dtypes = ",".join(
                        str(a.dtype).replace("torch.", "")
                        for a in args
                        if isinstance(a, torch.Tensor)
                    )
                    recorder.dtypes.setdefault((name, shapes), dtypes)
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._mode.__exit__(*exc)


def reference_gemm(sizes: tuple[int, ...] = (2048, 4096), iters: int = 50) -> dict[str, float]:
    """Measure square fp16 GEMM throughput in TFLOPS, timed with device events."""
    import torch

    out: dict[str, float] = {}
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            a @ b
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0
        out[str(n)] = round(2 * n**3 * iters / sec / 1e12, 2)
        del a, b
    torch.cuda.empty_cache()
    return out


def summarize_stage(
    prof: Any,
    dtypes: dict[tuple[str, tuple[tuple[int, ...], ...]], str],
    top: int,
) -> dict[str, Any]:
    """Turn one stage's profile into class totals, GEMM rows and top other ops."""
    events = prof.key_averages(group_by_input_shape=True)
    class_us: dict[str, float] = {}
    gemm_rows: list[dict[str, Any]] = []
    other_us: dict[str, float] = {}
    other_calls: dict[str, int] = {}
    for e in events:
        self_us = max(0.0, float(e.self_device_time_total))
        if self_us <= 0:
            continue
        cls = _class_of(e.key)
        class_us[cls] = class_us.get(cls, 0.0) + self_us
        if cls == "gemm":
            shapes = _tensor_shapes(e.input_shapes)
            mnkb = _gemm_mnkb(e.key, shapes)
            row: dict[str, Any] = {
                "op": e.key,
                "shapes": [list(s) for s in shapes],
                "dtype": dtypes.get((e.key, shapes), "?"),
                "count": int(e.count),
                "device_ms": round(self_us / 1000.0, 2),
            }
            if mnkb is not None:
                m, n, k, batch = mnkb
                row.update(m=m, n=n, k=k, batch=batch)
                row["tflops"] = round(2.0 * batch * m * n * k * e.count / self_us / 1e6, 2)
            gemm_rows.append(row)
        elif cls != "bookkeeping":
            other_us[e.key] = other_us.get(e.key, 0.0) + self_us
            other_calls[e.key] = other_calls.get(e.key, 0) + int(e.count)
    gemm_rows.sort(key=lambda r: -r["device_ms"])
    top_other = sorted(other_us.items(), key=lambda kv: -kv[1])[:top]
    # Bookkeeping is profiler noise, not GPU work, so ratios are taken without it.
    total_us = sum(v for k, v in class_us.items() if k != "bookkeeping")
    return {
        "device_total_ms": round(total_us / 1000.0, 1),
        "class_ms": {k: round(v / 1000.0, 1) for k, v in sorted(class_us.items())},
        "gemm_share": round(class_us.get("gemm", 0.0) / total_us, 3) if total_us else 0.0,
        "gemm_rows": gemm_rows,
        "top_other": [
            {"op": op, "device_ms": round(us / 1000.0, 2), "count": other_calls[op]}
            for op, us in top_other
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(REPO_ROOT / "assets" / "sample.png"))
    parser.add_argument("--json", dest="json_path", default=None, help="where to write the result")
    parser.add_argument("--ss_steps", type=int, default=None)
    parser.add_argument("--slat_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=15, help="rows kept per stage")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set before torch is imported (BLAS backends read these at import)",
    )
    args = parser.parse_args()

    for pair in args.env:
        key, _, value = pair.partition("=")
        os.environ[key] = value

    # Mirrors the runner: these must precede the first torch import to matter.
    if config.FAST_ATTENTION:
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    if config.PREFER_HIPBLASLT:
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
        os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "1")

    import torch
    from PIL import Image
    from torch.profiler import ProfilerActivity, profile

    from runners.trellis import gfxlight, pipeline

    started = time.perf_counter()

    def say(stage: str, message: str, **extra: Any) -> None:
        step = f" ({extra['step']}/{extra.get('total') or '?'})" if "step" in extra else ""
        print(f"[{time.perf_counter() - started:7.1f}s] {stage}: {message}{step}", flush=True)

    light = gfxlight.GfxLight()
    light.start()
    try:
        say("ref", "reference GEMM before")
        ref_before = reference_gemm()
        say("ref", f"TFLOPS {ref_before}")

        image = Image.open(args.image)
        say("warmup", "unprofiled generation (honest stage times)")
        warm = pipeline.generate_mesh(
            image,
            ss_steps=args.ss_steps,
            slat_steps=args.slat_steps,
            seed=args.seed,
            progress=say,
        )
        stage_walls = {
            "cond": warm.cond_sec,
            "structure": warm.structure_sec,
            "slat": warm.slat_sec,
            "decode": warm.decode_sec,
        }
        say("warmup", f"stage walls {json.dumps({k: round(v, 1) for k, v in stage_walls.items()})}")

        # The profiled run replays the same public upstream calls as
        # `pipeline.generate_mesh`, one profiler per stage, so device time can be
        # attributed per stage. Upstream code is not modified.
        pipe = pipeline.load_pipeline(say)
        ss = config.SS_STEPS if args.ss_steps is None else args.ss_steps
        slat_steps = config.SLAT_STEPS if args.slat_steps is None else args.slat_steps

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        stages: dict[str, dict[str, Any]] = {}
        n_voxels = 0

        def profiled(stage: str, fn: Any) -> Any:
            say("profile", f"{stage} begins")
            with _GemmDtypeRecorder() as rec:
                with profile(activities=activities, record_shapes=True) as prof:
                    result = fn()
                    torch.cuda.synchronize()
            stages[stage] = summarize_stage(prof, rec.dtypes, args.top)
            say(
                "profile",
                f"{stage}: device {stages[stage]['device_total_ms']}ms, "
                f"gemm share {stages[stage]['gemm_share']:.0%}",
            )
            return result

        with torch.no_grad():

            def _cond() -> Any:
                return pipe.get_cond([pipe.preprocess_image(image)])

            cond = profiled("cond", _cond)
            torch.manual_seed(args.seed)
            pipeline._STEPS.bind(say, "structure", "sampling the sparse structure")
            try:
                coords = profiled(
                    "structure",
                    lambda: pipe.sample_sparse_structure(
                        cond, 1, {"steps": ss, "cfg_strength": config.SS_GUIDANCE}
                    ),
                )
            finally:
                pipeline._STEPS.bind(None, "structure")
            n_voxels = int(coords.shape[0])
            pipeline._STEPS.bind(say, "slat", "sampling the latent")
            try:
                slat_latent = profiled(
                    "slat",
                    lambda: pipe.sample_slat(
                        cond, coords, {"steps": slat_steps, "cfg_strength": config.SLAT_GUIDANCE}
                    ),
                )
            finally:
                pipeline._STEPS.bind(None, "slat")
            profiled("decode", lambda: pipe.decode_slat(slat_latent, ["mesh"]))

        say("ref", "reference GEMM after")
        ref_after = reference_gemm()
        say("ref", f"TFLOPS {ref_after}")
    finally:
        light.stop()

    result: dict[str, Any] = {
        "runner": RUNNER,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
        "fast_attention": warm.fast_attention,
        "gfx_keepalive": True,
        "ref_gemm_tflops": {"before": ref_before, "after": ref_after},
        "params": {"ss_steps": warm.ss_steps, "slat_steps": warm.slat_steps, "seed": warm.seed},
        "n_voxels": n_voxels,
        "vram_peak_gb": round(warm.vram_peak_gb, 2),
        "stage_walls_sec": {k: round(v, 2) for k, v in stage_walls.items()},
        "stages": stages,
    }

    json_path = (
        Path(args.json_path)
        if args.json_path
        else REPO_ROOT
        / "docs"
        / "local"
        / "profile"
        / f"gemm_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    say("done", f"written to {json_path}")

    for stage, s in stages.items():
        print(f"\n== {stage}: device {s['device_total_ms']}ms, classes {s['class_ms']}")
        for row in s["gemm_rows"][: args.top]:
            mnkb = (
                f"M={row['m']} N={row['n']} K={row['k']} B={row['batch']}"
                if "m" in row
                else str(row["shapes"])
            )
            print(
                f"  {row['op']:<14} {mnkb:<38} {row['dtype']:<18} "
                f"x{row['count']:<5} {row['device_ms']:>9.1f}ms "
                f"{row.get('tflops', 0):>6.2f} TFLOPS"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
