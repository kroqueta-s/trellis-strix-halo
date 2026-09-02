# SPDX-License-Identifier: MIT
"""TRELLIS itself (image to mesh). **Only this module touches the GPU.**

Everything gfx1151 / Windows / ROCm requires is confined to `shims.install()`.
**The reasoning is in the `shims.py` docstring.**

**Preprocessing (background removal) is done by the upstream pipeline via
`rembg`.** It is the runner's responsibility under the contract, but TRELLIS
brings its own, so that is what runs.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import trimesh
from PIL import Image

from . import config, postprocess, shims
from .steps import StepCounter, count_tqdm

NAME = "trellis"
VERSION = "image-large"

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0
# Counts whichever sampling loop is running. Rebound for each stage.
_STEPS = StepCounter()
# Whether fast attention (AOTriton) is in effect. **Recorded in metrics.**
_FAST_ATTENTION: bool = False


class _DeviceWatch:
    """Watcher thread that tracks device memory and **reports liveness at a fixed interval**.

    It used to only take a peak reading. That alone leaves no way to tell from
    outside whether a long stage is progressing or stuck; on 2026-09-01 runs
    were repeatedly allowed to continue silently for more than 12 minutes.

    Three things are watched:

    1. **Liveness** (`heartbeat`): elapsed time and VRAM every 10 seconds by
       default. The caller can treat a gap as "not progressing".
    2. **Dedicated VRAM overflow** (`vram_over`). **There are only 32 GB of
       dedicated VRAM.** The total reported by `torch.cuda.mem_get_info`
       (43.87 GB) is a lie that counts shared memory, so spilling raises nothing
       and **silently becomes several times slower**. Crossing the line is
       reported the moment it happens.
    3. The peak (still reported in `metrics`).

    **This thread calls `progress`, so the caller's emit must be lock-protected.**
    """

    def __init__(
        self,
        progress: Callable[[str, str], None] | None = None,
        stage: str = "",
        interval: float = 2.0,
        heartbeat_sec: float = 10.0,
        limit_gb: float = 0.0,
    ) -> None:
        self.interval = interval
        self.heartbeat_sec = heartbeat_sec
        self.limit_gb = limit_gb
        self.stage = stage
        self.peak_used_gb = 0.0
        self.exceeded = False
        self._progress = progress
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _say(self, stage: str, message: str) -> None:
        if self._progress is not None:
            self._progress(stage, message)

    def _run(self) -> None:
        started = time.perf_counter()
        last_beat = started
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            if used > self.peak_used_gb:
                self.peak_used_gb = used
            now = time.perf_counter()
            if self.limit_gb > 0 and used > self.limit_gb and not self.exceeded:
                self.exceeded = True
                self._say(
                    "vram_over",
                    f"**dedicated VRAM exceeded** ({used:.2f}GB > {self.limit_gb:.2f}GB). "
                    "It is spilling into shared memory, so waiting only means slower",
                )
            if now - last_beat >= self.heartbeat_sec:
                last_beat = now
                self._say(
                    "heartbeat",
                    f"{self.stage or 'running'} {now - started:.0f}s elapsed / "
                    f"VRAM {used:.2f}GB (peak {self.peak_used_gb:.2f}GB)",
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> _DeviceWatch:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)


def apply_vram_limit() -> float:
    """Make exceeding dedicated VRAM **fail immediately instead of silently slowing down**.

    The total from `torch.cuda.mem_get_info` includes shared memory (43.87 GB on
    gfx1151), so passing the 32 GB of dedicated VRAM raises nothing. The excess
    lands in host memory and **becomes several times slower with no exception and
    no warning** (measured 2026-09-01 in sparse convolution, which eventually
    reached 42.02 GB and a `torch.OutOfMemoryError`).

    Passing an allocation cap to torch as a fraction of the total makes any
    allocation beyond it a `torch.OutOfMemoryError`, so **it surfaces at once**.

    Returns:
        The cap actually applied, in GB, or 0.0 if none could be applied.
    """
    limit = float(config.VRAM_LIMIT_GB)
    if limit <= 0 or not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    total_gb = total / 1024**3
    fraction = min(max(limit / total_gb, 0.05), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return limit


def device_memory_gb() -> tuple[float, float]:
    """Return device memory as (used, total) in GB.

    **`total` includes shared memory and does not match the 32 GB of dedicated
    VRAM.** Keep that in mind when comparing figures.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def _prepare_environment() -> None:
    """Set environment variables and install the shims **before** importing `trellis`.

    Upstream branches on environment variables at import time
    (`trellis/modules/sparse/__init__.py`), so **setting them afterwards has no
    effect.**
    """
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    repo = str(config.TRELLIS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    global _FAST_ATTENTION
    _FAST_ATTENTION = shims.install(head_chunk=config.ATTN_HEAD_CHUNK)


def load_pipeline(progress: Callable[[str, str], None] | None = None) -> Any:
    """Load the weights (a no-op on later calls).

    **Timing is reported per stage.** When switching is slow there is nothing to
    act on without knowing where the time went (a real problem on 2026-09-01).
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is not None:
        return _PIPELINE

    if not config.TRELLIS_REPO.is_dir():
        raise FileNotFoundError(f"no TRELLIS clone at: {config.TRELLIS_REPO}")
    if not (config.TRELLIS_WEIGHTS_DIR / "pipeline.json").is_file():
        raise FileNotFoundError(f"weights not found: {config.TRELLIS_WEIGHTS_DIR}/pipeline.json")

    def _say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    _say("import", "importing trellis (after installing the shims)")
    _prepare_environment()
    limit = apply_vram_limit()
    _say("vram_limit", f"dedicated VRAM capped at {limit:.1f}GB (exceeding it fails as OOM)")
    from trellis.pipelines import TrellisImageTo3DPipeline

    started = time.perf_counter()
    _say("weights", "loading the weights")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(config.TRELLIS_WEIGHTS_DIR))
    _say("to_gpu", "moving to the GPU")
    pipeline.cuda()
    _LOAD_SEC = time.perf_counter() - started

    # **Count the sampling steps.** Both samplers loop inside
    # `flow_euler.py` over a `tqdm`, so replacing that one module's `tqdm`
    # covers the sparse-structure pass and the latent pass alike. The stage name
    # comes from whichever call is running (see `generate_mesh`).
    from trellis.pipelines.samplers import flow_euler

    count_tqdm(flow_euler, _STEPS)

    _say("loaded", f"loading finished ({_LOAD_SEC:.1f}s)")
    _PIPELINE = pipeline
    return _PIPELINE


def unload_pipeline() -> bool:
    """Release the weights and give the VRAM back.

    Returns:
        Whether anything was actually released (False if nothing was loaded).
    """
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is None:
        return False
    _PIPELINE = None
    _LOAD_SEC = 0.0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return True


@dataclass
class MeshResult:
    """The generated mesh and its measurements."""

    mesh: trimesh.Trimesh
    load_sec: float
    gen_sec: float
    vram_peak_gb: float
    fast_attention: bool
    clean: dict[str, Any]
    cond_sec: float
    structure_sec: float
    slat_sec: float
    decode_sec: float
    n_voxels: int
    ss_steps: int
    slat_steps: int
    ss_guidance: float
    slat_guidance: float
    seed: int


def generate_mesh(
    image: Image.Image,
    ss_steps: int | None = None,
    slat_steps: int | None = None,
    ss_guidance: float | None = None,
    slat_guidance: float | None = None,
    seed: int = 0,
    progress: Callable[[str, str], None] | None = None,
) -> MeshResult:
    """Generate a mesh from one image.

    **The coordinate system is returned as upstream leaves it (Z-up).** Nothing
    is scaled to real-world size and nothing is reoriented, both of which are
    downstream work (`forge`).

    This **repeats the steps of upstream's `pipeline.run()` explicitly**, for two
    reasons:

    1. **To measure where the time goes.** Calling `run()` wholesale gives no
       breakdown, leaving nothing to act on when it is slow (a real problem on
       2026-09-01).
    2. **To record the active voxel count**, which largely determines the cost of
       the sparse stages and the decode.

    **Upstream code is not modified**; these are its public methods called in
    order.
    """
    pipeline = load_pipeline(progress)
    ss = config.SS_STEPS if ss_steps is None else int(ss_steps)
    slat = config.SLAT_STEPS if slat_steps is None else int(slat_steps)
    ss_cfg = config.SS_GUIDANCE if ss_guidance is None else float(ss_guidance)
    slat_cfg = config.SLAT_GUIDANCE if slat_guidance is None else float(slat_guidance)

    if progress is not None:
        progress("sample", f"generating (ss={ss} / slat={slat})")
    with _DeviceWatch(
        progress=progress,
        stage="generation",
        heartbeat_sec=config.HEARTBEAT_SEC,
        limit_gb=config.VRAM_LIMIT_GB,
    ) as sampler:
        # **Do not drop `torch.no_grad()`.** Upstream's `run()` carries it as a
        # decorator, but the individual methods such as `sample_sparse_structure`
        # do not. Splitting into measured stages loses it and **accumulates an
        # autograd graph that eats all the VRAM** (measured 2026-09-01: 29.66 GB
        # at the decode stage, then OOM -- caught within 100 seconds thanks to
        # the cap).
        with torch.no_grad():
            started = time.perf_counter()
            if progress is not None:
                progress("cond", "encoding the image into a conditioning vector")
            image = pipeline.preprocess_image(image)
            cond = pipeline.get_cond([image])
            cond_sec = time.perf_counter() - started

            torch.manual_seed(int(seed))

            if progress is not None:
                progress("structure", f"sampling the sparse structure (steps={ss})")
            step_started = time.perf_counter()
            # **The count belongs to the stage that is running.** Both samplers
            # share one loop, so the stage is named here rather than in the hook.
            _STEPS.bind(progress, "structure", "sampling the sparse structure")
            try:
                coords = pipeline.sample_sparse_structure(
                    cond, 1, {"steps": ss, "cfg_strength": ss_cfg}
                )
            finally:
                _STEPS.bind(None, "structure")
            structure_sec = time.perf_counter() - step_started
            n_voxels = int(coords.shape[0])

            if progress is not None:
                progress("slat", f"sampling the latent (steps={slat} / {n_voxels} active voxels)")
            step_started = time.perf_counter()
            _STEPS.bind(progress, "slat", "sampling the latent")
            try:
                slat_latent = pipeline.sample_slat(
                    cond, coords, {"steps": slat, "cfg_strength": slat_cfg}
                )
            finally:
                _STEPS.bind(None, "slat")
            slat_sec = time.perf_counter() - step_started

            if progress is not None:
                progress("decode", "decoding to a mesh")
            step_started = time.perf_counter()
            outputs = pipeline.decode_slat(slat_latent, ["mesh"])
            decode_sec = time.perf_counter() - step_started
            gen_sec = time.perf_counter() - started

    extracted = outputs["mesh"][0]
    if not bool(getattr(extracted, "success", True)):
        raise RuntimeError("the mesh came back empty (the foreground may not have been extracted)")

    mesh = trimesh.Trimesh(
        vertices=extracted.vertices.detach().float().cpu().numpy(),
        faces=extracted.faces.detach().cpu().numpy(),
        process=False,
    )
    # **Post-processing follows upstream.** The details, and the one addition
    # upstream lacks, are in the postprocess.py docstring.
    mesh, clean_stats = postprocess.clean(mesh, progress)
    return MeshResult(
        mesh=mesh,
        load_sec=_LOAD_SEC,
        gen_sec=gen_sec,
        vram_peak_gb=sampler.peak_used_gb,
        fast_attention=_FAST_ATTENTION,
        clean=clean_stats.as_dict(),
        cond_sec=cond_sec,
        structure_sec=structure_sec,
        slat_sec=slat_sec,
        decode_sec=decode_sec,
        n_voxels=n_voxels,
        ss_steps=ss,
        slat_steps=slat,
        ss_guidance=ss_cfg,
        slat_guidance=slat_cfg,
        seed=int(seed),
    )
