# SPDX-License-Identifier: MIT
"""TRELLIS.2 itself (image to mesh). **Only this module touches the GPU.**

Everything gfx1151 / Windows / ROCm requires is confined to
`runners.trellis.shims`, which both runners share - **fix it once and both are
fixed**. The reasoning is in that module's docstring.

**Upstream's `run()` is not used.** It samples texture whatever you ask for, and
finishes in `Mesh.fill_holes()`, which needs CuMesh. The public methods are
called in order instead, which also makes each stage measurable - the only way
to know which one is slow.

**Background removal is BiRefNet**, part of the upstream pipeline, pointed at a
local copy through `pipeline.local.json`.
"""

from __future__ import annotations

import gc
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image

from runners.trellis import close_holes as holes
from runners.trellis import postprocess, shims
from runners.trellis.steps import StepCounter, count_tqdm

from . import config

NAME = "trellis2"
VERSION = "4B"

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0
_FAST_ATTENTION: bool = False
# Counts whichever sampling loop is running. Rebound for each stage.
_STEPS = StepCounter()


class _DeviceWatch:
    """Report liveness, and **catch the moment dedicated VRAM is exceeded**.

    The same watcher the TRELLIS.1 runner carries, and for the same reasons:
    `torch.cuda.mem_get_info` counts shared memory as video memory, so passing
    the 32 GB of dedicated VRAM raises nothing and **silently becomes several
    times slower**. A long stage with no heartbeat is indistinguishable from a
    stuck one from outside.

    **This thread calls `progress`, so the caller's emit must be lock-protected.**
    """

    def __init__(
        self,
        progress: Callable[..., None] | None = None,
        stage: str = "",
        interval: float = 1.0,
    ) -> None:
        self.interval = interval
        self.stage = stage
        self.peak_used_gb = 0.0
        self.exceeded = False
        self._progress = progress
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _say(self, stage: str, message: str) -> None:
        if self._progress is not None:
            self._progress(stage, message)

    def _run(self) -> None:
        started = time.perf_counter()
        last_beat = started
        limit = config.VRAM_LIMIT_GB
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            self.peak_used_gb = max(self.peak_used_gb, used)
            now = time.perf_counter()
            if limit > 0 and used > limit and not self.exceeded:
                self.exceeded = True
                self._say(
                    "vram_over",
                    f"**dedicated VRAM exceeded** ({used:.2f}GB > {limit:.2f}GB). "
                    "It is spilling into shared memory, so waiting only means slower",
                )
            if now - last_beat >= config.HEARTBEAT_SEC:
                last_beat = now
                self._say(
                    "heartbeat",
                    f"{self.stage or 'running'} {now - started:.0f}s elapsed / "
                    f"VRAM {used:.2f}GB (peak {self.peak_used_gb:.2f}GB)",
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> _DeviceWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def blas_backend() -> str:
    """Name the BLAS backend torch prefers. **A timing means nothing without it.**"""
    try:
        name = str(torch.backends.cuda.preferred_blas_library())
    except (AttributeError, RuntimeError):
        return "unknown"
    lowered = name.rsplit(".", 1)[-1].lower()
    return {"cublaslt": "hipblaslt", "cublas": "rocblas"}.get(lowered, lowered)


def apply_vram_limit() -> float:
    """Make exceeding dedicated VRAM **fail at once instead of silently slowing down**."""
    limit = float(config.VRAM_LIMIT_GB)
    if limit <= 0 or not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(max(limit / (total / 1024**3), 0.05), 1.0))
    return limit


def device_memory_gb() -> tuple[float, float]:
    """Device memory as (used, total) in GB. **`total` counts shared memory.**"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def _prepare_environment() -> None:
    """Put the shims in place and the checkout on the path, **before importing trellis2**.

    Upstream reads its backend choice from the environment at import time, so
    these are set here and not later. **`flex_gemm` is the backend the published
    checkpoints were saved with**: through `spconv` the convolution weights are
    never loaded at all (see `shims._install_flex_gemm`).
    """
    import os

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

    repo = str(config.TRELLIS2_REPO)
    for entry in (repo, str(config.TRELLIS2_REPO / "o-voxel")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    global _FAST_ATTENTION
    _FAST_ATTENTION = shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
    shims.install_trellis2(close_mesh=config.CLOSE_MESH)


def _adapt(pipeline: Any) -> None:
    """Two API drifts, fixed on the launch side. **Upstream is not modified.**

    - The background remover comes back in the checkpoint's dtype. transformers
      5 keeps fp16 weights in fp16 while upstream's wrapper feeds it float32,
      and the first convolution refuses.
    - DINOv3's 24 blocks moved from `.layer` to `.model.layer`. The blocks
      themselves are unchanged, so an alias is the whole fix; it shares the
      module list rather than copying it.
    """
    inner = getattr(getattr(pipeline, "rembg_model", None), "model", None)
    if inner is not None and next(inner.parameters()).dtype != torch.float32:
        inner.float()

    cond_model = getattr(getattr(pipeline, "image_cond_model", None), "model", None)
    if cond_model is not None and not hasattr(cond_model, "layer"):
        nested = getattr(getattr(cond_model, "model", None), "layer", None)
        if nested is None:
            raise RuntimeError("DINOv3: no layer stack at .layer or .model.layer")
        cond_model.layer = nested


def load_pipeline(progress: Callable[..., None] | None = None) -> Any:
    """Load the weights (a no-op on later calls). **Timing is reported per stage.**"""
    global _PIPELINE, _LOAD_SEC
    if _PIPELINE is not None:
        return _PIPELINE

    if not config.TRELLIS2_REPO.is_dir():
        raise FileNotFoundError(f"no TRELLIS.2 checkout at: {config.TRELLIS2_REPO}")
    description = config.WEIGHTS_DIR / config.PIPELINE_CONFIG
    if not description.is_file():
        raise FileNotFoundError(f"pipeline description not found: {description}")

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    say("import", "importing trellis2 (after installing the shims)")
    _prepare_environment()
    limit = apply_vram_limit()
    say("vram_limit", f"dedicated VRAM capped at {limit:.1f}GB (exceeding it fails as OOM)")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    started = time.perf_counter()
    say("weights", "loading the weights")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        str(config.WEIGHTS_DIR), config_file=config.PIPELINE_CONFIG
    )
    pipeline.cuda()
    _adapt(pipeline)
    _LOAD_SEC = time.perf_counter() - started

    # **Count the sampling steps.** Both samplers loop inside `flow_euler` over
    # a tqdm, so replacing that one module's tqdm covers every stage; the stage
    # name comes from whichever call is running (see `generate_mesh`).
    from trellis2.pipelines.samplers import flow_euler

    count_tqdm(flow_euler, _STEPS)

    say("loaded", f"loading finished ({_LOAD_SEC:.1f}s)")
    _PIPELINE = pipeline
    return _PIPELINE


def unload_pipeline() -> bool:
    """Release the weights and give the VRAM back."""
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
    vram_over: bool
    fast_attention: bool
    resolution: int
    n_voxels: int
    seed: int
    stages: dict[str, float] = field(default_factory=dict)
    post: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)


def _sampler_params(steps: int, guidance: float) -> dict[str, Any]:
    """Only override what was actually set; upstream's own defaults stand otherwise."""
    params: dict[str, Any] = {}
    if steps > 0:
        params["steps"] = int(steps)
    if guidance > 0:
        params["cfg_strength"] = float(guidance)
    return params


def _topology(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Count what a caller needs to know before trusting the mesh.

    **Reported rather than fixed.** The winding is inconsistent by construction
    here (49.2% of faces at 512, measured 2026-09-11) because the dual grid's
    flags carry no inside or outside, and correcting that costs more than the
    generation - so it is downstream work, after decimation.
    """
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume": float(mesh.volume),
    }


def generate_mesh(
    image: Image.Image,
    resolution: int | None = None,
    seed: int = 0,
    max_tokens: int | None = None,
    progress: Callable[..., None] | None = None,
) -> MeshResult:
    """Generate one mesh from one image.

    **The coordinate system is returned as upstream leaves it.** Nothing is
    scaled to real-world size and nothing is reoriented: both are downstream
    work (`forge`).

    **The stages are called individually** rather than through `run()`, to
    measure where the time goes and because `run()` cannot skip texture.
    """
    pipeline = load_pipeline(progress)
    target = int(resolution or config.RESOLUTION)
    tokens = int(max_tokens or config.MAX_TOKENS)

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    stages: dict[str, float] = {}
    with _DeviceWatch(progress=progress, stage="generation") as watch:
        # **Do not drop `torch.no_grad()`.** Upstream carries it on `run()`, not
        # on the individual methods, and without it the autograd graph grows
        # through every stage until the card is full.
        with torch.no_grad():
            started = time.perf_counter()

            watch.stage = "preprocess"
            say("preprocess", "removing the background")
            mark = time.perf_counter()
            image = pipeline.preprocess_image(image)
            stages["preprocess_sec"] = time.perf_counter() - mark

            watch.stage = "cond"
            say("cond", "encoding the image into a conditioning vector")
            mark = time.perf_counter()
            cond = pipeline.get_cond([image], min(target, 1024))
            stages["cond_sec"] = time.perf_counter() - mark

            torch.manual_seed(int(seed))

            # **The sparse structure is sampled coarser than the output**: 32
            # for 512, 64 above it, which is what upstream's own `run()` does.
            structure_resolution = 32 if target <= 512 else 64
            watch.stage = "structure"
            say("structure", f"sampling the sparse structure (grid {structure_resolution})")
            _STEPS.bind(progress, "structure", "sampling the sparse structure")
            mark = time.perf_counter()
            try:
                coords = pipeline.sample_sparse_structure(
                    cond,
                    structure_resolution,
                    1,
                    _sampler_params(config.SS_STEPS, config.SS_GUIDANCE),
                )
            finally:
                _STEPS.bind(None, "structure")
            stages["structure_sec"] = time.perf_counter() - mark
            n_voxels = int(coords.shape[0])
            say("structure", f"{n_voxels} active voxels")

            watch.stage = "shape_slat"
            say("shape_slat", f"sampling the shape latent ({n_voxels} voxels)")
            _STEPS.bind(progress, "shape_slat", "sampling the shape latent")
            mark = time.perf_counter()
            try:
                slat, target = _sample_shape(pipeline, cond, image, coords, target, tokens)
            finally:
                _STEPS.bind(None, "shape_slat")
            stages["shape_slat_sec"] = time.perf_counter() - mark

            watch.stage = "decode"
            say("decode", f"decoding to a mesh at {target}")
            mark = time.perf_counter()
            meshes, _subs = pipeline.decode_shape_slat(slat, target)
            stages["decode_sec"] = time.perf_counter() - mark
            gen_sec = time.perf_counter() - started

    extracted = meshes[0]
    mesh = trimesh.Trimesh(
        vertices=extracted.vertices.detach().float().cpu().numpy(),
        faces=extracted.faces.detach().cpu().numpy(),
        process=False,
    )
    mesh, post = _postprocess(mesh, progress)

    return MeshResult(
        mesh=mesh,
        load_sec=_LOAD_SEC,
        gen_sec=gen_sec,
        vram_peak_gb=watch.peak_used_gb,
        vram_over=watch.exceeded,
        fast_attention=_FAST_ATTENTION,
        resolution=target,
        n_voxels=n_voxels,
        seed=int(seed),
        stages={k: round(v, 2) for k, v in stages.items()},
        post=post,
        topology=_topology(mesh),
    )


def _sample_shape(
    pipeline: Any,
    cond: dict[str, Any],
    image: Image.Image,
    coords: torch.Tensor,
    target: int,
    tokens: int,
) -> tuple[Any, int]:
    """Sample the shape latent, through the cascade when the target needs it.

    **The cascade decides its own resolution**: it drops the target in steps of
    128 until the token count fits, so what it settled on comes back with the
    latent rather than being assumed. Measured 2026-09-11: asking for 1536
    yields 1408.
    """
    params = _sampler_params(config.SLAT_STEPS, config.SLAT_GUIDANCE)
    if target <= 1024:
        model = pipeline.models[f"shape_slat_flow_model_{target}"]
        return pipeline.sample_shape_slat(cond, model, coords, params), target

    cond_hr = pipeline.get_cond([image], 1024)
    return pipeline.sample_shape_slat_cascade(
        cond,
        cond_hr,
        pipeline.models["shape_slat_flow_model_512"],
        pipeline.models["shape_slat_flow_model_1024"],
        512,
        target,
        coords,
        params,
        tokens,
    )


def _postprocess(
    mesh: trimesh.Trimesh, progress: Callable[..., None] | None
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Drop the debris and close the holes. **The visibility pass is not used.**

    Upstream's visibility test plus min-cut - the one the TRELLIS.1 runner
    calls - **destroys these meshes**. Measured 2026-09-11 in three
    configurations: on the raw mesh it removed 69% of the faces and 71% of the
    volume, after fixing the winding it removed the same, and on a 700k
    decimated mesh it removed 49% and turned the volume negative. It also costs
    530 s at 512. So it is off, and what remains is cheap: **20 s to drop the
    debris, 2.5 s to close the holes.**
    """
    report: dict[str, Any] = {"faces_before": int(len(mesh.faces))}

    if config.FIX_WINDING:
        mark = time.perf_counter()
        if progress is not None:
            progress("postprocess", "normalizing the winding")
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        report["fix_winding_sec"] = round(time.perf_counter() - mark, 2)

    if config.DROP_SMALL_PARTS > 0:
        mark = time.perf_counter()
        if progress is not None:
            progress("drop_parts", "dropping free-floating debris")
        mesh, dropped = drop_debris(mesh, config.DROP_SMALL_PARTS, config.DROP_THIN_PARTS)
        report.update(dropped)
        report["drop_parts_sec"] = round(time.perf_counter() - mark, 2)

    if config.CLOSE_HOLES:
        mark = time.perf_counter()
        if progress is not None:
            progress("close_holes", "closing the boundary loops")
        mesh, stats = holes.close_holes(mesh)
        report.update(stats.as_dict())
        report["close_holes_sec"] = round(time.perf_counter() - mark, 2)

    report["faces_after"] = int(len(mesh.faces))
    return mesh, report


def drop_debris(
    mesh: trimesh.Trimesh, fraction: float, thinness: float
) -> tuple[trimesh.Trimesh, dict[str, int]]:
    """Drop the parts that are too small or too thin to be part of the model.

    The TRELLIS.1 runner's own `drop_small_parts`, reused rather than rewritten:
    **the debris is the same kind**, only far more of it - 18,898 parts at 512
    and 81,316 at 1024, against a few hundred for TRELLIS.1.
    """
    stats = postprocess.CleanStats(faces_before=int(len(mesh.faces)))
    mesh = postprocess.drop_small_parts(mesh, fraction, thinness, None, stats)
    return mesh, {
        "parts_before": stats.parts_before,
        "parts_after": stats.parts_after,
        "dropped_parts": stats.dropped_parts,
        "dropped_faces": stats.dropped_faces,
    }
