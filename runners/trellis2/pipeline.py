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
from runners.trellis import postprocess, shims, split_manifold
from runners.trellis.steps import StepCounter, count_tqdm

from . import config, shell, texture

NAME = "trellis2"
VERSION = "4B"

_PIPELINE: Any = None
_LOAD_SEC: float = 0.0
_FAST_ATTENTION: bool = False
# Whether the compiled mesh -> dual grid conversion loaded (native/o_voxel_cpu/).
_NATIVE_O_VOXEL: bool = False
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

    global _FAST_ATTENTION, _NATIVE_O_VOXEL
    _FAST_ATTENTION = shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
    shims.install_trellis2(close_mesh=config.CLOSE_MESH)
    # **Optional, and after the stand-in exists.** The compiled conversion is
    # only there when the operator built it (native/o_voxel_cpu/).
    _NATIVE_O_VOXEL = shims.install_o_voxel_cpu(config.NATIVE_DIR)
    print(
        f"[trellis2] o_voxel_cpu {'loaded' if _NATIVE_O_VOXEL else 'not built'}",
        file=sys.stderr,
    )


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
    vertex_colors: dict[str, Any] = field(default_factory=dict)
    # **The textured mesh is a different mesh.** Its atlas is built before the
    # manifold conversion, so it carries that stage's geometry rather than the
    # one `mesh` ends up with. `None` when no texture was asked for.
    textured: trimesh.Trimesh | None = None
    texture: dict[str, Any] = field(default_factory=dict)


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
    boundary, non_manifold = split_manifold.count_non_manifold(mesh)
    return {
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume": float(mesh.volume),
    }


def generate_mesh(
    image: Image.Image,
    resolution: int | None = None,
    seed: int = 0,
    max_tokens: int | None = None,
    target_faces: int | None = None,
    vertex_colors: bool | None = None,
    texture: bool | None = None,
    texture_size: int | None = None,
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
    want_texture = config.TEXTURE if texture is None else bool(texture)
    want_colors = config.VERTEX_COLORS if vertex_colors is None else bool(vertex_colors)
    # **Both read the same voxels**, so asking for a texture is enough to make
    # the texture stage run; the vertex colours are then nearly free.
    want_voxels = want_colors or want_texture
    atlas = int(texture_size or config.TEXTURE_SIZE)

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
            # **The substructures are what the texture decoder is guided by.**
            # They are held only while they are needed, because at 1024 they are
            # the largest thing on the card after the weights.
            meshes, subs = pipeline.decode_shape_slat(slat, target)
            stages["decode_sec"] = time.perf_counter() - mark

            voxels = None
            if want_voxels:
                watch.stage = "tex_slat"
                say("tex_slat", "sampling the texture latent")
                _STEPS.bind(progress, "tex_slat", "sampling the texture latent")
                mark = time.perf_counter()
                try:
                    tex_slat = _sample_texture(pipeline, cond, slat, target)
                finally:
                    _STEPS.bind(None, "tex_slat")
                stages["tex_slat_sec"] = time.perf_counter() - mark

                watch.stage = "tex_decode"
                say("tex_decode", "decoding the texture latent into voxel colours")
                mark = time.perf_counter()
                voxels = pipeline.decode_tex_slat(tex_slat, subs)[0]
                stages["tex_decode_sec"] = time.perf_counter() - mark
                del tex_slat
            del subs, slat
            gen_sec = time.perf_counter() - started

    extracted = meshes[0]
    mesh = trimesh.Trimesh(
        vertices=extracted.vertices.detach().float().cpu().numpy(),
        faces=extracted.faces.detach().cpu().numpy(),
        process=False,
    )
    # **The post-processing needs a heartbeat too.** At 1024 it runs for
    # minutes, and a caller watching for liveness cannot tell a long stage from
    # a stuck one: the switch test gave up on exactly this gap (2026-09-12).
    query = _colour_query(voxels, pipeline, target) if voxels is not None else None
    with _DeviceWatch(progress=progress, stage="postprocess"):
        mesh, post, textured, bake_report = _postprocess(
            mesh, progress, target_faces, query if want_texture else None, atlas
        )

        colors: dict[str, Any] = {"enabled": bool(want_colors)}
        if voxels is not None and want_colors:
            mark = time.perf_counter()
            say("vertex_colors", f"colouring {len(mesh.vertices):,} vertices")
            mesh, colors = _apply_vertex_colors(mesh, query)
            colors["sec"] = round(time.perf_counter() - mark, 2)
        del voxels, query
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        vertex_colors=colors,
        textured=textured,
        texture=bake_report,
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


def _sample_texture(pipeline: Any, cond: dict[str, Any], slat: Any, target: int) -> Any:
    """Sample the texture latent on the shape latent's own voxels.

    **The conditioning is the one already computed.** Upstream feeds the texture
    flow the 512 conditioning at 512 and the 1024 one everywhere above, which is
    exactly `get_cond([image], min(target, 1024))` - the vector this runner
    already has.

    There is one texture flow per resolution and, as with the shape flow, only
    512 and 1024 exist; above 1024 the cascade still ends on the 1024 model.
    """
    name = "tex_slat_flow_model_512" if target <= 512 else "tex_slat_flow_model_1024"
    model = pipeline.models.get(name)
    if model is None:
        raise FileNotFoundError(
            f"{name} is not in the pipeline: the texture checkpoints were not downloaded. "
            f"Run install-trellis2.ps1 -WithTexture, or turn TRELLIS2_VERTEX_COLORS off"
        )
    return pipeline.sample_tex_slat(
        cond, model, slat, _sampler_params(config.TEX_STEPS, config.TEX_GUIDANCE)
    )


def _colour_query(voxels: Any, pipeline: Any, resolution: int) -> Callable[..., torch.Tensor]:
    """A function from positions to base colour. **The one place either path asks.**

    The texture decoder produces an attribute vector per active voxel, and a
    point in space is trilinearly interpolated from the eight around it. That
    the colour follows the *position* rather than the surface is the whole
    reason both the vertex colours and the bake can run wherever they like in
    the post-processing chain.

    Upstream reaches the same sampler through `MeshWithVoxel.query_attrs`; this
    calls it directly, because the bake has no mesh to hang it on - it asks
    about a texel's position, which lies on a triangle rather than at a vertex.

    The attributes carry metallic, roughness and alpha as well
    (`pipeline.pbr_attr_layout`). **Only the base colour is kept**: neither a
    PLY nor a plain glTF texture has anywhere to put the rest, and inventing a
    place for it would be guessing at what a caller wants.
    """
    base = pipeline.pbr_attr_layout["base_color"]
    shape = torch.Size([*voxels.shape, *voxels.spatial_shape])
    # `origin` is -0.5 and `voxel_size` is 1/resolution, as upstream sets them
    # in `decode_latent`, so this is the same mapping into voxel units.
    scale = float(resolution)

    def query(points: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid = ((points.to(voxels.feats.device) + 0.5) * scale).reshape(1, -1, 3)
            attrs = shims._grid_sample_3d(voxels.feats, voxels.coords, shape, grid)[0]
        return attrs[:, base].clamp(0, 1).float()

    return query


def _apply_vertex_colors(
    mesh: trimesh.Trimesh, query: Callable[..., torch.Tensor]
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Carry the decoded voxel colours onto the mesh's vertices.

    **This runs after the post-processing, not before.** Decimation, hole
    closing and manifolding can all rewrite the vertices first and the colours
    still land, because `_colour_query` asks about a position. A UV layout could
    not survive any of those, which is why `texture.bake` runs earlier instead.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    base = query(vertices)
    rgb = (base.float().cpu().numpy() * 255).round().astype(np.uint8)
    opaque = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, opaque], axis=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    # **A vertex outside every active voxel comes back black**, because there is
    # nothing around it to interpolate. Counting them says how much of the mesh
    # the texture decoder never saw, which is the only way to tell a black model
    # from a failed one.
    unreached = int((base.abs().sum(dim=1) == 0).sum())
    return mesh, {
        "enabled": True,
        "n_vertices": int(len(mesh.vertices)),
        "unreached_vertices": unreached,
        "mean_rgb": [round(float(v), 4) for v in base.float().mean(dim=0).tolist()],
    }


def decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Reduce to `target` faces with quadric edge collapse.

    **Placed before everything else on purpose.** Every later step costs in
    proportion to the face count, and this one costs almost nothing: measured
    2026-09-11, 3.33 M faces to 700 k in 3.3 s, for a mean error of 0.17% of the
    longest side and a volume within 0.9%.

    **It does not damage the topology here** - the opposite, measured on the
    same mesh: non-manifold edges 15,860 -> 10,097 and boundary edges
    118,777 -> 26,959, because the slivers collapse. That is worth stating
    because the usual worry about decimation is that it breaks a manifold; what
    it cannot do is *make* one.
    """
    import fast_simplification

    vertices, faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), target_count=int(target)
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _postprocess(
    mesh: trimesh.Trimesh,
    progress: Callable[..., None] | None,
    target_faces: int | None = None,
    bake_query: Callable[..., torch.Tensor] | None = None,
    texture_size: int = 2048,
) -> tuple[trimesh.Trimesh, dict[str, Any], trimesh.Trimesh | None, dict[str, Any]]:
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

    # **First, because everything below is proportional to the face count.**
    target = int(config.TARGET_FACES if target_faces is None else target_faces)
    if 0 < target < len(mesh.faces):
        mark = time.perf_counter()
        if progress is not None:
            progress("decimate", f"reducing {len(mesh.faces):,} faces to {target:,}")
        mesh = decimate(mesh, target)
        report["decimate_to"] = target
        report["decimate_sec"] = round(time.perf_counter() - mark, 2)
    elif target > 0:
        # **Asked for more faces than there are.** Saying so beats silently
        # doing nothing, because the caller's budget was not met for a reason.
        report["decimate_skipped"] = f"already at {len(mesh.faces)} faces, target {target}"

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
        mesh, stats = holes.close_holes(mesh, max_extent=config.CLOSE_MAX_EXTENT)
        report.update(stats.as_dict())
        report["close_holes_sec"] = round(time.perf_counter() - mark, 2)

    # **The bake goes here, and the position is the whole design.** A UV atlas
    # belongs to the vertices and faces it was built for, so it has to be made
    # after everything that rewrites them and before anything else does. What
    # comes next is `make_manifold`, which adds patch worth 2.2-2.5x the input
    # surface area (measured 2026-09-12): an atlas built after it would spend
    # about 69% of its texels on internal membrane nobody ever sees.
    textured: trimesh.Trimesh | None = None
    bake_report: dict[str, Any] = {"enabled": bake_query is not None}
    if bake_query is not None:
        mark = time.perf_counter()
        if progress is not None:
            progress("texture", f"unwrapping and baking a {texture_size}x{texture_size} texture")
        # **A coarser mesh than the one that gets printed**, because the
        # unwrap - not the bake - is what costs, and the texture carries the
        # detail the triangles no longer do.
        target = int(config.TEXTURE_TARGET_FACES)
        surface = decimate(mesh, target) if 0 < target < len(mesh.faces) else mesh
        textured, bake_report = texture.bake(
            surface, bake_query, texture_size, config.TEXTURE_DILATE
        )
        bake_report["faces"] = int(len(surface.faces))
        bake_report["enabled"] = True
        report["texture_sec"] = round(time.perf_counter() - mark, 2)

    # **The print mesh is a solid made from the surface, not the surface
    # sewn shut.** The decoder's output is a thin, incomplete, double-walled
    # skin (see `shell`), so sewing it gives a manifold that encloses a tenth
    # of the silhouette's volume with a quarter of it wound inside-out. Carving
    # the exterior out by visibility keeps the outer surface where the model
    # put it and fills everything behind it.
    if config.SHELL:
        mark = time.perf_counter()
        if progress is not None:
            what = (
                f"carving the exterior by visibility ({config.SHELL_VISIBILITY} of 98 rays)"
                if config.SHELL_MODE == "carve"
                else f"thickening into a wall {config.SHELL_THICKNESS:.4f} of the longest side"
            )
            progress("shell", f"{what} on a {config.SHELL_GRID} grid")
        mesh, shell_report = shell.solidify(
            mesh,
            grid=config.SHELL_GRID,
            mode=config.SHELL_MODE,
            visibility=config.SHELL_VISIBILITY,
            thickness=config.SHELL_THICKNESS,
            fill_cavities=config.SHELL_FILL_CAVITIES,
        )
        report["shell"] = shell_report.as_dict()
        report["shell_sec"] = round(time.perf_counter() - mark, 2)
        # The offset surface has about as many faces as the input; the same
        # budget applies to it.
        if 0 < target < len(mesh.faces):
            mark = time.perf_counter()
            if progress is not None:
                progress(
                    "decimate", f"reducing the shell's {len(mesh.faces):,} faces to {target:,}"
                )
            mesh = decimate(mesh, target)
            report["shell_decimate_sec"] = round(time.perf_counter() - mark, 2)

    if config.MAKE_MANIFOLD:
        mark = time.perf_counter()
        if progress is not None:
            progress("manifold", "separating the sheets and closing the seams")
        mesh, manifold_report = split_manifold.make_manifold(mesh)
        report["manifold"] = manifold_report
        report["manifold_sec"] = round(time.perf_counter() - mark, 2)

    report["faces_after"] = int(len(mesh.faces))
    return mesh, report, textured, bake_report


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
