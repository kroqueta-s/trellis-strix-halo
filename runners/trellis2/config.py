# SPDX-License-Identifier: MIT
"""Settings for the TRELLIS.2 runner. **`.env` is the authority; nothing is named here.**

Every default that came from a measurement says so, and the measurement is
dated. A default that is only a guess says that too.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    return str(os.environ.get(key, default) or default)


def _int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def _float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


def _bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _path(key: str) -> Path:
    return Path(_str(key)).expanduser()


# Where upstream is cloned and where the weights are. **Both absolute**: the
# runner is started as its own process, from its own directory.
TRELLIS2_REPO: Path = _path("TRELLIS2_REPO")
WEIGHTS_DIR: Path = _path("TRELLIS2_WEIGHTS_DIR")

# The pipeline description to load. **The downloaded `pipeline.json` is never
# edited**; this one sits beside it and names local files only, dropping the
# texture models and pointing the conditioner and the background remover at
# directories on this machine.
PIPELINE_CONFIG: str = _str("TRELLIS2_PIPELINE_CONFIG", "pipeline.local.json")

# Where `native/o_voxel_cpu/build.ps1` put the compiled mesh -> dual grid
# conversion. **Optional**: empty means it was not built, and nothing on the
# image-to-mesh path needs it. The runner says on stderr whether it loaded.
NATIVE_DIR: str = _str("TRELLIS2_NATIVE_DIR", "")

# Output resolution. **Measured 2026-09-11** on assets/sample.png: 512 takes
# 49.6 s and 3.45 M faces, 1024 takes 209.3 s and 14.86 M faces, and asking for
# 1536 gets 1408 back (upstream's own token limit) for 1602.6 s and 26.24 M
# faces. 1024 is the operating point: 1408 costs eight times the time for 1.8x
# the faces.
RESOLUTION: int = _int("TRELLIS2_RESOLUTION", 1024)

# The token budget upstream uses to decide how far the cascade can climb.
# **Upstream's own default**; lowering it lowers the resolution it settles on.
MAX_TOKENS: int = _int("TRELLIS2_MAX_TOKENS", 49152)

# Sampler settings. **Upstream's pipeline.json defaults** (12 steps, guidance
# 7.5 for both stages); leaving them at 0 means "whatever the checkpoint says".
SS_STEPS: int = _int("TRELLIS2_SS_STEPS", 0)
SLAT_STEPS: int = _int("TRELLIS2_SLAT_STEPS", 0)
SS_GUIDANCE: float = _float("TRELLIS2_SS_GUIDANCE", 0.0)
SLAT_GUIDANCE: float = _float("TRELLIS2_SLAT_GUIDANCE", 0.0)

# The texture flow has **its own** sampler settings upstream (12 steps, guidance
# 1.0), so it gets its own keys rather than borrowing the shape ones: changing
# how the geometry is sampled should not quietly change the colours too.
TEX_STEPS: int = _int("TRELLIS2_TEX_STEPS", 0)
TEX_GUIDANCE: float = _float("TRELLIS2_TEX_GUIDANCE", 0.0)

# Attention head chunk, used only when the fast attention path is unavailable.
ATTN_HEAD_CHUNK: int = _int("TRELLIS2_ATTN_HEAD_CHUNK", 4)

# Set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 before torch is imported.
# **Measured 2026-09-10 on gfx1151 in both fp16 and bf16**: flash runs 15x
# faster than the math backend (1.61 ms against 24.90 ms at H8/S4096/D64) and
# agrees with it to four decimals.
FAST_ATTENTION: bool = _bool("TRELLIS2_FAST_ATTENTION", True)

# Prefer hipBLASLt over rocBLAS (also before torch is imported). Measured on the
# TRELLIS.1 runner: the sparse-conv shim's skinny GEMM goes from 1.0 to 14
# TFLOPS. `metrics.blas_backend` records which one served.
PREFER_HIPBLASLT: bool = _bool("TRELLIS2_PREFER_HIPBLASLT", True)

# AMD_LOG_LEVEL for the HIP runtime (before torch is imported): 0 silent,
# 1 errors, 2 warnings, 3 information, 4 debug. **A kernel fault is reported
# asynchronously, at the next synchronization, and when a second failure hits
# during the unwinding the process aborts without a message** - seen once at
# 1024 (2026-09-12), inside `nonzero`, with nothing to say what failed. At 1
# the runtime names the failing call and the error on stderr, which hearth
# keeps; the cost is four lines at start-up (measured 2026-09-12). -1 leaves
# the environment alone.
HIP_LOG_LEVEL: int = _int("TRELLIS2_HIP_LOG_LEVEL", 1)

# Dedicated VRAM is 32 GB; `torch.cuda.mem_get_info` reports 43.87 GB because it
# counts shared memory. Without a cap, spilling into shared memory is silent and
# several times slower, so the limit is passed to torch as well. **Measured
# peaks: 6.74 GB at 512, 15.42 GB at 1024, 28.40 GB at 1408** - the last leaves
# only 1.6 GB under this cap.
VRAM_LIMIT_GB: float = _float("TRELLIS2_VRAM_LIMIT_GB", 30.0)
HEARTBEAT_SEC: float = _float("TRELLIS2_HEARTBEAT_SEC", 10.0)

# Sample the texture latent as well, and carry its colours onto the mesh's
# vertices. **This is not a texture map**: the decoder produces one colour per
# active voxel and every vertex is interpolated from the eight around it, so the
# colours survive decimation, hole closing and manifolding - none of which keeps
# a UV layout. A texture map needs UV unwrapping and a bake, which is separate
# work.
#
# **Off by default** because it costs a second 1.3B flow and a second decoder
# pass; `metrics.vertex_colors` reports what it cost when it is on.
VERTEX_COLORS: bool = _bool("TRELLIS2_VERTEX_COLORS", False)

# Unwrap the surface and bake the colours into a texture map as well. **This is
# a different thing from vertex colours**: the colour lives in an image, so its
# resolution is the texture's rather than the mesh's, and detail finer than a
# triangle survives. It needs the same texture checkpoints.
#
# **It produces a second mesh**, in `extra.textured_glb`. A UV atlas is bound to
# the vertices it was built for, so the bake happens before `make_manifold`
# rewrites them - `mesh_path` still holds the manifold geometry, and the GLB
# holds the surface the texture was made for.
#
# **On by default.** Measured 2026-09-12 on a detailed mecha: the projection
# charts put 95.6% of the faces in charts of 50 or more, and the decals read
# across the seams. The unwrap and bake cost 8.8 s at 512; the texture latent
# and its decoder add 50% to a 1024 generation (+78.6 s) and nothing to VRAM.
# `mesh_path` is unchanged either way, so what this buys is a second file to
# look at, not a different thing to print.
TEXTURE: bool = _bool("TRELLIS2_TEXTURE", True)

# The texture is this many pixels square. **The number that matters is texels
# per triangle**: 2048 gives 4.19 M texels, which on a 1.5 M-face mesh is 2.8
# per triangle - only 5.5x what the vertices already carried. The gain is real
# after a heavier decimation, or at 4096 (16.8 M texels).
TEXTURE_SIZE: int = _int("TRELLIS2_TEXTURE_SIZE", 2048)

# Reduce to this many faces **before unwrapping**. The textured GLB is a thing
# to look at; `mesh_path` is the thing to print, and it keeps every face.
#
# **This is what makes the unwrap affordable.** Measured 2026-09-12 on the mesh
# at the bake point: 13.0 s at 100 k faces, 31.9 s at 200 k, and the full
# 1.34 M was still running after ten minutes. The atlas is what carries the
# detail here, not the triangles - 200 k faces under a 2048 texture is 21 texels
# per triangle, against 2.8 at 1.5 M. 0 unwraps the mesh as it stands.
TEXTURE_TARGET_FACES: int = _int("TRELLIS2_TEXTURE_TARGET_FACES", 200_000)

# How many times the face normals are averaged over the adjacency before the
# chart direction is read off them (`charts.project_charts`). **Measured
# 2026-09-12** on the carved 512 mecha at 200,000 faces: 3 rounds put 64% of the
# faces in charts of fifty or more, 10 put 69%, 30 put 73% - diminishing
# returns against a growing blur of genuinely different directions.
CHART_SMOOTHING: int = _int("TRELLIS2_CHART_SMOOTHING", 10)

# Charts smaller than this are merged into the neighbour they share the most
# edges with. **A chart of five triangles carries no picture**, which is exactly
# what xatlas produced when it chose the charts itself (50,052 of them from
# 200,000 faces). 50 is the size at which a chart is worth a seam.
CHART_MIN_FACES: int = _int("TRELLIS2_CHART_MIN_FACES", 50)

# A face whose own normal points away from its chart's axis folds over in
# projection. It is moved to a neighbouring chart it does not fold in; when
# none takes it, it gets a chart of its own if its area is at least this many
# times the median face's, and is otherwise left folded, reading the atlas
# but not writing it. **Measured 2026-09-12** on the 200 k specimen: 4,250
# folded faces (2.1%), 1,215 with a neighbour to go to; of the rest, 73 are
# at least twice the median face (0.08% of the area) and 2,962 are slivers
# (0.4%). A chart for every one would have doubled the chart count.
CHART_FOLD_AREA: float = _float("TRELLIS2_CHART_FOLD_AREA", 2.0)

# Rounds of spreading the colours past the edge of each chart. **Without it a
# renderer filtering across a seam pulls in the empty background and draws a
# black line along every cut.** 4 is enough for bilinear filtering at any
# reasonable mip level; it stops early when nothing is left to fill.
TEXTURE_DILATE: int = _int("TRELLIS2_TEXTURE_DILATE", 4)


def texture_weights_present() -> bool:
    """Whether the pipeline description names a texture flow. **Reads no weights.**

    `capabilities` has to answer at once (contract §3), and what decides this is
    whether the texture checkpoints were downloaded - `write_local_pipeline.py`
    keeps those entries only when their files are there. So the description is
    read, not the 2.6 GB behind it.
    """
    import json

    description = WEIGHTS_DIR / PIPELINE_CONFIG
    if not description.is_file():
        return False
    try:
        models = json.loads(description.read_text(encoding="utf-8"))["args"]["models"]
    except (ValueError, KeyError, OSError):
        return False
    return any(name.startswith("tex_slat_flow_model") for name in models)


def encoder_weights_present() -> bool:
    """Whether the mesh-to-latent encoder is downloaded. **Reads no weights.**

    Only `texture_mesh` needs it, and it is not in the pipeline description, so
    it is checked as a file rather than through the models table.
    """
    return (WEIGHTS_DIR / "ckpts" / "shape_enc_next_dc_f16c32_fp16.safetensors").is_file()


def native_o_voxel_present() -> bool:
    """Whether the compiled mesh -> dual grid conversion was built.

    `texture_mesh` cannot start without it: upstream's
    `mesh_to_flexible_dual_grid` is C++ with no torch equivalent here, unlike
    the extraction the other way, which the shims reimplement. **This looks for
    the file**; whether it loads is settled at import time.
    """
    if not NATIVE_DIR:
        return False
    directory = Path(NATIVE_DIR).expanduser()
    return directory.is_dir() and any(directory.glob("o_voxel_cpu*.pyd"))


# --- Post-processing -------------------------------------------------------
# Decimate to this many faces **before anything else is done to the mesh**.
# 0 turns it off and the model's own tessellation is kept.
#
# **Everything after it is proportional to the face count**, so this is what
# makes the post-processing affordable. Measured 2026-09-12 at 512: closing the
# holes takes 24.2 s on 3.45 M faces and **2.5 s on 700 k**, and dropping the
# debris 19.2 s against a few seconds. Decimation itself costs **3.3 s** and
# improves the topology on the way (non-manifold edges 15,860 -> 10,097,
# boundary edges 118,777 -> 26,959) for a mean error of **0.17%** of the longest
# side.
#
# The default is the operator's staged target: a light first pass to 1-3 M,
# with the rest of the reduction downstream.
TARGET_FACES: int = _int("TRELLIS2_TARGET_FACES", 1_500_000)

# Drop free-floating parts smaller than this fraction of the longest side.
# **Measured 2026-09-11**: at 512 that is 18,893 parts and 121,716 faces (3.5%),
# at 1024 it is 81,313 parts and 651,488 faces (4.4%). Set to 0 to keep them.
DROP_SMALL_PARTS: float = _float("TRELLIS2_DROP_SMALL_PARTS", 0.10)

# Drop parts thinner than this fraction of the longest side. **The threshold is
# the TRELLIS.1 runner's measured one** (flakes there came out 0.1-1.4% thick
# and genuine parts 11.8% or more), adopted rather than re-measured: the flakes
# here are thinner still - 18,834 of them under 0.5% at 512 (2026-09-11).
DROP_THIN_PARTS: float = _float("TRELLIS2_DROP_THIN_PARTS", 0.02)

# Close the boundary loops. **Measured 2026-09-12**: 26,959 boundary edges fall
# to 3,123 in 2.5 s on a 700k mesh. What remains is the boundary non-manifold
# edges produce, which this cannot close.
CLOSE_HOLES: bool = _bool("TRELLIS2_CLOSE_HOLES", True)

# The widest loop still treated as a hole, as a fraction of the longest side.
# **A fan over a wide loop is a sail, not a repair.** Measured 2026-09-12 at
# 512: closing every loop covers **13.4% of the surface area** with patch,
# while capping here closes 15,135 of 15,352 loops for **3.3%**. The
# distribution has its knee at this value (99th percentile extent 0.0658).
# 0 closes everything. `metrics.post.fan_area_fraction` reports what it cost.
CLOSE_MAX_EXTENT: float = _float("TRELLIS2_CLOSE_MAX_EXTENT", 0.05)

# Make a solid out of the surface before the manifold conversion. **The
# decoder's output is a thin, incomplete, double-walled skin, not a solid**
# (measured 2026-09-12: its cross-sections are open arcs, and no flood from
# outside can tell the hollow from the air), and a manifold sewn out of it
# encloses a tenth of the silhouette's volume. `metrics.post.shell` reports
# what the solid took and holds.
SHELL: bool = _bool("TRELLIS2_SHELL", True)

# How the solid is decided. `carve` keeps the outer surface exactly where the
# model put it and fills everything behind it: a corner is air only when it can
# see the outside, unblocked, in enough of 98 directions. `band` makes every
# point within half a wall of the surface solid, which guarantees a wall but
# grows the silhouette by half of it and rounds off detail narrower than the
# wall. Measured 2026-09-12 on the 512 mecha: carve keeps the hydraulics and
# track links that a 3 mm band rounds away, and its rays take a few seconds
# on the GPU.
SHELL_MODE: str = _str("TRELLIS2_SHELL_MODE", "carve")

# Carving: air must see the outside in at least this many of 98 directions.
# **Measured 4**: a bowl twice as deep as it is wide keeps its hollow up to 4
# and starts to fill at 6, while the specimen's interior is filled from 2 on
# (its solid moves by 0.002 between 2 and 4).
SHELL_VISIBILITY: int = _int("TRELLIS2_SHELL_VISIBILITY", 4)

# Band only: the wall, as a fraction of the longest side. **The runner does
# not know millimetres**: 0.0375 is 3 mm on an 80 mm print.
SHELL_THICKNESS: float = _float("TRELLIS2_SHELL_THICKNESS", 0.0375)

# Cells along the longest side for the lattice. Memory is about lattice^3 x a
# dozen bytes: 512 peaks near 2.5 GB; 1024 would be eight times that.
SHELL_GRID: int = _int("TRELLIS2_SHELL_GRID", 512)

# Band only: fill every enclosed pocket. Carving always fills them - a print
# wants a solid, and `forge.hollow` is where a hollow one is made.
SHELL_FILL_CAVITIES: bool = _bool("TRELLIS2_SHELL_FILL_CAVITIES", True)

# Carve only: a piece of solid detached from the rest and smaller than this
# many lattice corners is air. **The rays leave specks in the shadows**
# (measured 2026-09-12 at 512: 254 detached pieces, 213 of them one corner,
# 2-11 cells from the body), and each speck is a part the manifold repair pays
# for. 64 corners is 4 cells across - 0.6 mm on an 80 mm print at 512 - and
# dropping everything under it leaves the volume unchanged to five digits.
# 0 keeps every piece. `metrics.post.shell.islands_dropped` counts them.
SHELL_ISLAND_CORNERS: int = _int("TRELLIS2_SHELL_ISLAND_CORNERS", 64)

# Separate the touching sheets, cut what cannot be oriented, and close the
# seams, so that the mesh is a closed orientable manifold. **After the shell
# this only separates surface-nets sheets that touch along an edge** and
# closes nothing (measured 2026-09-12: `fan_area_fraction` 0.0). **This is what makes
# `manifold3d` - and therefore forge's `repair_manifold` - accept it**:
# measured 2026-09-12, `Error.NoError`, 1,517,778 triangles, genus 1207, where
# without it manifold3d refuses and forge reports "repairing did not make this
# watertight".
#
# **It is not free.** The seams it opens have to be closed again, and the
# patches add **2.5x the input surface area** (`post.manifold.close`). Most of
# that is internal - the silhouette and the detail survive, checked by eye -
# but a caller who wants only what the model produced should turn this off.
MAKE_MANIFOLD: bool = _bool("TRELLIS2_MAKE_MANIFOLD", True)

# **Off, because it costs more than the generation.** `fix_winding` +
# `fix_normals` take 270 s on the 3.45 M-face mesh at 512 (measured 2026-09-11)
# and 66 s at 700k, against 49.6 s to generate it. The mesh comes out with
# **49.2% of its faces wound the other way** - inherent to a dual grid whose
# flags carry no inside/outside - so this is worth doing **after** decimation,
# downstream, not here. `metrics.winding_consistent` reports the state either way.
FIX_WINDING: bool = _bool("TRELLIS2_FIX_WINDING", False)

# Close the sparse region's edge during extraction. **Measured 2026-09-11: it
# changes almost nothing** (830 faces, boundary 111,454 -> 111,528), because
# only ~415 quads are dropped for want of a neighbour. Kept as a switch because
# it costs nothing and the next model may differ.
CLOSE_MESH: bool = _bool("TRELLIS2_CLOSE_MESH", True)
