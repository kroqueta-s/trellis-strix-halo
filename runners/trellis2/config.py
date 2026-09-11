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

# Dedicated VRAM is 32 GB; `torch.cuda.mem_get_info` reports 43.87 GB because it
# counts shared memory. Without a cap, spilling into shared memory is silent and
# several times slower, so the limit is passed to torch as well. **Measured
# peaks: 6.74 GB at 512, 15.42 GB at 1024, 28.40 GB at 1408** - the last leaves
# only 1.6 GB under this cap.
VRAM_LIMIT_GB: float = _float("TRELLIS2_VRAM_LIMIT_GB", 30.0)
HEARTBEAT_SEC: float = _float("TRELLIS2_HEARTBEAT_SEC", 10.0)

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
