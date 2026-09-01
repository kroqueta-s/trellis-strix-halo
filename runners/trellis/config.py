# SPDX-License-Identifier: MIT
"""Configuration for the TRELLIS runner, read from `.env`.

**This runner is self-contained.** It never reads hearth's configuration, so it
works unchanged as the standalone `trellis-strix-halo` repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/trellis/config.py -> the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw.strip() if raw is not None and raw.strip() != "" else default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw.strip() != "" else default


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw.strip() != "" else default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# The upstream clone (**never a fork**; the plain clone is used as-is).
TRELLIS_REPO: Path = Path(_str("TRELLIS_REPO"))
# Where the weights live (the directory holding `pipeline.json`).
TRELLIS_WEIGHTS_DIR: Path = Path(_str("TRELLIS_WEIGHTS_DIR"))

# Upstream defaults, from `pipeline.json`: 25 steps and cfg 5.0.
SS_STEPS: int = _int("TRELLIS_SS_STEPS", 25)
SLAT_STEPS: int = _int("TRELLIS_SLAT_STEPS", 25)
SS_GUIDANCE: float = _float("TRELLIS_SS_GUIDANCE", 5.0)
SLAT_GUIDANCE: float = _float("TRELLIS_SLAT_GUIDANCE", 5.0)

# Attention heads computed at once. Measured best on Hunyuan3D at 4 on gfx1151.
# **Do not change it without evidence.**
ATTN_HEAD_CHUNK: int = _int("TRELLIS_ATTN_HEAD_CHUNK", 4)

# Whether to set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL **before torch is
# imported**. Measured 2026-09-01 on gfx1151: it makes the flash and
# memory-efficient kernels available, taking seq=4096 from 0.135s to 0.012s and
# seq=9216 from 1.167s to 0.059s, with identical output. **Setting it afterwards
# has no effect**, so it goes at the top of `__main__.py`.
FAST_ATTENTION: bool = _bool("TRELLIS_FAST_ATTENTION", True)

# Whether to run the clock keepalive during generation (`gfxlight.py`). The AMD
# Windows driver does not raise the clock for compute-only work (measured: GEMM
# alone 600 MHz, with 3D alongside 2.35 GHz, a 4.3x difference). Generation
# works as before if it fails to start; `metrics.gfx_keepalive` records whether
# it was alive.
GFX_KEEPALIVE: bool = _bool("TRELLIS_GFX_KEEPALIVE", True)


# **Cap on dedicated VRAM (GB).** gfx1151 has 32 GB of dedicated VRAM, but the
# total from `torch.cuda.mem_get_info` is 43.87 GB because it counts shared
# memory. Overflow therefore raises nothing and **silently becomes several times
# slower** (measured on 2026-09-01). Passing the cap to torch as well turns that
# into an **immediate OOM**.
VRAM_LIMIT_GB: float = _float("TRELLIS_VRAM_LIMIT_GB", 30.0)

# Heartbeat interval in seconds. It exists so that **nothing runs silently for a
# long time**.
HEARTBEAT_SEC: float = _float("TRELLIS_HEARTBEAT_SEC", 10.0)


# --- Post-processing (following upstream's postprocess_mesh) ------------------
# **Whether to run upstream's `_fill_holes`**: their method of rasterizing many
# views, computing a visibility ratio per face, and cutting the faces with zero
# visibility (shells sealed inside) with a min-cut.
FILL_HOLES: bool = _bool("TRELLIS_FILL_HOLES", True)

# View count and resolution. **Upstream defaults to 1000 views at 1024^2, which
# takes 244 seconds on this machine** (measured on a 697,152-face mesh); 150
# views take about 37 seconds. TRELLIS-AMD also dropped to 100 views and
# reported no visible difference.
FILL_HOLES_VIEWS: int = _int("TRELLIS_FILL_HOLES_VIEWS", 150)
FILL_HOLES_RESOLUTION: int = _int("TRELLIS_FILL_HOLES_RESOLUTION", 1024)
FILL_HOLES_MAX_SIZE: float = _float("TRELLIS_FILL_HOLES_MAX_SIZE", 0.04)
# **Upstream's default of 32 is not enough here.** Upstream decimates to 0.95
# first, so its boundary loops have few edges; this runner does not decimate, so
# the same hole has several times as many (measured: a 146-vertex loop survived
# and watertightness was lost). At 250 there are zero boundary loops and the
# mesh is watertight again.
FILL_HOLES_MAX_NBE: int = _int("TRELLIS_FILL_HOLES_MAX_NBE", 250)

# **Added on top of upstream.** Drop free-floating debris by size: the fraction
# of the model's longest side below which a component's longest bounding-box
# extent is discarded. 0 disables it. Measured: at 10 % the arms and hands
# (15 %) survive and the visible debris (6.5 % and below) is gone.
DROP_SMALL_PARTS: float = _float("TRELLIS_DROP_SMALL_PARTS", 0.10)

# **Also added on top of upstream.** Drop detached components whose bounding-box
# minimum extent is below this fraction of the model's longest side (paper-thin
# flakes). Flakes hovering about 1 % off the surface are 11-29 % long, so
# DROP_SMALL_PARTS alone lets them through. Measured 2026-09-02: flakes are
# 1.4 % thick or less, genuine parts 11.8 % or more. 0 disables it.
DROP_THIN_PARTS: float = _float("TRELLIS_DROP_THIN_PARTS", 0.02)
