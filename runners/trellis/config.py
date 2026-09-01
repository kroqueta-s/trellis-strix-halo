# SPDX-License-Identifier: MIT
"""TRELLIS ランナーの設定（`.env` から読み込む）。

**このランナーは自分の中で閉じている。** hearth の設定を参照しないので、
`trellis-strix-halo` として独立リポジトリへ出しても、そのまま動く。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/trellis/config.py -> リポジトリのルート。
# 独立リポジトリへ出したときも、同じ位置関係になるよう配置する。
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


# 上流の clone（**フォークしない**。素の clone をそのまま使う）。
TRELLIS_REPO: Path = Path(_str("TRELLIS_REPO"))
# 重みの置き場（`pipeline.json` があるディレクトリ）。
TRELLIS_WEIGHTS_DIR: Path = Path(_str("TRELLIS_WEIGHTS_DIR"))

# 上流の既定は `pipeline.json` の 25 / 5.0。
SS_STEPS: int = _int("TRELLIS_SS_STEPS", 25)
SLAT_STEPS: int = _int("TRELLIS_SLAT_STEPS", 25)
SS_GUIDANCE: float = _float("TRELLIS_SS_GUIDANCE", 5.0)
SLAT_GUIDANCE: float = _float("TRELLIS_SLAT_GUIDANCE", 5.0)

# アテンションを何ヘッドずつ計算するか。gfx1151 では Hunyuan3D 実測で 4 が最良だった。
# **根拠なく変えない。**
ATTN_HEAD_CHUNK: int = _int("TRELLIS_ATTN_HEAD_CHUNK", 4)

# **torch を import する前に** TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL を立てるか。
# 実測（2026-09-01・gfx1151）：立てると flash / mem-efficient が使えるようになり、
# seq=4096 で 0.135s -> 0.012s、seq=9216 で 1.167s -> 0.059s。出力は一致する。
# **後から os.environ へ入れても効かない**ので、`__main__.py` の先頭で置く。
FAST_ATTENTION: bool = _bool("TRELLIS_FAST_ATTENTION", True)


# **専用 VRAM の上限（GB）。** gfx1151 の専用 VRAM は 32GB だが、
# `torch.cuda.mem_get_info` の total は共有メモリ込みの 43.87GB を返す。
# そのため溢れても例外にならず、**黙って数倍遅くなる**（2026-09-01 に実測で踏んだ）。
# ここを torch にも伝えて、超えたら OOM で**すぐ落ちる**ようにする。
VRAM_LIMIT_GB: float = _float("TRELLIS_VRAM_LIMIT_GB", 30.0)

# 生存確認を流す間隔（秒）。**黙って長時間走らせない**ためのもの。
HEARTBEAT_SEC: float = _float("TRELLIS_HEARTBEAT_SEC", 10.0)


# --- 後処理（上流の postprocess_mesh に倣う） -------------------------------
# **上流の `_fill_holes` を掛けるか。** 多視点ラスタライズで可視率を出し、
# 可視率 0 の面（内側に閉じ込められた殻）を min-cut で落とす本家の手法。
FILL_HOLES: bool = _bool("TRELLIS_FILL_HOLES", True)

# 視点数と解像度。**本家の既定は 1000 視点 / 1024^2 だが、本機では 244 秒かかる**
# （面 697,152 での実測）。150 視点なら約 37 秒。TRELLIS-AMD も 100 視点へ落として
# 「見た目に区別がつかない」と報告している。
FILL_HOLES_VIEWS: int = _int("TRELLIS_FILL_HOLES_VIEWS", 150)
FILL_HOLES_RESOLUTION: int = _int("TRELLIS_FILL_HOLES_RESOLUTION", 1024)
FILL_HOLES_MAX_SIZE: float = _float("TRELLIS_FILL_HOLES_MAX_SIZE", 0.04)
# **上流の既定 32 では足りない。** 上流は先に 0.95 で間引くので境界ループの辺が少ないが、
# こちらは間引かないので同じ穴でも辺が数倍多くなる（実測：最大 146 頂点のループが残り、
# watertight が崩れた）。250 にすると境界ループ 0 本・watertight に戻ることを確認した。
FILL_HOLES_MAX_NBE: int = _int("TRELLIS_FILL_HOLES_MAX_NBE", 250)

# **上流に無い、こちらの追加。** 外側に浮いた破片を大きさで落とす。
# 成分の外接箱の最長辺が全体の何割未満なら捨てるか。0 で無効。
# 実測では 10% で腕と手（15%）は残り、目に見える破片（6.5% 以下）が消えた。
DROP_SMALL_PARTS: float = _float("TRELLIS_DROP_SMALL_PARTS", 0.10)
