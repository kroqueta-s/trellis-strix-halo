# SPDX-License-Identifier: MIT
"""生成したメッシュの後処理。**上流の手法に倣い、独自の判断を足さない。**

上流 TRELLIS は `to_glb()` の中で `postprocess_mesh()` を呼ぶ。その中身は

1. `pyvista.decimate(0.95)`（面を 5% まで間引く）
2. `_fill_holes`：**多視点ラスタライズ → 面ごとの可視率 → min-cut → 小さな穴埋め**

**ここでは 2 だけを、上流のコードをそのまま呼んで実行する。**
1 の間引きは掛けない。上流は色付き GLB を出すのが目的で面数を削るが、こちらは
下流（`forge`）が実寸化と修復をする前提の素材なので、細部を捨てる理由が無い。

足りないのは `nvdiffrast`（CUDA 専用）のラスタライザだけなので、`raster.install()` で
`utils3d.torch` の該当関数を差し替えてから呼ぶ。**ベンダーコードは書き換えていない。**

## 上流に無い処理を 1 つだけ足している

**外側に浮いている小さな破片を、大きさで落とす。**（`.env` の `*_DROP_SMALL_PARTS`）

上流にこの手当ては無い。可視率で切る仕組みは「見えない面」を狙うので、**空中に浮いた
破片は見えてしまい、通り抜ける**。実測（2026-09-01・検体 `i2i_00038_.png`）：

- trellis：本体以外 831 個のうち、**内側 37 個 118,000 面（77.1%）／外側 794 個 35,064 面**
- hi3dgen：本体以外 968 個のうち、内側 27 個 3,008 面／**外側 941 個 30,048 面（90.9%）**

上流は間引き（0.95）で成分が 832 → 204 まで減るだけで、**完成品にも浮遊片は残る**。
印刷用途では実害になるので、ここだけ独自の処理を足す。**落とした量は必ず記録に残す。**
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import trimesh

from . import config, raster, shims


@dataclass
class CleanStats:
    """後処理で何がどれだけ変わったかの記録。**黙って消さないための数字。**"""

    faces_before: int = 0
    faces_after: int = 0
    fill_holes_sec: float = 0.0
    fill_holes_removed: int = 0
    parts_before: int = 0
    parts_after: int = 0
    dropped_parts: int = 0
    dropped_faces: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """`metrics` へそのまま載せられる形。"""
        return {
            "faces_before": self.faces_before,
            "faces_after": self.faces_after,
            "fill_holes_sec": round(self.fill_holes_sec, 2),
            "fill_holes_removed": self.fill_holes_removed,
            "parts_before": self.parts_before,
            "parts_after": self.parts_after,
            "dropped_parts": self.dropped_parts,
            "dropped_faces": self.dropped_faces,
            "warnings": self.warnings,
        }


def _say(progress: Callable[[str, str], None] | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)


def fill_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """**上流の `postprocess_mesh` をそのまま呼ぶ**（間引きは掛けない）。

    Args:
        vertices: `[V, 3]`。
        faces: `[F, 3]`。
        progress: 段の通知先。
        stats: 記録の置き場。

    Returns:
        後処理後の `(vertices, faces)`。**失敗したら入力をそのまま返す**
        （後処理は品質の改善であって、生成の成否ではない）。
    """
    import time

    repo = str(config.TRELLIS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    started = time.perf_counter()
    before = len(faces)
    try:
        # **`trellis` を import する前にシムを揃える。** 生成を通らずに後処理だけを
        # 呼ぶ道（既存のメッシュへ掛け直すとき）もあるので、ここでも念のため入れる。
        if "trellis" not in sys.modules:
            shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
        # nvdiffrast は本機に無い。**触ったら落ちる殻**を置いてから import する
        # （`postprocessing_utils` は先頭で import するが、使うのはテクスチャ焼き込みだけ）。
        shims.install_absent_nvdiffrast()
        raster.install()
        from trellis.utils import postprocessing_utils

        _say(
            progress,
            "fill_holes",
            f"見えない面を落とす（{config.FILL_HOLES_VIEWS} 視点 / "
            f"{config.FILL_HOLES_RESOLUTION}^2・面 {before:,}・上流の実装）",
        )
        # **上流の関数は中で数十秒〜数分黙る。** 心拍を出しておかないと
        # 「進んでいるのか止まっているのか」が外から分からない（T0 の方針）。
        from .pipeline import _DeviceWatch

        with _DeviceWatch(
            progress=progress,
            stage="後処理",
            heartbeat_sec=config.HEARTBEAT_SEC,
            limit_gb=config.VRAM_LIMIT_GB,
        ):
            vertices, faces = postprocessing_utils.postprocess_mesh(
                vertices,
                faces,
                simplify=False,  # **間引かない**（下流が実寸化するので細部を残す）
                fill_holes=True,
                fill_holes_max_hole_size=config.FILL_HOLES_MAX_SIZE,
                fill_holes_max_hole_nbe=config.FILL_HOLES_MAX_NBE,
                fill_holes_resolution=config.FILL_HOLES_RESOLUTION,
                fill_holes_num_views=config.FILL_HOLES_VIEWS,
            )
    except Exception as exc:  # noqa: BLE001 - 後処理の失敗で生成を落とさない
        message = f"上流の後処理に失敗した（メッシュはそのまま返す）: {type(exc).__name__}: {exc}"
        print(f"[postprocess] {message}", file=sys.stderr)
        if stats is not None:
            stats.warnings.append(message)
        return vertices, faces
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if stats is not None:
        stats.fill_holes_sec = time.perf_counter() - started
        stats.fill_holes_removed = before - len(faces)
    return vertices, faces


def drop_small_parts(
    mesh: trimesh.Trimesh,
    min_ratio: float,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> trimesh.Trimesh:
    """**外側に浮いた小さな破片を落とす**（上流に無い、こちらの追加）。

    判定は「その成分の外接箱の最長辺 ÷ 全体の最長辺」。面数ではなく**空間の大きさ**で
    見るのは、細かく分割された小片と、面数が少ないだけの正当な部品を分けるため。

    実測（検体 `i2i_00038_.png`）では、しきい値 10% で腕と手（全体の 15%）は残り、
    目に見える破片（6.5% 以下）は消えた。

    Args:
        mesh: 対象。
        min_ratio: 残す最小の大きさ（全体の最長辺に対する比）。0 以下なら何もしない。
        progress: 段の通知先。
        stats: 記録の置き場。

    Returns:
        破片を除いたメッシュ。**最大の成分だけは必ず残す。**
    """
    if min_ratio <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if stats is not None:
        stats.parts_before = len(parts)
    if len(parts) <= 1:
        if stats is not None:
            stats.parts_after = len(parts)
        return mesh

    whole = float(np.max(mesh.bounding_box.extents))
    sizes = np.array([float(np.max(p.bounding_box.extents)) for p in parts])
    face_counts = np.array([len(p.faces) for p in parts])
    keep = sizes / max(whole, 1e-12) >= min_ratio
    keep[int(np.argmax(face_counts))] = True  # 最大の成分は必ず残す

    if stats is not None:
        stats.parts_after = int(keep.sum())
        stats.dropped_parts = int((~keep).sum())
        stats.dropped_faces = int(face_counts[~keep].sum())
    _say(
        progress,
        "drop_parts",
        f"浮いた破片を落とす（{int((~keep).sum())} 個 / {int(face_counts[~keep].sum())} 面）",
    )
    if not (~keep).any():
        return mesh
    return trimesh.util.concatenate([p for p, k in zip(parts, keep, strict=True) if k])


def clean(
    mesh: trimesh.Trimesh,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[trimesh.Trimesh, CleanStats]:
    """後処理をまとめて掛ける。

    Returns:
        `(後処理後のメッシュ, 記録)`。
    """
    stats = CleanStats(faces_before=len(mesh.faces))

    if config.FILL_HOLES:
        vertices, faces = fill_holes(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            progress,
            stats,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh = drop_small_parts(mesh, config.DROP_SMALL_PARTS, progress, stats)
    stats.faces_after = len(mesh.faces)
    return mesh, stats
