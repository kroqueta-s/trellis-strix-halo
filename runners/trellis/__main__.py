# SPDX-License-Identifier: MIT
"""TRELLIS のランナー（`docs/00_runner_contract.md` の実装）。

**このプロセスだけが torch を持つ。** hearth 本体もアドオンも torch を import しない。

**将来 `trellis-strix-halo` として独立リポジトリへ出す。**
そのため hearth 側のモジュールを一切 import していない（依存は自分の中で閉じている）。
出すときは `.env` の `HEARTH_RUNNER_TRELLIS_CWD` を clone 先へ向けるだけでよい。

起動（通常は hearth が子プロセスとして起こす）::

    & $env:MESHFORGE_WORKER_PYTHON -m runners.trellis
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from . import config, gfxlight

# **torch より先に置かないと効かない**（後から os.environ へ入れても無視される）。
# 立てると gfx1151 で flash / mem-efficient が使えるようになり、実測で 10〜20 倍速い。
# config は dotenv しか読まないので、ここで import しても torch は入ってこない。
if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

NAME = "trellis"
VERSION = "image-large"


# --- プロトコル（契約 §1。hearth の rpc.py と同じ形式だが依存はしない） -------
def install_stdout_guard() -> TextIO:
    """本物の stdout を複製して隠し、**fd 1 ごと stderr へ向ける**。

    **最初に呼ぶこと。** ベンダーコードは平気で print する（契約 §1 の規則 2）。
    `sys.stdout` を差し替えるだけでは足りない。**C 拡張は fd 1 へ直接書く**ので、
    Python 側の差し替えを素通りしてプロトコルの流れに混ざる。
    実測（2026-09-01）：`pymeshfix` が `Loading ..0%` を数百回 fd 1 へ吐いた。

    そこで fd 1 を複製してプロトコル専用に取っておき、**fd 1 自体を fd 2 へ向け直す**。
    これで Python からもネイティブからも、プロトコル以外は必ず stderr へ落ちる。

    Returns:
        プロトコル専用の書き込み先。
    """
    fd = os.dup(1)
    os.dup2(2, 1)  # **fd 1 を stderr へ。C 拡張の出力もこちらへ落ちる。**
    protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = sys.stderr
    return protocol


# **別スレッド（生存確認）からも書くので鍵が要る。**
# 契約 §1 は「1 メッセージ＝1 行の JSON」。混ざると相手の解析が壊れる。
_EMIT_LOCK = threading.Lock()


def emit(out: TextIO, payload: dict[str, Any]) -> None:
    """1 行 1 メッセージで書き出し、必ず flush する。**スレッド安全。**"""
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _EMIT_LOCK:
        out.write(line)
        out.flush()


# --- メソッド ---------------------------------------------------------------
def m_capabilities(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """能力を返す。**重みを読み込まずに即答する**（契約 §3）。"""
    return {
        "name": NAME,
        "version": VERSION,
        "capabilities": {
            "image_to_mesh": True,
            "text_to_mesh": False,
            # 上流には run_multi_image があるが未検証なので申告しない。
            "multi_image_to_mesh": False,
            # テクスチャ段は nvdiffrast（CUDA 専用）が要るので本機では通らない。
            "texture": False,
        },
        "params": {
            "ss_steps": {"type": "int", "default": 25, "min": 1, "max": 100},
            "slat_steps": {"type": "int", "default": 25, "min": 1, "max": 100},
            "ss_guidance": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "slat_guidance": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0},
            "seed": {"type": "int", "default": 0, "min": 0},
        },
        "notes": (
            "spconv / flash_attn / kaolin / open3d は起動側の純 torch シムで置き換えている"
            "（Windows+ROCm に既製品が無い）。アテンションは AOTriton が使えるなら fp16 の"
            "flash、使えないなら fp32＋ヘッド分割へ落ちる。"
            "メッシュは Z-up・正規化スケールで返す。テクスチャは出せない。"
        ),
    }


def m_load(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """重みを読み込む（実測 14 秒前後。初回は dinov2 の取得を含む）。"""
    from . import pipeline

    progress("load", "TRELLIS の重みを読み込む")
    started = time.perf_counter()
    pipeline.load_pipeline(progress)
    return {"loaded": True, "elapsed_sec": round(time.perf_counter() - started, 2)}


def m_unload(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """重みを解放して VRAM を返す。"""
    from . import pipeline

    freed = pipeline.unload_pipeline()
    used_gb, _ = pipeline.device_memory_gb()
    return {"unloaded": freed, "vram_used_gb": round(used_gb, 2)}


_ALLOWED = frozenset({"ss_steps", "slat_steps", "ss_guidance", "slat_guidance", "seed"})


def m_image_to_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """画像 1 枚 → 生のメッシュ（契約 §4・§5）。

    **背景除去は上流のパイプラインが `rembg` で行う。**
    **実寸化はしない。** mm へのスケールは下流（meshforge の forge）の仕事である。
    """
    from PIL import Image

    from . import pipeline

    image_path = Path(str(params["image_path"]))
    out_dir = Path(str(params["out_dir"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"入力画像が無い: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = set(params) - _ALLOWED - {"image_path", "out_dir"}
    if unknown:
        raise ValueError(
            f"知らない引数がある: {sorted(unknown)}（受け付けるのは {sorted(_ALLOWED)}）"
        )

    progress("shape", "3D 形状を生成する（数分かかる）")
    result = pipeline.generate_mesh(
        Image.open(image_path),
        ss_steps=params.get("ss_steps"),
        slat_steps=params.get("slat_steps"),
        ss_guidance=params.get("ss_guidance"),
        slat_guidance=params.get("slat_guidance"),
        seed=int(params.get("seed", 0)),
        progress=progress,
    )

    progress("export", "書き出す")
    mesh_path = out_dir / "raw.ply"
    result.mesh.export(str(mesh_path))

    return {
        "mesh_path": str(mesh_path),
        "n_vertices": int(len(result.mesh.vertices)),
        "n_faces": int(len(result.mesh.faces)),
        "extra": {"up_axis": "z"},
        "metrics": {
            "load_sec": round(result.load_sec, 2),
            # **合否判定に使わない**（契約 §5）。
            "gen_sec": round(result.gen_sec, 2),
            "vram_peak_gb": round(result.vram_peak_gb, 2),
            # **速いアテンションが効いているか。** 効いていないと生成が数倍遅くなる。
            "fast_attention": result.fast_attention,
            # **生成の内訳。** どの段で待っているかが分からないと手が打てない。
            "cond_sec": round(result.cond_sec, 2),
            "structure_sec": round(result.structure_sec, 2),
            "slat_sec": round(result.slat_sec, 2),
            "decode_sec": round(result.decode_sec, 2),
            "n_voxels": result.n_voxels,
            # **後処理で何を落としたか。** 黙って消さないための記録。
            "clean": result.clean,
        },
        "params": {
            "ss_steps": result.ss_steps,
            "slat_steps": result.slat_steps,
            "ss_guidance": result.ss_guidance,
            "slat_guidance": result.slat_guidance,
            "seed": result.seed,
        },
    }


METHODS = {
    "capabilities": m_capabilities,
    "load": m_load,
    "unload": m_unload,
    "image_to_mesh": m_image_to_mesh,
}


def main() -> int:
    """要求を 1 件ずつ直列に処理する。

    Returns:
        終了コード。正常終了は 0。
    """
    out = install_stdout_guard()
    print(f"[{NAME}] ランナーを起動した。", file=sys.stderr)

    for raw in sys.stdin:
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = int(request["id"])
            method_name = str(request["method"])
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[{NAME}] 解析できない要求を読み飛ばした: {exc}", file=sys.stderr)
            continue

        if method_name == "shutdown":
            emit(out, {"id": request_id, "event": "result", "result": {"bye": True}})
            break

        method = METHODS.get(method_name)
        if method is None:
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": "ValueError", "message": f"知らないメソッド: {method_name}"},
                },
            )
            continue

        def progress(stage: str, message: str = "", _id: int = request_id) -> None:
            emit(out, {"id": _id, "event": "progress", "stage": stage, "message": message})

        # **3D の常夜灯**（gfxlight.py）。compute だけだとドライバがクロックを上げない。
        light: gfxlight.GfxLight | None = None
        if method_name == "image_to_mesh" and config.GFX_KEEPALIVE:
            light = gfxlight.GfxLight()
            light.start()
        try:
            result = method(dict(request.get("params") or {}), progress)
            if light is not None and isinstance(result.get("metrics"), dict):
                # 生成の終わりまで点いていたか。False なら効いていない可能性がある。
                result["metrics"]["gfx_keepalive"] = light.is_lit()
            emit(out, {"id": request_id, "event": "result", "result": result})
        except Exception as exc:  # noqa: BLE001 - 何が来ても応答を返しきる
            import traceback

            traceback.print_exc()
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        finally:
            if light is not None:
                light.stop()

    print(f"[{NAME}] ランナーを終了する。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
