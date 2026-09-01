# SPDX-License-Identifier: MIT
"""`drop_small_parts`（小ささ＋薄さ）の検算。

合成メッシュで「本体・正当な部品・小さな破片・表面の薄片」を作り、
**残るべきものが残り、消えるべきものが消える**ことを確かめる。

実行はこの venv で（trimesh が要る。torch は postprocess の import が引く）::

    .venv\\Scripts\\python.exe .\\tests\\test_drop_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = " OK " if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def build_specimen() -> tuple[trimesh.Trimesh, int]:
    """本体＋部品＋破片＋薄片。返り値は (メッシュ, 残るべき面数)。"""
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # 正当な部品：大きさ 20%・厚み 20%（腕のつもり）
    limb = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    limb.apply_translation((0.8, 0.0, 0.0))
    # 小さな破片：大きさ 5%（従来の基準で消える）
    crumb = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    crumb.apply_translation((0.0, 0.8, 0.0))
    # 表面の薄片：長さ 30%・厚み 0.5%（従来の基準を素通りしていたゴミ）
    flake = trimesh.creation.box(extents=(0.3, 0.3, 0.005))
    flake.apply_translation((0.0, 0.0, 0.51))
    mesh = trimesh.util.concatenate([body, limb, crumb, flake])
    return mesh, len(body.faces) + len(limb.faces)


def run(label: str, drop_small_parts) -> None:  # noqa: ANN001
    mesh, want_faces = build_specimen()
    out = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.02)
    parts = out.split(only_watertight=False)
    check(f"{label}: 本体と部品の 2 成分が残る", len(parts) == 2, f"got {len(parts)}")
    check(f"{label}: 面数が本体＋部品と一致する", len(out.faces) == want_faces)

    # 薄さ判定を切れば薄片は残る（0 で無効の約束）
    out2 = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.0)
    check(
        f"{label}: min_thick_ratio=0 なら薄片は残る",
        len(out2.split(only_watertight=False)) == 3,
    )

    # 両方 0 なら何もしない
    out3 = drop_small_parts(mesh, min_ratio=0.0, min_thick_ratio=0.0)
    check(f"{label}: 両方 0 なら何もしない", len(out3.faces) == len(mesh.faces))

    # 本体が薄くても最大成分は必ず残る
    thin_body = trimesh.creation.box(extents=(1.0, 1.0, 0.01))
    out4 = drop_small_parts(thin_body.copy(), min_ratio=0.10, min_thick_ratio=0.02)
    check(f"{label}: 最大成分は必ず残る", len(out4.faces) == len(thin_body.faces))


def main() -> int:
    from runners.trellis import postprocess as tre

    run("trellis", tre.drop_small_parts)

    total = 5
    print(f"\n{total - len(FAILURES)}/{total} 成功")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
