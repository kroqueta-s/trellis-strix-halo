# SPDX-License-Identifier: MIT
"""メッシュの健全性を数値で出す（**見た目の印象を数字で裏取りするための道具**）。

測るのは 3 つ：**連結成分**（浮いている破片）・**境界ループ**（穴）・寸法。
**これまでのチェックは連結成分と穴を見ておらず、破片と穴を素通しさせた。**
描画の点描（`tools/render_mesh.py` は点をばら撒くので、輪郭がまばらだと
小さな白い塊に見えることがある）と、**本当に浮いている破片**は別物である。

実行例::

    python tools/mesh_report.py mesh.ply [mesh2.ply ...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def boundary_loops(mesh: trimesh.Trimesh) -> list[dict[str, object]]:
    """境界辺（面が 1 枚しか接していない辺）をループごとにまとめる。

    **穴が「1 か所の大きな口」なのか「細かい割れの散らばり」なのかで手当てが変わる。**
    数だけ見ても判断できないので、ループごとの大きさと位置まで出す。
    """
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen: set[int] = set()
    loops: list[dict[str, object]] = []
    scale = float(np.max(mesh.bounding_box.extents))
    for start in list(adj):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            seen.add(v)
            stack.extend(adj[v])
        pts = mesh.vertices[list(comp)]
        extent = pts.max(0) - pts.min(0)
        loops.append(
            {
                "verts": len(comp),
                "extent": np.round(extent, 4).tolist(),
                "ratio": float(np.max(extent)) / max(scale, 1e-12),
                "center": np.round(pts.mean(0), 3).tolist(),
            }
        )
    return sorted(loops, key=lambda item: -int(item["verts"]))


def report(path: Path, top: int = 8) -> dict[str, object]:
    """1 個のメッシュを調べて表示し、要点を返す。"""
    mesh = trimesh.load(path, process=False)
    extents = mesh.bounding_box.extents
    parts = mesh.split(only_watertight=False)
    sizes = np.array([len(p.faces) for p in parts])
    order = np.argsort(sizes)[::-1]

    print(f"\n=== {path.name} ===")
    print(f"  頂点 {len(mesh.vertices):,} / 面 {len(mesh.faces):,}")
    print(f"  watertight={mesh.is_watertight}  外形={np.round(extents, 4).tolist()}")
    print(f"  連結成分 {len(parts)} 個")

    main_faces = int(sizes[order[0]]) if len(parts) else 0
    stray = int(sizes.sum() - main_faces)
    print(f"  最大の成分が全体の {main_faces / max(len(mesh.faces), 1) * 100:.2f}%")
    print(f"  それ以外の面数 {stray:,}（{stray / max(len(mesh.faces), 1) * 100:.2f}%）")

    loops = boundary_loops(mesh)
    if loops:
        stray_edges = sum(int(loop["verts"]) for loop in loops)
        print(f"  境界ループ {len(loops)} 本（境界の頂点 {stray_edges}）")
        for loop in loops[:top]:
            print(
                f"    頂点 {loop['verts']:>5}  大きさ {loop['extent']}  "
                f"最長辺が全体の {float(loop['ratio']) * 100:6.2f}%  中心 {loop['center']}"
            )
    else:
        print("  境界ループ 0 本（穴なし）")

    for rank, i in enumerate(order[: min(top, len(parts))]):
        part = parts[i]
        size = part.bounding_box.extents
        longest = float(np.max(size)) / float(np.max(extents))
        print(
            f"    #{rank + 1}: 面 {len(part.faces):>8,}  "
            f"最長辺が全体の {longest * 100:6.2f}%  中心 {np.round(part.centroid, 3).tolist()}"
        )
    return {
        "path": str(path),
        "components": len(parts),
        "main_face_ratio": main_faces / max(len(mesh.faces), 1),
        "stray_faces": stray,
        "boundary_loops": len(loops),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="メッシュの連結成分を調べる")
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    for m in args.meshes:
        report(Path(m), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
