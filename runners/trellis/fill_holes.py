# SPDX-License-Identifier: MIT
"""上流 TRELLIS の `_fill_holes` を、**手順と閾値をそのままに**書き写した実装。

## なぜ書き写したのか（**独自のアルゴリズムにしたわけではない**）

上流の `trellis/utils/postprocessing_utils.py:_fill_holes` は正しい。ただし
**Python 側に病的な無駄が 1 つ**あり、本機ではそこが支配的になっていた。

```python
g.add_edges([(f, "s") for f in inner_face_indices], ...)   # ← CUDA テンソルを 1 要素ずつ走査
```

実測（2026-09-01・面 697,152・150 視点）：

| 段 | 時間 | 割合 |
|---|--:|--:|
| ラスタライズ 150 視点 | 36.15 s | 44% |
| **source への辺（上の行）** | **21.64 s** | **27%** |
| **target への辺（同じ形の行）** | **13.52 s** | **17%** |
| `g.mincut` 本体 | 5.11 s | 6% |
| その他 | 5.0 s | 6% |

**min-cut そのものは 5 秒**で、遅かったのは 36 万回の GPU 読み出しだった。
一括で CPU へ移せば消える。**アルゴリズムも閾値も一切変えていない**ので、
出力は上流と同じである（`tests/test_fill_holes.py` が小さなメッシュで一致を確認する）。

**上流のヘルパはそのまま呼ぶ**（`utils3d.torch.*` / `sphere_hammersley_sequence` /
`pymeshfix`）。こちらが持っているのは段取りだけである。
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import torch


def _cameras(num_views: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """上流と同じ視点（Hammersley 列・半径 2.0・画角 40 度）を作る。"""
    import utils3d
    from trellis.utils.random_utils import sphere_hammersley_sequence

    yaws, pitchs = [], []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws_t = torch.tensor(yaws, device=device)
    pitchs_t = torch.tensor(pitchs, device=device)
    fov = torch.deg2rad(torch.tensor(40, device=device))
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    views = []
    for yaw, pitch in zip(yaws_t, pitchs_t, strict=True):
        eye = (
            torch.stack(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            ).float()
            * 2.0
        )
        views.append(utils3d.torch.view_look_at(eye, origin, up))
    return torch.stack(views, dim=0), projection


def visibility(
    verts: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    num_views: int,
    progress: Callable[[str, str], None] | None = None,
) -> torch.Tensor:
    """面ごとの可視率を返す（**上流と同じ定義**：見えた視点の割合）。"""
    import utils3d

    views, projection = _cameras(num_views, verts.device)
    seen = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    ctx = utils3d.torch.RastContext(backend="cuda")
    for i in range(views.shape[0]):
        buffers = utils3d.torch.rasterize_triangle_faces(
            ctx, verts[None], faces, resolution, resolution, view=views[i], projection=projection
        )
        face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        seen[torch.unique(face_id).long()] += 1
        if progress is not None and (i + 1) % 50 == 0:
            progress("raster", f"可視率を測る（{i + 1}/{views.shape[0]} 視点）")
    return seen.float() / num_views


def fill_holes(
    verts: torch.Tensor,
    faces: torch.Tensor,
    max_hole_size: float = 0.04,
    max_hole_nbe: int = 250,
    resolution: int = 1024,
    num_views: int = 150,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """見えない面を min-cut で切り、小さな穴を塞ぐ。**上流と同じ手順・同じ閾値。**

    Args:
        verts: `[V, 3]`（cuda）。
        faces: `[F, 3]`（cuda）。
        max_hole_size: 切り口が作る境界ループの面積の上限。これを超える切り方は採らない。
        max_hole_nbe: `pymeshfix` が塞ぐ境界ループの辺数の上限。
        resolution: ラスタライズの解像度。
        num_views: 視点の数。
        progress: 段の通知先。

    Returns:
        `(verts, faces)`。
    """
    import igraph
    import utils3d
    from pymeshfix import _meshfix

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    say("raster", f"可視率を測る（{num_views} 視点 / {resolution}^2）")
    visblity = visibility(verts, faces, resolution, num_views, progress)

    say("graph", "双対グラフを組む")
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    components = utils3d.torch.compute_connected_components(faces, edges, face2edge)

    # 成分ごとに「外側の面」を決める（上流と同じ適応しきい値）。
    outer_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for comp in components:
        threshold = min(max(visblity[comp].quantile(0.75).item(), 0.25), 0.5)
        outer_mask[comp] = visblity[comp] > threshold
    outer_face_indices = outer_mask.nonzero().reshape(-1)
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if inner_face_indices.shape[0] == 0:
        say("graph", "見えない面が無いので切らない")
        return verts, faces

    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)

    n_faces = int(faces.shape[0])
    g = igraph.Graph()
    g.add_vertices(n_faces + 2)  # 末尾の 2 つが source と target
    source, target = n_faces, n_faces + 1
    g.add_edges(dual_edges.cpu().numpy())
    weights = dual_weights.cpu().numpy().tolist()

    # **ここが上流との唯一の違い。** 添字を一括で CPU へ落としてから組む。
    # 上流は CUDA テンソルを Python で 1 要素ずつ走査していて、
    # 36 万回の GPU 読み出しに 35 秒を使っていた。**追加する辺は同じ。**
    inner_list = inner_face_indices.cpu().numpy().tolist()
    outer_list = outer_face_indices.cpu().numpy().tolist()
    g.add_edges([(f, source) for f in inner_list])
    g.add_edges([(f, target) for f in outer_list])
    weights.extend([1.0] * (len(inner_list) + len(outer_list)))

    say("mincut", f"min-cut を解く（内 {len(inner_list):,} / 外 {len(outer_list):,}）")
    capacities = (np.asarray(weights, dtype=np.float64) * 1000).tolist()
    cut = g.mincut(source, target, capacities)
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < n_faces], dtype=torch.long, device=faces.device
    )
    if remove_face_indices.shape[0] == 0:
        say("mincut", "切る面が無かった")
    else:
        remove_face_indices = _validate_cut(
            verts,
            faces,
            edges,
            face2edge,
            boundary_edge_indices,
            visblity,
            remove_face_indices,
            max_hole_size,
        )
        if remove_face_indices.shape[0] > 0:
            keep = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
            keep[remove_face_indices] = False
            faces = faces[keep]
            faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
            say("mincut", f"{int(remove_face_indices.shape[0]):,} 面を切った")

    say("meshfix", f"小さな穴を塞ぐ（辺 {max_hole_nbe} 本まで）")
    fixer = _meshfix.PyTMesh()
    fixer.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    fixer.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    new_verts, new_faces = fixer.return_arrays()
    return (
        torch.tensor(new_verts, device=verts.device, dtype=torch.float32),
        torch.tensor(new_faces, device=faces.device, dtype=torch.int32),
    )


def _validate_cut(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
    face2edge: torch.Tensor,
    boundary_edge_indices: torch.Tensor,
    visblity: torch.Tensor,
    remove_face_indices: torch.Tensor,
    max_hole_size: float,
) -> torch.Tensor:
    """切り口ごとに採否を決める（**上流と同じ 2 条件**）。

    1. その塊の可視率の中央値が 0.25 より大きければ**採らない**（見えている面を切らない）
    2. 新しくできる境界ループの面積が `max_hole_size` を超えるなら**採らない**
       （大きな口を開けない）
    """
    import utils3d

    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    valid: list[torch.Tensor] = []
    for cc in to_remove_cc:
        if visblity[remove_face_indices[cc]].median() > 0.25:
            continue
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary = cc_boundary[~torch.isin(cc_boundary, boundary_edge_indices)]
        if len(cc_new_boundary) > 0:
            loops = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary])
            too_big = False
            for loop in loops:
                pts = verts[edges[cc_new_boundary[loop]]]
                center = pts.mean(dim=1).mean(dim=0)
                e1 = verts[edges[cc_new_boundary[loop]][:, 0]] - center
                e2 = verts[edges[cc_new_boundary[loop]][:, 1]] - center
                area = torch.norm(torch.cross(e1, e2, dim=-1), dim=1).sum() * 0.5
                if area > max_hole_size:
                    too_big = True
                    break
            if too_big:
                continue
        valid.append(cc)
    if not valid:
        return torch.empty(0, dtype=torch.long, device=faces.device)
    return remove_face_indices[torch.cat(valid)]


def ensure_upstream_on_path(repo: str) -> None:
    """上流の clone を import できるようにする（ヘルパを借りるため）。"""
    if repo not in sys.path:
        sys.path.insert(0, repo)
