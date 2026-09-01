# SPDX-License-Identifier: MIT
"""`utils3d.torch` のラスタライザを **torch の Z バッファ**で置き換える。

上流 TRELLIS の後処理（`trellis/utils/postprocessing_utils.py` の `_fill_holes`）は、
**多視点からラスタライズして面ごとの可視率を出し**、可視率 0 の面を min-cut で切る。
これが「内側に閉じ込められた殻」を落とす本家の手法である。**大きさでは切っていない。**

そのラスタライズだけが `nvdiffrast`（CUDA 専用）に依存していて本機では使えない。
ここを差し替えれば、**上流の `postprocess_mesh` を 1 行も書き換えずに実行できる。**

## 実装の方針：**近似せずに三角形を塗る**

最初は「三角形の上に標本点をばら撒いて Z バッファに載せる」実装にしたが、
`tests/test_raster.py` が**大きな三角形で破綻すること**を捕まえた。
標本の隙間から奥の面が漏れて、**箱の中に隠した箱が「見えた」ことになってしまう**。
可視率 0 を内側の面の判定に使う以上、そこが狂うと本家の仕組みが働かない。

そこで、画素の中心が三角形の内側かを**辺関数の符号で厳密に判定**する。
三角形ごとの画面上の外接矩形の大きさで束に分け、束ごとに `K×K` の格子を
まとめて評価する。**この機のメッシュは面がほぼ画素以下**（解像度 256 の格子から出た
100 万面を 1024² へ落とす）なので、ほとんどの束は `K=1`、すなわち 1 画素の判定で済む。

深度は画面空間の重心座標で線形に補間する（NDC の z は画面空間で線形なので、これが正しい）。
"""

from __future__ import annotations

from typing import Any

import torch

# 1 度に評価する「三角形 × 画素」の数の上限。VRAM ピークを決める。
TILE_BUDGET = 16_000_000

# 深度を詰める幅（ビット）。下位 32 ビットは面の番号に使う。
_DEPTH_BITS = 21
_FACE_BITS = 32
_EMPTY = (1 << 62) - 1


class RastContext:
    """`utils3d.torch.RastContext` の代役。**状態を持たない。**

    本家は nvdiffrast の GL / CUDA コンテキストを抱えるが、こちらは純 torch なので
    持つものが無い。引数は受け取って捨てる（呼び出し側を変えないため）。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.backend = kwargs.get("backend", "torch")


def _rasterize(
    screen: torch.Tensor, depth: torch.Tensor, keep: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """画面座標の三角形を Z バッファへ塗り、画素ごとの詰めた鍵を返す。

    Args:
        screen: `[F, 3, 2]` の画面座標（画素単位）。
        depth: `[F, 3]` の NDC の z（-1 が手前・1 が奥）。
        keep: `[F]` の真偽。偽の面は捨てる（近平面より手前など）。
        width: 幅。
        height: 高さ。

    Returns:
        `[H*W]` の `int64`。上位に深度、下位に面の番号。空の画素は `_EMPTY`。
    """
    device = screen.device
    buffer = torch.full((height * width,), _EMPTY, dtype=torch.int64, device=device)
    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if index.numel() == 0:
        return buffer

    tri = screen[index]  # [f, 3, 2]
    lo = tri.amin(dim=1)
    hi = tri.amax(dim=1)
    x0 = lo[:, 0].floor().clamp(0, width - 1).long()
    y0 = lo[:, 1].floor().clamp(0, height - 1).long()
    x1 = hi[:, 0].ceil().clamp(0, width - 1).long()
    y1 = hi[:, 1].ceil().clamp(0, height - 1).long()
    span = torch.maximum(x1 - x0, y1 - y0) + 1  # [f]

    depth_scale = float((1 << _DEPTH_BITS) - 1)
    # 外接矩形の大きさを 2 の冪へ丸めて束にする。**ほとんどは 1 画素。**
    bucket = torch.pow(2, torch.ceil(torch.log2(span.float().clamp(min=1.0)))).long()
    for size in torch.unique(bucket).tolist():
        sel = torch.nonzero(bucket == size, as_tuple=False).squeeze(1)
        per_chunk = max(1, TILE_BUDGET // (size * size))
        for start in range(0, sel.numel(), per_chunk):
            part = sel[start : start + per_chunk]
            faces_here = index[part]
            v = tri[part]  # [n, 3, 2]
            grid = torch.arange(size, device=device)
            n_here = part.numel()
            shape = (n_here, size, size)
            px = (x0[part][:, None, None] + grid[None, :, None]).expand(shape)
            py = (y0[part][:, None, None] + grid[None, None, :]).expand(shape)
            sx = px.to(v.dtype) + 0.5
            sy = py.to(v.dtype) + 0.5

            ax, ay = v[:, 0, 0][:, None, None], v[:, 0, 1][:, None, None]
            bx, by = v[:, 1, 0][:, None, None], v[:, 1, 1][:, None, None]
            cx, cy = v[:, 2, 0][:, None, None], v[:, 2, 1][:, None, None]
            # 辺関数。3 つとも面積と同じ符号なら内側。
            w0 = (cx - bx) * (sy - by) - (cy - by) * (sx - bx)
            w1 = (ax - cx) * (sy - cy) - (ay - cy) * (sx - cx)
            w2 = (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)
            area = w0 + w1 + w2
            sign = torch.where(area >= 0, 1.0, -1.0)
            inside = (w0 * sign >= 0) & (w1 * sign >= 0) & (w2 * sign >= 0) & (area.abs() > 1e-12)
            inside &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not bool(inside.any()):
                continue

            safe_area = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
            z = depth[faces_here]  # [n, 3]
            zz = (
                w0 * z[:, 0][:, None, None]
                + w1 * z[:, 1][:, None, None]
                + w2 * z[:, 2][:, None, None]
            ) / safe_area
            inside &= (zz >= -1.0) & (zz <= 1.0)
            if not bool(inside.any()):
                continue

            pixel = (py * width + px)[inside]
            depth_q = (((zz.clamp(-1.0, 1.0) + 1.0) * 0.5) * depth_scale).long()[inside]
            face_idx = faces_here[:, None, None].expand(shape)[inside]
            # 深度を上位へ、面の番号を下位へ詰めて **最小値をとれば最前面が勝つ**。
            key = (depth_q << _FACE_BITS) | face_idx
            buffer.scatter_reduce_(0, pixel, key, reduce="amin")
    return buffer


def rasterize_triangle_faces(
    ctx: RastContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    attr: torch.Tensor | None = None,
    uv: torch.Tensor | None = None,
    texture: torch.Tensor | None = None,
    model: torch.Tensor | None = None,
    view: torch.Tensor | None = None,
    projection: torch.Tensor | None = None,
    antialiasing: bool | list[int] = True,
    diff_attrs: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """面 ID バッファを返す（`_fill_holes` が使うのは `face_id` と `mask` だけ）。

    Args:
        ctx: 使わない（本家と引数を揃えるためだけに受ける）。
        vertices: `[B, N, 3 or 4]`。**B は 1 のみ対応**（`_fill_holes` は 1 視点ずつ呼ぶ）。
        faces: `[F, 3]`。
        width: 出力の幅。
        height: 出力の高さ。
        view: `[4, 4]` のビュー行列。
        projection: `[4, 4]` の射影行列。

    Returns:
        `face_id`（`[1, H, W]`・**1 始まり。0 は背景**）／`mask`（`[1, H, W]` の float）／
        `depth`（`[1, H, W]`・0 が手前で 1 が奥）。

    Raises:
        NotImplementedError: 属性やテクスチャの補間を求められたとき（この経路では来ない）。
    """
    if attr is not None or uv is not None or texture is not None:
        raise NotImplementedError(
            "この代役は面 ID しか出せない（属性やテクスチャの補間は nvdiffrast が要る）"
        )
    if vertices.ndim != 3 or vertices.shape[0] != 1:
        raise NotImplementedError(f"バッチは 1 のみ対応: {tuple(vertices.shape)}")

    device = vertices.device
    verts = vertices[0].float()
    if verts.shape[-1] == 3:
        verts = torch.cat([verts, torch.ones_like(verts[..., :1])], dim=-1)

    eye = torch.eye(4, device=device, dtype=verts.dtype)
    mvp = projection.float() if projection is not None else eye
    if view is not None:
        mvp = mvp @ view.float()
    if model is not None:
        mvp = mvp @ model.float()
    pos_clip = verts @ mvp.transpose(-1, -2)

    idx = faces.long()
    tri_clip = pos_clip[idx]  # [F, 3, 4]
    w = tri_clip[..., 3]
    # **近平面より手前へ回り込む三角形は捨てる。** 割り算が破綻するため。
    # `_fill_holes` の視点は半径 2・近平面 1 で対象を外から見るので、実際には現れない。
    keep = (w > 1e-6).all(dim=1)
    inv_w = 1.0 / w.clamp(min=1e-6)
    ndc = tri_clip[..., :3] * inv_w.unsqueeze(-1)
    screen = torch.stack(
        [(ndc[..., 0] * 0.5 + 0.5) * width, (ndc[..., 1] * 0.5 + 0.5) * height], dim=-1
    )

    buffer = _rasterize(screen, ndc[..., 2], keep, width, height)

    hit = buffer != _EMPTY
    face_id = torch.zeros(height * width, dtype=torch.int64, device=device)
    face_id[hit] = (buffer[hit] & ((1 << _FACE_BITS) - 1)) + 1
    depth = torch.ones(height * width, dtype=torch.float32, device=device)
    depth[hit] = (buffer[hit] >> _FACE_BITS).float() / float((1 << _DEPTH_BITS) - 1)

    return {
        "face_id": face_id.view(1, height, width),
        "mask": hit.view(1, height, width).float(),
        "depth": depth.view(1, height, width),
    }


def install() -> None:
    """`utils3d.torch` のラスタライザを差し替える。

    **`postprocessing_utils` を import する前に呼ぶこと。** あちらは
    `utils3d.torch.RastContext(...)` と属性で引くので、モジュールの属性を差し替えれば効く。
    """
    import utils3d.torch as u3t

    u3t.RastContext = RastContext
    u3t.rasterize_triangle_faces = rasterize_triangle_faces
