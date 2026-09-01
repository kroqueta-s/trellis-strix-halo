# SPDX-License-Identifier: MIT
"""ラスタライザ代役の検算（**推論に使う前に、既知の形で数値を合わせる**）。

上流 TRELLIS の後処理は「面ごとの可視率」で内側の殻を落とす。可視率が狂うと
**落ちずに残る**か**必要な面まで落ちる**ので、ここで確かめておく。

確かめること：

- 外向きの箱を回りから見れば、**すべての面がいつかは見える**
- 箱の**中に隠した小さな箱**は、**どの視点からも見えない**（可視率 0）
- 手前の面が奥の面を隠す（Z バッファが効いている）

torch が要るので**ランナー側の venv で動かす**。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import raster  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _box(center: tuple[float, float, float], size: float) -> tuple[np.ndarray, np.ndarray]:
    """軸に沿った箱の頂点と面（外向き）を返す。"""
    c = np.array(center, dtype=np.float32)
    h = size / 2.0
    verts = (
        np.array(
            [
                [-1, -1, -1],
                [+1, -1, -1],
                [+1, +1, -1],
                [-1, +1, -1],
                [-1, -1, +1],
                [+1, -1, +1],
                [+1, +1, +1],
                [-1, +1, +1],
            ],
            dtype=np.float32,
        )
        * h
        + c
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -z
            [4, 5, 6],
            [4, 6, 7],  # +z
            [0, 1, 5],
            [0, 5, 4],  # -y
            [3, 7, 6],
            [3, 6, 2],  # +y
            [0, 4, 7],
            [0, 7, 3],  # -x
            [1, 2, 6],
            [1, 6, 5],  # +x
        ],
        dtype=np.int64,
    )
    return verts, faces


def _perspective(fov_deg: float, near: float, far: float) -> torch.Tensor:
    """正方画角の射影行列（`utils3d.torch.perspective_from_fov_xy` と同じ形）。"""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = torch.zeros(4, 4, device=DEVICE, dtype=torch.float32)
    m[0, 0] = f
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """ビュー行列。"""
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.cross(right, forward, dim=0)
    m = torch.eye(4, device=DEVICE, dtype=torch.float32)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -torch.stack([right @ eye, true_up @ eye, -(forward @ eye)])
    m[2, 3] = forward @ eye
    return m


def _visibility(
    verts: np.ndarray, faces: np.ndarray, views: int = 60, res: int = 256
) -> np.ndarray:
    """球面上の視点から見て、面ごとの可視率を返す。"""
    v = torch.tensor(verts, device=DEVICE, dtype=torch.float32)[None]
    f = torch.tensor(faces, device=DEVICE, dtype=torch.int64)
    projection = _perspective(40.0, 1.0, 10.0)
    ctx = raster.RastContext()
    counts = torch.zeros(faces.shape[0], device=DEVICE, dtype=torch.int32)
    up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    center = torch.zeros(3, device=DEVICE)
    for i in range(views):
        # 黄金角でおおよそ一様に散らす。
        yaw = 2.0 * math.pi * i * 0.618033988749895
        pitch = math.asin(2.0 * (i + 0.5) / views - 1.0)
        eye = (
            torch.tensor(
                [
                    math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                ],
                device=DEVICE,
            )
            * 3.0
        )
        if torch.allclose(torch.nn.functional.normalize(eye, dim=0).abs(), up.abs(), atol=1e-3):
            continue  # 真上・真下は view 行列が縮退する
        buffers = raster.rasterize_triangle_faces(
            ctx, v, f, res, res, view=_look_at(eye, center, up), projection=projection
        )
        seen = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        counts[torch.unique(seen).long()] += 1
    return (counts.float() / views).cpu().numpy()


def test_outer_box_is_fully_visible() -> None:
    """外向きの箱は、回りから見ればすべての面がどこかで見える。"""
    verts, faces = _box((0.0, 0.0, 0.0), 1.0)
    vis = _visibility(verts, faces)
    assert (vis > 0).all(), f"見えない面がある: {vis}"


def test_hidden_box_is_never_visible() -> None:
    """**箱の中に隠した箱はどの視点からも見えない**（可視率 0）。

    上流はこの「可視率 0」を内側の面として min-cut の source に使う。
    ここが 0 にならないと、内殻を落とす仕組みが働かない。
    """
    outer_v, outer_f = _box((0.0, 0.0, 0.0), 1.0)
    inner_v, inner_f = _box((0.0, 0.0, 0.0), 0.4)
    verts = np.concatenate([outer_v, inner_v], axis=0)
    faces = np.concatenate([outer_f, inner_f + len(outer_v)], axis=0)
    vis = _visibility(verts, faces)
    outer_vis, inner_vis = vis[: len(outer_f)], vis[len(outer_f) :]
    assert (outer_vis > 0).all(), f"外の箱に見えない面がある: {outer_vis}"
    assert (inner_vis == 0).all(), f"隠した箱が見えてしまった: {inner_vis}"


def test_z_buffer_prefers_the_near_face() -> None:
    """手前の面が奥の面を隠す（同じ画素で最前面が勝つ）。"""
    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],  # 奥
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],  # 手前
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    v = torch.tensor(verts, device=DEVICE)[None]
    f = torch.tensor(faces, device=DEVICE)
    eye = torch.tensor([0.0, 0.0, 5.0], device=DEVICE)
    view = _look_at(
        eye, torch.zeros(3, device=DEVICE), torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )
    buffers = raster.rasterize_triangle_faces(
        raster.RastContext(), v, f, 128, 128, view=view, projection=_perspective(40.0, 1.0, 10.0)
    )
    seen = torch.unique(buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1).tolist()
    assert seen == [1], f"手前の面だけが見えるはず: {seen}"


def test_background_is_zero() -> None:
    """何も無い画素は `face_id` が 0（背景）になる。"""
    verts, faces = _box((0.0, 0.0, 0.0), 0.2)
    v = torch.tensor(verts, device=DEVICE)[None]
    f = torch.tensor(faces, device=DEVICE)
    eye = torch.tensor([0.0, 0.0, 3.0], device=DEVICE)
    view = _look_at(
        eye, torch.zeros(3, device=DEVICE), torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )
    buffers = raster.rasterize_triangle_faces(
        raster.RastContext(), v, f, 128, 128, view=view, projection=_perspective(40.0, 1.0, 10.0)
    )
    mask = buffers["mask"][0]
    assert mask.sum() > 0, "何も写っていない"
    assert (buffers["face_id"][0][mask < 0.5] == 0).all(), "背景が 0 になっていない"


def main() -> int:
    """全テストを実行する。"""
    print(f"device: {DEVICE}")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 成功")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
