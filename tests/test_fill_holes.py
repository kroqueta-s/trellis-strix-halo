# SPDX-License-Identifier: MIT
"""`runners/trellis/fill_holes.py` が**上流と同じ結果を出す**ことを確かめる。

書き写した実装なので、**上流と突き合わせないと意味がない。**
上流の `_fill_holes` は Python 側の無駄で遅いが、**小さなメッシュなら数秒で済む**ので、
そこで 1 対 1 に比べる。

実行はランナー側の venv で（torch と上流の clone が要る）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import config  # noqa: E402

if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from runners.trellis import fill_holes as ours  # noqa: E402
from runners.trellis import raster, shims  # noqa: E402

VIEWS = 30
RESOLUTION = 256


def _prepare() -> None:
    """上流を import できる状態にして、ラスタライザを差し替える。"""
    shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
    shims.install_absent_nvdiffrast()
    raster.install()
    ours.ensure_upstream_on_path(str(config.TRELLIS_REPO))


def _sample_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """**内側に殻を持つ**小さなメッシュを作る（切るものが無いと比較にならない）。"""
    outer = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    verts = np.concatenate([outer.vertices, inner.vertices], axis=0)
    faces = np.concatenate([outer.faces, inner.faces + len(outer.vertices)], axis=0)
    return (
        torch.tensor(verts, dtype=torch.float32, device="cuda"),
        torch.tensor(faces, dtype=torch.int32, device="cuda"),
    )


def test_visibility_matches_upstream_definition() -> None:
    """**内側の殻は可視率 0**（上流が min-cut の source に使う条件）。"""
    _prepare()
    verts, faces = _sample_mesh()
    vis = ours.visibility(verts, faces, RESOLUTION, VIEWS)
    n_outer = 1280  # icosphere(3) の面数
    assert (vis[:n_outer] > 0).all(), "外の球に見えない面がある"
    assert (vis[n_outer:] == 0).all(), f"内の殻が見えている: {vis[n_outer:].max().item()}"


def test_same_result_as_upstream() -> None:
    """**上流の `_fill_holes` と同じ頂点・面が出る。**

    書き写した実装なので、ここが一致しなければ意味がない。
    """
    _prepare()
    from trellis.utils import postprocessing_utils

    verts, faces = _sample_mesh()
    theirs_v, theirs_f = postprocessing_utils._fill_holes(
        verts.clone(),
        faces.clone(),
        max_hole_size=0.04,
        max_hole_nbe=250,
        resolution=RESOLUTION,
        num_views=VIEWS,
    )
    ours_v, ours_f = ours.fill_holes(
        verts.clone(),
        faces.clone(),
        max_hole_size=0.04,
        max_hole_nbe=250,
        resolution=RESOLUTION,
        num_views=VIEWS,
    )
    assert ours_v.shape == theirs_v.shape, f"頂点数が違う: {ours_v.shape} != {theirs_v.shape}"
    assert ours_f.shape == theirs_f.shape, f"面数が違う: {ours_f.shape} != {theirs_f.shape}"
    assert torch.allclose(ours_v, theirs_v, atol=1e-5), "頂点が一致しない"
    assert torch.equal(ours_f, theirs_f), "面が一致しない"


def test_faster_than_upstream() -> None:
    """**上流より速い**（速くなっていないなら書き写した意味がない）。"""
    import time

    _prepare()
    from trellis.utils import postprocessing_utils

    verts, faces = _sample_mesh()
    t0 = time.perf_counter()
    postprocessing_utils._fill_holes(
        verts.clone(), faces.clone(), resolution=RESOLUTION, num_views=VIEWS, max_hole_nbe=250
    )
    upstream = time.perf_counter() - t0
    t0 = time.perf_counter()
    ours.fill_holes(
        verts.clone(), faces.clone(), resolution=RESOLUTION, num_views=VIEWS, max_hole_nbe=250
    )
    mine = time.perf_counter() - t0
    print(f"       上流 {upstream:.2f}s / こちら {mine:.2f}s")
    assert mine <= upstream * 1.1, f"速くなっていない: 上流 {upstream:.2f}s / こちら {mine:.2f}s"


def main() -> int:
    """全テストを実行する。"""
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
