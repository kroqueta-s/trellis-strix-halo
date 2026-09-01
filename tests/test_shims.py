# SPDX-License-Identifier: MIT
"""TRELLIS 用シムの検算（**推論を回す前に、ここで数値を合わせる**）。

実行は**ランナー側の venv で**（torch が要る）。TRELLIS の venv の python に
このファイルを渡す。hearth 本体の venv では torch が無いので動かない。

**このテストは「動いた」ではなく「同じ数が出る」を確かめる。**

- submanifold 畳み込みは「活性でないボクセルを 0 とした密な `F.conv3d`」と**厳密に一致する**。
  だから spconv が無くても**参照実装を自前で作れる**。重みの並び（KRSC）と畳み込みの向きを
  取り違えると、ここで必ず落ちる
- `flash_attn` の代替は素朴なアテンションと突き合わせる

torch が要るので**ランナー側の venv で動かす**（hearth 本体の venv では動かない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import shims  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _random_coords(n: int, res: int, batch: int = 1) -> torch.Tensor:
    """重複しない `[N, 4]` の座標を作る。"""
    g = torch.Generator().manual_seed(0)
    seen: set[tuple[int, int, int, int]] = set()
    while len(seen) < n:
        need = n - len(seen)
        raw = torch.randint(0, res, (need * 2, 3), generator=g)
        b = torch.randint(0, batch, (need * 2, 1), generator=g)
        for row in torch.cat([b, raw], dim=1).tolist():
            seen.add(tuple(row))
            if len(seen) >= n:
                break
    return torch.tensor(sorted(seen), dtype=torch.int32, device=DEVICE)


def _dense_reference(
    coords: torch.Tensor, feats: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, res: int
) -> torch.Tensor:
    """密な `F.conv3d` で submanifold 畳み込みの正解を作る。

    活性でないボクセルを 0 で埋めた密なボリュームに普通の畳み込みを掛け、
    活性ボクセルの位置だけ取り出せば submanifold 畳み込みそのものになる。
    """
    batch = int(coords[:, 0].max().item()) + 1
    out_c, k, _, _, in_c = weight.shape
    vol = torch.zeros(batch, res, res, res, in_c, device=feats.device, dtype=torch.float32)
    idx = coords.long()
    vol[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats.float()
    vol = vol.permute(0, 4, 1, 2, 3)  # [B, Cin, D, H, W]
    # spconv の KRSC [out, kD, kH, kW, in] を torch の [out, in, kD, kH, kW] へ
    w = weight.permute(0, 4, 1, 2, 3).float()
    dense = F.conv3d(vol, w, bias=bias.float(), padding=k // 2)
    return dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]


def test_submanifold_conv_matches_dense_conv() -> None:
    """カーネル 3 の submanifold 畳み込みが密な畳み込みと一致する。"""
    shims._install_spconv()
    res, n, in_c, out_c = 16, 400, 8, 6
    coords = _random_coords(n, res)
    feats = torch.randn(n, in_c, device=DEVICE)
    conv = shims._SubMConv3dImpl(in_c, out_c, 3, indice_key="t").to(DEVICE)
    torch.nn.init.normal_(conv.weight, std=0.2)
    torch.nn.init.normal_(conv.bias, std=0.1)

    x = shims.SparseConvTensor(feats, coords, [res, res, res], 1)
    got = conv(x).features
    want = _dense_reference(coords, feats, conv.weight.data, conv.bias.data, res)
    err = (got - want).abs().max().item()
    assert err < 2e-4, f"密な畳み込みと一致しない: 最大誤差 {err}"


def test_submanifold_conv_kernel1() -> None:
    """カーネル 1（skip connection）も一致する。"""
    shims._install_spconv()
    res, n, in_c, out_c = 12, 200, 5, 7
    coords = _random_coords(n, res)
    feats = torch.randn(n, in_c, device=DEVICE)
    conv = shims._SubMConv3dImpl(in_c, out_c, 1, indice_key=None).to(DEVICE)
    torch.nn.init.normal_(conv.weight, std=0.2)
    torch.nn.init.normal_(conv.bias, std=0.1)

    x = shims.SparseConvTensor(feats, coords, [res, res, res], 1)
    got = conv(x).features
    want = _dense_reference(coords, feats, conv.weight.data, conv.bias.data, res)
    err = (got - want).abs().max().item()
    assert err < 2e-4, f"カーネル 1 で一致しない: 最大誤差 {err}"


def test_submanifold_conv_multi_batch() -> None:
    """バッチが 2 でも隣のバッチの特徴を拾わない。"""
    shims._install_spconv()
    res, n, in_c, out_c = 10, 300, 4, 4
    coords = _random_coords(n, res, batch=2)
    feats = torch.randn(coords.shape[0], in_c, device=DEVICE)
    conv = shims._SubMConv3dImpl(in_c, out_c, 3).to(DEVICE)
    torch.nn.init.normal_(conv.weight, std=0.2)
    torch.nn.init.normal_(conv.bias, std=0.1)

    x = shims.SparseConvTensor(feats, coords, [res, res, res], 2)
    got = conv(x).features
    want = _dense_reference(coords, feats, conv.weight.data, conv.bias.data, res)
    err = (got - want).abs().max().item()
    assert err < 2e-4, f"バッチ 2 で一致しない: 最大誤差 {err}"


def test_submanifold_conv_chunking_matches() -> None:
    """**塊に切って処理しても結果が変わらない**（境界で取りこぼさない）。

    VRAM を溢れさせないために出力ボクセルを `VOXEL_CHUNK` ずつ処理している。
    切れ目でずれると、**落ちずに静かに違う形が出る**ので必ず検算する。
    """
    shims._install_spconv()
    res, n, in_c, out_c = 14, 500, 6, 5
    coords = _random_coords(n, res)
    feats = torch.randn(n, in_c, device=DEVICE)
    conv = shims._SubMConv3dImpl(in_c, out_c, 3, indice_key="chunked").to(DEVICE)
    torch.nn.init.normal_(conv.weight, std=0.2)
    torch.nn.init.normal_(conv.bias, std=0.1)
    want = _dense_reference(coords, feats, conv.weight.data, conv.bias.data, res)

    original = shims.VOXEL_CHUNK
    try:
        for chunk in (7, 64, 499, 500, 1000):
            shims.VOXEL_CHUNK = chunk
            x = shims.SparseConvTensor(feats, coords, [res, res, res], 1)
            got = conv(x).features
            err = (got - want).abs().max().item()
            assert err < 2e-4, f"VOXEL_CHUNK={chunk} で一致しない: 最大誤差 {err}"
    finally:
        shims.VOXEL_CHUNK = original


def test_rulebook_cache_is_keyed_by_coords() -> None:
    """`indice_key` の使い回しは、座標が同じときだけ効く。"""
    shims._install_spconv()
    res, n = 8, 50
    coords = _random_coords(n, res)
    feats = torch.randn(n, 3, device=DEVICE)
    conv = shims._SubMConv3dImpl(3, 3, 3, indice_key="res_16").to(DEVICE)
    x = shims.SparseConvTensor(feats, coords, [res, res, res], 1)
    conv(x)
    assert "_shim_rb_res_16" in x.indice_dict, "rulebook が載っていない"
    other = shims.SparseConvTensor(
        feats, coords.clone(), [res, res, res], 1, indice_dict=x.indice_dict
    )
    conv(other)  # 座標のオブジェクトが違うので作り直されるだけ。落ちないことを見る


def test_dense_expansion() -> None:
    """`SparseConvTensor.dense()` が座標どおりに散らす。"""
    shims._install_spconv()
    coords = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.int32, device=DEVICE)
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
    x = shims.SparseConvTensor(feats, coords, [4, 4, 4], 1)
    dense = x.dense()
    assert tuple(dense.shape) == (1, 2, 4, 4, 4), dense.shape
    assert dense[0, 0, 1, 2, 3].item() == 1.0
    assert dense[0, 1, 0, 0, 0].item() == 4.0
    assert dense.sum().item() == 10.0


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`[S, H, D]` を素朴に計算する参照実装。"""
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax((q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1)) * scale, dim=-1)
    return (attn @ v.transpose(0, 1)).transpose(0, 1)


def test_flash_attn_varlen_qkvpacked() -> None:
    """可変長の qkv パック版が素朴な実装と一致する（境界をまたがない）。"""
    fa = shims._make_flash_attn(head_chunk=2, fp32=True)
    lens = [7, 13]
    total, heads, dim = sum(lens), 4, 16
    qkv = torch.randn(total, 3, heads, dim, device=DEVICE)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32, device=DEVICE)
    got = fa.flash_attn_varlen_qkvpacked_func(qkv, cu, max(lens))
    want = torch.cat(
        [_naive_attention(*qkv[s:e].unbind(dim=1)) for s, e in [(0, lens[0]), (lens[0], total)]],
        dim=0,
    )
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"varlen qkvpacked が一致しない: 最大誤差 {err}"


def test_flash_attn_varlen_kvpacked() -> None:
    """cross attention（q と kv の長さが違う）版が一致する。"""
    fa = shims._make_flash_attn(head_chunk=3, fp32=True)
    tq, tk, heads, dim = 11, 19, 6, 8
    q = torch.randn(tq, heads, dim, device=DEVICE)
    kv = torch.randn(tk, 2, heads, dim, device=DEVICE)
    cu_q = torch.tensor([0, tq], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, tk], dtype=torch.int32, device=DEVICE)
    got = fa.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, tq, tk)
    want = _naive_attention(q, kv[:, 0], kv[:, 1])
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"varlen kvpacked が一致しない: 最大誤差 {err}"


def test_flash_attn_qkvpacked_batched() -> None:
    """バッチ付き（windowed attention が使う）版が一致する。"""
    fa = shims._make_flash_attn(head_chunk=2, fp32=True)
    batch, seq, heads, dim = 3, 9, 4, 8
    qkv = torch.randn(batch, seq, 3, heads, dim, device=DEVICE)
    got = fa.flash_attn_qkvpacked_func(qkv)
    want = torch.stack([_naive_attention(*qkv[b].unbind(dim=1)) for b in range(batch)], dim=0)
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"qkvpacked が一致しない: 最大誤差 {err}"


def test_kaolin_check_tensor() -> None:
    """FlexiCubes が使う `check_tensor` が形を見ている。"""
    shims._install_kaolin()
    from kaolin.utils.testing import check_tensor

    t = torch.zeros(5, 3)
    assert check_tensor(t, (None, 3), throw=False)
    assert not check_tensor(t, (None, 4), throw=False)
    assert not check_tensor(t, (5,), throw=False)


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
