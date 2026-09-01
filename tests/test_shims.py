# SPDX-License-Identifier: MIT
"""Verify the shims (**get the numbers to agree before running any inference**).

Run it with this repository's virtual environment (torch is required). It falls
back to CPU when no GPU is present, which is what CI does.

**This checks "the same numbers come out", not "it ran".**

- Submanifold convolution is exactly a dense `F.conv3d` restricted to active
  voxels, so **a reference implementation can be written without the original
  library**. Getting the weight layout (KRSC) or the convolution direction wrong
  fails here, every time.
- The `flash_attn` replacement is compared against a naive attention
  implementation.
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
    """Build `[N, 4]` coordinates with no duplicates."""
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
    """Produce the ground truth for submanifold convolution with a dense `F.conv3d`.

    Filling a dense volume with zeros at inactive voxels, running an ordinary
    convolution, and reading back only the active positions is submanifold
    convolution by definition.
    """
    batch = int(coords[:, 0].max().item()) + 1
    out_c, k, _, _, in_c = weight.shape
    vol = torch.zeros(batch, res, res, res, in_c, device=feats.device, dtype=torch.float32)
    idx = coords.long()
    vol[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats.float()
    vol = vol.permute(0, 4, 1, 2, 3)  # [B, Cin, D, H, W]
    # spconv KRSC [out, kD, kH, kW, in] -> torch [out, in, kD, kH, kW]
    w = weight.permute(0, 4, 1, 2, 3).float()
    dense = F.conv3d(vol, w, bias=bias.float(), padding=k // 2)
    return dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]


def test_submanifold_conv_matches_dense_conv() -> None:
    """Kernel-3 submanifold convolution agrees with dense convolution."""
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
    assert err < 2e-4, f"disagrees with dense convolution: max error {err}"


def test_submanifold_conv_kernel1() -> None:
    """Kernel 1 (the skip connection) agrees too."""
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
    assert err < 2e-4, f"disagrees at kernel 1: max error {err}"


def test_submanifold_conv_multi_batch() -> None:
    """With a batch of 2, features never leak across the batch boundary."""
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
    assert err < 2e-4, f"disagrees with a batch of 2: max error {err}"


def test_submanifold_conv_chunking_matches() -> None:
    """**Chunking does not change the result** (nothing is lost at the seams).

    Output voxels are processed `VOXEL_CHUNK` at a time to avoid overflowing
    VRAM. A mistake at a seam **produces a different shape without failing**, so
    it is always verified.
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
            assert err < 2e-4, f"disagrees at VOXEL_CHUNK={chunk}: max error {err}"
    finally:
        shims.VOXEL_CHUNK = original


def test_rulebook_cache_is_keyed_by_coords() -> None:
    """Reuse through `indice_key` applies only when the coordinates are the same."""
    shims._install_spconv()
    res, n = 8, 50
    coords = _random_coords(n, res)
    feats = torch.randn(n, 3, device=DEVICE)
    conv = shims._SubMConv3dImpl(3, 3, 3, indice_key="res_16").to(DEVICE)
    x = shims.SparseConvTensor(feats, coords, [res, res, res], 1)
    conv(x)
    assert "_shim_rb_res_16" in x.indice_dict, "the rulebook was not cached"
    other = shims.SparseConvTensor(
        feats, coords.clone(), [res, res, res], 1, indice_dict=x.indice_dict
    )
    conv(other)  # a different coordinate object just rebuilds it; check it does not raise


def test_dense_expansion() -> None:
    """`SparseConvTensor.dense()` scatters to the right coordinates."""
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
    """Reference implementation over `[S, H, D]`, written the obvious way."""
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax((q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1)) * scale, dim=-1)
    return (attn @ v.transpose(0, 1)).transpose(0, 1)


def test_flash_attn_varlen_qkvpacked() -> None:
    """The variable-length qkv-packed form agrees, never crossing a segment boundary."""
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
    assert err < 1e-5, f"varlen qkvpacked disagrees: max error {err}"


def test_flash_attn_varlen_kvpacked() -> None:
    """The cross-attention form (q and kv of different lengths) agrees."""
    fa = shims._make_flash_attn(head_chunk=3, fp32=True)
    tq, tk, heads, dim = 11, 19, 6, 8
    q = torch.randn(tq, heads, dim, device=DEVICE)
    kv = torch.randn(tk, 2, heads, dim, device=DEVICE)
    cu_q = torch.tensor([0, tq], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, tk], dtype=torch.int32, device=DEVICE)
    got = fa.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, tq, tk)
    want = _naive_attention(q, kv[:, 0], kv[:, 1])
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"varlen kvpacked disagrees: max error {err}"


def test_flash_attn_qkvpacked_batched() -> None:
    """The batched form, used by windowed attention, agrees."""
    fa = shims._make_flash_attn(head_chunk=2, fp32=True)
    batch, seq, heads, dim = 3, 9, 4, 8
    qkv = torch.randn(batch, seq, 3, heads, dim, device=DEVICE)
    got = fa.flash_attn_qkvpacked_func(qkv)
    want = torch.stack([_naive_attention(*qkv[b].unbind(dim=1)) for b in range(batch)], dim=0)
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"qkvpacked disagrees: max error {err}"


def test_kaolin_check_tensor() -> None:
    """The `check_tensor` FlexiCubes uses really does check the shape."""
    shims._install_kaolin()
    from kaolin.utils.testing import check_tensor

    t = torch.zeros(5, 3)
    assert check_tensor(t, (None, 3), throw=False)
    assert not check_tensor(t, (None, 4), throw=False)
    assert not check_tensor(t, (5,), throw=False)


def main() -> int:
    """Run every test."""
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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
