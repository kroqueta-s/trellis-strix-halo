# SPDX-License-Identifier: MIT
"""**Launch-time shims** that make TRELLIS run on gfx1151 / Windows / ROCm.

**This file is deliberately identical to the Hi3DGen runner's `shims.py`**
(`hi3dgen-strix-halo`), whose sparse modules are TRELLIS code. The two runners
never import each other, since each ships as its own repository. **Fix one and
fix the other.**

**Upstream code is never modified** (a re-clone or an update would undo it).
`install()` is called **before** importing `trellis` and puts the replacements
into `sys.modules`.

Four things are replaced. **Each is on the required path but unobtainable on
Windows + ROCm**, and the surface of API actually used is small.

- `spconv`: sparse convolution. **Only submanifold, stride-1 `SubMConv3d` is
  ever used**, so it is reproduced with coordinate linearization and a gather
  over neighbours.
- `flash_attn`: sparse attention, reproduced with
  `F.scaled_dot_product_attention`.
- `kaolin`: FlexiCubes imports `check_tensor` (a shape check, nothing more).
- `open3d`: imported at the top of `trellis_text_to_3d` and **never used on the
  image-to-mesh path**, so it gets a stand-in that raises if touched.

**Correctness here is verifiable.** Submanifold convolution is exactly a dense
`F.conv3d` restricted to active voxels, so it can be checked against a dense
reference implementation (`tests/test_shims.py`).
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Whether the dual-grid extraction closes the sparse region's edge (TRELLIS.2).
# **Upstream leaves it open**; see `flexible_dual_grid_to_mesh` for the measurement
# that made this the default.
_CLOSE_MESH = True

# Attention heads computed at once. Measured best on Hunyuan3D at 4 on gfx1151.
DEFAULT_HEAD_CHUNK = 4

# Output voxels processed at once in sparse convolution. **This sets the VRAM
# peak.** There are only 32 GB of dedicated VRAM, and spilling into shared
# memory is several times slower.
VOXEL_CHUNK = 65536

# **Capture the real function before replacing it.** Calling
# `F.scaled_dot_product_attention` from inside a shim would otherwise call the
# shim itself, chunking the heads twice and running slower.
_TORCH_SDPA = F.scaled_dot_product_attention


def _new_module(name: str, is_package: bool = False) -> types.ModuleType:
    """Create an empty module that has `__spec__` and `__file__`.

    **`__spec__` is mandatory.** `transformers` probes with
    `importlib.util.find_spec("flash_attn")`, and without `__spec__` that fails
    with `ValueError: flash_attn.__spec__ is None`, **taking an unrelated import
    down with it** (hit on 2026-09-01).
    """
    module = types.ModuleType(name)
    module.__file__ = f"<hearth shim: {name}>"
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------------------
# Replacement for spconv
# --------------------------------------------------------------------------------------


class SparseConvTensor:
    """Reproduces the container of `spconv.pytorch.SparseConvTensor`, nothing more.

    TRELLIS's `SparseTensor` (`trellis/modules/sparse/basic.py`) **touches these
    attributes directly**: `features`, `indices`, `spatial_shape`, `batch_size`,
    `grid`, `voxel_num`, `indice_dict`, `benchmark`, `benchmark_record`,
    `thrust_allocator`, `_timer`, `force_algo`, `int8_scale`, `_features`, plus
    `dense()`. **The positional order matches the real class too**, because
    `replace()` passes seven of them positionally.

    `features` being a property backed by `_features` also matches the real
    class: TRELLIS constructs with features reshaped to two dimensions and then
    writes the multi-dimensional tensor back into `_features`.
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[Any],
        batch_size: int,
        grid: Any = None,
        voxel_num: Any = None,
        indice_dict: dict[str, Any] | None = None,
        benchmark: bool = False,
        permanent_thrust_allocator: bool = False,
        force_algo: Any = None,
        int8_scale: Any = None,
    ) -> None:
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict: dict[str, Any] = {} if indice_dict is None else indice_dict
        self.benchmark = benchmark
        self.benchmark_record: dict[str, Any] = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = force_algo
        self.int8_scale = int8_scale

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @features.setter
    def features(self, value: torch.Tensor) -> None:
        self._features = value

    @property
    def dtype(self) -> torch.dtype:
        return self._features.dtype

    @property
    def device(self) -> torch.device:
        return self._features.device

    def replace_feature(self, feature: torch.Tensor) -> SparseConvTensor:
        """Return a new tensor with only the features replaced (coordinates and rulebook shared)."""
        return SparseConvTensor(
            feature,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
        )

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        """Expand to a dense tensor (called from `SparseTensor.dense()`)."""
        feats = self._features
        channels = feats.shape[1:]
        out = torch.zeros(
            (self.batch_size, *self.spatial_shape, *channels),
            dtype=feats.dtype,
            device=feats.device,
        )
        idx = self.indices.long()
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats
        if channels_first:
            ndim = len(self.spatial_shape)
            perm = [0, ndim + 1, *range(1, ndim + 1)]
            out = out.permute(*perm).contiguous()
        return out


class ConvAlgo:
    """Replacement for `spconv.ConvAlgo`. **The values are unused**, as there is no choice here."""

    Native = "native"
    MaskImplicitGemm = "implicit_gemm"
    MaskSplitImplicitGemm = "mask_split_implicit_gemm"


def _neighbor_index_map(coords: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Build the rulebook for submanifold convolution.

    For each output voxel `i` and kernel position `k`, return the **index of the
    input voxel**, or `-1` where the neighbour is not active. Lookup is by
    linearizing coordinates and running `searchsorted` over the sorted array, so
    no dense index table is needed and memory grows only with the coordinates.

    **Returns `int32`, written straight into a `[K, N]` buffer.** Building 27
    `int64` planes and calling `torch.stack` would keep two temporaries of over
    400 MB alive at once at resolution 256 (1.8 M active voxels). **There are
    only 32 GB of dedicated VRAM, and spilling into shared memory is several
    times slower** (hit on 2026-09-01, all the way to `torch.OutOfMemoryError`).

    Args:
        coords: `[N, 4]` of `(batch, z, y, x)`.
        kernel_size: Kernel edge length; only 1 and 3 occur.

    Returns:
        `[K, N]` of `int32`, where `K = kernel_size ** 3`. Inactive neighbours
        are `-1`.
    """
    n = coords.shape[0]
    device = coords.device
    c = coords.long()
    batch = c[:, 0]
    z, y, x = c[:, 1], c[:, 2], c[:, 3]

    dim_z = int(z.max().item()) + 1
    dim_y = int(y.max().item()) + 1
    dim_x = int(x.max().item()) + 1

    lin = ((batch * dim_z + z) * dim_y + y) * dim_x + x
    order = torch.argsort(lin)
    lin_sorted = lin[order]
    order32 = order.to(torch.int32)

    half = kernel_size // 2
    out = torch.empty((kernel_size**3, n), device=device, dtype=torch.int32)
    slot = 0
    for dz in range(-half, half + 1):
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    torch.arange(n, device=device, dtype=torch.int32, out=out[slot])
                    slot += 1
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                valid = (
                    (nz >= 0) & (nz < dim_z) & (ny >= 0) & (ny < dim_y) & (nx >= 0) & (nx < dim_x)
                )
                nlin = ((batch * dim_z + nz) * dim_y + ny) * dim_x + nx
                pos = torch.searchsorted(lin_sorted, nlin.clamp_(min=0))
                pos.clamp_(max=n - 1)
                valid &= lin_sorted[pos] == nlin
                out[slot] = torch.where(valid, order32[pos], torch.full_like(order32, -1))
                slot += 1
    return out


def _submanifold_conv(
    feats: torch.Tensor,
    rulebook: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """The convolution itself, shared by the two backends that ask for it.

    `weight` is KRSC (`[out, kD, kH, kW, in]`), `rulebook` is `[K, N]` of input
    indices with `-1` for an inactive neighbour, and the result is `[N, out]`.

    **Output voxels are processed in chunks.** Doing them all at once makes the
    gathered features `[N, in]` alone exceed 700 MB at resolution 256 with 192
    input channels, and there are only 32 GB of dedicated VRAM.

    **The accumulator is fp32 even when the features are fp16**: a 3^3 kernel
    over 1024 channels sums 27,648 products per output, which leaves fp16 no
    headroom.
    """
    out_channels, in_channels = weight.shape[0], weight.shape[-1]
    n = feats.shape[0]
    # [out, K, in] -> [K, in, out] (built once and reused)
    w_all = (
        weight.reshape(out_channels, -1, in_channels).permute(1, 2, 0).contiguous().to(feats.dtype)
    )
    kernels = w_all.shape[0]
    center = kernels // 2

    out = torch.empty(n, out_channels, device=feats.device, dtype=feats.dtype)
    bias_f = None if bias is None else bias.float()
    for start in range(0, n, VOXEL_CHUNK):
        end = min(start + VOXEL_CHUNK, n)
        acc = torch.zeros(end - start, out_channels, device=feats.device)
        for k in range(kernels):
            if kernels == 1 or k == center:
                # The centre is always the voxel itself: submanifold means
                # output coordinates equal input coordinates.
                acc += (feats[start:end] @ w_all[k]).float()
                continue
            idx = rulebook[k, start:end].long()
            valid = idx >= 0
            gathered = feats.index_select(0, idx.clamp_(min=0))
            gathered[~valid] = 0
            acc += (gathered @ w_all[k]).float()
        if bias_f is not None:
            acc += bias_f
        out[start:end] = acc.to(feats.dtype)
    return out


class _SubMConv3dImpl(nn.Module):
    """Submanifold sparse 3D convolution (stride 1, output coordinates equal input coordinates).

    **Weights are laid out as spconv 2.x KRSC**:
    `[out_channels, kD, kH, kW, in_channels]`. Confirmed against a real
    checkpoint (for example `upsample.0.out_layers.0.conv.weight` is
    `[192, 3, 3, 3, 768]`). Getting this wrong **silently produces a different
    shape**, so do not change it.

    The convolution is a correlation (`out[p] = sum_k W[k] . in[p + k - center]`),
    which is exactly a dense `F.conv3d(padding=k//2)` with inactive voxels set to
    zero. `tests/test_shims.py` verifies that agreement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        indice_key: str | None = None,
        algo: Any = None,
    ) -> None:
        super().__init__()
        if dilation != 1:
            raise NotImplementedError("dilation is unsupported (the TRELLIS path never uses it)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(
            torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _rulebook(self, x: SparseConvTensor) -> torch.Tensor:
        """Build the rulebook, reusing it through the tensor's `indice_dict` when keyed."""
        if self.kernel_size == 1:
            return torch.arange(
                x.indices.shape[0], device=x.indices.device, dtype=torch.int32
            ).unsqueeze(0)
        key = f"_shim_rb_{self.indice_key}" if self.indice_key is not None else None
        if key is not None:
            cached = x.indice_dict.get(key)
            if cached is not None and cached[0] is x.indices:
                return cached[1]
        rulebook = _neighbor_index_map(x.indices, self.kernel_size)
        if key is not None:
            x.indice_dict[key] = (x.indices, rulebook)
        return rulebook

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        feats = x.features
        if feats.dim() != 2:
            feats = feats.reshape(feats.shape[0], -1)
        rulebook = self._rulebook(x)
        out = _submanifold_conv(feats, rulebook, self.weight, self.bias)

        return SparseConvTensor(
            out,
            x.indices,
            x.spatial_shape,
            x.batch_size,
            x.grid,
            x.voxel_num,
            x.indice_dict,
        )


class _UnsupportedConv(nn.Module):
    """A convolution this path should never reach. **Always raises**, never passes silently."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError(
            "strided SparseConv3d / SparseInverseConv3d are not part of the spconv replacement. "
            "They should not appear on the TRELLIS image-to-mesh path, so reaching this means "
            "the path has changed"
        )


def _install_spconv() -> None:
    """Install `spconv` and `spconv.pytorch` (**even if the real package exists**)."""
    spconv = _new_module("spconv", is_package=True)
    pytorch = _new_module("spconv.pytorch")

    pytorch.SparseConvTensor = SparseConvTensor  # type: ignore[attr-defined]
    pytorch.SubMConv3d = _SubMConv3dImpl  # type: ignore[attr-defined]
    pytorch.SparseConv3d = _UnsupportedConv  # type: ignore[attr-defined]
    pytorch.SparseInverseConv3d = _UnsupportedConv  # type: ignore[attr-defined]
    pytorch.ConvAlgo = ConvAlgo  # type: ignore[attr-defined]
    spconv.pytorch = pytorch  # type: ignore[attr-defined]
    spconv.ConvAlgo = ConvAlgo  # type: ignore[attr-defined]

    sys.modules["spconv"] = spconv
    sys.modules["spconv.pytorch"] = pytorch


# --------------------------------------------------------------------------------------
# Replacement for flash_attn
# --------------------------------------------------------------------------------------


def _attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_chunk: int, fp32: bool
) -> torch.Tensor:
    """Take a `[B, S, H, D]` triple and return `[B, S, H, D]`.

    Without flash or mem_efficient on gfx1151 the math backend is what actually
    runs. It materialises `q @ k^T` in the input dtype, so **fp16 exceeds 65504
    and breaks** (reproduced deterministically on Hunyuan3D with a mirrored
    input). Hence fp32 arithmetic, with the heads chunked to keep the
    intermediate tensors small.
    """
    out_dtype = q.dtype
    qt = q.transpose(1, 2)  # [B, H, S, D]
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if not fp32:
        # When flash or mem-efficient is available, its online softmax never
        # overflows, so fp16 is safe as-is.
        return _TORCH_SDPA(qt, kt, vt).transpose(1, 2).contiguous()
    heads = qt.shape[1]
    outs: list[torch.Tensor] = []
    for i in range(0, heads, head_chunk):
        o = _TORCH_SDPA(
            qt[:, i : i + head_chunk].float(),
            kt[:, i : i + head_chunk].float(),
            vt[:, i : i + head_chunk].float(),
        )
        outs.append(o.to(out_dtype))
    return torch.cat(outs, dim=1).transpose(1, 2).contiguous()


def _segments(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    """Turn `cu_seqlens` (cumulative lengths) into `[(start, end), ...]`."""
    bounds = cu_seqlens.tolist()
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def _make_flash_attn(head_chunk: int, fp32: bool) -> types.ModuleType:
    """Build a module holding **only the four functions actually called**.

    TRELLIS calls them from five places in `full_attn.py`, `windowed_attn.py`
    and `serialized_attn.py`, always with positional arguments only. Neither
    `causal` nor `dropout` is used.
    """
    module = _new_module("flash_attn")

    def flash_attn_qkvpacked_func(qkv: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # qkv: [B, S, 3, H, D] -> [B, S, H, D]
        q, k, v = qkv.unbind(dim=2)
        return _attend(q, k, v, head_chunk, fp32)

    def flash_attn_varlen_qkvpacked_func(
        qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # qkv: [T, 3, H, D] -> [T, H, D]
        out = torch.empty_like(qkv[:, 0])
        for start, end in _segments(cu_seqlens):
            if end <= start:
                continue
            q, k, v = qkv[start:end].unsqueeze(0).unbind(dim=2)
            out[start:end] = _attend(q, k, v, head_chunk, fp32)[0]
        return out

    def flash_attn_varlen_kvpacked_func(
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # q: [Tq, H, D], kv: [Tk, 2, H, D] -> [Tq, H, D]
        out = torch.empty_like(q)
        segments = zip(_segments(cu_seqlens_q), _segments(cu_seqlens_k), strict=True)
        for (qs, qe), (ks, ke) in segments:
            if qe <= qs:
                continue
            k_, v_ = kv[ks:ke].unsqueeze(0).unbind(dim=2)
            out[qs:qe] = _attend(q[qs:qe].unsqueeze(0), k_, v_, head_chunk, fp32)[0]
        return out

    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        segments = zip(_segments(cu_seqlens_q), _segments(cu_seqlens_k), strict=True)
        for (qs, qe), (ks, ke) in segments:
            if qe <= qs:
                continue
            out[qs:qe] = _attend(
                q[qs:qe].unsqueeze(0),
                k[ks:ke].unsqueeze(0),
                v[ks:ke].unsqueeze(0),
                head_chunk,
                fp32,
            )[0]
        return out

    module.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func  # type: ignore[attr-defined]
    module.flash_attn_varlen_qkvpacked_func = (  # type: ignore[attr-defined]
        flash_attn_varlen_qkvpacked_func
    )
    module.flash_attn_varlen_kvpacked_func = (  # type: ignore[attr-defined]
        flash_attn_varlen_kvpacked_func
    )
    module.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore[attr-defined]
    module.__version__ = "0.0.0+hearth-shim"  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------------------
# Stand-ins for kaolin and open3d
# --------------------------------------------------------------------------------------


def _install_kaolin() -> None:
    """Provide only `kaolin.utils.testing.check_tensor`, which FlexiCubes imports.

    FlexiCubes uses it exclusively as `assert check_tensor(x, shape, throw=False)`.
    **The shape check is implemented for real**, because always returning true
    would make the assertion meaningless.
    """

    def check_tensor(
        tensor: torch.Tensor,
        shape: Iterable[int | None] | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        throw: bool = True,
    ) -> bool:
        problems: list[str] = []
        if shape is not None:
            expected = list(shape)
            if tensor.dim() != len(expected):
                problems.append(f"wrong number of dimensions: {tensor.dim()} != {len(expected)}")
            else:
                for i, (got, want) in enumerate(zip(tensor.shape, expected, strict=True)):
                    if want is not None and got != want:
                        problems.append(f"wrong size on axis {i}: {got} != {want}")
        if dtype is not None and tensor.dtype != dtype:
            problems.append(f"wrong dtype: {tensor.dtype} != {dtype}")
        if device is not None and torch.device(device).type != tensor.device.type:
            problems.append(f"wrong device: {tensor.device} != {device}")
        if problems and throw:
            raise ValueError("; ".join(problems))
        return not problems

    kaolin = _new_module("kaolin", is_package=True)
    utils = _new_module("kaolin.utils", is_package=True)
    testing = _new_module("kaolin.utils.testing")
    testing.check_tensor = check_tensor  # type: ignore[attr-defined]
    utils.testing = testing  # type: ignore[attr-defined]
    kaolin.utils = utils  # type: ignore[attr-defined]
    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing


class _AbsentAttribute:
    """A stand-in that **allows name lookup but raises when called**.

    Raising on attribute access alone would break **annotation evaluation** such
    as `class Foo: def f(self, m: o3d.geometry.TriangleMesh)`, which is exactly
    what `trellis_text_to_3d` does. Annotations pass; only an **actual call**
    raises.
    """

    def __init__(self, path: str, reason: str) -> None:
        self._path = path
        self._reason = reason

    def __getattr__(self, attr: str) -> _AbsentAttribute:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentAttribute(f"{self._path}.{attr}", self._reason)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{self._path} is unavailable on this machine ({self._reason})")

    def __repr__(self) -> str:
        return f"<absent {self._path}>"


def _install_absent(name: str, reason: str) -> None:
    """Install a stand-in that imports fine but **raises when called**.

    It never returns a wrong result silently.

    **Dunders alone raise `AttributeError`.** `inspect.getmodule` walks all of
    `sys.modules` looking for `__file__`, so raising anything else there **takes
    an unrelated import (torchvision) down with it** (hit on 2026-09-01).
    """

    module = _new_module(name, is_package=True)

    def _absent(attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentAttribute(f"{name}.{attr}", reason)

    module.__getattr__ = _absent  # type: ignore[attr-defined]
    sys.modules[name] = module


# --------------------------------------------------------------------------------------
# Replacing dense SDPA (the same reason as Hunyuan3D)
# --------------------------------------------------------------------------------------


def _install_dense_sdpa(head_chunk: int) -> None:
    """Replace `F.scaled_dot_product_attention` with fp32 arithmetic over chunked heads.

    TRELLIS's **dense** attention (`ATTN_BACKEND=sdpa`) runs seq=4096 through 24
    DiT blocks. That is the same regime where the math backend broke down in
    fp16 on Hunyuan3D, so the same treatment is applied pre-emptively.
    """
    torch.backends.cuda.sdp_kernel = lambda *a, **kw: contextlib.nullcontext()

    def sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if attn_mask is not None or is_causal or dropout_p:
            raise NotImplementedError("the replacement SDPA accepts plain attention only")
        out_dtype = query.dtype
        heads = query.shape[1]
        outs: list[torch.Tensor] = []
        for i in range(0, heads, head_chunk):
            # **fp32 arithmetic, torch's own implementation.** Writing the
            # softmax by hand would build intermediates just as large while
            # giving up the fused kernel.
            o = _TORCH_SDPA(
                query[:, i : i + head_chunk].float(),
                key[:, i : i + head_chunk].float(),
                value[:, i : i + head_chunk].float(),
                scale=scale,
            )
            outs.append(o.to(out_dtype))
        return torch.cat(outs, dim=1)

    F.scaled_dot_product_attention = sdpa  # type: ignore[assignment]


# --------------------------------------------------------------------------------------
# Replacement for o_voxel's hashmap (**TRELLIS.2 only**)
# --------------------------------------------------------------------------------------
#
# `o_voxel.convert.flexible_dual_grid_to_mesh` - the function that turns the
# decoder's dual grid into a mesh - is pure torch except for two calls into that
# package's CUDA extension: a hashmap from a voxel coordinate to the row it came
# from. Nothing else on the image-to-mesh path touches `_C`, so replacing those
# two calls is enough to run the extraction without building anything.
#
# **This section has no counterpart in the Hi3DGen runner**, whose upstream is
# TRELLIS.1 and never imports `o_voxel`. The rest of this file stays identical.
#
# **What a miss is**: `hashmap_lookup_3d_cuda` returns the largest value of the
# value type both for an absent key and for a coordinate outside the grid
# (`o-voxel/src/hash/hash.cu`), and the caller tests against `0xffffffff`.


def _linearize_3d(coords: torch.Tensor, w: int, h: int, d: int) -> torch.Tensor:
    """Flatten `[N, 4]` (batch, x, y, z) exactly as the kernels do.

    `b*W*H*D + x*H*D + y*D + z`, in int64 so the product cannot wrap: the
    largest grid TRELLIS.2 offers (1536^3) is 3.6e9, already past a signed
    32-bit range.
    """
    c = coords.long()
    return ((c[:, 0] * w + c[:, 1]) * h + c[:, 2]) * d + c[:, 3]


def _table_as_int64(table: torch.Tensor) -> torch.Tensor:
    """Read a hashmap array as int64, refusing the case that would wrap.

    The caller picks uint32 or uint64 by grid volume
    (`o_voxel/convert/flexible_dual_grid.py`), and **uint64 is only reached past
    2^32 voxels** - beyond every resolution TRELLIS.2 offers. Rather than carry
    an unsigned path that nothing exercises, a key that would not survive the
    conversion raises.
    """
    if table.dtype == torch.uint64:
        sentinel = torch.iinfo(torch.uint64).max
        as_int = table.to(torch.int64)
        if bool(((as_int < 0) & (table != sentinel)).any()):
            raise NotImplementedError("o_voxel hashmap keys past 2^63 are not supported")
        return torch.where(table == sentinel, torch.iinfo(torch.int64).max, as_int)
    return table.to(torch.int64)


def _hashmap_insert_3d_idx_as_val(
    hashmap_keys: torch.Tensor,
    hashmap_values: torch.Tensor,
    coords: torch.Tensor,
    w: int,
    h: int,
    d: int,
) -> None:
    """Store `coordinate -> row index` **in the caller's own two tensors**.

    Upstream fills them by open addressing on the GPU. Here they hold the keys
    **sorted at the front**, with the empty sentinel - the largest value of the
    key type, which is what upstream also leaves in an empty slot - filling the
    tail. The tail keeps the array sorted, so a lookup is one `searchsorted`
    over the whole array with no count to carry. It is the idiom the
    submanifold convolution already uses for its neighbour index.

    **Duplicate coordinates keep the first row.** Upstream's kernel races for
    them; the pipeline produces none, and being deterministic is worth more
    here than reproducing a race.
    """
    keys = _linearize_3d(coords, w, h, d)
    order = torch.argsort(keys, stable=True)
    ordered = keys[order]
    first = torch.ones_like(ordered, dtype=torch.bool)
    first[1:] = ordered[1:] != ordered[:-1]
    unique_keys = ordered[first]
    unique_rows = order[first]

    count = int(unique_keys.numel())
    if count > hashmap_keys.numel():
        raise ValueError(f"hashmap too small: {count} keys into {hashmap_keys.numel()} slots")

    hashmap_keys.fill_(torch.iinfo(hashmap_keys.dtype).max)
    hashmap_keys[:count] = unique_keys.to(hashmap_keys.dtype)
    hashmap_values[:count] = unique_rows.to(hashmap_values.dtype)


def _hashmap_lookup_3d(
    hashmap_keys: torch.Tensor,
    hashmap_values: torch.Tensor,
    coords: torch.Tensor,
    w: int,
    h: int,
    d: int,
) -> torch.Tensor:
    """Look coordinates up, returning the stored row or the miss value."""
    c = coords.long()
    inside = (
        (c[:, 1] >= 0)
        & (c[:, 1] < w)
        & (c[:, 2] >= 0)
        & (c[:, 2] < h)
        & (c[:, 3] >= 0)
        & (c[:, 3] < d)
    )
    query = _linearize_3d(coords, w, h, d)
    table = _table_as_int64(hashmap_keys)
    # A coordinate outside the grid can linearize to a negative number, which
    # `searchsorted` would place before every key; `inside` rejects it anyway.
    position = torch.searchsorted(table, query.clamp(min=0)).clamp(max=table.numel() - 1)
    hit = inside & (table[position] == query)

    out = torch.full_like(query, torch.iinfo(hashmap_values.dtype).max)
    out[hit] = _table_as_int64(hashmap_values)[position[hit]]
    return out.to(hashmap_values.dtype)


# Offsets of the four voxels that share an edge, per axis. **The order is
# upstream's** (`o_voxel/convert/flexible_dual_grid.py`): it decides which way
# round a quad is wound, so it is copied exactly rather than re-derived.
_EDGE_NEIGHBOURS = (
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
)


def _voxel_index(active: torch.Tensor, query: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Index of each queried voxel in `active`, or -1. The same sorted lookup as everywhere here."""
    scale = torch.tensor(
        [int(grid[1]) * int(grid[2]), int(grid[2]), 1], device=active.device, dtype=torch.long
    )
    keys = (active.long() * scale).sum(dim=1)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    wanted = (query.long() * scale).sum(dim=1)
    position = torch.searchsorted(sorted_keys, wanted).clamp(max=sorted_keys.numel() - 1)
    found = sorted_keys[position] == wanted
    inside = ((query >= 0) & (query < grid.to(query.device))).all(dim=1)
    return torch.where(found & inside, order[position], torch.full_like(wanted, -1))


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: torch.Tensor | None = None,
    aabb: Any = None,
    voxel_size: Any = None,
    grid_size: Any = None,
    train: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the mesh from the dual grid, **and close it at the edge of the sparse region**.

    Upstream drops every quad whose four voxels are not all active
    (`connected_voxel_valid`), which is where TRELLIS.2's meshes are open: not
    because the geometry is broken, but because the sparse structure stops
    there. Measured on this machine at resolution 512: **118,777 boundary
    edges**, and nothing downstream could close them - `trimesh.repair` reached
    20,677, `manifold3d` refused the mesh outright, MeshFix ran single-threaded
    for 23 minutes without finishing.

    So the hole is closed where it is made. A missing neighbour contributes a
    vertex at **the centre of its own voxel**, which is where a dual vertex
    would sit with no surface to pull it, and the quad is emitted as usual.
    **The dual vertices the model produced are used unchanged**, so the sharp
    features that TRELLIS.2 exists for are not touched - this is not a
    re-meshing and nothing is smoothed.

    Set `TRELLIS2_CLOSE_MESH=off` to get upstream's behaviour back.
    """
    if train:
        raise NotImplementedError("training-mode extraction is not used by this runner")

    device = coords.device
    n = dual_vertices.shape[0]
    if isinstance(aabb, (list, tuple, np.ndarray)):
        aabb = torch.tensor(np.asarray(aabb), dtype=torch.float32, device=device)
    if grid_size is None:
        raise ValueError("grid_size is required")
    if isinstance(grid_size, int):
        grid_size = [grid_size] * 3
    grid = torch.tensor(np.asarray(grid_size), dtype=torch.long, device=device).reshape(3)
    if voxel_size is None:
        voxel_size = (aabb[1] - aabb[0]) / grid.to(torch.float32)

    vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)

    offsets = torch.tensor(_EDGE_NEIGHBOURS, dtype=torch.long, device=device).unsqueeze(0)
    neighbours = coords.long().reshape(n, 1, 1, 3) + offsets  # [N, 3, 4, 3]
    connected = neighbours[intersected_flag]  # [M, 4, 3]
    m = connected.shape[0]
    if m == 0:
        return vertices, torch.zeros((0, 3), dtype=torch.long, device=device)

    index = _voxel_index(coords, connected.reshape(-1, 3), grid).reshape(m, 4)
    missing = index < 0

    if _CLOSE_MESH and bool(missing.any()):
        # One synthetic vertex per missing voxel, at the centre of that voxel.
        absent = connected.reshape(-1, 3)[missing.reshape(-1)]
        unique, inverse = torch.unique(absent, dim=0, return_inverse=True)
        extra = (unique.float() + 0.5) * voxel_size + aabb[0].reshape(1, 3)
        vertices = torch.cat([vertices, extra], dim=0)
        index = index.clone()
        index[missing] = inverse + n
    else:
        keep = ~missing.any(dim=1)
        index = index[keep]
    if index.shape[0] == 0:
        return vertices, torch.zeros((0, 3), dtype=torch.long, device=device)

    quad = vertices[index]  # [L, 4, 3]
    if split_weight is None:
        # Upstream's rule: split the quad the way that keeps the two triangles
        # most nearly coplanar.
        first = torch.cross(quad[:, 1] - quad[:, 0], quad[:, 2] - quad[:, 0], dim=1)
        second = torch.cross(quad[:, 2] - quad[:, 1], quad[:, 3] - quad[:, 1], dim=1)
        align_a = (first * second).sum(dim=1).abs()
        first = torch.cross(quad[:, 2] - quad[:, 1], quad[:, 3] - quad[:, 1], dim=1)
        second = torch.cross(quad[:, 3] - quad[:, 2], quad[:, 0] - quad[:, 2], dim=1)
        align_b = (first * second).sum(dim=1).abs()
        pick = (align_a > align_b).unsqueeze(1)
    else:
        weight = split_weight.reshape(-1)
        padded = torch.cat([weight, weight.new_zeros(vertices.shape[0] - weight.shape[0])])
        corners = padded[index]
        pick = (corners[:, 0] * corners[:, 2] > corners[:, 1] * corners[:, 3]).unsqueeze(1)

    split_a = index[:, [0, 1, 2, 0, 2, 3]]
    split_b = index[:, [0, 1, 3, 3, 1, 2]]
    return vertices, torch.where(pick, split_a, split_b).reshape(-1, 3)


def install_o_voxel_extraction(close: bool = True) -> bool:
    """Replace `o_voxel.convert.flexible_dual_grid_to_mesh` **before the VAE imports it**.

    `fdg_vae` does `from o_voxel.convert import ...` at module level, so the
    replacement has to be in place before `trellis2.models` is imported.

    Returns:
        Whether the replacement was installed.
    """
    global _CLOSE_MESH
    _CLOSE_MESH = close
    try:
        import o_voxel.convert as convert
    except ImportError as exc:
        print(f"[shims] o_voxel is not importable: {type(exc).__name__}: {exc}")
        return False
    convert.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    sys.modules["o_voxel.convert"].flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    return True


def _make_o_voxel_c() -> types.ModuleType:
    """Build the `o_voxel._C` stand-in holding **only the two functions used**."""
    module = _new_module("o_voxel._C")
    module.hashmap_insert_3d_idx_as_val_cuda = _hashmap_insert_3d_idx_as_val  # type: ignore[attr-defined]
    module.hashmap_lookup_3d_cuda = _hashmap_lookup_3d  # type: ignore[attr-defined]
    return module


def install_o_voxel_hashmap() -> None:
    """Take the place of `o_voxel._C` **before `o_voxel` is imported**.

    The rest of that package is pure python and is imported from the upstream
    checkout unchanged; only the compiled half is replaced.
    """
    sys.modules["o_voxel._C"] = _make_o_voxel_c()


def _flex_gemm_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    _spatial_size: Any,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    neighbor_cache: torch.Tensor | None,
    dilation: Any = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`flex_gemm.ops.spconv.sparse_submanifold_conv3d`, on the same rulebook.

    Returns the features **and the rulebook**, which the caller caches on the
    sparse tensor and hands back for the next convolution at the same
    resolution - the same reuse the spconv path gets from `indice_key`.
    """
    if dilation not in (1, (1, 1, 1), [1, 1, 1]):
        raise NotImplementedError(f"dilation {dilation} is unsupported (this path never uses it)")
    kernel = int(weight.shape[1])
    if neighbor_cache is None:
        if kernel == 1:
            neighbor_cache = torch.arange(
                coords.shape[0], device=coords.device, dtype=torch.int32
            ).unsqueeze(0)
        else:
            neighbor_cache = _neighbor_index_map(coords, kernel)
    return _submanifold_conv(feats, neighbor_cache, weight, bias), neighbor_cache


def _install_flex_gemm() -> None:
    """Stand in for `flex_gemm`, **with a working sparse convolution**.

    **This is the backend the published checkpoints were saved with.** Its
    convolution keeps the weight on the module itself (`conv.weight`), while the
    spconv backend nests it one level down (`conv.conv.weight`), so loading a
    TRELLIS.2 checkpoint through the spconv path leaves **every convolution
    weight uninitialized** - and the decoder then predicts no subdivision at
    all, which surfaces far away as an empty tensor. Measured 2026-09-11:
    `blocks.0.0.conv.conv.weight` held 32 NaNs and 2 infinities, because it was
    never written.

    The arithmetic is the same submanifold convolution as the spconv shim, and
    the weight layout is the same KRSC, so `tests/test_shims.py` covers both.
    `grid_sample` stays absent: it belongs to texture.
    """
    module = _new_module("flex_gemm", is_package=True)
    ops = _new_module("flex_gemm.ops", is_package=True)
    spconv_ops = _new_module("flex_gemm.ops.spconv", is_package=True)

    spconv_ops.sparse_submanifold_conv3d = _flex_gemm_submanifold_conv3d  # type: ignore[attr-defined]
    # The algorithm and the hashmap ratio choose between CUDA kernels that do
    # not exist here; there is one implementation, so both are accepted and
    # ignored rather than raising in the middle of a forward pass.
    spconv_ops.set_algorithm = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    spconv_ops.set_hashmap_ratio = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

    ops.spconv = spconv_ops  # type: ignore[attr-defined]
    module.ops = ops  # type: ignore[attr-defined]
    sys.modules["flex_gemm"] = module
    sys.modules["flex_gemm.ops"] = ops
    sys.modules["flex_gemm.ops.spconv"] = spconv_ops

    _install_absent("flex_gemm.ops.grid_sample", "texture sampling only (not produced here)")
    ops.grid_sample = sys.modules["flex_gemm.ops.grid_sample"]  # type: ignore[attr-defined]


def install_trellis2(close_mesh: bool = True) -> None:
    """Everything TRELLIS.2 needs on top of `install()`. **Call it before importing.**

    Three names are imported by the module that carries `Mesh`, and only one of
    them does any work on the image-to-mesh path:

    - `o_voxel._C`: **the mesh extraction runs through it**, so it is replaced
      with a working implementation above.
    - `cumesh`: a GPU mesh library, reached only through `Mesh.fill_holes`,
      `simplify` and `remove_faces`. **This runner calls none of them** - it
      drives the pipeline stage by stage and does its own postprocessing - so a
      stand-in that raises when called says so rather than guessing.
    - `flex_gemm`: **the sparse convolution the checkpoints were saved with**,
      so its half is implemented rather than stubbed (`_install_flex_gemm`).
      Only `ops.grid_sample`, which samples texture attributes, stays absent.

    A stand-in that raises is the point: **it can never return a wrong mesh
    quietly.** If a future change reaches one of these, it stops with the name
    that was called.
    """
    install_o_voxel_hashmap()
    _install_absent("cumesh", "a CUDA mesh library with no Windows + ROCm build")
    _install_flex_gemm()
    # **The postprocessing this runner shares with TRELLIS.1 imports `rembg`
    # at module level and never calls it there** - background removal belongs to
    # the pipeline, and TRELLIS.2 does it with BiRefNet instead. Without a
    # stand-in the invisible-face removal fails and the mesh comes back
    # unchanged, which is easy to miss because it is caught and reported as a
    # warning (seen 2026-09-11: 118,871 boundary edges left behind).
    if "rembg" not in sys.modules:
        _install_absent("rembg", "background removal is BiRefNet's job in this pipeline")
    # `o_voxel`'s package __init__ imports every submodule, and two of them
    # (`postprocess`, `rasterize`) import nvdiffrast at the top for work this
    # runner never asks for. **Importing the package at all needs the name.**
    install_absent_nvdiffrast()
    # **Last, because it imports the real `o_voxel`**, which needs every
    # stand-in above to be in place first.
    if not install_o_voxel_extraction(close=close_mesh):
        raise ImportError("o_voxel is not importable (is the checkout on sys.path?)")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def fast_attention_available() -> bool:
    """Test on a small tensor whether flash or mem-efficient **actually runs**.

    On gfx1151 the AOTriton implementations become available only when
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set **before torch is
    imported** (measured: setting `os.environ` afterwards has no effect).
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return False
    probe = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                _TORCH_SDPA(probe, probe, probe)
            return True
        except Exception:  # noqa: BLE001 - only the availability matters
            continue
    return False


def _make_numba() -> types.ModuleType:
    """A stand-in for `numba` holding only what PyMatting imports from it.

    PyMatting asks for `njit`, `prange` and `pndindex` and nothing else. Without
    the JIT its functions still compute the same values, just slowly - and on
    this path they are **imported and never called**, because background removal
    runs with alpha matting off.
    """
    module = _new_module("numba")

    def njit(*args: Any, **kwargs: Any) -> Any:
        """Accept both `@njit` and `@njit(...)`, and change nothing."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(function: Any) -> Any:
            return function

        return decorate

    module.njit = njit  # type: ignore[attr-defined]
    module.jit = njit  # type: ignore[attr-defined]
    module.prange = range  # type: ignore[attr-defined]
    module.pndindex = np.ndindex  # type: ignore[attr-defined]
    return module


def install_numba_fallback() -> bool:
    """Stand in for `numba` **only when the real one will not load**.

    Smart App Control blocks unsigned binaries, and `numba/_devicearray.pyd` is
    one it blocks **intermittently** on this class of machine: the same file
    loads one hour and is refused the next (confirmed in the CodeIntegrity log,
    events 3033 and 3077). Every path through `rembg` imports it, because
    `rembg` imports PyMatting eagerly and PyMatting imports numba at module
    level - so an intermittently blocked file becomes an intermittently failing
    generation, tens of seconds in.

    **The real numba is tried first and kept whenever it works.** This is not a
    replacement for it; it is what keeps a generation alive on the runs where
    the operating system refuses to load it.

    Returns:
        True if the real numba is in use, False if the stand-in was installed.
    """
    if "numba" in sys.modules:
        return True
    try:
        import numba  # noqa: F401  - imported for the side effect of proving it loads
    except Exception as exc:  # noqa: BLE001 - ImportError, OSError, anything the loader raises
        sys.modules["numba"] = _make_numba()
        print(
            f"[shims] numba would not load ({type(exc).__name__}: {exc}); "
            "using a stand-in so background removal still runs",
            file=sys.stderr,
        )
        return False
    return True


def _miopen_batch_norm_works() -> bool:
    """Try one tiny batch norm on the GPU and report whether MIOpen served it.

    On the ROCm 10.0 Windows wheels (2026-09-02), MIOpen compiles its
    batch-norm kernels with hiprtc at first use and the compile fails with
    `'type_traits' file not found` — the wheels ship no C++ standard library
    for hiprtc to include — surfacing as `miopenStatusUnknownError`.
    Convolutions use precompiled solvers and are unaffected.
    """
    if not torch.cuda.is_available():
        return True
    try:
        # Weight and bias are what routes the call to MIOpen; without them
        # torch already picks its native kernel and the probe would prove
        # nothing.
        F.batch_norm(
            torch.randn(2, 4, 8, 8, device="cuda"),
            torch.zeros(4, device="cuda"),
            torch.ones(4, device="cuda"),
            weight=torch.ones(4, device="cuda"),
            bias=torch.zeros(4, device="cuda"),
        )
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def install_native_batch_norm_if_needed() -> bool:
    """Route `F.batch_norm` around MIOpen when MIOpen cannot run it (see above).

    torch's own HIP kernel takes over — same operation, no MIOpen involved —
    while every other operator keeps its MIOpen path. The probe costs one tiny
    kernel at load time and nothing thereafter.

    Returns:
        True if the reroute was installed, False if MIOpen batch norm works.
    """
    if _miopen_batch_norm_works():
        return False
    original = F.batch_norm

    def _batch_norm_without_miopen(*args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            return original(*args, **kwargs)

    F.batch_norm = _batch_norm_without_miopen
    print(
        "[shims] MIOpen cannot run batch_norm (hiprtc build failure); "
        "using torch's native kernel for batch_norm only",
        file=sys.stderr,
    )
    return True


def install(head_chunk: int = DEFAULT_HEAD_CHUNK, fp32_attention: bool | None = None) -> bool:
    """**Call this before importing the upstream package.**

    Puts every replacement into `sys.modules`.

    Args:
        head_chunk: Attention heads computed at once. Only applies to fp32
            arithmetic.
        fp32_attention: Whether to compute attention in fp32 over chunked heads.
            None **decides by measurement**: False when flash or mem-efficient
            is available (its online softmax cannot overflow in fp16, and it is
            **10-20x faster in practice**), True otherwise, because only the
            math backend remains and fp16 exceeds 65504 there.

    Returns:
        **Whether fast attention is in effect.** This goes into `metrics` as a
        record; it was what made the "4x slower through hearth" problem of
        2026-09-01 diagnosable.
    """
    use_fp32 = (not fast_attention_available()) if fp32_attention is None else fp32_attention
    install_numba_fallback()
    install_native_batch_norm_if_needed()
    _install_spconv()
    sys.modules["flash_attn"] = _make_flash_attn(head_chunk, use_fp32)
    _install_kaolin()
    _install_absent("open3d", "only used on the text-to-3D path")
    if use_fp32:
        _install_dense_sdpa(head_chunk)
    print(
        f"[shims] fast attention={'no' if use_fp32 else 'yes'} (fp32={use_fp32})",
        file=sys.stderr,
    )
    return not use_fp32


def install_absent_nvdiffrast() -> None:
    """Install a stand-in for `nvdiffrast`.

    Upstream's `postprocessing_utils` imports `nvdiffrast.torch as dr` at the
    top, but `dr` is only ever used for **texture baking and UV unwrapping**,
    neither of which appears on the `postprocess_mesh(fill_holes=True)` path
    this runner calls. Rasterization is replaced on the `utils3d.torch` side by
    `raster.install()`.
    """
    _install_absent("nvdiffrast", "texture baking only (not produced on this machine)")
    _install_absent("nvdiffrast.torch", "texture baking only (not produced on this machine)")
