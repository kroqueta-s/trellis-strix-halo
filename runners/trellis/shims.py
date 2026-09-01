# SPDX-License-Identifier: MIT
"""TRELLIS を gfx1151 / Windows / ROCm で動かすための**起動側のシム**。

**ベンダーコードは書き換えない**（再 clone・更新で壊れるため）。`trellis` を import する
**前**に `install()` を呼び、`sys.modules` へ代替を差し込む。`runners/hunyuan3d/shape.py`
の `_install_sdpa_shim` と同じ流儀である。

差し替えるのは 4 つ。**どれも「必須経路にあるが Windows+ROCm では入手できない」もの**で、
実際に使われる API の面積は小さい（`docs/02_port_report.md` の G-1 を見よ）。

- `spconv`：疎な畳み込み。**submanifold・stride 1 の `SubMConv3d` しか使われない**ので、
  座標の線形化＋近傍の gather で再現する
- `flash_attn`：疎なアテンション。`F.scaled_dot_product_attention` で再現する
- `kaolin`：FlexiCubes が `check_tensor`（形の検査だけ）を import する
- `open3d`：`trellis_text_to_3d` が先頭で import するだけ。**画像→メッシュの経路では
  1 度も使わない**ので、触ったら落ちる殻を置く

**正しさは検算できる。** submanifold 畳み込みは「密な `F.conv3d` を活性ボクセルだけに
制限したもの」と厳密に一致するので、密な参照実装と突き合わせられる
（`tests/test_trellis_shims.py`）。
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 一度に計算するアテンションのヘッド数。gfx1151 では Hunyuan3D で 4 が最良だった。
DEFAULT_HEAD_CHUNK = 4

# 疎畳み込みで一度に処理する出力ボクセル数。**VRAM ピークを決める。**
# 専用 VRAM は 32GB しかなく、溢れると共有メモリへ落ちて数倍遅くなる。
VOXEL_CHUNK = 65536

# **差し替える前の本物**を捕まえておく。シムの中から `F.scaled_dot_product_attention` を
# 呼ぶと、差し替えたあとは自分自身を呼ぶことになり、二重にヘッド分割して遅くなる。
_TORCH_SDPA = F.scaled_dot_product_attention


def _new_module(name: str, is_package: bool = False) -> types.ModuleType:
    """`__spec__` と `__file__` を持つ空モジュールを作る。

    **`__spec__` は必須。** `transformers` は `importlib.util.find_spec("flash_attn")` で
    有無を調べるので、`__spec__` が無いと `ValueError: flash_attn.__spec__ is None` で
    **無関係な import が落ちる**（2026-09-01 に踏んだ）。
    """
    module = types.ModuleType(name)
    module.__file__ = f"<hearth shim: {name}>"
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------------------
# spconv の代替
# --------------------------------------------------------------------------------------


class SparseConvTensor:
    """`spconv.pytorch.SparseConvTensor` の器だけを再現する。

    TRELLIS の `SparseTensor`（`trellis/modules/sparse/basic.py`）は、この型の
    **属性を直接触る**：`features` / `indices` / `spatial_shape` / `batch_size` /
    `grid` / `voxel_num` / `indice_dict` / `benchmark` / `benchmark_record` /
    `thrust_allocator` / `_timer` / `force_algo` / `int8_scale` / `_features`、および
    `dense()`。**位置引数の順序も本物に合わせる**（`replace()` が 7 個を位置で渡すため）。

    `features` が `_features` を返すプロパティであることも本物と同じにする。TRELLIS は
    2 次元へ reshape した特徴で構築したあと `_features` に多次元の実体を入れ直す。
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
        """特徴だけ差し替えた新しいテンソルを返す（座標と rulebook は共有する）。"""
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
        """密なテンソルへ展開する（`SparseTensor.dense()` から呼ばれる）。"""
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
    """`spconv.ConvAlgo` の代替。**値は使わない**（アルゴリズムの選択肢が無いため）。"""

    Native = "native"
    MaskImplicitGemm = "implicit_gemm"
    MaskSplitImplicitGemm = "mask_split_implicit_gemm"


def _neighbor_index_map(coords: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """submanifold 畳み込みの rulebook を作る。

    出力ボクセル `i` について、カーネル位置 `k` に対応する**入力ボクセルの添字**を返す。
    活性でない近傍は `-1`。座標の線形化＋ソート済み配列への `searchsorted` で引く
    （密な索引表を持たないので、解像度が上がっても記憶量は座標の分しか要らない）。

    **`int32` で返して `[K, N]` の器へ直接書き込む。** `int64` で 27 枚作って
    `torch.stack` すると、解像度 256（活性ボクセル 180 万）で 400MB を超える一時領域が
    2 つ同時に生きる。**専用 VRAM は 32GB しかなく、溢れると共有メモリへ落ちて
    数倍遅くなる**（2026-09-01 に実際に踏んだ。`torch.OutOfMemoryError` まで行った）。

    Args:
        coords: `[N, 4]` の `(batch, z, y, x)`。
        kernel_size: カーネルの一辺。1 か 3 しか来ない。

    Returns:
        `[K, N]` の `int32`。`K = kernel_size ** 3`。活性でない近傍は `-1`。
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


class _SubMConv3dImpl(nn.Module):
    """submanifold な 3D 疎畳み込み（stride 1・出力座標＝入力座標）。

    **重みの並びは spconv 2.x の KRSC**：`[out_channels, kD, kH, kW, in_channels]`。
    実際の ckpt で確認した（例：`upsample.0.out_layers.0.conv.weight` が
    `[192, 3, 3, 3, 768]`）。ここを取り違えると**黙って違う形が出る**ので変えないこと。

    畳み込みの向きは相関（`out[p] = Σ_k W[k] · in[p + k - center]`）で、
    これは活性でないボクセルを 0 とした密な `F.conv3d(padding=k//2)` と厳密に一致する。
    `tests/test_trellis_shims.py` がその一致を検算している。
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
            raise NotImplementedError("dilation は未対応（TRELLIS の経路では使われない）")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(
            torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _rulebook(self, x: SparseConvTensor) -> torch.Tensor:
        """rulebook を作る。`indice_key` があれば疎テンソルの `indice_dict` へ載せて使い回す。"""
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
        n = feats.shape[0]
        # [out, K, in] -> [K, in, out]（一度だけ作って使い回す）
        weight = self.weight.reshape(self.out_channels, -1, self.in_channels)
        w_all = weight.permute(1, 2, 0).contiguous().to(feats.dtype)
        kernels = w_all.shape[0]
        center = kernels // 2

        out = torch.empty(n, self.out_channels, device=feats.device, dtype=feats.dtype)
        bias = None if self.bias is None else self.bias.float()
        # **出力ボクセルを塊に切って処理する。** 全部まとめてやると、集めた特徴
        # `[N, in]` だけで解像度 256・入力 192ch のとき 700MB を超える。
        for start in range(0, n, VOXEL_CHUNK):
            end = min(start + VOXEL_CHUNK, n)
            acc = torch.zeros(end - start, self.out_channels, device=feats.device)
            for k in range(kernels):
                if kernels == 1 or k == center:
                    # 中心は必ず自分自身（submanifold なので出力座標＝入力座標）。
                    acc += (feats[start:end] @ w_all[k]).float()
                    continue
                idx = rulebook[k, start:end].long()
                valid = idx >= 0
                gathered = feats.index_select(0, idx.clamp_(min=0))
                gathered[~valid] = 0
                acc += (gathered @ w_all[k]).float()
            if bias is not None:
                acc += bias
            out[start:end] = acc.to(feats.dtype)

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
    """この経路では使われないはずの畳み込み。**呼ばれたら必ず落ちる**（黙って通さない）。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError(
            "stride 付きの SparseConv3d / SparseInverseConv3d は spconv の代替に無い。"
            "TRELLIS の画像→メッシュの経路には現れないはずなので、来たら経路が変わっている"
        )


def _install_spconv() -> None:
    """`spconv` と `spconv.pytorch` を差し込む（**本物があっても差し込む**）。"""
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
# flash_attn の代替
# --------------------------------------------------------------------------------------


def _attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_chunk: int, fp32: bool
) -> torch.Tensor:
    """`[B, S, H, D]` の 3 つ組を受けて `[B, S, H, D]` を返す。

    gfx1151 では flash も mem_efficient も無いので、実際に走るのは math backend である。
    math は `q @ k^T` を入力 dtype のまま実体化するので、**fp16 だと 65504 を超えて壊れる**
    （Hunyuan3D で鏡像入力を使い決定論的に再現した実績がある）。だから fp32 で計算し、
    ヘッドを分割して中間テンソルを小さく保つ。
    """
    out_dtype = q.dtype
    qt = q.transpose(1, 2)  # [B, H, S, D]
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if not fp32:
        # flash / mem-efficient が使えるときは online softmax なので fp16 のままで壊れない。
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
    """`cu_seqlens`（累積長）を `[(start, end), ...]` へ直す。"""
    bounds = cu_seqlens.tolist()
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def _make_flash_attn(head_chunk: int, fp32: bool) -> types.ModuleType:
    """`flash_attn` の**実際に呼ばれる 4 関数だけ**を持つモジュールを作る。

    TRELLIS が呼ぶのは `full_attn.py` / `windowed_attn.py` / `serialized_attn.py` の
    5 箇所で、いずれも位置引数だけを渡す。`causal` も `dropout` も使わない。
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
# kaolin / open3d の殻
# --------------------------------------------------------------------------------------


def _install_kaolin() -> None:
    """FlexiCubes が import する `kaolin.utils.testing.check_tensor` だけを用意する。

    FlexiCubes は `assert check_tensor(x, shape, throw=False)` の形でしか使わない。
    **形の検査をそのまま実装する**（真を返すだけにすると assert が意味を失うため）。
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
                problems.append(f"次元数が違う: {tensor.dim()} != {len(expected)}")
            else:
                for i, (got, want) in enumerate(zip(tensor.shape, expected, strict=True)):
                    if want is not None and got != want:
                        problems.append(f"軸 {i} の大きさが違う: {got} != {want}")
        if dtype is not None and tensor.dtype != dtype:
            problems.append(f"dtype が違う: {tensor.dtype} != {dtype}")
        if device is not None and torch.device(device).type != tensor.device.type:
            problems.append(f"device が違う: {tensor.device} != {device}")
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
    """**名前を辿るのは許すが、呼んだら落ちる**代役。

    属性の参照だけで落とすと `class Foo: def f(self, m: o3d.geometry.TriangleMesh)` のような
    **注釈の評価**で巻き込まれる（`trellis_text_to_3d` がまさにそれ）。注釈は通し、
    **実際に呼ばれたときだけ**落とす。
    """

    def __init__(self, path: str, reason: str) -> None:
        self._path = path
        self._reason = reason

    def __getattr__(self, attr: str) -> _AbsentAttribute:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentAttribute(f"{self._path}.{attr}", self._reason)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{self._path} は本機では使えない（{self._reason}）")

    def __repr__(self) -> str:
        return f"<absent {self._path}>"


def _install_absent(name: str, reason: str) -> None:
    """import は通るが**呼んだら落ちる**殻を置く。黙って違う結果を返させない。

    **dunder だけは `AttributeError` を返す。** `inspect.getmodule` は `sys.modules` を
    総なめして `__file__` を見に来るので、ここで例外を投げると
    **無関係な import（torchvision）を巻き込んで落ちる**（2026-09-01 に踏んだ）。
    """

    module = _new_module(name, is_package=True)

    def _absent(attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _AbsentAttribute(f"{name}.{attr}", reason)

    module.__getattr__ = _absent  # type: ignore[attr-defined]
    sys.modules[name] = module


# --------------------------------------------------------------------------------------
# 密な SDPA の差し替え（Hunyuan3D と同じ理由）
# --------------------------------------------------------------------------------------


def _install_dense_sdpa(head_chunk: int) -> None:
    """`F.scaled_dot_product_attention` を fp32 計算＋ヘッド分割へ差し替える。

    TRELLIS の**密**なアテンション（`ATTN_BACKEND=sdpa`）は seq=4096 の DiT を 24 段
    通す。Hunyuan3D で同じ規模の math backend が fp16 で破綻したのと同じ条件なので、
    先回りして同じ手当てをしておく。**根拠は `runners/hunyuan3d/shape.py` の docstring。**
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
            raise NotImplementedError("差し替えた SDPA は素のアテンションだけを受ける")
        out_dtype = query.dtype
        heads = query.shape[1]
        outs: list[torch.Tensor] = []
        for i in range(0, heads, head_chunk):
            # **計算は fp32、実装は torch の本物**。自前で softmax を書くと、
            # 同じ大きさの中間テンソルを作ったうえに融合カーネルを捨てることになる。
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
# 入口
# --------------------------------------------------------------------------------------


def fast_attention_available() -> bool:
    """flash / mem-efficient が**実際に走るか**を小さなテンソルで試す。

    gfx1151 では `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を
    **torch を import する前に**置いたときだけ AOTriton の実装が有効になる
    （後から `os.environ` へ入れても効かないことを実測した）。
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
        except Exception:  # noqa: BLE001 - 使えないことを知りたいだけ
            continue
    return False


def install(head_chunk: int = DEFAULT_HEAD_CHUNK, fp32_attention: bool | None = None) -> bool:
    """**`trellis` を import する前に呼ぶ。** 代替をすべて `sys.modules` へ置く。

    Args:
        head_chunk: 一度に計算するアテンションのヘッド数。fp32 で計算するときだけ効く。
        fp32_attention: アテンションを fp32＋ヘッド分割で計算するか。
            None なら**実測で決める**：flash / mem-efficient が使えるなら False
            （online softmax なので fp16 でも溢れない・**実測で 10〜20 倍速い**）、
            使えないなら True（math backend しか無く、fp16 だと 65504 を超えて壊れる）。

    Returns:
        **速いアテンションが有効か。** これを `metrics` へ載せて記録に残す
        （2026-09-01 に「hearth 経由だと生成が 4 倍遅い」を切り分けるのに要った）。
    """
    use_fp32 = (not fast_attention_available()) if fp32_attention is None else fp32_attention
    _install_spconv()
    sys.modules["flash_attn"] = _make_flash_attn(head_chunk, use_fp32)
    _install_kaolin()
    _install_absent("open3d", "text→3D の経路でしか使わない")
    if use_fp32:
        _install_dense_sdpa(head_chunk)
    print(
        f"[shims] fast attention={'no' if use_fp32 else 'yes'} (fp32={use_fp32})",
        file=sys.stderr,
    )
    return not use_fp32


def install_absent_nvdiffrast() -> None:
    """`nvdiffrast` の殻を置く。

    上流の `postprocessing_utils` は先頭で `import nvdiffrast.torch as dr` するが、
    `dr` を実際に使うのは**テクスチャの焼き込みと UV 展開**だけで、こちらが呼ぶ
    `postprocess_mesh(fill_holes=True)` の経路には現れない。
    ラスタライズは `utils3d.torch` 側を `raster.install()` で差し替える。
    """
    _install_absent("nvdiffrast", "テクスチャ焼き込み専用（本機では出せない）")
    _install_absent("nvdiffrast.torch", "テクスチャ焼き込み専用（本機では出せない）")
