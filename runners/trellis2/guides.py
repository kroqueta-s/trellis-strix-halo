# SPDX-License-Identifier: MIT
"""Decoding the texture latent onto a grid of our choosing.

**The colours land where the decoder's octree says, not where the mesh is.**
The texture decoder turns a latent at `resolution / 16` into voxels by
subdividing four times, and at each subdivision something has to say which of
the eight children of a voxel exist. In `image_to_mesh` that came from the
shape decoder's own subdivisions, so the colours sit on the surface the decoder
drew - and the mesh that gets printed is not that surface: the carve rebuilds
it on a lattice, up to a cell away, which at a 1024 decode is two voxels and
outside the eight a trilinear sample reads. Measured 2026-09-13: 52% of the
print mesh's vertices came back black at 1024.

`texture_mesh` does not have the problem, because the encoder leaves the
subdivisions it used in the latent's spatial cache and the decoder undoes them
exactly. That is the whole trick, and this module is that trick without an
encoder: build the subdivisions from **the mesh we want coloured**, and the
decoder puts a colour on every voxel of it that the latent can reach.

**Nothing here edits the model.** `decode_on_grid` walks the decoder's own
blocks the way its `forward` does, and passes a mask where upstream passes the
shape decoder's.
"""

from __future__ import annotations

from typing import Any

import torch

# The child order the upsampling uses: `subidx // 2**i % 2` is the offset along
# axis i (`SparseChannel2Spatial.forward`), so the index is x + 2y + 4z.
_OFFSETS = torch.tensor([[i % 2, (i // 2) % 2, (i // 4) % 2] for i in range(8)], dtype=torch.int64)

# Coordinates are packed into one integer to be compared. 2048 covers every
# resolution this runner can be asked for; the constructor checks it.
_STRIDE = 2048


def _keys(coords: torch.Tensor) -> torch.Tensor:
    """One integer per coordinate, so that membership is a search, not a join."""
    return (coords[:, 0] * _STRIDE + coords[:, 1]) * _STRIDE + coords[:, 2]


def _contains(sorted_keys: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Which of `keys` are in `sorted_keys`."""
    if sorted_keys.numel() == 0:
        return torch.zeros_like(keys, dtype=torch.bool)
    index = torch.searchsorted(sorted_keys, keys).clamp(max=sorted_keys.numel() - 1)
    return sorted_keys[index] == keys


class SubdivisionGuides:
    """The subdivision masks that grow a latent's coords into a given grid.

    Args:
        target: `[M, 3]` integer voxel coordinates at the finest resolution -
            the grid the colours are wanted on.
        levels: How many times the decoder subdivides (its blocks minus one).
    """

    def __init__(self, target: torch.Tensor, levels: int) -> None:
        target = torch.as_tensor(target, dtype=torch.int64).reshape(-1, 3)
        if int(target.max()) >= _STRIDE << levels or int(target.min()) < 0:
            raise ValueError(f"target coordinates do not fit {_STRIDE} per axis")
        self.levels = int(levels)
        # One sorted key set per level, coarsest first: a voxel exists at a
        # level when any of its descendants exists at the finest one.
        self._levels: list[torch.Tensor] = []
        for level in range(self.levels + 1):
            shift = self.levels - level
            coords = target >> shift if shift else target
            self._levels.append(torch.unique(_keys(coords)))

    def to(self, device: Any) -> SubdivisionGuides:
        """Move the key sets to `device`, so the masks are not copied per level."""
        self._levels = [keys.to(device) for keys in self._levels]
        return self

    def mask(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Which children of `coords` to keep when subdividing out of `level`.

        Returns `[N, 8]`, positive where the child is wanted: the decoder's own
        blocks read the sign (`subdiv.feats > 0`).
        """
        coords = torch.as_tensor(coords, dtype=torch.int64).reshape(-1, 3)
        children = (coords[:, None, :] << 1) + _OFFSETS.to(coords.device)
        present = _contains(
            self._levels[level + 1].to(coords.device), _keys(children.reshape(-1, 3))
        )
        return torch.where(present.reshape(-1, 8), 1.0, -1.0)

    def reachable(self, latent_coords: torch.Tensor) -> dict[str, int]:
        """How much of the target the latent can reach, counted before decoding.

        **A voxel with no ancestor in the latent has nothing to subdivide**, and
        no guide can conjure it. Reporting the number is the difference between
        a texture that came out dark and a texture that could not have come out
        otherwise.
        """
        latent = torch.as_tensor(latent_coords, dtype=torch.int64).reshape(-1, 3)
        finest = self._levels[-1]
        coords = torch.stack(
            [finest // (_STRIDE * _STRIDE), finest // _STRIDE % _STRIDE, finest % _STRIDE], dim=1
        )
        under = _contains(torch.unique(_keys(latent)), _keys(coords >> self.levels))
        return {
            "target_voxels": int(finest.numel()),
            "reachable_voxels": int(under.sum()),
            "latent_voxels": int(latent.shape[0]),
        }


def decode_on_grid(decoder: Any, latent: Any, guides: SubdivisionGuides) -> Any:
    """Run `decoder` over `latent`, subdividing where `guides` says.

    **This is upstream's `SparseUnetVaeDecoder.forward` with our mask.** It is
    repeated here rather than called because the mask has to be built against
    the coordinates the decoder actually holds at each level: handing it a list
    made in advance would line the masks up by position and misplace them if
    anything about the order changed.

    **Gradients are off explicitly.** Upstream decodes inside the generation's
    own `no_grad`; this runs later, in the post-processing, where nothing has
    turned autograd off - and keeping the activations of a four-level decode
    asks for 29 GB and an out-of-memory at 512 (measured 2026-09-13).
    """
    import torch.nn.functional as functional

    levels = len(decoder.blocks) - 1
    if levels != guides.levels:
        raise ValueError(f"the decoder subdivides {levels} times, the guides {guides.levels}")

    with torch.no_grad():
        h = decoder.from_latent(latent)
        h = h.type(decoder.dtype)
        for i, stage in enumerate(decoder.blocks):
            for j, block in enumerate(stage):
                if i < levels and j == len(stage) - 1:
                    mask = guides.mask(h.coords[:, 1:], i).to(h.feats.dtype)
                    if not bool((mask > 0).any()):
                        raise RuntimeError(f"no voxel of the target survives subdivision {i}")
                    h = block(h, subdiv=h.replace(mask))
                else:
                    h = block(h)
        h = h.type(latent.dtype)
        h = h.replace(functional.layer_norm(h.feats, h.feats.shape[-1:]))
        return decoder.output_layer(h)
