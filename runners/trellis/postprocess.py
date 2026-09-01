# SPDX-License-Identifier: MIT
"""Post-processing for the generated mesh. **Follow upstream; add no judgement of our own.**

Upstream TRELLIS calls `postprocess_mesh()` from inside `to_glb()`, which does

1. `pyvista.decimate(0.95)` (reduce to 5 % of the faces), then
2. `_fill_holes`: **rasterize many views, compute a visibility ratio per face,
   min-cut, then fill small holes**.

**Only step 2 runs here, with its procedure and thresholds unchanged** (the
implementation is in `fill_holes.py`). Step 1 is skipped: upstream decimates
because it is producing a coloured GLB, whereas this mesh is raw material for
downstream scaling and repair (`forge`), so there is no reason to discard detail.

The missing `nvdiffrast` rasterizer is replaced by `raster.install()`.
**Upstream code is not modified.**

## One thing is added on top of upstream

**Free-floating debris is dropped by size and by thinness**
(`TRELLIS_DROP_SMALL_PARTS` and `TRELLIS_DROP_THIN_PARTS` in `.env`).

Upstream has no such treatment. Its visibility test targets faces that are never
visible, so **debris floating in open air is visible and passes straight
through**. Measured on the sample `i2i_00038_.png` (2026-09-01):

- trellis: of the 831 components outside the body, **37 components and 118,000
  faces (77.1 %) were inside it, and 794 components with 35,064 faces outside**.
- hi3dgen: of the 968 components outside the body, 27 with 3,008 faces were
  inside, and **941 with 30,048 faces (90.9 %) outside**.

Upstream's decimation (0.95) only takes the count from 832 to 204, so **the
finished result still contains floating debris**. That is a real defect for
printing, which is the one place this runner adds something of its own.
**Always record how much was dropped.**
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import trimesh

from . import config, raster, shims


@dataclass
class CleanStats:
    """What post-processing changed, and by how much. **Numbers so nothing vanishes silently.**"""

    faces_before: int = 0
    faces_after: int = 0
    fill_holes_sec: float = 0.0
    fill_holes_removed: int = 0
    parts_before: int = 0
    parts_after: int = 0
    dropped_parts: int = 0
    dropped_faces: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The shape that goes straight into `metrics`."""
        return {
            "faces_before": self.faces_before,
            "faces_after": self.faces_after,
            "fill_holes_sec": round(self.fill_holes_sec, 2),
            "fill_holes_removed": self.fill_holes_removed,
            "parts_before": self.parts_before,
            "parts_after": self.parts_after,
            "dropped_parts": self.dropped_parts,
            "dropped_faces": self.dropped_faces,
            "warnings": self.warnings,
        }


def _say(progress: Callable[[str, str], None] | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)


def fill_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut invisible faces and fill small holes (**upstream's procedure and thresholds**).

    The implementation lives in `fill_holes.py`. It was transcribed rather than
    called directly because **upstream's Python side has one pathological
    inefficiency that dominated on this machine**: it walks a CUDA tensor element
    by element, spending 35 seconds on 360,000 GPU reads.
    **Neither the algorithm nor the thresholds were changed**, which
    `tests/test_fill_holes.py` verifies by comparing against upstream one to one.

    Args:
        vertices: `[V, 3]`.
        faces: `[F, 3]`.
        progress: Where to report the stage.
        stats: Where to record what happened.

    Returns:
        The post-processed `(vertices, faces)`. **Returns the input unchanged on
        failure**, because post-processing improves quality and does not decide
        whether generation succeeded.
    """
    import time

    from . import fill_holes as impl

    impl.ensure_upstream_on_path(str(config.TRELLIS_REPO))
    started = time.perf_counter()
    before = len(faces)
    try:
        if "trellis" not in sys.modules:
            shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
        shims.install_absent_nvdiffrast()
        raster.install()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        v = torch.tensor(vertices, dtype=torch.float32, device=device)
        f = torch.tensor(faces, dtype=torch.int32, device=device)
        # **The upstream function goes quiet for tens of seconds.** Without a
        # heartbeat there is no way to tell progress from a hang.
        from .pipeline import _DeviceWatch

        with _DeviceWatch(
            progress=progress,
            stage="post-processing",
            heartbeat_sec=config.HEARTBEAT_SEC,
            limit_gb=config.VRAM_LIMIT_GB,
        ):
            v, f = impl.fill_holes(
                v,
                f,
                max_hole_size=config.FILL_HOLES_MAX_SIZE,
                max_hole_nbe=config.FILL_HOLES_MAX_NBE,
                resolution=config.FILL_HOLES_RESOLUTION,
                num_views=config.FILL_HOLES_VIEWS,
                progress=progress,
            )
        vertices = v.cpu().numpy()
        faces = f.cpu().numpy()
    except Exception as exc:  # noqa: BLE001 - never fail generation over post-processing
        message = (
            f"removing invisible faces failed (mesh returned unchanged): "
            f"{type(exc).__name__}: {exc}"
        )
        print(f"[postprocess] {message}", file=sys.stderr)
        if stats is not None:
            stats.warnings.append(message)
        return vertices, faces
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if stats is not None:
        stats.fill_holes_sec = time.perf_counter() - started
        stats.fill_holes_removed = before - len(faces)
    return vertices, faces


def drop_small_parts(
    mesh: trimesh.Trimesh,
    min_ratio: float,
    min_thick_ratio: float = 0.0,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> trimesh.Trimesh:
    """**Drop free-floating debris by size and thinness** (added on top of upstream).

    Components are judged by their axis-aligned bounding box against the model's
    longest side, and only components passing **both** tests are kept:

    1. **Longest extent at least `min_ratio`** (drops small debris). Spatial size
       rather than face count separates finely tessellated crumbs from genuine
       parts that merely have few faces.
    2. **Shortest extent at least `min_thick_ratio`** (drops flakes). Paper-thin
       flakes hovering about 1 % off the surface and lying parallel to it are
       **11-29 % long, so test 1 alone lets them straight through** (they render
       as dark speckles and tabs on the surface).

    Measured on the sample `i2i_00038_.png`: at 10 % the arms and hands (15 % of
    the model) survive and the visible debris (6.5 % and below) is gone. For
    thinness (2026-09-02), the remaining flakes were **0.1-1.4 % thick** and
    genuine parts (arms, panels) **11.8 % or more**, so 2 % separates them with
    an order of magnitude to spare. A tilted flake would measure thicker in an
    axis-aligned box, but every flake measured lay parallel to the surface and
    close to axis-aligned, so it did not matter.

    Args:
        mesh: The mesh to treat.
        min_ratio: Minimum size to keep, relative to the model's longest side.
            0 or less skips the test.
        min_thick_ratio: Minimum thickness to keep, on the same scale. 0 or less
            skips the test.
        progress: Where to report the stage.
        stats: Where to record what happened.

    Returns:
        The mesh without the debris. **The largest component is always kept.**
    """
    if min_ratio <= 0 and min_thick_ratio <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if stats is not None:
        stats.parts_before = len(parts)
    if len(parts) <= 1:
        if stats is not None:
            stats.parts_after = len(parts)
        return mesh

    whole = max(float(np.max(mesh.bounding_box.extents)), 1e-12)
    face_counts = np.array([len(p.faces) for p in parts])
    keep = np.ones(len(parts), dtype=bool)
    if min_ratio > 0:
        sizes = np.array([float(np.max(p.bounding_box.extents)) for p in parts])
        keep &= sizes / whole >= min_ratio
    if min_thick_ratio > 0:
        thicks = np.array([float(np.min(p.bounding_box.extents)) for p in parts])
        keep &= thicks / whole >= min_thick_ratio
    keep[int(np.argmax(face_counts))] = True  # always keep the largest component

    if stats is not None:
        stats.parts_after = int(keep.sum())
        stats.dropped_parts = int((~keep).sum())
        stats.dropped_faces = int(face_counts[~keep].sum())
    _say(
        progress,
        "drop_parts",
        f"dropping free-floating debris "
        f"({int((~keep).sum())} parts / {int(face_counts[~keep].sum())} faces)",
    )
    if not (~keep).any():
        return mesh
    return trimesh.util.concatenate([p for p, k in zip(parts, keep, strict=True) if k])


def clean(
    mesh: trimesh.Trimesh,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[trimesh.Trimesh, CleanStats]:
    """Apply all post-processing.

    Returns:
        `(post-processed mesh, record)`.
    """
    stats = CleanStats(faces_before=len(mesh.faces))

    if config.FILL_HOLES:
        vertices, faces = fill_holes(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            progress,
            stats,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh = drop_small_parts(mesh, config.DROP_SMALL_PARTS, config.DROP_THIN_PARTS, progress, stats)
    stats.faces_after = len(mesh.faces)
    return mesh, stats
