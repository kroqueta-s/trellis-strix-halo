# SPDX-License-Identifier: MIT
"""Report mesh health as numbers (**to back up visual impressions with data**).

Three things are measured: **connected components** (free-floating debris),
**boundary loops** (holes), and dimensions. **Earlier checks looked at neither
components nor holes, so debris and holes passed straight through.** Stippling
in a render (`tools/render_mesh.py` scatters points, so a sparse silhouette can
look like a small white blob) and **actual free-floating debris** are different
things.

Example::

    python tools/mesh_report.py mesh.ply [mesh2.ply ...]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def boundary_loops(mesh: trimesh.Trimesh) -> list[dict[str, object]]:
    """Group boundary edges (edges with exactly one adjacent face) into loops.

    **The fix differs depending on whether a hole is one large opening or a
    scatter of small cracks.** A count alone cannot tell them apart, so the size
    and position of every loop are reported as well.
    """
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen: set[int] = set()
    loops: list[dict[str, object]] = []
    scale = float(np.max(mesh.bounding_box.extents))
    for start in list(adj):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            seen.add(v)
            stack.extend(adj[v])
        pts = mesh.vertices[list(comp)]
        extent = pts.max(0) - pts.min(0)
        loops.append(
            {
                "verts": len(comp),
                "extent": np.round(extent, 4).tolist(),
                "ratio": float(np.max(extent)) / max(scale, 1e-12),
                "center": np.round(pts.mean(0), 3).tolist(),
            }
        )
    return sorted(loops, key=lambda item: -int(item["verts"]))


def report(path: Path, top: int = 8) -> dict[str, object]:
    """Inspect one mesh, print the findings, and return the summary."""
    mesh = trimesh.load(path, process=False)
    extents = mesh.bounding_box.extents
    parts = mesh.split(only_watertight=False)
    sizes = np.array([len(p.faces) for p in parts])
    order = np.argsort(sizes)[::-1]

    print(f"\n=== {path.name} ===")
    print(f"  vertices {len(mesh.vertices):,} / faces {len(mesh.faces):,}")
    print(f"  watertight={mesh.is_watertight}  extents={np.round(extents, 4).tolist()}")
    print(f"  connected components: {len(parts)}")

    main_faces = int(sizes[order[0]]) if len(parts) else 0
    stray = int(sizes.sum() - main_faces)
    print(f"  largest component holds {main_faces / max(len(mesh.faces), 1) * 100:.2f}% of faces")
    print(f"  faces outside it: {stray:,} ({stray / max(len(mesh.faces), 1) * 100:.2f}%)")

    loops = boundary_loops(mesh)
    if loops:
        stray_edges = sum(int(loop["verts"]) for loop in loops)
        print(f"  boundary loops: {len(loops)} (boundary vertices {stray_edges})")
        for loop in loops[:top]:
            print(
                f"    vertices {loop['verts']:>5}  size {loop['extent']}  "
                f"longest edge is {float(loop['ratio']) * 100:6.2f}% of the model  "
                f"center {loop['center']}"
            )
    else:
        print("  boundary loops: 0 (no holes)")

    for rank, i in enumerate(order[: min(top, len(parts))]):
        part = parts[i]
        size = part.bounding_box.extents
        longest = float(np.max(size)) / float(np.max(extents))
        print(
            f"    #{rank + 1}: faces {len(part.faces):>8,}  "
            f"longest edge is {longest * 100:6.2f}% of the model  "
            f"center {np.round(part.centroid, 3).tolist()}"
        )
    return {
        "path": str(path),
        "components": len(parts),
        "main_face_ratio": main_faces / max(len(mesh.faces), 1),
        "stray_faces": stray,
        "boundary_loops": len(loops),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the connected components of a mesh")
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    for m in args.meshes:
        report(Path(m), args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
