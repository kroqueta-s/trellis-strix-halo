# SPDX-License-Identifier: MIT
"""Verify the textured renderer **without a GPU, weights or a model**.

The splatted views draw one atlas sample per vertex, which is vertex colours
again: they cannot show a seam, and a seam is the whole reason to look at a
texture. `_render_textured` interpolates the UV over each triangle instead and
reads the atlas per pixel, and the two things it can get quietly wrong are the
direction of the V axis and what happens to a triangle smaller than a pixel.
Both have an exact answer here, on shapes small enough to check by hand.

Run it with either virtual environment; nothing in it touches the device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.render_mesh import _covered_pixels, _render_textured, _sample_bilinear  # noqa: E402

SIZE = 64


def _quad(z: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A square facing the camera, its UVs spanning the whole atlas."""
    verts = np.array(
        [[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    return verts, normals, uvs, faces


def test_v_axis_points_up() -> None:
    """**v = 0 reads the bottom row of the image**, which is trimesh's convention.

    Getting this backwards flips every texture vertically, and on an atlas of
    scattered charts a flip does not look like a flip - it looks like noise.
    """
    image = np.zeros((2, 2, 3))
    image[0] = [1.0, 0.0, 0.0]  # top row red
    image[1] = [0.0, 1.0, 0.0]  # bottom row green
    uv = np.array([[0.5, 0.0], [0.5, 1.0]])
    sampled = _sample_bilinear(image, uv)
    assert np.allclose(sampled[0], [0.0, 1.0, 0.0]), sampled[0]
    assert np.allclose(sampled[1], [1.0, 0.0, 0.0]), sampled[1]


def test_the_atlas_lands_on_the_right_half_of_the_shape() -> None:
    """A left-right split in the atlas comes out as a left-right split on screen."""
    image = np.zeros((2, 2, 3))
    image[:, 0] = [1.0, 0.0, 0.0]
    image[:, 1] = [0.0, 1.0, 0.0]
    verts, normals, uvs, faces = _quad()
    view = _render_textured(verts, normals, uvs, faces, image, SIZE, 0.0, 0.0, True)
    middle = SIZE // 2
    left = view[middle, SIZE // 4]
    right = view[middle, 3 * SIZE // 4]
    assert left[0] > left[1], left
    assert right[1] > right[0], right


def test_a_sub_pixel_triangle_still_draws() -> None:
    """**A triangle smaller than a pixel keeps the pixel its centroid is in.**

    At 200,000 faces in a 512-pixel view most triangles are sub-pixel; without
    this the model would be drawn full of holes, which reads as a bad mesh.
    """
    corners = np.array([[[10.2, 10.2], [10.4, 10.2], [10.3, 10.4]]])
    pixel, triangle, weights = _covered_pixels(corners, SIZE)
    assert pixel.size == 1, pixel
    assert int(pixel[0]) == 10 * SIZE + 10, int(pixel[0])
    assert triangle.tolist() == [0]
    assert np.allclose(weights, 1.0 / 3.0)


def test_the_nearer_surface_wins() -> None:
    """Two quads over each other: the one in front is what is drawn."""
    image_far = np.zeros((2, 2, 3))
    image_far[:] = [1.0, 0.0, 0.0]
    image_near = np.zeros((2, 2, 3))
    image_near[:] = [0.0, 0.0, 1.0]
    far_v, far_n, uvs, faces = _quad(z=-0.5)
    near_v, near_n, _, _ = _quad(z=0.5)

    verts = np.concatenate([far_v, near_v])
    normals = np.concatenate([far_n, near_n])
    all_uvs = np.concatenate([uvs, uvs])
    all_faces = np.concatenate([faces, faces + 4])
    # One atlas for both quads: the near one's UVs point at the blue half.
    image = np.zeros((2, 4, 3))
    image[:, :2] = [1.0, 0.0, 0.0]
    image[:, 2:] = [0.0, 0.0, 1.0]
    scaled = all_uvs.copy()
    scaled[:4, 0] = all_uvs[:4, 0] * 0.5
    scaled[4:, 0] = 0.5 + all_uvs[4:, 0] * 0.5

    view = _render_textured(verts, normals, scaled, all_faces, image, SIZE, 0.0, 0.0, True)
    centre = view[SIZE // 2, SIZE // 2]
    assert centre[2] > centre[0], centre


def main() -> int:
    """Run every test."""
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
