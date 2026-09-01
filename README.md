# trellis-strix-halo

**[TRELLIS](https://github.com/microsoft/TRELLIS) image-to-mesh on AMD Strix Halo
(gfx1151), Windows, ROCm — with no CUDA-only package installed.**

Upstream TRELLIS needs `spconv`, `flash-attn` or `xformers`, `kaolin` and
`nvdiffrast`. None of those exist for Windows + ROCm. This repository supplies
pure-torch replacements that are injected at launch time, so **upstream code is
cloned and run unmodified**.

This is a runner for [hearth](https://github.com/kroqueta-s/hearth): it speaks
one JSON object per line over stdin/stdout. It also runs standalone.

## Install

```powershell
git clone https://github.com/kroqueta-s/trellis-strix-halo
cd trellis-strix-halo
.\install.ps1
```

That creates a virtual environment, installs ROCm PyTorch, clones upstream at a
pinned commit, downloads the weights (3.1 GB), writes `.env`, and **verifies the
replacements against exact references** before you trust any mesh.

Requirements: Windows, an AMD GPU with ROCm 7.2.1 drivers, Python 3.12, ~10 GB
of disk, and about 16 GB of VRAM at peak.

## Use

```powershell
.venv\Scripts\python.exe -m runners.trellis
```

Then write one request per line:

```json
{"id": 1, "method": "capabilities"}
{"id": 2, "method": "image_to_mesh", "params": {"image_path": "C:/in.png", "out_dir": "C:/out"}}
```

`capabilities` answers without loading weights. `image_to_mesh` writes `raw.ply`
and returns vertex/face counts plus timings. Parameters: `ss_steps`,
`slat_steps`, `ss_guidance`, `slat_guidance`, `seed`.

## What is replaced, and why it is safe

| Upstream dependency | Replacement | Verified by |
|---|---|---|
| `spconv` (sparse conv) | `runners/trellis/shims.py` — submanifold convolution in torch | Exact agreement with a dense `F.conv3d` reference |
| `flash_attn` (sparse attention) | Same file — `F.scaled_dot_product_attention` | Agreement with a naive attention reference |
| `nvdiffrast` (rasterizer for post-processing) | `runners/trellis/raster.py` — z-buffer rasterizer in torch | A box hidden inside a box is never visible; the near face wins |
| `kaolin.utils.testing`, `open3d` | Small stands-in; unused on this path | Import-time only |

Run the checks yourself:

```powershell
.venv\Scripts\python.exe tests\test_shims.py
.venv\Scripts\python.exe tests\test_raster.py
```

Submanifold convolution is exactly a dense convolution restricted to occupied
voxels, so it can be checked against a reference without the original library.
That is the reason this approach is trustworthy rather than merely plausible.

## Measurements (gfx1151, Radeon 8060S, 32 GB dedicated VRAM)

One image, `ss_steps = slat_steps = 6`, 2026-09-01:

| Stage | Time |
|---|--:|
| Load weights | 24 s |
| Sparse structure | 26 s |
| Structured latent | 40 s |
| Decode to mesh | 10 s |
| Post-processing (150 views, 1024²) | 97 s |
| **Total** | **211 s** |

Peak VRAM 12.6 GB. Output 468,994 faces, watertight, no boundary edges and no
non-manifold edges after downstream repair.

Rasterizer throughput on a 697k-face mesh: 61 ms per view at 128², 244 ms at
1024². Upstream's default of 1000 views would take 244 s, which is why the
default here is 150.

Attention is 10–20× faster when `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is
set **before** torch is imported (4096-token attention: 135 ms → 12 ms). The
runner sets it for you; setting it afterwards has no effect.

## Limits

- **No texture.** Texture baking needs `nvdiffrast` for real; only the
  rasterizer used by hole filling is replaced here.
- **No decimation.** Upstream reduces to 5 % of faces before post-processing.
  This runner keeps every face, because the mesh is an input to downstream
  scaling and repair.
- **One addition that upstream does not have.** Upstream removes faces that are
  never visible; parts floating in open air stay visible and survive. Measured
  on one sample, 794 of 831 stray parts were outside the body. The runner drops
  free-floating parts below 10 % of the model's longest side, and always records
  how much it dropped. Set `TRELLIS_DROP_SMALL_PARTS=0` to disable.
- Generation time varies widely on this hardware for identical settings. Do not
  use it as a pass/fail signal.

## License

MIT (see [LICENSE](LICENSE)). Upstream TRELLIS is MIT; its weights
(`microsoft/TRELLIS-image-large`) are MIT. This repository contains no upstream
code and no weights.
