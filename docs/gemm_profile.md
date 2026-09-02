# Where the GPU time goes in one generation

Measured with [`tools/profile_gemm.py`](../tools/profile_gemm.py) on an ASUS
ProArt PX13 (Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, factory power
limits), Windows 11, ROCm 7.2.1, torch 2.9.1+rocm7.2.1, fast
attention on, 2026-09-02. Reference fp16 GEMM taken alongside:
25 TFLOPS at 2048³, 31 TFLOPS at 4096³ (rocBLAS). Sample:
`assets/sample.png`, upstream defaults (`ss_steps=slat_steps=25`), 23 215
active voxels.

Shares are of profiled device time; walls are from an unprofiled run. This
torch build has no Kineto, so shares are decision-grade rather than exact.

| Stage | Wall | GEMM | Attention | Other |
|---|--:|--:|--:|--:|
| structure | 22.9 s | 5.7 s (25 %) | 12.7 s (56 %) | 4.9 s |
| slat (dominant) | 52.4 s | 21.4 s (33 %) | 10.3 s (16 %) | 34.0 s |
| decode | 3.0 s | 0.4 s (9 %) | 1.0 s | 3.1 s |

The structure stage is attention-bound; the slat stage splits between GEMM,
attention, and the elementwise machinery of the sparse-convolution shim
(masks, gathers, copies). The largest GEMMs, all fp16:

| Role | M | N | K | Calls | rocBLAS TFLOPS |
|---|--:|--:|--:|--:|--:|
| **Sparse-conv shim (slat)** | **23215** | **128** | **2048** | **1188** | **1.0** |
| MLP down (structure) | 4096 | 1024 | 4096 | 1056 | 18.5 |
| MLP up (structure) | 4096 | 4096 | 1024 | 1056 | 26.1 |
| QKV (structure) | 4096 | 3072 | 1024 | 1056 | 31.5 |
| Out proj (structure) | 4096 | 1024 | 1024 | 3168 | 29.4 |
| Slat transformer | 4383 | 1024–4096 | 1024–4096 | ~6300 | 19–31 |
| Decode | 65536 | 96–192 | 96–768 | ~1300 | 11–21 |

**The single largest GEMM cost is the skinny sparse-conv projection
(N = 128): 14.5 s of the slat stage at 1.0 TFLOPS** — a 30× gap to what the
same silicon does at 4096³.

## With hipBLASLt (now the default)

Same measurement with `TORCH_BLAS_PREFER_HIPBLASLT=1` and
`ROCBLAS_USE_HIPBLASLT=1` (what `TRELLIS_PREFER_HIPBLASLT=on` sets):

| Stage | rocBLAS | hipBLASLt | Speedup |
|---|--:|--:|--:|
| structure | 22.9 s | 22.9 s | 1.00× (attention-bound) |
| slat | 52.4 s | **39.1 s** | **1.34×** |
| whole generation | 79.4 s | 66.1 s | **1.20×** |

The gain is almost entirely the skinny sparse-conv GEMM: 14.5 s at 1.0 TFLOPS
under rocBLAS, 1.0 s at 14.3 TFLOPS under hipBLASLt. `metrics.blas_backend`
records which backend a run used.

## After the ROCm 10.0 update (torch 2.13.0+rocm10.0.0)

Same sample and settings, 2026-09-02. The 10.0 column is the **median of 5
fresh-process runs**, each bracketed by a 4096³ reference GEMM (all within
30.9–31.5 TFLOPS) with the GPU clock traced throughout; the 7.2.1 column is
a single pre-upgrade measurement.

| Stage | 7.2.1 + hipBLASLt (single) | 10.0 (median of 5) |
|---|--:|--:|
| conditioning | — | 0.8 s (0.8–0.9) |
| structure | 22.9 s | **13.8 s** (13.8–14.0) |
| slat | 39.1 s | **32.0 s** (31.9–32.1) |
| decode | 3.0 s | 2.9 s (2.9–2.9) |
| whole generation | 66.1 s | **49.6 s (1.60× over the 7.2.1 baseline)** |

The attention-bound structure stage dropped by a third (newer AOTriton flash
kernels), and the skinny-GEMM fix is part of the default path. The ~5 s
conditioning readings seen on the first post-upgrade runs were MIOpen
3.6.0's one-time tuning of the DINOv2 convolutions, not a regression; with
the cache in place conditioning is back to 0.8 s. The install traps and the
full measurement set are in gfx1151-gemm's `docs/rocm10.md`.

TRELLIS shares its architecture (and therefore these shapes, modulo the voxel
count) with Hi3DGen; Hunyuan3D looks completely different. The three-pipeline
comparison, the shape-overlap analysis, and everything about this GPU that
does not depend on the model live in
[gfx1151-gemm](https://github.com/kroqueta-s/gfx1151-gemm).
