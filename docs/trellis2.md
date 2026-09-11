# TRELLIS.2 on this machine

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) is a 4B image-to-3D model
built on a *flexible dual grid*: one dual vertex per active voxel, plus a
per-axis flag saying which grid edges the surface crosses. It reaches far more
detail than TRELLIS.1 — **millions of faces rather than hundreds of
thousands** — and it needs six CUDA-only packages to do it.

This runner (`runners/trellis2/`) runs the image-to-mesh half of it on
Windows + ROCm with **nothing compiled**: the CUDA halves are replaced at launch
time, the same way the TRELLIS.1 runner replaces `spconv` and `flash_attn`.
**A texture map is not implemented; colour per vertex is** — see
[Vertex colours](#vertex-colours).

Everything below was measured on an **ASUS ProArt PX13** (Ryzen AI MAX+ 395,
Radeon 8060S / gfx1151, 32 GB dedicated VRAM, factory power limits), Windows 11,
torch 2.13.0+rocm10.0.0, on 2026-09-11 and 2026-09-12.

## How these numbers were taken

**Read this before comparing anything.** On this machine a measurement taken
carelessly is off by an order of magnitude.

- **One image, stated every time.** Two specimens appear here: the robot in
  `assets/sample.png`, and a construction mecha with far more small detail.
  They are not interchangeable — the mecha produces 14,340 active voxels and
  9.3 M faces at 1024, the robot 18,534 voxels and 14.9 M faces. **Comparing
  two runners on different pictures says nothing.**
- **Never the first run.** MIOpen tunes its convolutions once, and the first
  run of a loop can be an order of magnitude slower than every run after it.
- **Stage times come from the runner itself** (`metrics`), measured around
  upstream's own public methods, not from a wall clock around the process.
- **VRAM is read twice**: `torch.cuda.mem_get_info` inside the runner, and
  Windows' own performance counters outside it (hearth's `vram.py`). The first
  one **counts shared memory as if it were video memory** — it reports 43.87 GB
  of "total" on a 32 GB card — so a peak that looks comfortable there can
  already be spilling. `metrics.vram_over` says whether the dedicated limit was
  crossed.
- **Topology is counted, not claimed**: `metrics.topology` reports boundary
  edges, non-manifold edges, whether the mesh is watertight, whether the winding
  is consistent, and the signed volume — every run.
- **Generation time varies** by several times for identical settings on this
  class of machine. **It is not a pass/fail signal.**

## What is replaced, and why it is safe

| Upstream dependency | Replacement | Verified by |
|---|---|---|
| `flex_gemm.ops.spconv` (sparse conv) | `runners/trellis/shims.py` — submanifold convolution in torch, shared with the TRELLIS.1 runner | Exact agreement with a dense `F.conv3d` reference (`tests/test_shims.py`) |
| `flash_attn` (sparse attention) | Same file — `F.scaled_dot_product_attention` | Agreement with a naive attention reference |
| `o_voxel._C` (GPU hashmap) | Same file — coordinate linearization and `searchsorted` | A python dictionary is an exact reference (`tests/test_shims2.py`) |
| `o_voxel.convert.flexible_dual_grid_to_mesh` | Same file — the extraction, reimplemented | With closing off it reproduces upstream **to the face**: 3,454,810 |
| `flex_gemm.ops.grid_sample` (sparse trilinear sampling) | Same file — the CUDA kernel's rule in torch, over the same hashmap | `F.grid_sample` agrees on a full grid; a dictionary states the sparse rule (`tests/test_shims2.py`) |
| `cumesh`, `nvdiffrast` | Stands-in that **raise when called** | Nothing on the image-to-mesh path calls them |

**The dense attention needs no shim at all**: upstream accepts
`ATTN_BACKEND=sdpa`, and on gfx1151 the AOTriton flash kernels are available in
both fp16 and bf16 — measured 15× faster than the math backend (1.61 ms against
24.90 ms at 8 heads / 4096 tokens / 64 channels) and in agreement with it to
four decimals. `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` must be set **before
torch is imported**; the runner does that for you.

## Measurements

### Resolution

`assets/sample.png`, generation only (loading the weights costs 35–45 s on top),
a 30 GB cap on dedicated VRAM, no decimation:

| Setting | Active voxels | Structure | Shape latent | Decode | **Generate** | Faces | **Peak VRAM** |
|---|--:|--:|--:|--:|--:|--:|--:|
| 512 | 4,415 | 17.0 s | 22.3 s | 5.8 s | **49.6 s** | 3,454,810 | **6.74 GB** |
| 1024 | 18,534 | 19.9 s | 163.0 s | 20.6 s | **209.3 s** | 14,860,594 | **15.42 GB** |
| 1408 | 4,415 | 17.1 s | 1,540.9 s | 40.3 s | **1,602.6 s** | 26,236,350 | **28.40 GB** |

**1536 cannot be asked for.** Upstream drops the target by 128 at a time until
the token count fits its own budget, and settles on 1408 — it says so on its
own output. `metrics.resolution` reports what it actually ran at.

**What limits this machine is time, not memory.** The 1408 shape stage runs at
140 s per step against 13.6 s at 1024. 28.40 GB also leaves only 1.6 GB under
the cap, so a different subject could cross it.

**And the time is quadratic in the token count, by design.** `SLatFlowModel`
runs its sparse attention with `attn_mode='full'`
(`trellis2/models/structured_latent_flow.py:71`), so every token attends to
every other one. The cascade holds the count just under `max_num_tokens`
(49,152), against roughly 14,000 at 1024 - **3.4x the tokens, and 3.4 squared
is 11.6x** against the 10.3x measured. Nothing about this port changes that
exponent: the stage is already on the fast attention path, and a faster
attention kernel moves the constant, not the shape of the curve. **The only
lever is fewer tokens**, which is the same thing as a lower resolution.

**1024 is the default** for that reason: 1408 costs eight times the time for
1.8× the faces.

### Post-processing

Upstream finishes in `CuMesh.fill_holes`, which does not exist here, and the
TRELLIS.1 runner's visibility pass **destroys these meshes** (see Limits). What
runs instead is decimation, debris removal and hole closing. On the mecha at
1024:

| Stage | Undecimated (9.3 M faces) | **Decimated first (1.5 M)** |
|---|--:|--:|
| Decimate | — | **12.7 s** |
| Drop debris | 55.2 s | 40.6 s |
| Close holes | 70.3 s | **10.0 s** |
| **Total** | **125.5 s** | **63.3 s** |

With `make_manifold` on as well, a 512 run of the mecha spends 3.1 s
decimating, 34.9 s dropping debris, 9.3 s closing holes and **59.7 s becoming a
manifold**, against 44.0 s to generate. **The post-processing is the expensive
half again**, and the two stages worth attacking are the debris removal — it
scales with the component count rather than the faces — and the manifold
conversion.

**Decimation runs first because everything after it is proportional to
something it reduces.** It costs 3.3 s to take 3.33 M faces to 700 k, for a
mean error of 0.17 % of the longest side (95th percentile 0.31 %) and a volume
within 0.9 % — and it *improves* the topology on the way: non-manifold edges
15,860 → 10,097, boundary edges 118,777 → 26,959, because the slivers collapse.
`target_faces` controls it; **0 turns it off**.

Dropping debris scales with the **component count**, not the face count (87,630
parts on that mesh), which is why it barely moves between the two columns.

### What comes out

The model's raw output carries a great deal of debris — **18,898 parts at 512
and 81,316 at 1024**, of which all but a handful are smaller than a tenth of the
model or thinner than a fiftieth. Dropping them costs 3.5–4.4 % of the faces.

Holes are the other half: thousands of pinholes rather than a few openings
(**2,978 loops, median 3 vertices, 95.2 % under 32** on a 700 k mesh).
`runners/trellis/close_holes.py` closes each with a fan from its centroid.

### Vertex colours

`TRELLIS2_VERTEX_COLORS=on` (or `vertex_colors` on the call) runs the texture
flow and its decoder as well, and carries the result onto the mesh's vertices.
**It is not a texture map.** The decoder produces one attribute vector per
active voxel, and each vertex is trilinearly interpolated from the eight voxels
around it — so the colour follows the *position*, not the surface. That is what
makes it survive decimation, hole closing and the manifold conversion, all of
which replace the vertices outright; a UV atlas would not survive any of them.

On the mecha at 512, against the same run without it:

| Stage | Seconds |
|---|--:|
| Texture latent (12 steps) | 10.4 |
| Texture decode | 4.8 |
| **Added to generation** | **15.1 of 64.4 (+31 %)** |
| Colouring 738,847 vertices | **0.12** |

Loading costs more too — **64.7 s against 42–45 s**, because the pipeline holds
eight models rather than five — and it needs 6.1 GB more on disk
(`install-trellis2.ps1 -WithTexture`). **Peak VRAM did not move**: 5.22 GB, with
no spill.

**511 vertices of 738,847 (0.07 %) came back black** — the ones no active voxel
surrounds, which cannot be interpolated from anything.
`metrics.vertex_colors.unreached_vertices` counts them every run, because a
model that is black because the texture stage failed and one that is black
because it is black look identical otherwise.

Only the base colour is kept. The decoder also produces metallic, roughness and
alpha (`pipeline.pbr_attr_layout`), and a PLY has nowhere to put them.
`tools/render_mesh.py --color` draws what came out.

## Gotchas

**The published checkpoints are `flex_gemm`-shaped.** Every sparse convolution
weight sits at `conv.weight`; the `spconv` backend expects `conv.conv.weight`,
one level down. Loading through `spconv` therefore leaves **every convolution
uninitialized** — and nothing says so. The first decoder block returns NaN and
the failure surfaces four blocks later as a subdivision that selects nothing.
Measured: `blocks.0.0.conv.conv.weight` held 32 NaNs and 2 infinities, while the
checkpoint's own key was `blocks.0.0.conv.weight` with the same shape. **Run
with `SPARSE_CONV_BACKEND=flex_gemm`** (the runner does).

**`Pipeline.from_pretrained` swallows every load error and retries against the
hub.** A local failure surfaces as `RepositoryNotFoundError: 401` for a
repository that does not exist. Diagnose by loading the model outside the
pipeline.

**The visibility pass removes most of the mesh.** Upstream's
`postprocess_mesh` (visibility plus min-cut) was measured in three
configurations on TRELLIS.2 output: on the raw mesh it removed 69 % of the faces
and 71 % of the volume, after normalizing the winding it removed the same, and
on a 700 k decimated mesh it removed 49 % and turned the volume negative. It is
off for this runner. It costs 530 s at 512 doing it.

**`PYTORCH_TUNABLEOP` does not help here.** Tuned, the stages come out at 17.2 s
/ 22.2 s / 5.9 s against 17.0 / 22.3 / 5.8 untuned — inside the run-to-run
spread — while the tuning pass itself costs 21 minutes and the results are no
longer bit-reproducible (3,454,810 faces became 3,413,090). hipBLASLt is
already doing this job.

**FlexGEMM's own torch reference disagrees with its CUDA kernel.**
`grid_sample_3d` takes `floor(q - 0.5)` as the base voxel in
`grid_sample.cu`, and `.int()` — truncation toward zero — in
`grid_sample_torch.py`. They differ for any query below 0.5. **The kernel is
what produced the published results**, so it is what this shim reproduces.

**Smart App Control blocks freshly installed binaries intermittently.** Two
instances here: `torch/lib/aotriton_v2.dll` and
`fast_simplification/_replay.pyd`, both `WinError 4551`, both loading on the
next attempt with no change. **Retry once before changing versions.** Do not
disable Smart App Control.

**transformers 5 moved two things.** The background remover comes back in the
checkpoint's dtype (fp16) while upstream's wrapper feeds it float32, and
DINOv3's 24 blocks moved from `.layer` to `.model.layer`. Both are fixed on the
launch side, in `runners/trellis2/pipeline.py`.

**The conditioner's weights are gated.** `facebook/dinov3-vitl16-pretrain-lvd1689m`
needs an accepted licence and a read token. A fine-grained token with the
Read-only preset is enough.

## Limits

- **No texture map.** Colour reaches the vertices (see below) but not a UV
  atlas: that needs unwrapping (`xatlas` can do it rather than CuMesh) and a
  bake, and neither is implemented here.
- **What the decoder produces is neither closed nor orientable.** Its flags say
  which grid edges the surface crosses, and **about 5 % of the primal faces
  carry an odd number of crossings** (4.99 % at 512, 5.02 % at 1024, measured as
  the share of quad edges with an odd number of quads on them). A dual-grid
  surface closes only where that count is even — and the consequence goes
  further than holes. Where three sheets meet on one edge a path can return to
  itself half a turn over, and **26,843 edges of roughly 2.2 M then cannot be
  given a consistent orientation**. That is genuine rather than an artefact of
  the analysis: an independent two-colouring returns the same count, a Möbius
  strip reproduces the effect and a torus does not, and 99.99 % of the faces sit
  in components carrying it. Recovering an inside/outside labelling to fix it at
  the source was tried, and **left 6.8 % of its constraints violated** — worse
  than the 5 % it set out to repair.

  `make_manifold` deals with it by separating and cutting rather than arguing:
  the touching sheets are split apart, the edges that cannot agree are cut too,
  and every seam that opens is closed again. **The result is watertight,
  edge-manifold and consistently wound, and `manifold3d` accepts it**
  (`Error.NoError`, genus 1207), which is what meshforge's `repair_manifold`
  needs. `TRELLIS2_MAKE_MANIFOLD=off` returns the model's own surface instead.

  **It costs, and the cost is in `metrics`.** Closing those seams adds patch
  worth **2.2–2.5× the input surface area**
  (`post.manifold.close.fan_area_fraction`), nearly all of it internal — the
  silhouette and the detail survive, checked by eye — while the enclosed volume
  comes out at 0.0023 against 0.024 for the open surface, which says the inside
  is a nest of shells rather than a solid. **Topologically clean is not the same
  as printable**, and that part is not verified yet.
- **Half the faces are wound the other way** (49.2 % at 512). Correcting that
  costs more than the generation on an undecimated mesh, so it is left to the
  caller, after decimation. `tools/render_mesh.py --two-sided` exists because a
  visual check is otherwise impossible: the render is salt-and-pepper speckle
  and the detail being checked for is exactly what disappears.
- **Millimetres, orientation and printability are downstream work.** The mesh
  comes back Z-up at normalized scale, as upstream leaves it.

## Licences

This repository is MIT, and **nothing this runner needs at runtime is more
restrictive**: trimesh (MIT), numpy and scipy (BSD), fast-simplification (MIT),
manifold3d (Apache), transformers (Apache).

**Two of the TRELLIS.1 runner's dependencies are not in that list**, and they
are deliberately absent here: `pymeshfix` is **AGPL-3.0** and `igraph` is
**GPL**. Both are needed by upstream's visibility-and-min-cut post-processing,
which this runner does not run.

The weights have their own terms, which are not this repository's: TRELLIS.2
itself is MIT, `facebook/dinov3-vitl16-pretrain-lvd1689m` is under the DINOv3
licence and gated, and the background remover this runner points at is whatever
the operator configured. **Read the licence of a model before using what it
produces.**
