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

With `make_manifold` on as well, a 512 run of the mecha spent 3.1 s
decimating, 34.9 s dropping debris, 9.3 s closing holes and **59.7 s becoming a
manifold**, against 44.0 s to generate — the post-processing was the expensive
half again. **Those numbers were the implementation's, not the problem's**
(measured 2026-09-12, same specimens):

| Stage | Before | After | What it was |
|---|--:|--:|---|
| Drop debris, 28,608 parts (1.5 M faces) | 12.7 s | **0.8 s** | `trimesh.split` built a `Trimesh` per part, and ran `fill_holes()` on each |
| `make_manifold`, six steps (1.34 M faces) | 36.0 s | **6.4 s** | `np.unique(axis=0)` over 4 M edge rows, 2.1 s a call, a dozen calls |
| `orient_faces`, 200 k faces in 9,973 components | 56.9 s | **0.2 s** | One breadth-first search per component, each allocating the whole mesh |

The debris removal no longer depends on the part count, and it no longer fills
holes as a side effect — that was never asked of it, and `close_holes` is the
stage that does it on purpose. `tests/test_edge_keys.py` holds the replaced
implementations and checks that every answer is unchanged.

**Decimation runs first because everything after it is proportional to
something it reduces.** It costs 3.3 s to take 3.33 M faces to 700 k, for a
mean error of 0.17 % of the longest side (95th percentile 0.31 %) and a volume
within 0.9 % — and it *improves* the topology on the way: non-manifold edges
15,860 → 10,097, boundary edges 118,777 → 26,959, because the slivers collapse.
`target_faces` controls it; **0 turns it off**.

Dropping debris used to scale with the **component count** (87,630 parts on
that mesh), which is why it barely moved between the two columns; see the
table above for what that cost actually was.

### The print mesh is a shell, because the model does not produce a solid

**What the decoder emits is a thin, double-walled skin.** Cross-sections of
the 512 specimen on the primal lattice show every part outlined twice — an
outer and an inner surface two or three cells apart — and the space between
them, and the space inside, connected to the outside through openings wider
than 24 cells (5 % of the model): flooding the lattice from outside reaches
the interior however the openings are closed, up to a closing radius of 12
cells. That is why sewing the surface shut (`make_manifold` alone) gave a
manifold enclosing a volume of 0.0023 against a silhouette of about 0.1, with
a quarter of the points near the surface at winding number −1: a surface that
does not separate an inside from an outside has no orientation to propagate,
and the sewn result depends on which fill rule a slicer applies.

So the print mesh is made from the surface instead (`runners/trellis2/shell.py`):
every point within half a wall of the surface is solid, every pocket the
outside cannot reach is filled, and the boundary of that solid is extracted
with surface nets. **Closed by construction, oriented by which side is solid,
and no wall thinner than asked** — which is what Blender's solidify and
OpenVDB's mesh-to-volume do with a surface soup. `TRELLIS2_SHELL_THICKNESS` is
the wall as a fraction of the longest side; the runner does not know
millimetres, and the default 0.0375 is 3 mm on an 80 mm print.

Measured on the 512 mecha (1.34 M faces at the shell's input, a 536-cell lattice):

| Wall | Shell | Decimate to 1.5 M | `make_manifold` | `manifold3d` | Volume |
|---|--:|--:|--:|---|--:|
| 0.0375 (3 mm / 80 mm) | 17.5 s (14.1 s of it the distance transform) | 1.2 s | 5.0 s, nothing to close | `NoError` | 0.0977 |
| 0.015 (1.2 mm / 80 mm) | 15.3 s | 2.5 s | 5.5 s | `NoError` | 0.0557 |

Through the runner, on the same image at 512 with the shell on: generation
56.2 s, then decimation 2.7 s, debris 1.5 s (71,473 parts), holes 1.4 s,
**shell 28.4 s** (23.7 s of it the distance transform, measured while a
compiler was using the other cores), decimation of the shell 1.4 s and
`make_manifold` 5.8 s — **41 s of post-processing against 80 s before the
shell existed**, for a mesh that is watertight, edge-manifold, consistently
wound and 0.0977 in volume.

**The price is detail narrower than the wall, and half a wall of growth
outward.** At 3 mm the hydraulics and track links round off; at 1.2 mm they
survive. Both renders are checked by eye; the wall is the operator's choice
and `metrics.post.shell` reports it in cells, together with how many pockets
were filled and what they hold.

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

On the mecha, against the same run without it:

| Stage | 512 | 1024 |
|---|--:|--:|
| Texture latent (12 steps) | 10.4 s | 64.3 s |
| Texture decode | 4.8 s | 14.7 s |
| **Added to generation** | **+31 %** (15.1 of 64.4) | **+50 %** (78.6 of 236.1) |
| Colouring the vertices | 0.12 s (738,847) | **0.15 s** (755,968) |
| Peak VRAM, with / without | 5.22 / 5.22 GB | 13.53 / 13.52 GB |

**Carrying the colours onto the mesh is free; sampling them is not.** The
texture flow runs the same full attention as the shape flow, so it grows with
the square of the token count — but at guidance 1.0 it makes no negative pass,
which is why it costs about half what the shape stage does (113.3 s against
64.3 s at 1024).

**Peak VRAM does not move**, because `low_vram` puts each model on the card only
while it is needed. What it costs instead is loading — **63–65 s against
42–45 s**, for eight models rather than five — and 6.1 GB more on disk
(`install-trellis2.ps1 -WithTexture`). The geometry is unchanged: 9,273,134
faces before decimation either way, and the same topology afterwards.

**1,161 vertices of 755,968 (0.15 %) came back black** at 1024, 511 of 738,847
(0.07 %) at 512 — the ones no active voxel surrounds, which cannot be
interpolated from anything. `metrics.vertex_colors.unreached_vertices` counts
them every run, because a model that is black because the texture stage failed
and one that is black because it is black look identical otherwise.

Only the base colour is kept. The decoder also produces metallic, roughness and
alpha (`pipeline.pbr_attr_layout`), and a PLY has nowhere to put them.
`tools/render_mesh.py --color` draws what came out.

### A texture map

`TRELLIS2_TEXTURE=on` bakes the same colours into a UV map as well, and writes
it as a second mesh at `extra.textured_glb`. **Unwrapping is xatlas and the bake
is 40 lines of barycentric arithmetic in UV space** — an atlas's triangles do
not overlap, so there is no depth test and no need for `nvdiffrast`.

It runs **before `make_manifold`**, because an atlas belongs to the vertices it
was built for and that stage replaces them. So the two outputs are different
meshes: the PLY is the watertight one to print, the GLB is the one to look at.

Measured on the mecha at 512:

| Stage | Seconds |
|---|--:|
| Unwrap (200,000 faces, in a child process) | **34.4** |
| Bake into a 2048² texture | **0.25** |

**The unwrap is the whole cost, and it grows brutally with the face count**:
13.0 s at 100 k, 31.9 s at 200 k, and the full 1.34 M mesh was still running
after ten minutes. `TRELLIS2_TEXTURE_TARGET_FACES` (default 200,000) is what
keeps it affordable; the texture carries the detail the triangles no longer do.

**Two things about it are worth knowing before turning it on.**

*It runs in a child process*, because `xatlas.parametrize` holds the GIL for its
whole run and would otherwise stop the heartbeat — measured at ten minutes of
silence, which a caller watching for liveness reads as a stall.

*The atlas comes out shattered.* This surface has no large flat regions for
xatlas to grow charts over, so it produces **35,697 charts from 200,000 faces**
— about five triangles each — and duplicates the vertices **3.1×**. Coverage is
43.8 %. The colours are right and the largest charts carry readable markings,
but **detail finer than a chart still breaks across a seam**, which is the one
thing a texture map was supposed to buy over vertex colours. Tuning does not
help: a relaxed chart search gave the same chart count in 13.8 s against 14.8 s,
and the packer's resolution changed the time by 12 % and the charts not at all.
**What this needs is a retopology first**, and that is not in this repository.

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

- **The texture map is not worth turning on yet.** It works and it is measured
  (see above), but the atlas shatters into ~5-triangle charts on this surface,
  so it buys little over vertex colours. Both are off by default.
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

  **Sewn on its own it is not a solid.** Closing those seams adds patch worth
  **2.2–2.5× the input surface area**
  (`post.manifold.close.fan_area_fraction`), nearly all of it internal, and
  the enclosed volume comes out at 0.0023 with a quarter of the space near the
  surface wound inside-out — because the surface is a double-walled skin with
  wide openings (see *The print mesh is a shell*). With `TRELLIS2_SHELL=on`,
  the default, `make_manifold` runs on the shell instead and has nothing to
  close; the sewn manifold is what `TRELLIS2_SHELL=off` gives.
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
