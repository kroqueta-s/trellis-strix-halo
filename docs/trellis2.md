# TRELLIS.2 on this machine

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) is a 4B image-to-3D model
built on a *flexible dual grid*: one dual vertex per active voxel, plus a
per-axis flag saying which grid edges the surface crosses. It reaches far more
detail than TRELLIS.1 — **millions of faces rather than hundreds of
thousands** — and it needs six CUDA-only packages to do it.

This runner (`runners/trellis2/`) runs the image-to-mesh half of it on
Windows + ROCm with **nothing compiled**: the CUDA halves are replaced at launch
time, the same way the TRELLIS.1 runner replaces `spconv` and `flash_attn`.
(One optional module can be compiled for the `texture_mesh` path — see
`native/o_voxel_cpu/` — and the runner works without it.)
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
| `o_voxel._C.mesh_to_flexible_dual_grid_cpu` (mesh → dual grid, for `texture_mesh`) | **Optional compiled module**, `native/o_voxel_cpu/` — upstream's CPU-only C++ built alone with MSVC | A box and a sphere round-trip through the runner's own extraction within a cell (`tests/test_o_voxel_cpu.py`); the mecha's 1.34 M faces convert in 7.7 s |

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

### The print mesh is carved out of the surface, because the model does not produce a solid

**What the decoder emits is a thin, double-walled, incomplete skin.**
Cross-sections of the 512 specimen on the primal lattice show every part
outlined twice — an outer and an inner surface two or three cells apart — and
the outlines are open arcs as often as closed loops. So the space between the
walls and the space inside are connected to the outside through gaps at every
scale: flooding the lattice from outside reaches the interior with the floor
sealed, and with every opening up to 24 cells wide closed. That is why sewing
the surface shut (`make_manifold` alone) gave a manifold enclosing a volume of
0.0023 against a silhouette near 0.05, with a quarter of the points near the
surface at winding number −1: a surface that does not separate an inside from
an outside has no orientation to propagate, and no fill can say which side is
which.

**Visibility can.** A point of the exterior is seen from far away in many
directions; a point in the hollow behind the skin is seen, if at all, only
through a gap, from a few. `runners/trellis2/shell.py` casts rays in 98
directions across the lattice with the surface as the occluder and keeps, as
air, every corner that escapes in at least `TRELLIS2_SHELL_VISIBILITY` of them
(the space carving of a visual hull); everything else is solid, and the
boundary of that solid is extracted with surface nets. **The outer surface
stays where the model put it, nothing grows outward, and the inside is
filled** — what a slicer wants, and what `forge.hollow` takes apart again
downstream when a print should be hollow. Closed by construction, and
oriented by which side is solid.

The threshold is measured. On the specimen the count of visible directions is
sharply bimodal — 3.2 M corners see none, a plateau from twelve to sixteen,
the exterior at ninety and more — and the solid moves by 0.002 between a
threshold of 2 and 4. A bowl twice as deep as it is wide (an annulus with a
floor, `tests/test_shell.py`) keeps its hollow up to 4 and starts to fill at 6.
So the default is 4: deep concavities stay open, gaps do not let the inside
leak out.

Measured on the 512 mecha (1.34 M faces at the input, a 516-cell lattice):

| Step | Seconds |
|---|--:|
| Rasterize the surface (13 samples per triangle, a half-cell grid on any wider one) | 3.0 |
| 98 rays on the GPU, pockets, erosion, chamfer distance | 7.2 |
| Surface nets (2.26 M faces) | 1.3 |
| Decimate to 1.5 M | 1.5 |
| `make_manifold` (nothing to close) | 5.3 |

`manifold3d` accepts it (`NoError`), the volume is 0.042, and the solid's
surface sits within a cell of the input's (median 0.94 cells, 95th percentile
3.6 — the larger distances are the caps over gaps, where there was no input
surface to be near). Hydraulics, track links and panel lines survive by eye.

Through the runner on the same image, with the texture on, **measured
2026-09-12, second run of two** (the first pays for MIOpen's tuning):

| | 512 | 1024 |
|---|--:|--:|
| Generate the shape (preprocess, conditioning, structure, latent, decode) | **45.2 s** | **154.2 s** |
| Sample and decode the texture latent | 14.1 s | 78.4 s |
| Decimate | 2.5 s | 10.1 s |
| Drop debris | 1.4 s | 1.3 s |
| Close holes | 1.3 s | 1.2 s |
| **Carve** | **15.7 s** | **17.3 s** |
| Decimate the solid | 1.3 s | 2.1 s |
| Drop debris again | 1.3 s | 2.0 s |
| Unwrap and bake | 3.6 s | 3.8 s |
| `make_manifold` | 7.0 s | 7.0 s |
| **Post-processing** | **34 s** | **45 s** |
| Faces out | 1,508,632 | 1,502,124 |
| Volume | 0.04316 | 0.04317 |

Both come out watertight, edge-manifold, consistently wound and in one
part. **The generation column is the shape alone**: the texture flow is a
separate 14 s at 512 and 78 s at 1024, and comparing a run that sampled it
with one that did not is the easiest way to think this machine has slowed
down.

**And it stays that way downstream.** meshforge's `prepare_mesh` starts by
welding vertices by position, dropping degenerate faces and dropping
duplicate faces, and the first carved meshes came out of it "not watertight"
although the runner had reported them so: coincident vertices (surface nets
placing crossings on lattice corners, split copies left at one point) and
isolated two-face pillows did not survive the welding. `make_manifold` now
keeps every copy 1e-5 of the longest side apart, the crossings sit a quarter
of the way along their edges, and pillows are dropped whole — measured
through `forge.prepare_mesh` at 80 mm:

| Resolution | Parts handed to forge | forge repair | Watertight | Size (mm) | Faces |
|---|--:|--:|---|---|--:|
| 512 | 1 (was 453) | 4.4 s (was 36 s) | yes, no warnings | 80.0 × 36.7 × 65.8 | 1,508,632 |
| 1024 | 1 (was 4,729) | 5.2 s (was 354 s) | yes, no warnings | 80.0 × 37.9 × 67.2 | 1,502,124 |

**The repair used to be paid per part.** The 354 s were `manifold3d`'s
`decompose`, which scans the whole mesh once for every connected component
(0.12 s a part on 1.5 M faces, whatever the part's size), and the 4,729
parts were not the model. Two things made them, measured 2026-09-12 on the
512 specimen: **strays the rays could not reach** - 254 detached pieces of
solid, 213 of them a single lattice corner, floating 2-11 cells from the body
in the shadows of struts - and **slivers the decimation of the solid pinches
off its surface**, 229 two-face pairs at 1.5 M faces and 2,827 at 1.0 M,
which `close_holes` then sealed into four- and six-face bits. So the carve
drops every detached piece of solid under `TRELLIS2_SHELL_ISLAND_CORNERS`
(64, four cells across; 1.0-1.4 s at 512, `metrics.post.shell.islands_dropped`
counts them: 248 at 512, 1,327 at 1024), and **the debris pass runs a second
time after the solid's decimation**, with the same thresholds as the first
(`metrics.post.shell_dropped_parts`: 232 at 512, 6,658 at 1024, 2.0 s), and
what reaches forge is one part with the volume unchanged to four digits.
forge's `repair_manifold` also stopped calling `decompose` and labels the
parts itself in one pass, so a mesh from elsewhere with many parts no longer
costs minutes there either.

The earlier two meshes, the unions of those 453 and 4,729 parts, were sliced
in Bambu Studio as they came, with no repair and no error (2026-09-12).

**A finer lattice and a bigger face budget were both measured, and neither is
worth taking** (2026-09-12, the same 1024 decode through all three):

| | 512 lattice, 1.5 M faces | **1024 lattice** | 512 lattice, **3 M faces** |
|---|--:|--:|--:|
| Carve | 17.3 s | **110.5 s** | 19.5 s |
| Faces the carve produced | 2,872,796 | 16,050,260 | 2,845,736 |
| One-corner-thick plates (`sheet_corners`) | 58,821 | **375,687** | 56,269 |
| Post-processing | 44 s | **158 s** | 54 s |
| Volume through forge at 80 mm | 22,023 mm³ | **18,672 mm³** | 22,199 mm³ |

**The 1024 lattice carves a different solid, not a finer one.** Rays pass
through narrower gaps, so 15 % of the volume goes, and what is left has six
times the plate one corner thick - 0.078 mm at 80 mm, which no printer will
make. The surface comes out pitted rather than detailed.

**And the face budget is not what limits the detail**: the carve produces
about 2.87 M faces whichever budget it is given, because the lattice decides
them. Asking for 3 M only skips the decimation, which is doing something
useful - it averages away the surface-nets staircase on curved surfaces, for a
mean error of 0.17 %. The one thing it buys is that nothing pinches slivers
off the surface, so the second debris pass finds a single part to begin with.

`TRELLIS2_SHELL_MODE=band` is the older construction: every point within
half of `TRELLIS2_SHELL_THICKNESS` of the surface is solid, which guarantees a
wall thickness (0.0375 is 3 mm on an 80 mm print) but grows the silhouette by
half a wall and rounds off detail narrower than the wall — measured on the
same specimen, 17.5 s and a volume of 0.0977 at 3 mm, with the hydraulics
rounded away. `metrics.post.shell` reports which mode ran, the threshold, the
pockets filled and the solid's volume.

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

**The colours reach the decoder's own mesh and not much else.** A vertex no
active voxel surrounds cannot be interpolated from anything and comes back
black; on the mesh as the decoder extracted it that was 1,161 of 755,968
(0.15 %) at 1024 and 511 of 738,847 (0.07 %) at 512. **On the carved solid it
is 390,100 of 750,214 (52 %) at 1024 and 91,935 of 754,118 (12 %) at 512**
(measured 2026-09-12). The carve makes a new surface out of a lattice, up to a
cell away from the one the decoder drew, and a cell of the 512 lattice is two
voxels wide at a 1024 decode - outside the eight that a trilinear sample reads.
Carving on a 1024 lattice does not fix it (51.9 %): the finer carve moves the
surface somewhere else again.

`metrics.vertex_colors.unreached_vertices` counts them every run, because a
model that is black because the texture stage failed and one that is black
because it is black look identical otherwise. `tools/render_mesh.py --color`
draws what came out; at 512 the black areas are the tracks and dark panels,
which are black in the image too, and at 1024 they are mottled over the whole
model.

A PLY carries the base colour alone, because it has nowhere to put the rest;
the GLB carries all four (below).

### A texture map

`TRELLIS2_TEXTURE` bakes the same colours into a UV map as well, and writes it
as a second mesh at `extra.textured_glb`. **It is on by default**; `off` skips
the unwrap and the bake, and `mesh_path` is the same mesh either way. **No `nvdiffrast` and no
`cumesh`**: the charts are cut here, xatlas only packs them, and the bake is
barycentric arithmetic in UV space - an atlas's triangles do not overlap, so
there is no depth test to do.

**xatlas cannot choose the charts on this surface.** Asked to cut as well as
pack, it made **35,697 charts from 200,000 faces** - five triangles each -
tripled the vertices and covered 44 % of the texture, and no chart option
moved it: a relaxed search gave the same charts in 13.8 s against 14.8 s, and
the packer's resolution changed the time by 12 % and the charts not at all.
Two causes, both in the input. Half the faces were wound the other way, and an
edge whose faces disagree is a boundary no chart may cross; and the surface is
rough at the scale of a triangle, so chart growth that follows face normals
stops after a few faces.

**So the charts are cut by direction in space** (`runners/trellis2/charts.py`).
Face normals are smoothed over the adjacency, each face is assigned to the one
of six axis directions its smoothed normal points along, a chart is a connected
run of faces sharing a direction, and each chart is laid flat by orthographic
projection along its axis. Charts too small to carry a picture are merged into
the neighbour they share the most edges with. Blender's *Smart UV Project* is
the same idea. The surface it works on is the **carved solid**, whose winding
is consistent - which removes the first of the two causes outright.

Measured on the mecha at 512, 200,000 faces into a 2048² texture:

| | xatlas chose the charts | Projected charts |
|---|--:|--:|
| Charts | 35,697 | **2,248** |
| Faces in charts of 50 or more | — | **95.6 %** |
| Vertex growth | 3.1× | **1.29×** |
| Coverage | 44 % | **55 %** |
| Cut | 31.9 s | **0.5 s** |
| Pack | (included) | **5.7 s** |
| Bake | 0.25 s | **0.5 s** |
| **Total** | **34.4 s** | **8.8 s** |

**The markings are readable now**, which is the whole point of a texture over
vertex colours: the hazard stripes on the armour, the beacons, the tracks and
the decals on the chest survive, where a chart of five triangles broke them
across a seam.

**Faces that fold.** A face whose own normal points away from its chart's
axis turns over in projection and lands on its neighbours: 4,250 of 200,000
(2.1 %) on the 512 specimen, 2,488 of them against their own smoothed
normal, because the surface is rough at the scale of a triangle. Left
alone they would have written their colour over the faces under them -
1.7 % of the covered texels, 5,551 of those in *other* charts. So each one
is moved to a neighbouring chart it does not fold in (1,215), or given a
chart of its own along its own normal when it is at least
`TRELLIS2_CHART_FOLD_AREA` times the median face (2; 73 faces), and the
slivers that remain - 2,962, holding 0.6 % of the area, a third of a median
face each - keep their place but **read the atlas without writing it**
(the bake skips them; they are read off the projection before packing,
because the packer mirrors whole charts and a sliver's orientation does not
survive its float32 rounding). Charts 2,105 → 2,178, faces in charts of
fifty or more 95.5 % → 95.4 %. `metrics.texture` reports `folds_moved`,
`folds_own_chart` and `folded_faces`.

**Metallic, roughness and alpha ride along.** The decoder produces all four
attributes per voxel (`pipeline.pbr_attr_layout`) and they come out of the same
trilinear sample as the colour, so carrying them costs the bake 0.06 s of 3.63
(measured 2026-09-12 at 512). glTF has a place for each: the alpha is the base
colour texture's fourth channel, and a second texture holds roughness in green
and metallic in blue. The material is `OPAQUE` even so - the decoder's alpha is
material information, and a viewer that blended on it would put holes in a mesh
meant to be printed.

**A texel the decoder never saw is counted, not filled.** `texels_black` is the
covered texels whose colour came back exactly zero - the same test the vertex
colours use - and `texels_black_after_dilate` says what the dilation did about
them, which is nothing: it fills texels no face wrote, and these were written.
Measured 2026-09-12 at 512: 221,065 of 2,262,652 covered texels (9.8 %), all
still there afterwards. At 1024 it is 44 %, for the reason under Vertex colours.

The bake runs **after the carve and before the manifold conversion**, on a mesh
decimated to `TRELLIS2_TEXTURE_TARGET_FACES` (200,000). The textured GLB is
therefore a different, coarser mesh than `mesh_path`: the PLY is the solid to
print, the GLB is the thing to look at. **Packing runs in a child process**,
because xatlas holds the GIL for its whole run and would otherwise silence the
heartbeat - measured at ten minutes of silence, which a caller watching for
liveness reads as a stall.

### Colouring a mesh that came from somewhere else

`texture_mesh` takes a mesh and a reference image and puts the same colours on
it. **The geometry is not touched.** What happens is the reverse of the
generating path's first half: the mesh is turned back into the latent the
texture flow conditions on, and the same flow, decoder and colour query run.

It needs two things the image-to-mesh path does not: the **shape encoder**
(`ckpts/shape_enc_next_dc_f16c32_fp16`, 709 MB, loaded the first time it is
asked for) and the **compiled dual-grid conversion** (`native/o_voxel_cpu`).
Upstream's `mesh_to_flexible_dual_grid` is C++ with no torch equivalent here -
unlike the extraction the other way, which the shims reimplement.
`capabilities.texture_mesh` is false unless all three are present, and it is
answered without loading any of them.

Measured on this runner's own 512 output (1,511,942 faces) with the mecha as
the reference:

| Stage | Seconds |
|---|--:|
| Shape encoder, first call only | 3.3 |
| Mesh to dual grid (1,084,913 voxels) | 7.7 |
| Encode to a shape latent | 4.9 |
| Conditioning | 5.8 |
| Texture latent (12 steps) | 9.9 |
| Texture decode | 4.2 |
| **Total** | **32.6** |
| Vertex colours (756,679 vertices) | 0.25 |
| Charting and baking a 2048² texture | 12.1 |

Peak VRAM 5.96 GB. 15 vertices of 756,679 came back black. The colours match
what `image_to_mesh` produces on the same subject: mean RGB (0.332, 0.287,
0.158).

**`up_axis` is asked for rather than assumed.** Upstream's own preprocessing
swaps Y and Z because it assumes a Y-up file; applied to this runner's output,
which is Z-up, that lays the model on its face. The default is `z`, which is
what `image_to_mesh` reports for the meshes it writes.

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

- **The texture map costs a second 1.3B flow**, like the vertex colours it
  shares its query with, and 8.8 s more for the charts, the packing and the
  bake. The vertex colours are off by default and the texture map is on.
  What it does not do is match the print mesh:
  the atlas belongs to a coarser surface (see above).
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
  gaps at every scale (see *The print mesh is carved out of the surface*). With `TRELLIS2_SHELL=on`,
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
