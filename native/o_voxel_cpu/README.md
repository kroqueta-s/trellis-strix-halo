# o_voxel_cpu

**Optional.** The image-to-mesh path builds nothing; this is the one compiled
piece the `texture_mesh` path will need, and the runner works without it.

`o-voxel`, TRELLIS.2's grid library, turns a mesh into a flexible dual grid in
`src/convert/flexible_dual_grid.cpp`: 775 lines of C++ with no CUDA in them.
Upstream compiles that file into one extension together with five CUDA
kernels, so the whole package cannot be built on Windows + ROCm; this
directory compiles the CPU file alone, with a binding of its own
(`ext.cpp`), into a module the runner's `o_voxel._C` stand-in picks up.

## Building

Visual Studio 2022 Build Tools with the *Desktop development with C++*
workload (it brings the Windows SDK), then:

```powershell
.\native\o_voxel_cpu\build.ps1 -Python C:\path\to\trellis2-venv\Scripts\python.exe
```

The module lands in `TRELLIS2_NATIVE_DIR` from `.env` (or beside the script);
set that key and the runner loads it at start, reporting on stderr whether it
did. Eigen 3.4.0 (header-only, MPL-2.0) is downloaded once next to it, because
upstream keeps Eigen as a submodule that a shallow clone leaves empty.

**Upstream's source is not modified.** It uses GCC's `d` suffix on two
floating-point literals, which MSVC refuses; `setup.py` copies the file into
the build directory with those two literals rewritten and stops if the
rewrite no longer applies.

## Measured (2026-09-12, Ryzen AI MAX+ 395, torch 2.13.0+rocm10.0.0, MSVC 14.44)

| Input | Grid | Voxels | Crossings | Time |
|---|--:|--:|--:|--:|
| Construction mecha, 1.34 M faces | 512 | 2,774,956 | 2,811,234 | **7.7 s** |
| Icosphere (subdivisions 4) | 64 | 18,752 | 18,750 | 0.03 s |

Round trip through the runner's own extraction (`shims.flexible_dual_grid_to_mesh`)
lands within a cell of the input: median 0.8 cells on the mecha, 0.5 on a box.
A sphere comes back a cell larger, because the regularised least squares
places dual vertices outside their cells (offsets up to 2.02); the encoder
consumes the converter's output directly, so that convention is upstream's
own and is not re-interpreted here.

Two things Smart App Control may do to a freshly built module: refuse it once
(`WinError 4551`, loads on the next attempt) or refuse it for good. The runner
treats an import failure as "not built" and carries on without `texture_mesh`.
