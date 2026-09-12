# SPDX-License-Identifier: MIT
"""Build `o_voxel_cpu`: upstream's mesh -> flexible dual grid conversion, CPU only.

**Nothing on the image-to-mesh path needs this.** It exists for `texture_mesh`,
which has to turn a mesh from anywhere into the dual grid the encoder reads,
and upstream implements that in 775 lines of C++ with no CUDA in them
(`o-voxel/src/convert/flexible_dual_grid.cpp`). Upstream's own `setup.py`
compiles that file into one extension together with five CUDA kernels, which
is why it cannot be built here; this one compiles the CPU file alone.

Run it through `build.ps1`, which sets up the compiler. It needs:

- `TRELLIS2_REPO` (from `.env`): the upstream checkout, for the source file.
- `O_VOXEL_EIGEN_DIR`: Eigen's headers. Upstream keeps them as a submodule
  that a shallow clone leaves empty; `build.ps1` downloads a copy.

**Upstream's source is not modified.** It uses GCC's `d` suffix on two
floating-point literals (`1e-6d`, `0.0d`), which MSVC and clang both refuse;
the file is copied into the build directory with those two literals rewritten,
and the copy is what gets compiled. The check below stops the build if upstream
changes and the rewrite no longer applies, so it cannot silently compile
something else.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

upstream = os.environ.get("TRELLIS2_REPO", "")
eigen = os.environ.get("O_VOXEL_EIGEN_DIR", "")
if not upstream:
    raise SystemExit("TRELLIS2_REPO is not set (in .env or the environment)")
if not eigen or not (Path(eigen) / "Eigen" / "Dense").is_file():
    raise SystemExit(f"O_VOXEL_EIGEN_DIR does not hold Eigen/Dense: {eigen!r}")

source_file = Path(upstream) / "o-voxel" / "src" / "convert" / "flexible_dual_grid.cpp"
if not source_file.is_file():
    raise SystemExit(f"upstream source not found: {source_file}")

patched_dir = HERE / "build" / "patched"
patched_dir.mkdir(parents=True, exist_ok=True)
source = source_file.read_text(encoding="utf-8")
patched = source.replace("1e-6d", "1e-6").replace("0.0d", "0.0")
if patched == source:
    raise SystemExit("the literal rewrite no longer applies: upstream changed, check the file")
patched_file = patched_dir / "flexible_dual_grid.cpp"
patched_file.write_text(patched, encoding="utf-8")

setup(
    name="o_voxel_cpu",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="o_voxel_cpu",
            sources=[str(HERE / "ext.cpp"), str(patched_file)],
            include_dirs=[
                str(Path(upstream) / "o-voxel" / "src"),
                str(Path(upstream) / "o-voxel" / "src" / "convert"),
                eigen,
            ],
            # torch 2.13's headers need C++20 under MSVC.
            extra_compile_args=["/O2", "/std:c++20", "/EHsc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
