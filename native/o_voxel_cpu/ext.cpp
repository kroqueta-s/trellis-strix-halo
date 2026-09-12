// SPDX-License-Identifier: MIT
//
// A CPU-only slice of o-voxel: the mesh -> flexible dual grid conversion that
// `texture_mesh` needs. Upstream's source file is compiled as it is (see
// setup.py for the one build-time rewrite); only this binding is ours, and it
// exposes exactly the one function the runner's `o_voxel._C` stand-in lacks.
#include <torch/extension.h>
#include "convert/api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mesh_to_flexible_dual_grid_cpu", &mesh_to_flexible_dual_grid_cpu,
          py::call_guard<py::gil_scoped_release>());
}
