#pragma once

#include <vector>

namespace fsc::mesh {

struct MeshBuildOptions {
    int columns = 28;
    int rows = 34;
};

std::vector<std::vector<double>> buildSyntheticFaceMesh3d(
    const std::vector<std::vector<double>>& landmarks3d,
    const MeshBuildOptions& options = {});

} // namespace fsc::mesh
