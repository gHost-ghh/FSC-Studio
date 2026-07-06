#include "fsc/mesh/FaceMesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fsc::mesh {
namespace {

struct Point3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Point3 toPoint(const std::vector<double>& row) {
    return {
        row.size() > 0 ? row[0] : 0.0,
        row.size() > 1 ? row[1] : 0.0,
        row.size() > 2 ? row[2] : 0.0,
    };
}

std::vector<Point3> validPoints(const std::vector<std::vector<double>>& rows) {
    std::vector<Point3> points;
    points.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size() >= 2) {
            points.push_back(toPoint(row));
        }
    }
    return points;
}

double distance2(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

double interpolatedZ(const std::vector<Point3>& landmarks, double x, double y, double fallback) {
    double weighted = 0.0;
    double totalWeight = 0.0;
    for (const auto& landmark : landmarks) {
        const double weight = 1.0 / std::max(distance2(x, y, landmark.x, landmark.y), 16.0);
        weighted += landmark.z * weight;
        totalWeight += weight;
    }
    return totalWeight > 0.0 ? weighted / totalWeight : fallback;
}

} // namespace

std::vector<std::vector<double>> buildSyntheticFaceMesh3d(
    const std::vector<std::vector<double>>& landmarks3d,
    const MeshBuildOptions& options) {
    const auto landmarks = validPoints(landmarks3d);
    if (landmarks.size() < 8) {
        throw std::runtime_error("At least 8 3D landmarks are required to build a native face mesh.");
    }

    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();
    double maxZ = std::numeric_limits<double>::lowest();
    for (const auto& point : landmarks) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
        minZ = std::min(minZ, point.z);
        maxZ = std::max(maxZ, point.z);
    }

    const double width = std::max(maxX - minX, 1.0);
    const double height = std::max(maxY - minY, 1.0);
    const double cx = (minX + maxX) * 0.5;
    const double cy = (minY + maxY) * 0.5;
    const double cz = (minZ + maxZ) * 0.5;
    const double rx = width * 0.56;
    const double ry = height * 0.56;
    const int columns = std::clamp(options.columns, 8, 80);
    const int rows = std::clamp(options.rows, 8, 100);

    std::vector<std::vector<double>> mesh;
    mesh.reserve(static_cast<size_t>(columns * rows) + landmarks.size());

    for (int row = 0; row < rows; ++row) {
        const double v = rows == 1 ? 0.0 : -1.0 + 2.0 * static_cast<double>(row) / static_cast<double>(rows - 1);
        for (int column = 0; column < columns; ++column) {
            const double u = columns == 1 ? 0.0 : -1.0 + 2.0 * static_cast<double>(column) / static_cast<double>(columns - 1);
            const double oval = u * u + v * v;
            if (oval > 1.0) {
                continue;
            }
            const double x = cx + u * rx;
            const double y = cy + v * ry;
            const double surface = std::sqrt(std::max(0.0, 1.0 - oval));
            const double landmarkZ = interpolatedZ(landmarks, x, y, cz);
            const double z = landmarkZ + surface * (maxZ - minZ) * 0.18;
            mesh.push_back({x, y, z});
        }
    }

    for (const auto& point : landmarks) {
        mesh.push_back({point.x, point.y, point.z});
    }

    return mesh;
}

} // namespace fsc::mesh
