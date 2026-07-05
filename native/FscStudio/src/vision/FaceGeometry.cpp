#include "fsc/vision/FaceGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fsc::vision {
namespace {

void addEquation(const std::array<double, 4>& row, double value, double normal[4][4], double rhs[4]) {
    for (int i = 0; i < 4; ++i) {
        rhs[i] += row[static_cast<size_t>(i)] * value;
        for (int j = 0; j < 4; ++j) {
            normal[i][j] += row[static_cast<size_t>(i)] * row[static_cast<size_t>(j)];
        }
    }
}

std::array<double, 4> solve4(double matrix[4][4], double vector[4]) {
    double augmented[4][5]{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            augmented[row][col] = matrix[row][col];
        }
        augmented[row][4] = vector[row];
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col])) {
                pivot = row;
            }
        }
        for (int item = col; item < 5; ++item) {
            std::swap(augmented[col][item], augmented[pivot][item]);
        }

        const double divisor = std::abs(augmented[col][col]) < 1e-12 ? 1e-12 : augmented[col][col];
        for (int item = col; item < 5; ++item) {
            augmented[col][item] /= divisor;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = augmented[row][col];
            for (int item = col; item < 5; ++item) {
                augmented[row][item] -= factor * augmented[col][item];
            }
        }
    }

    return {augmented[0][4], augmented[1][4], augmented[2][4], augmented[3][4]};
}

} // namespace

std::array<Point2f, 5> arcFace112ReferencePoints() {
    return {
        Point2f{38.2946f, 51.6963f},
        Point2f{73.5318f, 51.5014f},
        Point2f{56.0252f, 71.7366f},
        Point2f{41.5493f, 92.3655f},
        Point2f{70.7299f, 92.2041f},
    };
}

SimilarityTransform estimateSimilarityTransform(
    const std::array<Point2f, 5>& source,
    const std::array<Point2f, 5>& destination) {
    double normal[4][4]{};
    double rhs[4]{};
    for (size_t i = 0; i < source.size(); ++i) {
        const double sx = source[i].x;
        const double sy = source[i].y;
        addEquation({sx, -sy, 1.0, 0.0}, destination[i].x, normal, rhs);
        addEquation({sy, sx, 0.0, 1.0}, destination[i].y, normal, rhs);
    }
    const auto solution = solve4(normal, rhs);
    return {
        static_cast<float>(solution[0]),
        static_cast<float>(solution[1]),
        static_cast<float>(solution[2]),
        static_cast<float>(solution[3]),
    };
}

Point2f applyTransform(const SimilarityTransform& transform, Point2f point) {
    return {
        transform.a * point.x - transform.b * point.y + transform.tx,
        transform.b * point.x + transform.a * point.y + transform.ty,
    };
}

float area(Rectf rect) {
    return std::max(0.0f, rect.x2 - rect.x1 + 1.0f) * std::max(0.0f, rect.y2 - rect.y1 + 1.0f);
}

float intersectionOverUnion(Rectf left, Rectf right) {
    const Rectf intersection{
        std::max(left.x1, right.x1),
        std::max(left.y1, right.y1),
        std::min(left.x2, right.x2),
        std::min(left.y2, right.y2),
    };
    const float intersectionArea = area(intersection);
    const float unionArea = area(left) + area(right) - intersectionArea;
    return unionArea <= 1e-6f ? 0.0f : intersectionArea / unionArea;
}

std::vector<Detection> nonMaximumSuppression(std::vector<Detection> detections, float threshold, int maxResults) {
    std::sort(detections.begin(), detections.end(), [](const Detection& left, const Detection& right) {
        return left.score > right.score;
    });

    std::vector<Detection> keep;
    while (!detections.empty()) {
        Detection current = detections.front();
        detections.erase(detections.begin());
        keep.push_back(current);
        detections.erase(
            std::remove_if(
                detections.begin(),
                detections.end(),
                [&](const Detection& other) {
                    return intersectionOverUnion(current.box, other.box) > threshold;
                }),
            detections.end());
        if (maxResults > 0 && keep.size() >= static_cast<size_t>(maxResults)) {
            break;
        }
    }
    return keep;
}

} // namespace fsc::vision
