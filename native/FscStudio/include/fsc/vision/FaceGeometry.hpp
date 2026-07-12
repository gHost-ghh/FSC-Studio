#pragma once

#include <array>
#include <vector>

namespace fsc::vision {

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

struct Point3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Rectf {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct Detection {
    Rectf box;
    std::array<Point2f, 5> keypoints{};
    float score = 0.0f;
};

struct SimilarityTransform {
    // Forward transform:
    // x' = a*x - b*y + tx
    // y' = b*x + a*y + ty
    float a = 1.0f;
    float b = 0.0f;
    float tx = 0.0f;
    float ty = 0.0f;
};

std::array<Point2f, 5> arcFace112ReferencePoints();
SimilarityTransform estimateSimilarityTransform(
    const std::array<Point2f, 5>& source,
    const std::array<Point2f, 5>& destination);
Point2f applyTransform(const SimilarityTransform& transform, Point2f point);
float area(Rectf rect);
float intersectionOverUnion(Rectf left, Rectf right);
std::vector<Detection> nonMaximumSuppression(std::vector<Detection> detections, float threshold, int maxResults);

} // namespace fsc::vision
