#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/core/VectorMath.hpp"
#include "fsc/mesh/FaceMesh.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <QApplication>
#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMouseEvent>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QStyle>
#include <QTabBar>
#include <QTableWidget>
#include <QTabWidget>
#include <QTextEdit>
#include <QTime>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

#ifdef FSC_ENABLE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

QString qs(const std::string& value) {
    return QString::fromUtf8(value.c_str());
}

std::filesystem::path pathFrom(const QString& value) {
    return std::filesystem::path(value.toStdWString());
}

QString defaultModelRoot() {
    const auto packaged = QApplication::applicationDirPath() + "/models/insightface/models";
    if (std::filesystem::exists(pathFrom(packaged))) {
        return packaged;
    }
    return "D:\\FSC\\model\\insightface\\models";
}

QTableWidgetItem* item(const QString& value) {
    auto* output = new QTableWidgetItem(value);
    output->setFlags(output->flags() & ~Qt::ItemIsEditable);
    return output;
}

QTableWidgetItem* numberItem(double value, int decimals = 4) {
    return item(QString::number(value, 'f', decimals));
}

std::string csvEscape(const std::string& value) {
    const bool quote = value.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) {
        return value;
    }
    std::string output = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            output += "\"\"";
        } else {
            output += ch;
        }
    }
    output += '"';
    return output;
}

bool isSupportedImageFile(const QString& path) {
    const auto suffix = QFileInfo(path).suffix().toLower();
    return suffix == "jpg" || suffix == "jpeg" || suffix == "png" || suffix == "bmp" || suffix == "ppm";
}

int writeFacesCsv(const std::vector<fsc::core::FaceRecord>& records, const std::filesystem::path& outputPath) {
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }
    std::ofstream stream(outputPath, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Unable to open CSV output file.");
    }
    stream << "id,file_name,person,tags,review,ignored,quality,detection,source,notes,created_at\n";
    for (const auto& record : records) {
        stream << record.id << ','
               << csvEscape(record.fileName) << ','
               << csvEscape(record.personName) << ','
               << csvEscape(record.tagText) << ','
               << csvEscape(record.reviewState) << ','
               << (record.ignored ? "1" : "0") << ','
               << record.qualityScore << ','
               << record.detectionScore << ','
               << csvEscape(record.sourcePath) << ','
               << csvEscape(record.notes) << ','
               << csvEscape(record.createdAt) << '\n';
    }
    return static_cast<int>(records.size());
}

QString translatedText(const QString& key, const QString& language) {
    if (language == "en") {
        return key;
    }
    static const std::map<std::string, std::map<std::string, const char*>> translations = {
        {"zh", {
            {"FSC Studio Native", "FSC Studio 原生版"},
            {"Database", "数据库"},
            {"Language", "语言"},
            {"Identity Mode", "身份模式"},
            {"Strict", "严格"},
            {"Balanced", "均衡"},
            {"Broad", "宽松"},
            {"Overview", "概览"},
            {"Library", "人脸库"},
            {"People", "人物"},
            {"Search", "搜索"},
            {"Camera", "摄像头"},
            {"Review", "复核"},
            {"Clusters", "聚类"},
            {"Compare", "比对"},
            {"Dense Mesh", "稠密网格"},
            {"Runtime", "运行"},
            {"Ready", "就绪"},
        }},
        {"ja", {
            {"FSC Studio Native", "FSC Studio ネイティブ"},
            {"Database", "データベース"},
            {"Language", "言語"},
            {"Identity Mode", "識別モード"},
            {"Strict", "厳格"},
            {"Balanced", "標準"},
            {"Broad", "広め"},
            {"Overview", "概要"},
            {"Library", "ライブラリ"},
            {"People", "人物"},
            {"Search", "検索"},
            {"Camera", "カメラ"},
            {"Review", "レビュー"},
            {"Clusters", "クラスタ"},
            {"Compare", "比較"},
            {"Dense Mesh", "高密度メッシュ"},
            {"Runtime", "実行環境"},
            {"Ready", "準備完了"},
        }},
        {"ko", {
            {"FSC Studio Native", "FSC Studio 네이티브"},
            {"Database", "데이터베이스"},
            {"Language", "언어"},
            {"Identity Mode", "식별 모드"},
            {"Strict", "엄격"},
            {"Balanced", "균형"},
            {"Broad", "넓게"},
            {"Overview", "개요"},
            {"Library", "라이브러리"},
            {"People", "인물"},
            {"Search", "검색"},
            {"Camera", "카메라"},
            {"Review", "검토"},
            {"Clusters", "클러스터"},
            {"Compare", "비교"},
            {"Dense Mesh", "고밀도 메시"},
            {"Runtime", "런타임"},
            {"Ready", "준비됨"},
        }},
    };
    const auto languageIt = translations.find(language.toStdString());
    if (languageIt == translations.end()) {
        return key;
    }
    const auto valueIt = languageIt->second.find(key.toStdString());
    return valueIt == languageIt->second.end() ? key : QString::fromUtf8(valueIt->second);
}

void fitTable(QTableWidget* table) {
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->horizontalHeader()->setStretchLastSection(true);
    table->verticalHeader()->setVisible(false);
}

std::vector<std::vector<double>> keypointRows(const std::array<fsc::vision::Point2f, 5>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point2f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y});
    }
    return rows;
}

std::vector<std::vector<double>> landmarkRows(const std::vector<fsc::vision::Point3f>& points) {
    std::vector<std::vector<double>> rows;
    rows.reserve(points.size());
    for (const auto& point : points) {
        rows.push_back({point.x, point.y, point.z});
    }
    return rows;
}

std::string qualityJson(const fsc::vision::AnalyzedFace& face) {
    std::ostringstream stream;
    stream << std::fixed
           << "{\"det_score\":" << face.detection.score
           << ",\"area_ratio\":" << face.qualityAreaRatio
           << ",\"sharpness\":" << face.qualitySharpness
           << ",\"brightness\":" << face.qualityBrightness
           << ",\"contrast\":" << face.qualityContrast
           << ",\"native\":true}";
    return stream.str();
}

fsc::core::FaceInsertRecord insertRecordFromFace(
    const std::filesystem::path& imagePath,
    const fsc::vision::AnalyzedFace& face,
    const std::string& imageHash,
    bool duplicate) {
    fsc::core::FaceInsertRecord record;
    record.fileName = imagePath.filename().string();
    record.sourcePath = imagePath.string();
    record.embedding = face.embedding;
    record.embeddingDim = static_cast<int>(face.embedding.size());
    record.bbox = {
        face.detection.box.x1,
        face.detection.box.y1,
        face.detection.box.x2,
        face.detection.box.y2,
    };
    record.keypoints = keypointRows(face.detection.keypoints);
    record.landmarks2d = landmarkRows(face.landmarks2d);
    record.landmarks3d = landmarkRows(face.landmarks3d);
    record.detectionScore = face.detection.score;
    record.qualityScore = face.qualityScore;
    record.qualityJson = qualityJson(face);
    record.imageHash = imageHash;
    if (duplicate) {
        record.reviewState = "duplicate";
        record.notes = "Same source image hash already exists in this database.";
    }
    return record;
}

const fsc::vision::AnalyzedFace& bestFace(const std::vector<fsc::vision::AnalyzedFace>& faces) {
    if (faces.empty()) {
        throw std::runtime_error("No face detected.");
    }
    return *std::max_element(faces.begin(), faces.end(), [](const auto& left, const auto& right) {
        return left.detection.score < right.detection.score;
    });
}

struct ClusterSummary {
    std::vector<fsc::core::FaceRecord> members;
    std::vector<std::string> knownPeople;
    std::string suggestedName;
    int64_t representativeId = 0;
    double meanSimilarity = 0.0;
    double maxSimilarity = 0.0;
    double averageQuality = 0.0;
};

std::vector<ClusterSummary> buildClusters(
    std::vector<fsc::core::FaceRecord> records,
    double threshold,
    int minSize,
    double minQuality = 0.0,
    bool unassignedOnly = false,
    int maxFaces = 0) {
    records.erase(
        std::remove_if(records.begin(), records.end(), [minQuality, unassignedOnly](const auto& record) {
            return record.embedding.empty() ||
                record.qualityScore < minQuality ||
                (unassignedOnly && record.personId > 0);
        }),
        records.end());
    if (maxFaces > 0 && records.size() > static_cast<size_t>(maxFaces)) {
        std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
            return left.qualityScore > right.qualityScore;
        });
        records.resize(static_cast<size_t>(maxFaces));
    }
    const int n = static_cast<int>(records.size());
    std::vector<int> parent(static_cast<size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);
    const auto findRoot = [&](int value) {
        int root = value;
        while (parent[static_cast<size_t>(root)] != root) {
            root = parent[static_cast<size_t>(root)];
        }
        while (parent[static_cast<size_t>(value)] != value) {
            const int next = parent[static_cast<size_t>(value)];
            parent[static_cast<size_t>(value)] = root;
            value = next;
        }
        return root;
    };
    const auto unite = [&](int left, int right) {
        const int leftRoot = findRoot(left);
        const int rightRoot = findRoot(right);
        if (leftRoot != rightRoot) {
            parent[static_cast<size_t>(rightRoot)] = leftRoot;
        }
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (fsc::core::dot(records[static_cast<size_t>(i)].embedding, records[static_cast<size_t>(j)].embedding) >= threshold) {
                unite(i, j);
            }
        }
    }

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) {
        groups[findRoot(i)].push_back(i);
    }

    std::vector<ClusterSummary> clusters;
    for (const auto& [_, indexes] : groups) {
        if (indexes.size() < static_cast<size_t>(std::max(2, minSize))) {
            continue;
        }
        ClusterSummary cluster;
        cluster.members.reserve(indexes.size());
        double qualityTotal = 0.0;
        std::map<std::string, int> peopleCounts;
        for (const int index : indexes) {
            const auto& record = records[static_cast<size_t>(index)];
            qualityTotal += record.qualityScore;
            if (!record.personName.empty()) {
                ++peopleCounts[record.personName];
            }
            cluster.members.push_back(record);
        }
        cluster.averageQuality = qualityTotal / static_cast<double>(cluster.members.size());
        for (const auto& [name, _] : peopleCounts) {
            cluster.knownPeople.push_back(name);
        }
        if (!peopleCounts.empty()) {
            const auto bestPerson = std::max_element(peopleCounts.begin(), peopleCounts.end(), [](const auto& left, const auto& right) {
                return left.second < right.second;
            });
            cluster.suggestedName = bestPerson->first;
        }
        const auto representative = std::max_element(cluster.members.begin(), cluster.members.end(), [](const auto& left, const auto& right) {
            return left.qualityScore < right.qualityScore;
        });
        if (representative != cluster.members.end()) {
            cluster.representativeId = representative->id;
        }
        double similarityTotal = 0.0;
        int pairCount = 0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double score = fsc::core::dot(cluster.members[i].embedding, cluster.members[j].embedding);
                similarityTotal += score;
                cluster.maxSimilarity = std::max(cluster.maxSimilarity, score);
                ++pairCount;
            }
        }
        cluster.meanSimilarity = pairCount > 0 ? similarityTotal / static_cast<double>(pairCount) : 0.0;
        clusters.push_back(std::move(cluster));
    }
    std::sort(clusters.begin(), clusters.end(), [](const auto& left, const auto& right) {
        if (left.members.size() != right.members.size()) {
            return left.members.size() > right.members.size();
        }
        return left.meanSimilarity > right.meanSimilarity;
    });
    return clusters;
}

#ifdef FSC_ENABLE_OPENCV
fsc::vision::RgbImage rgbImageFromBgrMat(const cv::Mat& frame) {
    if (frame.empty() || frame.channels() != 3) {
        throw std::runtime_error("Camera frame is empty or not BGR.");
    }
    fsc::vision::RgbImage image;
    image.width = frame.cols;
    image.height = frame.rows;
    image.pixels.resize(static_cast<size_t>(image.width * image.height * 3));
    for (int y = 0; y < frame.rows; ++y) {
        const auto* source = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            const size_t offset = static_cast<size_t>((y * frame.cols + x) * 3);
            image.pixels[offset + 0] = source[x][2];
            image.pixels[offset + 1] = source[x][1];
            image.pixels[offset + 2] = source[x][0];
        }
    }
    return image;
}

void scaleAnalyzedFaceCoordinates(std::vector<fsc::vision::AnalyzedFace>& faces, double scaleX, double scaleY) {
    const float sx = static_cast<float>(scaleX);
    const float sy = static_cast<float>(scaleY);
    const float sz = static_cast<float>((scaleX + scaleY) * 0.5);
    for (auto& face : faces) {
        face.detection.box.x1 *= sx;
        face.detection.box.x2 *= sx;
        face.detection.box.y1 *= sy;
        face.detection.box.y2 *= sy;
        for (auto& point : face.detection.keypoints) {
            point.x *= sx;
            point.y *= sy;
        }
        for (auto& point : face.landmarks2d) {
            point.x *= sx;
            point.y *= sy;
        }
        for (auto& point : face.landmarks3d) {
            point.x *= sx;
            point.y *= sy;
            point.z *= sz;
        }
    }
}
#endif

class PointCloudWidget final : public QWidget {
public:
    explicit PointCloudWidget(QWidget* parent = nullptr)
        : QWidget(parent) {
        setMinimumSize(420, 320);
        setMouseTracking(true);
        setMessage("Select a face");
    }

    void setData(
        std::vector<std::vector<double>> points,
        std::vector<std::vector<double>> overlayPoints,
        QString message) {
        points_ = std::move(points);
        overlayPoints_ = std::move(overlayPoints);
        message_ = std::move(message);
        update();
    }

    void setMessage(QString message) {
        points_.clear();
        overlayPoints_.clear();
        message_ = std::move(message);
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(12, 20, 32));
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QColor(220, 232, 245));
        painter.drawText(QPoint(14, 24), message_.isEmpty() ? QString("3D preview") : message_);
        painter.setPen(QColor(140, 170, 205));
        painter.drawText(QPoint(14, 46), "drag to rotate");

        if (points_.empty() && overlayPoints_.empty()) {
            return;
        }

        Bounds bounds = computeBounds();
        std::vector<ProjectedPoint> projected;
        projected.reserve(points_.size() + overlayPoints_.size());
        for (const auto& point : points_) {
            if (point.size() >= 2) {
                projected.push_back(project(point, bounds, false));
            }
        }
        for (const auto& point : overlayPoints_) {
            if (point.size() >= 2) {
                projected.push_back(project(point, bounds, true));
            }
        }
        std::sort(projected.begin(), projected.end(), [](const auto& left, const auto& right) {
            return left.depth < right.depth;
        });

        painter.setPen(Qt::NoPen);
        for (const auto& point : projected) {
            const QColor color = point.overlay ? QColor(43, 219, 235) : QColor(239, 181, 78);
            painter.setBrush(color);
            const double radius = point.overlay ? 2.2 : 1.45;
            painter.drawEllipse(point.screen, radius, radius);
        }
    }

    void mousePressEvent(QMouseEvent* event) override {
        lastMouse_ = event->position();
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if (event->buttons() & Qt::LeftButton) {
            const QPointF delta = event->position() - lastMouse_;
            yaw_ += delta.x() * 0.01;
            pitch_ += delta.y() * 0.01;
            pitch_ = std::clamp(pitch_, -1.45, 1.45);
            lastMouse_ = event->position();
            update();
        }
    }

private:
    struct Bounds {
        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        double span = 1.0;
    };

    struct ProjectedPoint {
        QPointF screen;
        double depth = 0.0;
        bool overlay = false;
    };

    Bounds computeBounds() const {
        bool any = false;
        double minX = 0.0;
        double maxX = 0.0;
        double minY = 0.0;
        double maxY = 0.0;
        double minZ = 0.0;
        double maxZ = 0.0;
        const auto visit = [&](const std::vector<std::vector<double>>& rows) {
            for (const auto& row : rows) {
                if (row.size() < 2) {
                    continue;
                }
                const double x = row[0];
                const double y = row[1];
                const double z = row.size() > 2 ? row[2] : 0.0;
                if (!any) {
                    minX = maxX = x;
                    minY = maxY = y;
                    minZ = maxZ = z;
                    any = true;
                } else {
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                    minZ = std::min(minZ, z);
                    maxZ = std::max(maxZ, z);
                }
            }
        };
        visit(points_);
        visit(overlayPoints_);
        Bounds bounds;
        bounds.cx = (minX + maxX) * 0.5;
        bounds.cy = (minY + maxY) * 0.5;
        bounds.cz = (minZ + maxZ) * 0.5;
        bounds.span = std::max({maxX - minX, maxY - minY, maxZ - minZ, 1.0});
        return bounds;
    }

    ProjectedPoint project(const std::vector<double>& point, const Bounds& bounds, bool overlay) const {
        double x = point[0] - bounds.cx;
        double y = -(point[1] - bounds.cy);
        double z = (point.size() > 2 ? point[2] : 0.0) - bounds.cz;

        const double cy = std::cos(yaw_);
        const double sy = std::sin(yaw_);
        const double cp = std::cos(pitch_);
        const double sp = std::sin(pitch_);

        const double xYaw = x * cy + z * sy;
        const double zYaw = -x * sy + z * cy;
        const double yPitch = y * cp - zYaw * sp;
        const double zPitch = y * sp + zYaw * cp;

        const double scale = 0.78 * std::min(width(), height()) / bounds.span;
        return {
            QPointF(width() * 0.5 + xYaw * scale, height() * 0.54 - yPitch * scale),
            zPitch,
            overlay,
        };
    }

    std::vector<std::vector<double>> points_;
    std::vector<std::vector<double>> overlayPoints_;
    QString message_;
    double yaw_ = 0.15;
    double pitch_ = -0.08;
    QPointF lastMouse_;
};

class FaceSelectionPreview final : public QLabel {
public:
    explicit FaceSelectionPreview(QWidget* parent = nullptr)
        : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(320, 260);
        setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        setText("Select image");
        setMouseTracking(true);
    }

    void setImagePath(QString path) {
        imagePath_ = std::move(path);
        selectedIndex_ = 0;
        focusOnFace_ = false;
        faces_.clear();
        refreshPixmap();
    }

    void setFaces(std::vector<fsc::vision::AnalyzedFace> faces, int selectedIndex) {
        faces_ = std::move(faces);
        selectedIndex_ = clampIndex(selectedIndex);
        refreshPixmap();
    }

    void setSelectedIndex(int selectedIndex) {
        selectedIndex_ = clampIndex(selectedIndex);
        refreshPixmap();
    }

    void setFocusOnFace(bool enabled) {
        focusOnFace_ = enabled;
        refreshPixmap();
    }

    [[nodiscard]] bool focusOnFace() const noexcept {
        return focusOnFace_;
    }

    std::function<void(int)> faceClicked;

protected:
    void resizeEvent(QResizeEvent* event) override {
        QLabel::resizeEvent(event);
        refreshPixmap();
    }

    void mousePressEvent(QMouseEvent* event) override {
        for (int index = 0; index < static_cast<int>(drawnRects_.size()); ++index) {
            if (drawnRects_[static_cast<size_t>(index)].contains(event->position())) {
                if (faceClicked) {
                    faceClicked(index);
                }
                return;
            }
        }
        QLabel::mousePressEvent(event);
    }

private:
    int clampIndex(int index) const {
        if (faces_.empty()) {
            return 0;
        }
        return std::clamp(index, 0, static_cast<int>(faces_.size()) - 1);
    }

    static QRectF detectionRect(const fsc::vision::AnalyzedFace& face) {
        const auto& box = face.detection.box;
        return QRectF(QPointF(box.x1, box.y1), QPointF(box.x2, box.y2)).normalized();
    }

    void refreshPixmap() {
        drawnRects_.clear();
        if (imagePath_.isEmpty()) {
            clear();
            setText("Select image");
            return;
        }
        QImage source(imagePath_);
        if (source.isNull()) {
            clear();
            setText("Image unavailable");
            return;
        }

        QPointF offset(0.0, 0.0);
        QImage view = source;
        if (focusOnFace_ && !faces_.empty()) {
            const QRectF box = detectionRect(faces_[static_cast<size_t>(selectedIndex_)]);
            const double pad = std::max(box.width(), box.height()) * 0.75;
            QRect crop(
                static_cast<int>(std::floor(box.left() - pad)),
                static_cast<int>(std::floor(box.top() - pad)),
                static_cast<int>(std::ceil(box.width() + pad * 2.0)),
                static_cast<int>(std::ceil(box.height() + pad * 2.0)));
            crop = crop.intersected(QRect(0, 0, source.width(), source.height()));
            if (crop.isValid() && crop.width() > 0 && crop.height() > 0) {
                view = source.copy(crop);
                offset = crop.topLeft();
            }
        }

        QPixmap pixmap = QPixmap::fromImage(view).scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        if (pixmap.isNull()) {
            clear();
            setText("Image unavailable");
            return;
        }

        const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, view.width()));
        const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, view.height()));
        const double xShift = (width() - pixmap.width()) * 0.5;
        const double yShift = (height() - pixmap.height()) * 0.5;

        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing, true);
        for (int index = 0; index < static_cast<int>(faces_.size()); ++index) {
            const QRectF box = detectionRect(faces_[static_cast<size_t>(index)]);
            const QRectF drawBox(
                (box.left() - offset.x()) * sx,
                (box.top() - offset.y()) * sy,
                box.width() * sx,
                box.height() * sy);
            const bool selected = index == selectedIndex_;
            painter.setPen(QPen(selected ? QColor(0, 230, 70) : QColor(20, 180, 235), selected ? 2.4 : 1.6));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(drawBox);
            if (drawBox.intersects(QRectF(0, 0, pixmap.width(), pixmap.height()))) {
                drawnRects_.push_back(drawBox.translated(xShift, yShift));
            } else {
                drawnRects_.push_back(QRectF());
            }
        }
        if (!faces_.empty()) {
            const auto& selectedFace = faces_[static_cast<size_t>(selectedIndex_)];
            painter.setPen(QPen(QColor(20, 170, 220), 1.1));
            painter.setBrush(QColor(20, 210, 235));
            for (const auto& point : selectedFace.landmarks2d) {
                const QPointF drawPoint((point.x - offset.x()) * sx, (point.y - offset.y()) * sy);
                if (drawPoint.x() >= 0.0 && drawPoint.y() >= 0.0 && drawPoint.x() <= pixmap.width() && drawPoint.y() <= pixmap.height()) {
                    painter.drawEllipse(drawPoint, 1.8, 1.8);
                }
            }
        }
        setPixmap(pixmap);
    }

    QString imagePath_;
    std::vector<fsc::vision::AnalyzedFace> faces_;
    std::vector<QRectF> drawnRects_;
    int selectedIndex_ = 0;
    bool focusOnFace_ = false;
};

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr)
        : QMainWindow(parent) {
        setWindowTitle("FSC Studio Native");
        resize(1180, 760);
        buildUi();
    }

    void openDatabasePath(const QString& path) {
        openDatabase(path);
    }

private:
    QString trUi(const QString& key) const {
        const QString language = languageCombo_ == nullptr ? QString("en") : languageCombo_->currentData().toString();
        return translatedText(key, language);
    }

    void addMainTab(QWidget* page, const QString& key) {
        const int index = tabs_->addTab(page, trUi(key));
        if (tabKeys_.size() <= static_cast<size_t>(index)) {
            tabKeys_.resize(static_cast<size_t>(index + 1));
        }
        tabKeys_[static_cast<size_t>(index)] = key;
        if (sidebar_ != nullptr) {
            auto* entry = new QListWidgetItem(trUi(key), sidebar_);
            entry->setData(Qt::UserRole, key);
        }
    }

    void applyLanguage() {
        setWindowTitle(trUi("FSC Studio Native"));
        if (databaseLabel_ != nullptr) {
            databaseLabel_->setText(trUi("Database"));
        }
        if (languageLabel_ != nullptr) {
            languageLabel_->setText(trUi("Language"));
        }
        if (identityModeLabel_ != nullptr) {
            identityModeLabel_->setText(trUi("Identity Mode"));
        }
        if (identityModeCombo_ != nullptr) {
            for (int index = 0; index < identityModeCombo_->count(); ++index) {
                const auto key = identityModeCombo_->itemData(index, Qt::UserRole + 1).toString();
                if (!key.isEmpty()) {
                    identityModeCombo_->setItemText(index, trUi(key));
                }
            }
        }
        if (sidebar_ != nullptr) {
            for (int index = 0; index < sidebar_->count(); ++index) {
                auto* entry = sidebar_->item(index);
                entry->setText(trUi(entry->data(Qt::UserRole).toString()));
            }
        }
        if (tabs_ != nullptr) {
            for (int index = 0; index < tabs_->count() && index < static_cast<int>(tabKeys_.size()); ++index) {
                tabs_->setTabText(index, trUi(tabKeys_[static_cast<size_t>(index)]));
            }
        }
        statusBar()->showMessage(trUi("Ready"));
    }

    fsc::core::IdentityMode selectedIdentityMode() const {
        const QString value = identityModeCombo_ == nullptr ? QString("strict") : identityModeCombo_->currentData().toString();
        if (value == "balanced") {
            return fsc::core::IdentityMode::Balanced;
        }
        if (value == "broad") {
            return fsc::core::IdentityMode::Broad;
        }
        return fsc::core::IdentityMode::Strict;
    }

    void buildUi() {
        auto* root = new QWidget(this);
        auto* rootLayout = new QHBoxLayout(root);
        rootLayout->setContentsMargins(0, 0, 0, 0);
        rootLayout->setSpacing(0);

        auto* sidebarPanel = new QWidget(root);
        sidebarPanel->setFixedWidth(190);
        sidebarPanel->setStyleSheet("background:#101827;");
        auto* sidebarLayout = new QVBoxLayout(sidebarPanel);
        sidebarLayout->setContentsMargins(0, 0, 0, 0);
        sidebarLayout->setSpacing(0);
        sidebar_ = new QListWidget(sidebarPanel);
        sidebar_->setFrameShape(QFrame::NoFrame);
        sidebar_->setStyleSheet(
            "QListWidget{background:#101827;color:#dce8f5;border:0;padding:8px;}"
            "QListWidget::item{padding:10px 12px;border-radius:6px;margin:2px 0;}"
            "QListWidget::item:selected{background:#2f80ed;color:white;}");
        sidebarLayout->addWidget(sidebar_, 1);

        auto* languageBox = new QWidget(sidebarPanel);
        auto* languageLayout = new QVBoxLayout(languageBox);
        languageLayout->setContentsMargins(10, 8, 10, 12);
        languageLayout->setSpacing(6);
        languageLabel_ = new QLabel(languageBox);
        languageLabel_->setStyleSheet("color:#9fb4ce;");
        languageCombo_ = new QComboBox(languageBox);
        languageCombo_->addItem("English", "en");
        languageCombo_->addItem(QString::fromUtf8("中文"), "zh");
        languageCombo_->addItem(QString::fromUtf8("日本語"), "ja");
        languageCombo_->addItem(QString::fromUtf8("한국어"), "ko");
        identityModeLabel_ = new QLabel(languageBox);
        identityModeLabel_->setStyleSheet("color:#9fb4ce;");
        identityModeCombo_ = new QComboBox(languageBox);
        identityModeCombo_->addItem("Strict", "strict");
        identityModeCombo_->setItemData(0, "Strict", Qt::UserRole + 1);
        identityModeCombo_->addItem("Balanced", "balanced");
        identityModeCombo_->setItemData(1, "Balanced", Qt::UserRole + 1);
        identityModeCombo_->addItem("Broad", "broad");
        identityModeCombo_->setItemData(2, "Broad", Qt::UserRole + 1);
        languageLayout->addWidget(languageLabel_);
        languageLayout->addWidget(languageCombo_);
        languageLayout->addWidget(identityModeLabel_);
        languageLayout->addWidget(identityModeCombo_);
        sidebarLayout->addWidget(languageBox);
        rootLayout->addWidget(sidebarPanel);

        auto* content = new QWidget(root);
        auto* contentLayout = new QVBoxLayout(content);
        contentLayout->setContentsMargins(10, 10, 10, 10);
        contentLayout->setSpacing(8);

        auto* toolbar = new QWidget(content);
        auto* toolbarLayout = new QHBoxLayout(toolbar);
        toolbarLayout->setContentsMargins(0, 0, 0, 0);
        toolbarLayout->setSpacing(6);

        databasePathEdit_ = new QLineEdit(toolbar);
        databasePathEdit_->setPlaceholderText("Open an FSC .fscdb database");
        auto* openButton = new QToolButton(toolbar);
        openButton->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
        openButton->setToolTip("Open Database");
        auto* createButton = new QToolButton(toolbar);
        createButton->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
        createButton->setToolTip("Create Database");
        auto* refreshButton = new QToolButton(toolbar);
        refreshButton->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
        refreshButton->setToolTip("Refresh");
        databaseLabel_ = new QLabel(toolbar);

        toolbarLayout->addWidget(databaseLabel_);
        toolbarLayout->addWidget(databasePathEdit_, 1);
        toolbarLayout->addWidget(openButton);
        toolbarLayout->addWidget(createButton);
        toolbarLayout->addWidget(refreshButton);
        contentLayout->addWidget(toolbar);

        tabs_ = new QTabWidget(content);
        tabs_->tabBar()->hide();
        tabs_->setDocumentMode(true);
        contentLayout->addWidget(tabs_, 1);
        rootLayout->addWidget(content, 1);
        buildOverviewTab();
        buildLibraryTab();
        buildPeopleTab();
        buildSearchTab();
        buildCameraTab();
        buildReviewTab();
        buildClustersTab();
        buildCompareTab();
        buildDenseMeshTab();
        buildRuntimeTab();

        setCentralWidget(root);
        applyLanguage();

        connect(openButton, &QToolButton::clicked, this, [this] { chooseDatabase(); });
        connect(createButton, &QToolButton::clicked, this, [this] { createDatabase(); });
        connect(refreshButton, &QToolButton::clicked, this, [this] { reloadAll(); });
        connect(databasePathEdit_, &QLineEdit::returnPressed, this, [this] { openDatabase(databasePathEdit_->text()); });
        connect(languageCombo_, &QComboBox::currentIndexChanged, this, [this] { applyLanguage(); });
        connect(sidebar_, &QListWidget::currentRowChanged, tabs_, &QTabWidget::setCurrentIndex);
        sidebar_->setCurrentRow(0);
    }

    void buildOverviewTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QFormLayout(page);
        formatLabel_ = new QLabel("-");
        modelLabel_ = new QLabel("-");
        metricLabel_ = new QLabel("-");
        facesLabel_ = new QLabel("-");
        peopleLabel_ = new QLabel("-");
        reviewLabel_ = new QLabel("-");
        qualityLabel_ = new QLabel("-");
        layout->addRow("Format", formatLabel_);
        layout->addRow("Model", modelLabel_);
        layout->addRow("Metric", metricLabel_);
        layout->addRow("Faces", facesLabel_);
        layout->addRow("People", peopleLabel_);
        layout->addRow("Review Queue", reviewLabel_);
        layout->addRow("Average Quality", qualityLabel_);
        addMainTab(page, "Overview");
    }

    void buildLibraryTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* mainSplitter = new QSplitter(Qt::Horizontal, page);
        auto* leftPanel = new QWidget(mainSplitter);
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        leftLayout->setSpacing(6);

        auto* controls = new QWidget(leftPanel);
        auto* form = new QFormLayout(controls);
        modelRootEdit_ = new QLineEdit(controls);
        modelRootEdit_->setText(defaultModelRoot());
        importImageEdit_ = new QLineEdit(controls);
        auto* modelButton = new QPushButton("Browse", controls);
        auto* imageButton = new QPushButton("Browse", controls);
        auto* importButton = new QPushButton("Import Image", controls);
        auto* importFolderButton = new QPushButton("Import Folder", controls);
        auto* exportButton = new QPushButton("Export CSV", controls);
        auto* reloadLibraryButton = new QPushButton("Reload", controls);
        auto* modelRow = new QWidget(controls);
        auto* modelRowLayout = new QHBoxLayout(modelRow);
        modelRowLayout->setContentsMargins(0, 0, 0, 0);
        modelRowLayout->addWidget(modelRootEdit_, 1);
        modelRowLayout->addWidget(modelButton);
        auto* imageRow = new QWidget(controls);
        auto* imageRowLayout = new QHBoxLayout(imageRow);
        imageRowLayout->setContentsMargins(0, 0, 0, 0);
        imageRowLayout->addWidget(importImageEdit_, 1);
        imageRowLayout->addWidget(imageButton);
        imageRowLayout->addWidget(importButton);
        imageRowLayout->addWidget(importFolderButton);
        imageRowLayout->addWidget(reloadLibraryButton);
        imageRowLayout->addWidget(exportButton);
        form->addRow("Models", modelRow);
        form->addRow("Image", imageRow);
        libraryImportMinQualitySpin_ = new QDoubleSpinBox(controls);
        libraryImportMinQualitySpin_->setRange(0.0, 1.0);
        libraryImportMinQualitySpin_->setDecimals(3);
        libraryImportMinQualitySpin_->setSingleStep(0.050);
        libraryImportMinQualitySpin_->setValue(0.0);
        form->addRow("Import min quality", libraryImportMinQualitySpin_);
        leftLayout->addWidget(controls);

        auto* filterControls = new QWidget(leftPanel);
        auto* filterLayout = new QGridLayout(filterControls);
        filterLayout->setContentsMargins(0, 0, 0, 0);
        libraryFilterTextEdit_ = new QLineEdit(filterControls);
        libraryFilterTextEdit_->setPlaceholderText("name, path, person, tag, notes");
        libraryFilterPersonCombo_ = new QComboBox(filterControls);
        libraryFilterTagCombo_ = new QComboBox(filterControls);
        libraryFilterReviewCombo_ = new QComboBox(filterControls);
        libraryFilterReviewCombo_->addItem("All", "");
        libraryFilterReviewCombo_->addItems({"open", "reviewed", "duplicate", "low_quality", "ignored"});
        libraryFilterMinQualitySpin_ = new QDoubleSpinBox(filterControls);
        libraryFilterMinQualitySpin_->setRange(0.0, 1.0);
        libraryFilterMinQualitySpin_->setDecimals(3);
        libraryFilterMinQualitySpin_->setSingleStep(0.050);
        libraryFilterMinQualitySpin_->setValue(0.0);
        libraryFilterIncludeIgnoredCheck_ = new QCheckBox("Include ignored", filterControls);
        libraryFilterIncludeIgnoredCheck_->setChecked(true);
        auto* applyFilterButton = new QPushButton("Apply Filter", filterControls);
        auto* resetFilterButton = new QPushButton("Reset Filter", filterControls);
        filterLayout->addWidget(new QLabel("Filter", filterControls), 0, 0);
        filterLayout->addWidget(libraryFilterTextEdit_, 0, 1, 1, 2);
        filterLayout->addWidget(new QLabel("Person", filterControls), 0, 3);
        filterLayout->addWidget(libraryFilterPersonCombo_, 0, 4);
        filterLayout->addWidget(new QLabel("Tag", filterControls), 0, 5);
        filterLayout->addWidget(libraryFilterTagCombo_, 0, 6);
        filterLayout->addWidget(libraryFilterIncludeIgnoredCheck_, 0, 7);
        filterLayout->addWidget(applyFilterButton, 0, 8);
        filterLayout->addWidget(resetFilterButton, 0, 9);
        filterLayout->addWidget(new QLabel("Review", filterControls), 1, 0);
        filterLayout->addWidget(libraryFilterReviewCombo_, 1, 1);
        filterLayout->addWidget(new QLabel("Min quality", filterControls), 1, 2);
        filterLayout->addWidget(libraryFilterMinQualitySpin_, 1, 3);
        filterLayout->setColumnStretch(10, 1);
        leftLayout->addWidget(filterControls);

        auto* splitter = new QSplitter(Qt::Vertical, leftPanel);
        libraryTable_ = new QTableWidget(splitter);
        libraryTable_->setColumnCount(9);
        libraryTable_->setHorizontalHeaderLabels({"ID", "Name", "Person", "Tags", "Review", "Ignored", "Dupes", "Quality", "Source"});
        fitTable(libraryTable_);
        libraryTable_->setSelectionMode(QAbstractItemView::ExtendedSelection);
        importLog_ = new QTableWidget(splitter);
        importLog_->setColumnCount(6);
        importLog_->setHorizontalHeaderLabels({"Inserted ID", "Face", "Detection", "Quality", "2D", "3D"});
        fitTable(importLog_);
        importLog_->setMaximumHeight(150);
        splitter->addWidget(libraryTable_);
        splitter->addWidget(importLog_);
        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 0);
        leftLayout->addWidget(splitter, 1);

        auto* visualPanel = new QWidget(mainSplitter);
        auto* visualLayout = new QVBoxLayout(visualPanel);
        visualLayout->setContentsMargins(8, 0, 0, 0);

        libraryVisualTabs_ = new QTabWidget(visualPanel);
        auto* imageTab = new QWidget(libraryVisualTabs_);
        auto* imageLayout = new QVBoxLayout(imageTab);
        imageLayout->setContentsMargins(0, 0, 0, 0);
        libraryFocusButton_ = new QPushButton("Focus on Face", imageTab);
        libraryFocusButton_->setMaximumWidth(132);
        libraryPreviewLabel_ = new QLabel("Select a face", imageTab);
        libraryPreviewLabel_->setAlignment(Qt::AlignCenter);
        libraryPreviewLabel_->setMinimumWidth(300);
        libraryPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        imageLayout->addWidget(libraryFocusButton_, 0, Qt::AlignLeft);
        imageLayout->addWidget(libraryPreviewLabel_, 1);

        auto* landmarksTab = new QWidget(libraryVisualTabs_);
        auto* landmarksLayout = new QVBoxLayout(landmarksTab);
        landmarksLayout->setContentsMargins(0, 0, 0, 0);
        libraryLandmarksView_ = new PointCloudWidget(landmarksTab);
        landmarksLayout->addWidget(libraryLandmarksView_, 1);

        auto* denseTab = new QWidget(libraryVisualTabs_);
        auto* denseLayout = new QVBoxLayout(denseTab);
        denseLayout->setContentsMargins(0, 0, 0, 0);
        auto* denseControls = new QWidget(denseTab);
        auto* denseControlsLayout = new QHBoxLayout(denseControls);
        denseControlsLayout->setContentsMargins(0, 0, 0, 0);
        libraryMeshOverlayCheck_ = new QCheckBox("3D Landmarks", denseControls);
        libraryMeshOverlayCheck_->setChecked(true);
        auto* generateLibraryMeshButton = new QPushButton("Generate Native Mesh", denseControls);
        libraryMeshStatusLabel_ = new QLabel("Select a face", denseControls);
        denseControlsLayout->addWidget(libraryMeshOverlayCheck_);
        denseControlsLayout->addWidget(generateLibraryMeshButton);
        denseControlsLayout->addWidget(libraryMeshStatusLabel_, 1);
        libraryDenseMeshView_ = new PointCloudWidget(denseTab);
        denseLayout->addWidget(denseControls);
        denseLayout->addWidget(libraryDenseMeshView_, 1);

        libraryVisualTabs_->addTab(imageTab, "Image");
        libraryVisualTabs_->addTab(landmarksTab, "3D Landmarks");
        libraryVisualTabs_->addTab(denseTab, "Dense Mesh");
        visualLayout->addWidget(libraryVisualTabs_, 2);
        auto* metadataTabs = new QTabWidget(visualPanel);
        auto* selectedTab = new QWidget(metadataTabs);
        auto* selectedForm = new QFormLayout(selectedTab);
        libraryPersonEdit_ = new QLineEdit(selectedTab);
        libraryTagsEdit_ = new QLineEdit(selectedTab);
        libraryReviewCombo_ = new QComboBox(selectedTab);
        libraryReviewCombo_->addItems({"open", "reviewed", "duplicate", "low_quality", "ignored"});
        libraryIgnoredCheck_ = new QCheckBox("Ignore in search", selectedTab);
        libraryNotesEdit_ = new QTextEdit(selectedTab);
        libraryNotesEdit_->setMinimumHeight(62);
        auto* saveMetadataButton = new QPushButton("Save Metadata", selectedTab);
        selectedForm->addRow("Person", libraryPersonEdit_);
        selectedForm->addRow("Tags", libraryTagsEdit_);
        selectedForm->addRow("Review", libraryReviewCombo_);
        selectedForm->addRow("", libraryIgnoredCheck_);
        selectedForm->addRow("Notes", libraryNotesEdit_);
        selectedForm->addRow("", saveMetadataButton);

        auto* batchTab = new QWidget(metadataTabs);
        auto* batchForm = new QFormLayout(batchTab);
        libraryBatchPersonEdit_ = new QLineEdit(batchTab);
        libraryBatchTagsEdit_ = new QLineEdit(batchTab);
        libraryBatchAppendTagsCheck_ = new QCheckBox("Append tags", batchTab);
        libraryBatchReviewCombo_ = new QComboBox(batchTab);
        libraryBatchReviewCombo_->addItems({"No change", "open", "reviewed", "duplicate", "low_quality", "ignored"});
        libraryBatchIgnoredCombo_ = new QComboBox(batchTab);
        libraryBatchIgnoredCombo_->addItems({"No change", "Ignore", "Restore"});
        libraryBatchNotesEdit_ = new QLineEdit(batchTab);
        libraryBatchNotesEdit_->setPlaceholderText("leave blank for no change");
        auto* applyBatchButton = new QPushButton("Apply to Selection", batchTab);
        batchForm->addRow("Person", libraryBatchPersonEdit_);
        batchForm->addRow("Tags", libraryBatchTagsEdit_);
        batchForm->addRow("", libraryBatchAppendTagsCheck_);
        batchForm->addRow("Review", libraryBatchReviewCombo_);
        batchForm->addRow("Ignored", libraryBatchIgnoredCombo_);
        batchForm->addRow("Notes", libraryBatchNotesEdit_);
        batchForm->addRow("", applyBatchButton);
        metadataTabs->addTab(selectedTab, "Selected");
        metadataTabs->addTab(batchTab, "Batch");
        auto* activityTab = new QWidget(metadataTabs);
        auto* activityLayout = new QVBoxLayout(activityTab);
        activityLayout->setContentsMargins(0, 0, 0, 0);
        libraryProgressBar_ = new QProgressBar(activityTab);
        libraryProgressBar_->setRange(0, 100);
        libraryProgressBar_->setValue(0);
        libraryActivityLog_ = new QTextEdit(activityTab);
        libraryActivityLog_->setReadOnly(true);
        libraryActivityLog_->setMinimumHeight(130);
        activityLayout->addWidget(libraryProgressBar_);
        activityLayout->addWidget(libraryActivityLog_, 1);
        metadataTabs->addTab(activityTab, "Activity");
        visualLayout->addWidget(metadataTabs, 1);

        mainSplitter->addWidget(leftPanel);
        mainSplitter->addWidget(visualPanel);
        mainSplitter->setStretchFactor(0, 3);
        mainSplitter->setStretchFactor(1, 2);
        layout->addWidget(mainSplitter, 1);
        addMainTab(page, "Library");
        connect(libraryTable_, &QTableWidget::itemSelectionChanged, this, [this] {
            const auto selected = libraryTable_->selectedItems();
            if (selected.empty()) {
                return;
            }
            const int row = selected.front()->row();
            const auto* idItem = libraryTable_->item(row, 0);
            if (idItem == nullptr) {
                return;
            }
            const int faceId = idItem->text().toInt();
            if (faceId > 0) {
                if (faceIdSpin_ != nullptr) {
                    faceIdSpin_->setValue(faceId);
                }
                if (assignFaceSpin_ != nullptr) {
                    assignFaceSpin_->setValue(faceId);
                }
                if (meshFaceIdSpin_ != nullptr) {
                    meshFaceIdSpin_->setValue(faceId);
                }
                libraryFocusOnFace_ = false;
                loadLibraryMetadata(faceId);
                updateLibraryVisuals(faceId);
            }
        });
        connect(modelButton, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getExistingDirectory(this, "Select model root", modelRootEdit_->text());
            if (!path.isEmpty()) {
                modelRootEdit_->setText(path);
            }
        });
        connect(imageButton, &QPushButton::clicked, this, [this] { chooseImage(importImageEdit_); });
        connect(importButton, &QPushButton::clicked, this, [this] { importImage(); });
        connect(importFolderButton, &QPushButton::clicked, this, [this] { importFolder(); });
        connect(reloadLibraryButton, &QPushButton::clicked, this, [this] { loadLibrary(); });
        connect(exportButton, &QPushButton::clicked, this, [this] { exportLibraryCsv(); });
        connect(applyFilterButton, &QPushButton::clicked, this, [this] { loadLibrary(); });
        connect(resetFilterButton, &QPushButton::clicked, this, [this] { resetLibraryFilters(); });
        connect(libraryFilterTextEdit_, &QLineEdit::returnPressed, this, [this] { loadLibrary(); });
        connect(libraryFocusButton_, &QPushButton::clicked, this, [this] {
            libraryFocusOnFace_ = !libraryFocusOnFace_;
            if (libraryPreviewFaceId_ > 0) {
                updateLibraryPreview(libraryPreviewFaceId_);
            }
        });
        connect(libraryMeshOverlayCheck_, &QCheckBox::toggled, this, [this] {
            if (libraryPreviewFaceId_ > 0) {
                updateLibrary3dPreview(libraryPreviewFaceId_);
            }
        });
        connect(generateLibraryMeshButton, &QPushButton::clicked, this, [this] { generateLibraryMeshForSelectedFace(); });
        connect(saveMetadataButton, &QPushButton::clicked, this, [this] { saveLibrarySelectedMetadata(); });
        connect(applyBatchButton, &QPushButton::clicked, this, [this] { applyLibraryBatchMetadata(); });
    }

    void buildPeopleTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        personNameEdit_ = new QLineEdit(controls);
        personNameEdit_->setPlaceholderText("Person name");
        assignFaceSpin_ = new QSpinBox(controls);
        assignFaceSpin_->setRange(1, 999999999);
        assignFaceSpin_->setPrefix("Face ");
        assignPersonSpin_ = new QSpinBox(controls);
        assignPersonSpin_->setRange(1, 999999999);
        assignPersonSpin_->setPrefix("Person ");
        auto* addButton = new QPushButton("Add Person", controls);
        auto* assignButton = new QPushButton("Assign Face", controls);
        auto* trainButton = new QPushButton("Train Profiles", controls);
        controlsLayout->addWidget(personNameEdit_, 1);
        controlsLayout->addWidget(addButton);
        controlsLayout->addWidget(assignFaceSpin_);
        controlsLayout->addWidget(assignPersonSpin_);
        controlsLayout->addWidget(assignButton);
        controlsLayout->addWidget(trainButton);
        layout->addWidget(controls);

        peopleTable_ = new QTableWidget(page);
        peopleTable_->setColumnCount(8);
        peopleTable_->setHorizontalHeaderLabels({"ID", "Name", "Faces", "Avg Quality", "Identity", "Samples", "Exemplars", "Health"});
        fitTable(peopleTable_);
        layout->addWidget(peopleTable_, 1);
        addMainTab(page, "People");
        connect(peopleTable_, &QTableWidget::itemSelectionChanged, this, [this] {
            const auto selected = peopleTable_->selectedItems();
            if (selected.empty()) {
                return;
            }
            const int row = selected.front()->row();
            const auto* idItem = peopleTable_->item(row, 0);
            if (idItem != nullptr) {
                const int personId = idItem->text().toInt();
                if (personId > 0) {
                    assignPersonSpin_->setValue(personId);
                }
            }
        });
        connect(addButton, &QPushButton::clicked, this, [this] { addPerson(); });
        connect(assignButton, &QPushButton::clicked, this, [this] { assignFace(); });
        connect(trainButton, &QPushButton::clicked, this, [this] { trainProfiles(); });
    }

    void buildReviewTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        reviewStateCombo_ = new QComboBox(controls);
        reviewStateCombo_->addItems({"open", "reviewed", "duplicate", "low_quality", "ignored"});
        reviewIgnoredCombo_ = new QComboBox(controls);
        reviewIgnoredCombo_->addItems({"Not ignored", "Ignored"});
        reviewNotesEdit_ = new QLineEdit(controls);
        reviewNotesEdit_->setPlaceholderText("Notes");
        auto* applyButton = new QPushButton("Apply", controls);
        auto* reviewedButton = new QPushButton("Reviewed", controls);
        auto* ignoredButton = new QPushButton("Ignore", controls);
        auto* suggestButton = new QPushButton("Suggest Person", controls);
        auto* confirmSuggestionButton = new QPushButton("Confirm Suggestion", controls);
        controlsLayout->addWidget(reviewStateCombo_);
        controlsLayout->addWidget(reviewIgnoredCombo_);
        controlsLayout->addWidget(reviewNotesEdit_, 1);
        controlsLayout->addWidget(applyButton);
        controlsLayout->addWidget(reviewedButton);
        controlsLayout->addWidget(ignoredButton);
        controlsLayout->addWidget(suggestButton);
        controlsLayout->addWidget(confirmSuggestionButton);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        reviewTable_ = new QTableWidget(splitter);
        reviewTable_->setColumnCount(8);
        reviewTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Quality", "Detection", "Review", "Ignored", "Notes"});
        fitTable(reviewTable_);
        auto* detailPanel = new QWidget(splitter);
        auto* detailLayout = new QVBoxLayout(detailPanel);
        detailLayout->setContentsMargins(8, 0, 0, 0);
        reviewPreviewLabel_ = new QLabel("Select a review item", detailPanel);
        reviewPreviewLabel_->setAlignment(Qt::AlignCenter);
        reviewPreviewLabel_->setMinimumWidth(320);
        reviewPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        reviewSuggestionLabel_ = new QLabel("AI Suggested Person: -", detailPanel);
        reviewSuggestionLabel_->setWordWrap(true);
        detailLayout->addWidget(reviewPreviewLabel_, 1);
        detailLayout->addWidget(reviewSuggestionLabel_);
        splitter->addWidget(reviewTable_);
        splitter->addWidget(detailPanel);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);
        addMainTab(page, "Review");

        connect(reviewTable_, &QTableWidget::itemSelectionChanged, this, [this] {
            const auto selected = reviewTable_->selectedItems();
            if (selected.empty()) {
                return;
            }
            const int row = selected.front()->row();
            if (const auto* idItem = reviewTable_->item(row, 0); idItem != nullptr && faceIdSpin_ != nullptr) {
                const int faceId = idItem->text().toInt();
                faceIdSpin_->setValue(faceId);
                if (meshFaceIdSpin_ != nullptr) {
                    meshFaceIdSpin_->setValue(faceId);
                }
            }
            if (const auto* stateItem = reviewTable_->item(row, 5); stateItem != nullptr) {
                reviewStateCombo_->setCurrentText(stateItem->text());
            }
            if (const auto* ignoredItem = reviewTable_->item(row, 6); ignoredItem != nullptr) {
                reviewIgnoredCombo_->setCurrentIndex(ignoredItem->text() == "yes" ? 1 : 0);
            }
            if (const auto* notesItem = reviewTable_->item(row, 7); notesItem != nullptr) {
                reviewNotesEdit_->setText(notesItem->text());
            }
            if (const auto* idItem = reviewTable_->item(row, 0); idItem != nullptr) {
                updateReviewDetail(idItem->text().toInt());
            }
        });
        connect(applyButton, &QPushButton::clicked, this, [this] { applyReviewFromControls(); });
        connect(reviewedButton, &QPushButton::clicked, this, [this] { applyReviewState("reviewed", false); });
        connect(ignoredButton, &QPushButton::clicked, this, [this] { applyReviewState("ignored", true); });
        connect(suggestButton, &QPushButton::clicked, this, [this] { suggestReviewPerson(); });
        connect(confirmSuggestionButton, &QPushButton::clicked, this, [this] { confirmReviewSuggestion(); });
    }

    void buildSearchTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* imageControls = new QWidget(page);
        auto* imageLayout = new QHBoxLayout(imageControls);
        imageLayout->setContentsMargins(0, 0, 0, 0);
        searchImageEdit_ = new QLineEdit(imageControls);
        searchImageEdit_->setPlaceholderText("Query image");
        auto* browseImageButton = new QPushButton("Browse", imageControls);
        auto* analyzeImageButton = new QPushButton("Analyze Query", imageControls);
        searchFaceCombo_ = new QComboBox(imageControls);
        searchFaceCombo_->setMinimumWidth(180);
        imageLayout->addWidget(searchImageEdit_, 1);
        imageLayout->addWidget(browseImageButton);
        imageLayout->addWidget(analyzeImageButton);
        imageLayout->addWidget(searchFaceCombo_);
        layout->addWidget(imageControls);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        faceIdSpin_ = new QSpinBox(controls);
        faceIdSpin_->setRange(1, 999999999);
        faceIdSpin_->setPrefix("Face ");
        topKSpin_ = new QSpinBox(controls);
        topKSpin_->setRange(1, 500);
        topKSpin_->setValue(30);
        topKSpin_->setPrefix("Top ");
        searchThresholdSpin_ = new QDoubleSpinBox(controls);
        searchThresholdSpin_->setRange(-1.0, 1.0);
        searchThresholdSpin_->setDecimals(3);
        searchThresholdSpin_->setSingleStep(0.010);
        searchThresholdSpin_->setValue(-1.0);
        searchThresholdSpin_->setPrefix("Threshold ");
        searchMinQualitySpin_ = new QDoubleSpinBox(controls);
        searchMinQualitySpin_->setRange(0.0, 1.0);
        searchMinQualitySpin_->setDecimals(3);
        searchMinQualitySpin_->setSingleStep(0.050);
        searchMinQualitySpin_->setValue(0.0);
        searchMinQualitySpin_->setPrefix("Min quality ");
        searchIncludeIgnoredCheck_ = new QCheckBox("Include ignored", controls);
        auto* searchButton = new QPushButton("Search", controls);
        auto* identifyButton = new QPushButton("Identify", controls);
        identityLabel_ = new QLabel("Identity: -", controls);
        controlsLayout->addWidget(faceIdSpin_);
        controlsLayout->addWidget(topKSpin_);
        controlsLayout->addWidget(searchThresholdSpin_);
        controlsLayout->addWidget(searchMinQualitySpin_);
        controlsLayout->addWidget(searchIncludeIgnoredCheck_);
        controlsLayout->addWidget(searchButton);
        controlsLayout->addWidget(identifyButton);
        controlsLayout->addWidget(identityLabel_, 1);
        layout->addWidget(controls);

        auto* filterControls = new QWidget(page);
        auto* filterLayout = new QHBoxLayout(filterControls);
        filterLayout->setContentsMargins(0, 0, 0, 0);
        searchPersonFilterCombo_ = new QComboBox(filterControls);
        searchPersonFilterCombo_->setMinimumWidth(180);
        searchTagFilterCombo_ = new QComboBox(filterControls);
        searchTagFilterCombo_->setMinimumWidth(160);
        filterLayout->addWidget(new QLabel("Person", filterControls));
        filterLayout->addWidget(searchPersonFilterCombo_);
        filterLayout->addWidget(new QLabel("Tag", filterControls));
        filterLayout->addWidget(searchTagFilterCombo_);
        filterLayout->addStretch(1);
        layout->addWidget(filterControls);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        auto* leftPanel = new QWidget(splitter);
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        searchIdentityTable_ = new QTableWidget(leftPanel);
        searchIdentityTable_->setColumnCount(6);
        searchIdentityTable_->setHorizontalHeaderLabels({"Rank", "Person", "Decision", "Score", "Confidence", "Evidence"});
        fitTable(searchIdentityTable_);
        searchIdentityTable_->setMaximumHeight(130);
        leftLayout->addWidget(searchIdentityTable_);
        searchTable_ = new QTableWidget(leftPanel);
        searchTable_->setColumnCount(8);
        searchTable_->setHorizontalHeaderLabels({"Rank", "ID", "File", "Person", "Tags", "Cosine", "Similarity", "Quality"});
        fitTable(searchTable_);
        leftLayout->addWidget(searchTable_, 1);

        auto* actionRow = new QWidget(leftPanel);
        auto* actionLayout = new QHBoxLayout(actionRow);
        actionLayout->setContentsMargins(0, 0, 0, 0);
        searchAssignPersonEdit_ = new QLineEdit(actionRow);
        searchAssignPersonEdit_->setPlaceholderText("Person for selected result");
        auto* assignResultButton = new QPushButton("Assign Result", actionRow);
        auto* confirmIdentityButton = new QPushButton("Confirm Identity", actionRow);
        actionLayout->addWidget(searchAssignPersonEdit_, 1);
        actionLayout->addWidget(assignResultButton);
        actionLayout->addWidget(confirmIdentityButton);
        leftLayout->addWidget(actionRow);

        auto* previewPanel = new QWidget(splitter);
        auto* previewLayout = new QVBoxLayout(previewPanel);
        previewLayout->setContentsMargins(8, 0, 0, 0);
        searchPreviewLabel_ = new QLabel("Select or analyze a query image", previewPanel);
        searchPreviewLabel_->setAlignment(Qt::AlignCenter);
        searchPreviewLabel_->setMinimumWidth(320);
        searchPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        searchResultPreviewLabel_ = new QLabel("Result preview", previewPanel);
        searchResultPreviewLabel_->setAlignment(Qt::AlignCenter);
        searchResultPreviewLabel_->setMinimumWidth(320);
        searchResultPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        previewLayout->addWidget(searchPreviewLabel_, 1);
        previewLayout->addWidget(searchResultPreviewLabel_, 1);
        splitter->addWidget(leftPanel);
        splitter->addWidget(previewPanel);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);

        searchPreviewTimer_ = new QTimer(this);
        searchPreviewTimer_->setInterval(70);
        addMainTab(page, "Search");
        connect(browseImageButton, &QPushButton::clicked, this, [this] { chooseImage(searchImageEdit_); });
        connect(analyzeImageButton, &QPushButton::clicked, this, [this] { analyzeSearchImage(); });
        connect(searchFaceCombo_, &QComboBox::currentIndexChanged, this, [this](int index) {
            searchQueryFaceIndex_ = index;
            updateSearchQueryPreview();
        });
        connect(searchButton, &QPushButton::clicked, this, [this] { runSearch(); });
        connect(identifyButton, &QPushButton::clicked, this, [this] { runIdentify(); });
        connect(searchTable_, &QTableWidget::itemSelectionChanged, this, [this] { updateSelectedSearchResultPreview(); });
        connect(searchIdentityTable_, &QTableWidget::itemSelectionChanged, this, [this] { updateSelectedSearchIdentityPreview(); });
        connect(assignResultButton, &QPushButton::clicked, this, [this] { assignSelectedSearchResult(); });
        connect(confirmIdentityButton, &QPushButton::clicked, this, [this] { confirmSearchIdentity(); });
        connect(searchPreviewTimer_, &QTimer::timeout, this, [this] { advanceSearchPreviewAnimation(); });
        refreshSearchFilterOptions();
    }

    void buildCameraTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        cameraIndexSpin_ = new QSpinBox(controls);
        cameraIndexSpin_->setRange(0, 16);
        cameraIndexSpin_->setPrefix("Camera ");
        cameraThresholdSpin_ = new QDoubleSpinBox(controls);
        cameraThresholdSpin_->setRange(-1.0, 1.0);
        cameraThresholdSpin_->setDecimals(3);
        cameraThresholdSpin_->setSingleStep(0.010);
        cameraThresholdSpin_->setValue(0.350);
        cameraThresholdSpin_->setPrefix("Threshold ");
        cameraTopKSpin_ = new QSpinBox(controls);
        cameraTopKSpin_->setRange(1, 10);
        cameraTopKSpin_->setValue(3);
        cameraTopKSpin_->setPrefix("Top ");
        cameraIntervalSpin_ = new QSpinBox(controls);
        cameraIntervalSpin_->setRange(100, 5000);
        cameraIntervalSpin_->setSingleStep(50);
        cameraIntervalSpin_->setValue(1200);
        cameraIntervalSpin_->setSuffix(" ms");
        cameraProcessSizeSpin_ = new QSpinBox(controls);
        cameraProcessSizeSpin_->setRange(320, 1920);
        cameraProcessSizeSpin_->setSingleStep(80);
        cameraProcessSizeSpin_->setValue(640);
        cameraProcessSizeSpin_->setPrefix("Process ");
        cameraAutoCheck_ = new QCheckBox("Auto Identify", controls);
        cameraAutoCheck_->setChecked(true);
        auto* startButton = new QPushButton("Start", controls);
        auto* stopButton = new QPushButton("Stop", controls);
        auto* identifyButton = new QPushButton("Identify Frame", controls);
        auto* useLibraryButton = new QPushButton("Use Library DB", controls);
        cameraStatusLabel_ = new QLabel("Camera stopped", controls);
        cameraDatabaseLabel_ = new QLabel("Database: not opened", controls);
        controlsLayout->addWidget(cameraIndexSpin_);
        controlsLayout->addWidget(cameraThresholdSpin_);
        controlsLayout->addWidget(cameraTopKSpin_);
        controlsLayout->addWidget(cameraIntervalSpin_);
        controlsLayout->addWidget(cameraProcessSizeSpin_);
        controlsLayout->addWidget(cameraAutoCheck_);
        controlsLayout->addWidget(startButton);
        controlsLayout->addWidget(stopButton);
        controlsLayout->addWidget(identifyButton);
        controlsLayout->addWidget(useLibraryButton);
        controlsLayout->addWidget(cameraStatusLabel_, 1);
        layout->addWidget(controls);

        auto* databaseRow = new QWidget(page);
        auto* databaseRowLayout = new QHBoxLayout(databaseRow);
        databaseRowLayout->setContentsMargins(0, 4, 0, 4);
        databaseRowLayout->addWidget(cameraDatabaseLabel_, 1);
        layout->addWidget(databaseRow);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        cameraPreviewLabel_ = new QLabel(page);
        cameraPreviewLabel_->setMinimumSize(640, 360);
        cameraPreviewLabel_->setAlignment(Qt::AlignCenter);
        cameraPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;");
        cameraPreviewLabel_->setText("Camera preview");
        splitter->addWidget(cameraPreviewLabel_);

        auto* resultsPanel = new QWidget(splitter);
        auto* resultsLayout = new QVBoxLayout(resultsPanel);
        resultsLayout->setContentsMargins(0, 0, 0, 0);
        cameraMatchPreviewLabel_ = new QLabel(resultsPanel);
        cameraMatchPreviewLabel_->setMinimumSize(320, 240);
        cameraMatchPreviewLabel_->setAlignment(Qt::AlignCenter);
        cameraMatchPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        cameraMatchPreviewLabel_->setText("Best match preview");
        resultsLayout->addWidget(cameraMatchPreviewLabel_, 1);
        cameraMatchStatusLabel_ = new QLabel("Match: --", resultsPanel);
        resultsLayout->addWidget(cameraMatchStatusLabel_);

        cameraIdentityLabel_ = new QLabel("Identity: -", page);
        resultsLayout->addWidget(cameraIdentityLabel_);
        cameraResultTable_ = new QTableWidget(page);
        cameraResultTable_->setColumnCount(11);
        cameraResultTable_->setHorizontalHeaderLabels({
            "Face",
            "Identity",
            "Decision",
            "Conf",
            "Evidence",
            "ID",
            "Name",
            "Person",
            "Cosine",
            "Similarity",
            "Quality",
        });
        fitTable(cameraResultTable_);
        resultsLayout->addWidget(cameraResultTable_, 2);
        splitter->addWidget(resultsPanel);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);

        cameraFrameTimer_ = new QTimer(this);
        cameraFrameTimer_->setInterval(33);
        cameraIdentifyTimer_ = new QTimer(this);
        cameraIdentifyTimer_->setInterval(cameraIntervalSpin_->value());
        addMainTab(page, "Camera");

        connect(startButton, &QPushButton::clicked, this, [this] { startCamera(); });
        connect(stopButton, &QPushButton::clicked, this, [this] { stopCamera(); });
        connect(identifyButton, &QPushButton::clicked, this, [this] { identifyCameraFrame(); });
        connect(useLibraryButton, &QPushButton::clicked, this, [this] { refreshCameraDatabaseLabel(); });
        connect(cameraIntervalSpin_, &QSpinBox::valueChanged, this, [this](int value) {
            if (cameraIdentifyTimer_ != nullptr) {
                cameraIdentifyTimer_->setInterval(value);
            }
        });
        connect(cameraFrameTimer_, &QTimer::timeout, this, [this] { captureCameraFrame(); });
        connect(cameraIdentifyTimer_, &QTimer::timeout, this, [this] {
            if (cameraAutoCheck_ != nullptr && cameraAutoCheck_->isChecked()) {
                identifyCameraFrame();
            }
        });
    }

    void buildCompareTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* formWidget = new QWidget(page);
        auto* form = new QFormLayout(formWidget);
        compareImageAEdit_ = new QLineEdit(formWidget);
        compareImageBEdit_ = new QLineEdit(formWidget);
        compareImageAEdit_->setReadOnly(true);
        compareImageBEdit_->setReadOnly(true);
        auto* rowA = new QWidget(formWidget);
        auto* rowALayout = new QHBoxLayout(rowA);
        rowALayout->setContentsMargins(0, 0, 0, 0);
        auto* browseA = new QPushButton("Image A", rowA);
        rowALayout->addWidget(compareImageAEdit_, 1);
        rowALayout->addWidget(browseA);
        auto* rowB = new QWidget(formWidget);
        auto* rowBLayout = new QHBoxLayout(rowB);
        rowBLayout->setContentsMargins(0, 0, 0, 0);
        auto* browseB = new QPushButton("Image B", rowB);
        rowBLayout->addWidget(compareImageBEdit_, 1);
        rowBLayout->addWidget(browseB);
        auto* compareButton = new QPushButton("Compare", formWidget);
        compareResultLabel_ = new QLabel("Cosine: --    Similarity: --", formWidget);
        compareResultLabel_->setAlignment(Qt::AlignCenter);
        form->addRow("Image A", rowA);
        form->addRow("Image B", rowB);
        form->addRow("", compareButton);
        form->addRow("Result", compareResultLabel_);
        layout->addWidget(formWidget);

        auto* previews = new QWidget(page);
        auto* previewsLayout = new QHBoxLayout(previews);
        previewsLayout->setContentsMargins(0, 0, 0, 0);
        previewsLayout->setSpacing(8);
        auto* panelA = new QWidget(previews);
        auto* panelALayout = new QVBoxLayout(panelA);
        panelALayout->setContentsMargins(0, 0, 0, 0);
        compareFocusAButton_ = new QPushButton("Focus on Face", panelA);
        compareFocusAButton_->setMaximumWidth(132);
        comparePreviewA_ = new FaceSelectionPreview(panelA);
        compareFaceListA_ = new QListWidget(panelA);
        compareFaceListA_->setMaximumHeight(110);
        panelALayout->addWidget(compareFocusAButton_, 0, Qt::AlignLeft);
        panelALayout->addWidget(comparePreviewA_, 1);
        panelALayout->addWidget(compareFaceListA_);
        auto* panelB = new QWidget(previews);
        auto* panelBLayout = new QVBoxLayout(panelB);
        panelBLayout->setContentsMargins(0, 0, 0, 0);
        compareFocusBButton_ = new QPushButton("Focus on Face", panelB);
        compareFocusBButton_->setMaximumWidth(132);
        comparePreviewB_ = new FaceSelectionPreview(panelB);
        compareFaceListB_ = new QListWidget(panelB);
        compareFaceListB_->setMaximumHeight(110);
        panelBLayout->addWidget(compareFocusBButton_, 0, Qt::AlignLeft);
        panelBLayout->addWidget(comparePreviewB_, 1);
        panelBLayout->addWidget(compareFaceListB_);
        previewsLayout->addWidget(panelA);
        previewsLayout->addWidget(panelB);
        layout->addWidget(previews, 1);

        compareFaceTable_ = new QTableWidget(page);
        compareFaceTable_->setColumnCount(5);
        compareFaceTable_->setHorizontalHeaderLabels({"Image", "Detection", "Quality", "2D", "3D"});
        fitTable(compareFaceTable_);
        compareFaceTable_->setMaximumHeight(110);
        layout->addWidget(compareFaceTable_);
        addMainTab(page, "Compare");

        comparePreviewA_->faceClicked = [this](int index) { selectCompareFace('a', index); };
        comparePreviewB_->faceClicked = [this](int index) { selectCompareFace('b', index); };
        connect(browseA, &QPushButton::clicked, this, [this] { selectCompareImage('a'); });
        connect(browseB, &QPushButton::clicked, this, [this] { selectCompareImage('b'); });
        connect(compareFocusAButton_, &QPushButton::clicked, this, [this] { toggleCompareFocus('a'); });
        connect(compareFocusBButton_, &QPushButton::clicked, this, [this] { toggleCompareFocus('b'); });
        connect(compareFaceListA_, &QListWidget::itemSelectionChanged, this, [this] {
            if (!updatingCompareLists_) {
                selectCompareFace('a', compareFaceListA_->currentRow());
            }
        });
        connect(compareFaceListB_, &QListWidget::itemSelectionChanged, this, [this] {
            if (!updatingCompareLists_) {
                selectCompareFace('b', compareFaceListB_->currentRow());
            }
        });
        connect(compareButton, &QPushButton::clicked, this, [this] { compareImages(); });
    }

    void buildClustersTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        clusterThresholdSpin_ = new QDoubleSpinBox(controls);
        clusterThresholdSpin_->setRange(-1.0, 1.0);
        clusterThresholdSpin_->setSingleStep(0.01);
        clusterThresholdSpin_->setDecimals(3);
        clusterThresholdSpin_->setValue(0.62);
        clusterMinSizeSpin_ = new QSpinBox(controls);
        clusterMinSizeSpin_->setRange(2, 50);
        clusterMinSizeSpin_->setValue(2);
        clusterMaxFacesSpin_ = new QSpinBox(controls);
        clusterMaxFacesSpin_->setRange(100, 100000);
        clusterMaxFacesSpin_->setValue(5000);
        clusterMaxFacesSpin_->setPrefix("Max ");
        clusterMinQualitySpin_ = new QDoubleSpinBox(controls);
        clusterMinQualitySpin_->setRange(0.0, 1.0);
        clusterMinQualitySpin_->setDecimals(3);
        clusterMinQualitySpin_->setSingleStep(0.050);
        clusterMinQualitySpin_->setValue(0.0);
        clusterMinQualitySpin_->setPrefix("Min quality ");
        clusterUnassignedOnlyCheck_ = new QCheckBox("Unassigned only", controls);
        clusterIncludeIgnoredCheck_ = new QCheckBox("Include ignored", controls);
        auto* clusterButton = new QPushButton("Build Clusters", controls);
        controlsLayout->addWidget(new QLabel("Threshold", controls));
        controlsLayout->addWidget(clusterThresholdSpin_);
        controlsLayout->addWidget(new QLabel("Min Size", controls));
        controlsLayout->addWidget(clusterMinSizeSpin_);
        controlsLayout->addWidget(clusterMaxFacesSpin_);
        controlsLayout->addWidget(clusterMinQualitySpin_);
        controlsLayout->addWidget(clusterUnassignedOnlyCheck_);
        controlsLayout->addWidget(clusterIncludeIgnoredCheck_);
        controlsLayout->addWidget(clusterButton);
        controlsLayout->addStretch(1);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(page);
        clusterTable_ = new QTableWidget(splitter);
        clusterTable_->setColumnCount(6);
        clusterTable_->setHorizontalHeaderLabels({"Cluster", "Faces", "Mean", "Max", "Avg Quality", "Known People"});
        fitTable(clusterTable_);
        clusterMemberTable_ = new QTableWidget(splitter);
        clusterMemberTable_->setColumnCount(6);
        clusterMemberTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Tags", "Quality", "Review"});
        fitTable(clusterMemberTable_);

        auto* actionPanel = new QWidget(splitter);
        auto* actionLayout = new QVBoxLayout(actionPanel);
        actionLayout->setContentsMargins(8, 0, 0, 0);
        clusterPreviewLabel_ = new QLabel("Select a cluster", actionPanel);
        clusterPreviewLabel_->setAlignment(Qt::AlignCenter);
        clusterPreviewLabel_->setMinimumWidth(320);
        clusterPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        auto* assignBox = new QGroupBox("Batch Assign", actionPanel);
        auto* assignForm = new QFormLayout(assignBox);
        clusterPersonEdit_ = new QLineEdit(assignBox);
        clusterTagsEdit_ = new QLineEdit(assignBox);
        clusterTagsEdit_->setText("cluster-suggested");
        clusterMarkReviewedCheck_ = new QCheckBox("Mark reviewed", assignBox);
        clusterMarkReviewedCheck_->setChecked(true);
        auto* assignClusterButton = new QPushButton("Assign Cluster", assignBox);
        assignForm->addRow("Person", clusterPersonEdit_);
        assignForm->addRow("Tags", clusterTagsEdit_);
        assignForm->addRow("", clusterMarkReviewedCheck_);
        assignForm->addRow("", assignClusterButton);
        clusterSummaryLabel_ = new QLabel("No clusters built", actionPanel);
        clusterSummaryLabel_->setWordWrap(true);
        actionLayout->addWidget(clusterPreviewLabel_, 1);
        actionLayout->addWidget(assignBox);
        actionLayout->addWidget(clusterSummaryLabel_);
        actionLayout->addStretch(1);

        splitter->addWidget(clusterTable_);
        splitter->addWidget(clusterMemberTable_);
        splitter->addWidget(actionPanel);
        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 2);
        splitter->setStretchFactor(2, 1);
        layout->addWidget(splitter, 1);
        addMainTab(page, "Clusters");

        connect(clusterButton, &QPushButton::clicked, this, [this] { refreshClusters(); });
        connect(clusterTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedClusterMembers(); });
        connect(clusterMemberTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedClusterMemberPreview(); });
        connect(assignClusterButton, &QPushButton::clicked, this, [this] { assignSelectedCluster(); });
    }

    void buildDenseMeshTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QHBoxLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        meshFaceIdSpin_ = new QSpinBox(controls);
        meshFaceIdSpin_->setRange(1, 999999999);
        meshFaceIdSpin_->setPrefix("Face ");
        meshOverlayCheck_ = new QCheckBox("3D Landmarks", controls);
        meshOverlayCheck_->setChecked(true);
        auto* loadButton = new QPushButton("Load 3D Data", controls);
        auto* generateButton = new QPushButton("Generate Native Mesh", controls);
        meshStatusLabel_ = new QLabel("Select a face", controls);
        controlsLayout->addWidget(meshFaceIdSpin_);
        controlsLayout->addWidget(meshOverlayCheck_);
        controlsLayout->addWidget(loadButton);
        controlsLayout->addWidget(generateButton);
        controlsLayout->addWidget(meshStatusLabel_, 1);
        layout->addWidget(controls);

        meshView_ = new PointCloudWidget(page);
        layout->addWidget(meshView_, 1);
        addMainTab(page, "Dense Mesh");

        connect(loadButton, &QPushButton::clicked, this, [this] { loadDenseMeshFace(); });
        connect(generateButton, &QPushButton::clicked, this, [this] { generateNativeMeshForSelectedFace(); });
        connect(meshOverlayCheck_, &QCheckBox::toggled, this, [this] { loadDenseMeshFace(); });
    }

    void buildRuntimeTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* engineBox = new QGroupBox("Engine", page);
        auto* engineForm = new QFormLayout(engineBox);
        runtimeModeCombo_ = new QComboBox(engineBox);
        runtimeModeCombo_->addItem("Auto", "auto");
        runtimeModeCombo_->addItem("CPU", "cpu");
        runtimeModeCombo_->addItem("DirectML", "directml");
        runtimeModeCombo_->setCurrentIndex(0);
        runtimeBuildLabel_ = new QLabel(engineBox);
        runtimeProviderLabel_ = new QLabel(engineBox);
        runtimeNoteLabel_ = new QLabel(engineBox);
        runtimeNoteLabel_->setWordWrap(true);
        auto* refreshButton = new QPushButton("Refresh Runtime", engineBox);
        engineForm->addRow("Mode", runtimeModeCombo_);
        engineForm->addRow("Build", runtimeBuildLabel_);
        engineForm->addRow("Provider", runtimeProviderLabel_);
        engineForm->addRow("Status", runtimeNoteLabel_);
        engineForm->addRow("", refreshButton);
        layout->addWidget(engineBox);

        auto* databaseBox = new QGroupBox("Current Database", page);
        auto* databaseForm = new QFormLayout(databaseBox);
        runtimeDatabasePathLabel_ = new QLabel("--", databaseBox);
        runtimeDatabasePathLabel_->setWordWrap(true);
        runtimeDatabaseStatsLabel_ = new QLabel("--", databaseBox);
        runtimeDatabaseStatsLabel_->setWordWrap(true);
        auto* refreshDatabaseButton = new QPushButton("Refresh Database Stats", databaseBox);
        databaseForm->addRow("Path", runtimeDatabasePathLabel_);
        databaseForm->addRow("Stats", runtimeDatabaseStatsLabel_);
        databaseForm->addRow("", refreshDatabaseButton);
        layout->addWidget(databaseBox);

        auto* maintenanceBox = new QGroupBox("Maintenance", page);
        auto* maintenanceLayout = new QGridLayout(maintenanceBox);
        auto* integrityButton = new QPushButton("Check Integrity", maintenanceBox);
        auto* backupButton = new QPushButton("Backup DB", maintenanceBox);
        auto* checkpointButton = new QPushButton("Checkpoint WAL", maintenanceBox);
        auto* vacuumButton = new QPushButton("VACUUM", maintenanceBox);
        runtimeMaintenanceLog_ = new QTextEdit(maintenanceBox);
        runtimeMaintenanceLog_->setReadOnly(true);
        runtimeMaintenanceLog_->setMinimumHeight(120);
        runtimeMaintenanceLog_->setPlaceholderText("Runtime operation results");
        maintenanceLayout->addWidget(integrityButton, 0, 0);
        maintenanceLayout->addWidget(backupButton, 0, 1);
        maintenanceLayout->addWidget(checkpointButton, 0, 2);
        maintenanceLayout->addWidget(vacuumButton, 0, 3);
        maintenanceLayout->addWidget(runtimeMaintenanceLog_, 1, 0, 1, 4);
        layout->addWidget(maintenanceBox);
        layout->addStretch(1);
        addMainTab(page, "Runtime");

        connect(refreshButton, &QPushButton::clicked, this, [this] { refreshRuntimeInfo(); });
        connect(refreshDatabaseButton, &QPushButton::clicked, this, [this] { refreshRuntimeDatabaseInfo(); });
        connect(integrityButton, &QPushButton::clicked, this, [this] { runRuntimeIntegrityCheck(); });
        connect(backupButton, &QPushButton::clicked, this, [this] { runRuntimeBackup(); });
        connect(checkpointButton, &QPushButton::clicked, this, [this] { runRuntimeCheckpoint(); });
        connect(vacuumButton, &QPushButton::clicked, this, [this] { runRuntimeVacuum(); });
        connect(runtimeModeCombo_, &QComboBox::currentTextChanged, this, [this] {
            resetCameraEngine();
            refreshRuntimeInfo();
        });
        refreshRuntimeInfo();
        refreshRuntimeDatabaseInfo();
    }

    void chooseDatabase() {
        const auto path = QFileDialog::getOpenFileName(this, "Open FSC database", {}, "FSC Database (*.fscdb);;SQLite Database (*.sqlite *.db);;All Files (*)");
        if (!path.isEmpty()) {
            openDatabase(path);
        }
    }

    void createDatabase() {
        const auto path = QFileDialog::getSaveFileName(this, "Create FSC database", {}, "FSC Database (*.fscdb)");
        if (path.isEmpty()) {
            return;
        }
        try {
            fsc::core::Database::createEmpty(pathFrom(path), true);
            openDatabase(path);
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void openDatabase(const QString& path) {
        try {
            database_ = std::make_unique<fsc::core::Database>(pathFrom(path));
            databasePathEdit_->setText(path);
            reloadAll();
            statusBar()->showMessage("Opened " + path);
        } catch (const std::exception& ex) {
            database_.reset();
            cameraIdentityProfiles_.clear();
            refreshCameraDatabaseLabel();
            showError(ex);
        }
    }

    void reloadAll() {
        if (!database_) {
            return;
        }
        loadOverview();
        refreshLibraryFilterOptions();
        loadLibrary();
        loadPeople();
        loadReview();
        cameraIdentityProfiles_ = database_->loadIdentityProfiles();
        refreshSearchFilterOptions();
        refreshRuntimeDatabaseInfo();
        refreshCameraDatabaseLabel();
    }

    void loadOverview() {
        const auto stats = database_->statistics();
        formatLabel_->setText(qs(stats.formatVersion));
        modelLabel_->setText(qs(stats.modelName));
        metricLabel_->setText(qs(stats.metric));
        facesLabel_->setText(QString::number(stats.faceCount));
        peopleLabel_->setText(QString::number(stats.peopleCount));
        reviewLabel_->setText(QString::number(stats.reviewCount));
        qualityLabel_->setText(QString::number(stats.averageQuality, 'f', 4));
    }

    void loadLibrary() {
        if (!database_ || libraryTable_ == nullptr) {
            return;
        }
        const bool includeIgnored = libraryFilterIncludeIgnoredCheck_ == nullptr || libraryFilterIncludeIgnoredCheck_->isChecked();
        auto records = database_->loadFaces(includeIgnored);
        records.erase(
            std::remove_if(records.begin(), records.end(), [this](const auto& record) {
                return !recordMatchesLibraryFilters(record);
            }),
            records.end());
        libraryTable_->setRowCount(static_cast<int>(records.size()));
        for (int row = 0; row < static_cast<int>(records.size()); ++row) {
            const auto& record = records[static_cast<size_t>(row)];
            libraryTable_->setItem(row, 0, item(QString::number(record.id)));
            libraryTable_->setItem(row, 1, item(qs(record.fileName)));
            libraryTable_->setItem(row, 2, item(qs(record.personName)));
            libraryTable_->setItem(row, 3, item(qs(record.tagText)));
            libraryTable_->setItem(row, 4, item(qs(record.reviewState)));
            libraryTable_->setItem(row, 5, item(record.ignored ? "yes" : "no"));
            libraryTable_->setItem(row, 6, item(record.reviewState == "duplicate" ? "yes" : "no"));
            libraryTable_->setItem(row, 7, numberItem(record.qualityScore, 3));
            libraryTable_->setItem(row, 8, item(qs(record.sourcePath)));
        }
        libraryTable_->resizeColumnsToContents();
        appendLibraryActivity(QString("Loaded %1 face(s)").arg(records.size()));
    }

    bool recordMatchesLibraryFilters(const fsc::core::FaceRecord& record) const {
        if (libraryFilterMinQualitySpin_ != nullptr && record.qualityScore < libraryFilterMinQualitySpin_->value()) {
            return false;
        }
        const QString personFilter = libraryFilterPersonCombo_ == nullptr ? QString() : libraryFilterPersonCombo_->currentData().toString();
        if (!personFilter.isEmpty() && qs(record.personName).compare(personFilter, Qt::CaseInsensitive) != 0) {
            return false;
        }
        const QString tagFilter = libraryFilterTagCombo_ == nullptr ? QString() : libraryFilterTagCombo_->currentData().toString();
        if (!tagFilter.isEmpty()) {
            QString tagText = qs(record.tagText);
            tagText.replace(';', ',');
            tagText.replace('|', ',');
            bool found = false;
            for (const auto& tag : tagText.split(',', Qt::SkipEmptyParts)) {
                if (tag.trimmed().compare(tagFilter, Qt::CaseInsensitive) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        const QString reviewFilter = libraryFilterReviewCombo_ == nullptr ? QString() : libraryFilterReviewCombo_->currentText();
        if (!reviewFilter.isEmpty() && reviewFilter != "All" && qs(record.reviewState).compare(reviewFilter, Qt::CaseInsensitive) != 0) {
            return false;
        }
        const QString text = libraryFilterTextEdit_ == nullptr ? QString() : libraryFilterTextEdit_->text().trimmed();
        if (!text.isEmpty()) {
            const QString haystack = QString("%1\n%2\n%3\n%4\n%5")
                                         .arg(qs(record.fileName), qs(record.sourcePath), qs(record.personName), qs(record.tagText), qs(record.notes));
            if (!haystack.contains(text, Qt::CaseInsensitive)) {
                return false;
            }
        }
        return true;
    }

    void appendLibraryActivity(const QString& message) {
        if (libraryActivityLog_ != nullptr) {
            const QString stamp = QTime::currentTime().toString("HH:mm:ss");
            libraryActivityLog_->append(QString("[%1] %2").arg(stamp, message));
        }
        statusBar()->showMessage(message);
    }

    void refreshLibraryFilterOptions() {
        if (libraryFilterPersonCombo_ == nullptr || libraryFilterTagCombo_ == nullptr) {
            return;
        }
        const QString currentPerson = libraryFilterPersonCombo_->currentData().toString();
        const QString currentTag = libraryFilterTagCombo_->currentData().toString();
        libraryFilterPersonCombo_->blockSignals(true);
        libraryFilterTagCombo_->blockSignals(true);
        libraryFilterPersonCombo_->clear();
        libraryFilterTagCombo_->clear();
        libraryFilterPersonCombo_->addItem("All", "");
        libraryFilterTagCombo_->addItem("All", "");
        if (database_) {
            try {
                for (const auto& person : database_->loadPeople()) {
                    libraryFilterPersonCombo_->addItem(qs(person.name), qs(person.name));
                }
                for (const auto& tag : database_->loadTags()) {
                    libraryFilterTagCombo_->addItem(qs(tag), qs(tag));
                }
            } catch (...) {
            }
        }
        const int personIndex = libraryFilterPersonCombo_->findData(currentPerson);
        const int tagIndex = libraryFilterTagCombo_->findData(currentTag);
        libraryFilterPersonCombo_->setCurrentIndex(personIndex >= 0 ? personIndex : 0);
        libraryFilterTagCombo_->setCurrentIndex(tagIndex >= 0 ? tagIndex : 0);
        libraryFilterPersonCombo_->blockSignals(false);
        libraryFilterTagCombo_->blockSignals(false);
    }

    void resetLibraryFilters() {
        if (libraryFilterTextEdit_ != nullptr) {
            libraryFilterTextEdit_->clear();
        }
        if (libraryFilterPersonCombo_ != nullptr) {
            libraryFilterPersonCombo_->setCurrentIndex(0);
        }
        if (libraryFilterTagCombo_ != nullptr) {
            libraryFilterTagCombo_->setCurrentIndex(0);
        }
        if (libraryFilterReviewCombo_ != nullptr) {
            libraryFilterReviewCombo_->setCurrentIndex(0);
        }
        if (libraryFilterMinQualitySpin_ != nullptr) {
            libraryFilterMinQualitySpin_->setValue(0.0);
        }
        if (libraryFilterIncludeIgnoredCheck_ != nullptr) {
            libraryFilterIncludeIgnoredCheck_->setChecked(true);
        }
        loadLibrary();
    }

    void loadLibraryMetadata(int faceId) {
        if (!database_) {
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                return;
            }
            if (libraryPersonEdit_ != nullptr) {
                libraryPersonEdit_->setText(qs(face->personName));
            }
            if (libraryTagsEdit_ != nullptr) {
                libraryTagsEdit_->setText(qs(face->tagText));
            }
            if (libraryReviewCombo_ != nullptr) {
                libraryReviewCombo_->setCurrentText(qs(face->reviewState));
            }
            if (libraryIgnoredCheck_ != nullptr) {
                libraryIgnoredCheck_->setChecked(face->ignored);
            }
            if (libraryNotesEdit_ != nullptr) {
                libraryNotesEdit_->setPlainText(qs(face->notes));
            }
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    std::vector<int64_t> selectedLibraryFaceIds() const {
        std::vector<int64_t> ids;
        if (libraryTable_ == nullptr || libraryTable_->selectionModel() == nullptr) {
            return ids;
        }
        const auto rows = libraryTable_->selectionModel()->selectedRows();
        ids.reserve(static_cast<size_t>(rows.size()));
        for (const auto& index : rows) {
            const auto* idItem = libraryTable_->item(index.row(), 0);
            if (idItem != nullptr) {
                const auto id = idItem->text().toLongLong();
                if (id > 0) {
                    ids.push_back(id);
                }
            }
        }
        return ids;
    }

    void assignFaceToPersonName(int64_t faceId, const QString& name) {
        const auto cleanName = name.trimmed();
        if (cleanName.isEmpty()) {
            database_->assignFaceToPerson(faceId, 0);
            return;
        }
        const auto personId = database_->upsertPerson(cleanName.toUtf8().constData());
        database_->assignFaceToPerson(faceId, personId);
    }

    void saveLibrarySelectedMetadata() {
        if (!database_) {
            return;
        }
        try {
            const auto faceIds = selectedLibraryFaceIds();
            if (faceIds.empty()) {
                throw std::runtime_error("Select a face first.");
            }
            const auto faceId = faceIds.front();
            assignFaceToPersonName(faceId, libraryPersonEdit_->text());
            database_->setFaceTags(faceId, libraryTagsEdit_->text().toUtf8().constData(), false);
            database_->updateFaceReview(
                faceId,
                libraryReviewCombo_->currentText().toStdString(),
                libraryIgnoredCheck_->isChecked(),
                libraryNotesEdit_->toPlainText().toUtf8().constData());
            reloadAll();
            statusBar()->showMessage(QString("Updated face %1").arg(faceId));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void applyLibraryBatchMetadata() {
        if (!database_) {
            return;
        }
        try {
            const auto faceIds = selectedLibraryFaceIds();
            if (faceIds.empty()) {
                throw std::runtime_error("Select one or more faces first.");
            }
            const bool changePerson = !libraryBatchPersonEdit_->text().trimmed().isEmpty();
            const bool changeTags = !libraryBatchTagsEdit_->text().trimmed().isEmpty();
            const bool appendTags = libraryBatchAppendTagsCheck_->isChecked();
            const QString reviewState = libraryBatchReviewCombo_->currentText();
            const QString ignoredMode = libraryBatchIgnoredCombo_->currentText();
            const QString notes = libraryBatchNotesEdit_->text();
            for (const auto faceId : faceIds) {
                if (changePerson) {
                    assignFaceToPersonName(faceId, libraryBatchPersonEdit_->text());
                }
                if (changeTags) {
                    database_->setFaceTags(faceId, libraryBatchTagsEdit_->text().toUtf8().constData(), appendTags);
                }
                if (reviewState != "No change" || ignoredMode != "No change" || !notes.isEmpty()) {
                    const auto face = database_->loadFace(faceId);
                    if (!face.has_value()) {
                        continue;
                    }
                    bool ignored = face->ignored;
                    if (ignoredMode == "Ignore") {
                        ignored = true;
                    } else if (ignoredMode == "Restore") {
                        ignored = false;
                    }
                    database_->updateFaceReview(
                        faceId,
                        reviewState == "No change" ? face->reviewState : reviewState.toStdString(),
                        ignored,
                        notes.isEmpty() ? face->notes : std::string(notes.toUtf8().constData()));
                }
            }
            reloadAll();
            statusBar()->showMessage(QString("Batch updated %1 selected face(s)").arg(faceIds.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void setDatabaseFacePreview(QLabel* label, const fsc::core::FaceRecord& face, const QString& fallbackText) {
        if (label == nullptr) {
            return;
        }
        QImage image(qs(face.sourcePath));
        if (image.isNull()) {
            label->setText(fallbackText);
            label->setPixmap(QPixmap());
            return;
        }
        QPixmap pixmap = QPixmap::fromImage(image).scaled(
            label->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        if (!pixmap.isNull()) {
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing, true);
            const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, image.width()));
            const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, image.height()));
            if (face.bbox.size() >= 4) {
                const QRectF box(
                    QPointF(face.bbox[0] * sx, face.bbox[1] * sy),
                    QPointF(face.bbox[2] * sx, face.bbox[3] * sy));
                painter.setPen(QPen(QColor(0, 230, 70), 2.0));
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(box.normalized());
            }
            painter.setPen(QPen(QColor(20, 170, 220), 1.1));
            painter.setBrush(QColor(20, 210, 235));
            for (const auto& point : face.landmarks2d) {
                if (point.size() < 2) {
                    continue;
                }
                painter.drawEllipse(QPointF(point[0] * sx, point[1] * sy), 1.7, 1.7);
            }
        }
        label->setPixmap(pixmap);
    }

    void updateLibraryPreview(int faceId) {
        if (libraryPreviewLabel_ == nullptr || !database_) {
            return;
        }
        libraryPreviewFaceId_ = faceId;
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                libraryPreviewLabel_->setText("Face not found");
                libraryPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            QImage image(qs(face->sourcePath));
            if (image.isNull()) {
                libraryPreviewLabel_->setText("Image unavailable");
                libraryPreviewLabel_->setPixmap(QPixmap());
                return;
            }

            QRectF bbox;
            if (face->bbox.size() >= 4) {
                bbox = QRectF(
                    QPointF(face->bbox[0], face->bbox[1]),
                    QPointF(face->bbox[2], face->bbox[3])).normalized();
            }

            QPointF offset(0.0, 0.0);
            QImage view = image;
            if (libraryFocusOnFace_ && bbox.isValid() && bbox.width() > 1.0 && bbox.height() > 1.0) {
                const double pad = std::max(bbox.width(), bbox.height()) * 0.75;
                QRect crop(
                    static_cast<int>(std::floor(bbox.left() - pad)),
                    static_cast<int>(std::floor(bbox.top() - pad)),
                    static_cast<int>(std::ceil(bbox.width() + pad * 2.0)),
                    static_cast<int>(std::ceil(bbox.height() + pad * 2.0)));
                crop = crop.intersected(QRect(0, 0, image.width(), image.height()));
                if (crop.isValid() && crop.width() > 0 && crop.height() > 0) {
                    view = image.copy(crop);
                    offset = crop.topLeft();
                }
            }

            QPixmap pixmap = QPixmap::fromImage(view).scaled(
                libraryPreviewLabel_->size(),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation);
            if (!pixmap.isNull()) {
                QPainter painter(&pixmap);
                painter.setRenderHint(QPainter::Antialiasing, true);
                const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, view.width()));
                const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, view.height()));
                if (bbox.isValid() && bbox.width() > 1.0 && bbox.height() > 1.0) {
                    const QRectF drawBox(
                        (bbox.left() - offset.x()) * sx,
                        (bbox.top() - offset.y()) * sy,
                        bbox.width() * sx,
                        bbox.height() * sy);
                    painter.setPen(QPen(QColor(0, 230, 70), 2.0));
                    painter.setBrush(Qt::NoBrush);
                    painter.drawRect(drawBox);
                }
                painter.setPen(QPen(QColor(20, 170, 220), 1.2));
                painter.setBrush(QColor(20, 210, 235));
                for (const auto& point : face->landmarks2d) {
                    if (point.size() < 2) {
                        continue;
                    }
                    const QPointF drawPoint((point[0] - offset.x()) * sx, (point[1] - offset.y()) * sy);
                    if (drawPoint.x() >= 0.0 && drawPoint.y() >= 0.0 && drawPoint.x() <= pixmap.width() && drawPoint.y() <= pixmap.height()) {
                        painter.drawEllipse(drawPoint, 2.0, 2.0);
                    }
                }
            }
            libraryPreviewLabel_->setPixmap(pixmap);
            libraryFocusButton_->setText(libraryFocusOnFace_ ? "Full Image" : "Focus on Face");
        } catch (const std::exception& ex) {
            libraryPreviewLabel_->setText(ex.what());
            libraryPreviewLabel_->setPixmap(QPixmap());
        }
    }

    void updateLibraryVisuals(int faceId) {
        updateLibraryPreview(faceId);
        updateLibrary3dPreview(faceId);
    }

    void updateLibrary3dPreview(int faceId) {
        if (libraryLandmarksView_ == nullptr || libraryDenseMeshView_ == nullptr) {
            return;
        }
        if (!database_) {
            libraryLandmarksView_->setMessage("No database");
            libraryDenseMeshView_->setMessage("No database");
            if (libraryMeshStatusLabel_ != nullptr) {
                libraryMeshStatusLabel_->setText("Open a database first");
            }
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                libraryLandmarksView_->setMessage("Face not found");
                libraryDenseMeshView_->setMessage("Face not found");
                if (libraryMeshStatusLabel_ != nullptr) {
                    libraryMeshStatusLabel_->setText("Face not found");
                }
                return;
            }

            if (face->landmarks3d.empty()) {
                libraryLandmarksView_->setMessage("No cached 3D landmarks");
            } else {
                libraryLandmarksView_->setData(
                    face->landmarks3d,
                    {},
                    QString("Face %1 3D landmarks (%2)").arg(face->id).arg(face->landmarks3d.size()));
            }

            const bool hasDenseMesh = !face->faceMesh3d.empty();
            const bool hasLandmarks = !face->landmarks3d.empty();
            if (!hasDenseMesh && !hasLandmarks) {
                libraryDenseMeshView_->setMessage("No cached dense mesh or 3D landmarks");
                if (libraryMeshStatusLabel_ != nullptr) {
                    libraryMeshStatusLabel_->setText("No cached 3D data");
                }
                return;
            }

            std::vector<std::vector<double>> meshPoints = hasDenseMesh
                ? face->faceMesh3d
                : fsc::mesh::buildSyntheticFaceMesh3d(face->landmarks3d);
            std::vector<std::vector<double>> overlay;
            if (hasLandmarks && libraryMeshOverlayCheck_ != nullptr && libraryMeshOverlayCheck_->isChecked()) {
                overlay = face->landmarks3d;
            }
            const QString source = hasDenseMesh ? "cached dense mesh" : "native fallback mesh";
            if (libraryMeshStatusLabel_ != nullptr) {
                libraryMeshStatusLabel_->setText(QString("Face %1: %2 point(s) from %3")
                                                     .arg(face->id)
                                                     .arg(meshPoints.size())
                                                     .arg(source));
            }
            libraryDenseMeshView_->setData(
                std::move(meshPoints),
                std::move(overlay),
                QString("Face %1 dense mesh").arg(face->id));
        } catch (const std::exception& ex) {
            libraryLandmarksView_->setMessage(ex.what());
            libraryDenseMeshView_->setMessage(ex.what());
            if (libraryMeshStatusLabel_ != nullptr) {
                libraryMeshStatusLabel_->setText(ex.what());
            }
        }
    }

    void generateLibraryMeshForSelectedFace() {
        if (!database_ || libraryPreviewFaceId_ <= 0) {
            return;
        }
        try {
            const auto face = database_->loadFace(libraryPreviewFaceId_);
            if (!face.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            if (face->landmarks3d.empty()) {
                throw std::runtime_error("This face has no cached 3D landmarks to build a native mesh from.");
            }
            const auto mesh = fsc::mesh::buildSyntheticFaceMesh3d(face->landmarks3d);
            database_->updateFaceMesh3d(libraryPreviewFaceId_, mesh);
            updateLibrary3dPreview(libraryPreviewFaceId_);
            if (meshFaceIdSpin_ != nullptr && meshFaceIdSpin_->value() == libraryPreviewFaceId_) {
                loadDenseMeshFace();
            }
            statusBar()->showMessage(QString("Generated native mesh for face %1 (%2 points)").arg(libraryPreviewFaceId_).arg(mesh.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void loadPeople() {
        const auto people = database_->loadPeople();
        peopleTable_->setRowCount(static_cast<int>(people.size()));
        for (int row = 0; row < static_cast<int>(people.size()); ++row) {
            const auto& person = people[static_cast<size_t>(row)];
            peopleTable_->setItem(row, 0, item(QString::number(person.id)));
            peopleTable_->setItem(row, 1, item(qs(person.name)));
            peopleTable_->setItem(row, 2, item(QString::number(person.faceCount)));
            peopleTable_->setItem(row, 3, numberItem(person.averageQuality, 3));
            peopleTable_->setItem(row, 4, item(qs(person.identityStatus)));
            peopleTable_->setItem(row, 5, item(QString::number(person.identitySampleCount)));
            peopleTable_->setItem(row, 6, item(QString::number(person.identityExemplarCount)));
            peopleTable_->setItem(row, 7, item(qs(person.identityHealth)));
        }
        peopleTable_->resizeColumnsToContents();
    }

    void loadReview() {
        if (reviewTable_ == nullptr) {
            return;
        }
        const auto records = database_->loadFaces(true, 1000);
        reviewTable_->setRowCount(0);
        for (const auto& record : records) {
            if (!record.ignored && record.reviewState == "reviewed") {
                continue;
            }
            const int row = reviewTable_->rowCount();
            reviewTable_->insertRow(row);
            reviewTable_->setItem(row, 0, item(QString::number(record.id)));
            reviewTable_->setItem(row, 1, item(qs(record.fileName)));
            reviewTable_->setItem(row, 2, item(qs(record.personName)));
            reviewTable_->setItem(row, 3, numberItem(record.qualityScore, 3));
            reviewTable_->setItem(row, 4, numberItem(record.detectionScore, 3));
            reviewTable_->setItem(row, 5, item(qs(record.reviewState)));
            reviewTable_->setItem(row, 6, item(record.ignored ? "yes" : "no"));
            reviewTable_->setItem(row, 7, item(qs(record.notes)));
        }
        reviewTable_->resizeColumnsToContents();
    }

    int selectedReviewFaceId() const {
        const auto selected = reviewTable_->selectedItems();
        if (selected.empty()) {
            return 0;
        }
        const int row = selected.front()->row();
        const auto* idItem = reviewTable_->item(row, 0);
        return idItem == nullptr ? 0 : idItem->text().toInt();
    }

    void updateReviewDetail(int faceId) {
        reviewSuggestedPersonId_ = 0;
        if (reviewSuggestionLabel_ != nullptr) {
            reviewSuggestionLabel_->setText("AI Suggested Person: -");
        }
        if (reviewPreviewLabel_ == nullptr || !database_) {
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                reviewPreviewLabel_->setText("Face not found");
                reviewPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            QImage image(qs(face->sourcePath));
            if (image.isNull()) {
                reviewPreviewLabel_->setText("Image unavailable");
                reviewPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            QPixmap pixmap = QPixmap::fromImage(image).scaled(reviewPreviewLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            if (!pixmap.isNull()) {
                QPainter painter(&pixmap);
                painter.setRenderHint(QPainter::Antialiasing, true);
                const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, image.width()));
                const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, image.height()));
                if (face->bbox.size() >= 4) {
                    QRectF box(QPointF(face->bbox[0], face->bbox[1]), QPointF(face->bbox[2], face->bbox[3]));
                    box = box.normalized();
                    painter.setPen(QPen(QColor(0, 230, 70), 2.0));
                    painter.setBrush(Qt::NoBrush);
                    painter.drawRect(QRectF(box.left() * sx, box.top() * sy, box.width() * sx, box.height() * sy));
                }
                painter.setPen(QPen(QColor(20, 170, 220), 1.1));
                painter.setBrush(QColor(20, 210, 235));
                for (const auto& point : face->landmarks2d) {
                    if (point.size() < 2) {
                        continue;
                    }
                    painter.drawEllipse(QPointF(point[0] * sx, point[1] * sy), 1.8, 1.8);
                }
            }
            reviewPreviewLabel_->setPixmap(pixmap);
        } catch (const std::exception& ex) {
            reviewPreviewLabel_->setText(ex.what());
            reviewPreviewLabel_->setPixmap(QPixmap());
        }
    }

    fsc::vision::RuntimeMode selectedRuntimeMode() const {
        if (runtimeModeCombo_ == nullptr) {
            return fsc::vision::RuntimeMode::Auto;
        }
        return fsc::vision::parseRuntimeMode(runtimeModeCombo_->currentData().toString().toStdString());
    }

    void refreshRuntimeInfo() {
        if (runtimeBuildLabel_ == nullptr || runtimeProviderLabel_ == nullptr || runtimeNoteLabel_ == nullptr) {
            return;
        }
#ifdef FSC_ENABLE_ONNX
        runtimeBuildLabel_->setText("ONNX Runtime enabled");
#else
        runtimeBuildLabel_->setText("ONNX Runtime disabled");
#endif
        const auto mode = selectedRuntimeMode();
        runtimeProviderLabel_->setText(qs(fsc::vision::toString(mode)));
#ifdef FSC_ONNXRUNTIME_HAS_DML
        runtimeNoteLabel_->setText(
            "Import, Compare, and Camera use this setting when creating native InsightFace sessions. "
            "Auto tries DirectML first and falls back to CPU if unavailable.");
#else
        runtimeNoteLabel_->setText(
            "Import, Compare, and Camera use this setting when creating native InsightFace sessions. "
            "This build does not include the DirectML-enabled ONNX Runtime package, so use CPU or rebuild the DirectML flavor.");
#endif
    }

    void refreshRuntimeDatabaseInfo() {
        if (runtimeDatabasePathLabel_ == nullptr || runtimeDatabaseStatsLabel_ == nullptr) {
            return;
        }
        if (!database_) {
            runtimeDatabasePathLabel_->setText("--");
            runtimeDatabaseStatsLabel_->setText("No database loaded");
            return;
        }
        const auto stats = database_->statistics();
        runtimeDatabasePathLabel_->setText(qs(database_->path().string()));
        runtimeDatabaseStatsLabel_->setText(
            QString("v%1 | faces %2 | people %3 | tags %4 | review %5 | ignored %6 | avg quality %7")
                .arg(qs(stats.formatVersion))
                .arg(stats.faceCount)
                .arg(stats.peopleCount)
                .arg(stats.tagCount)
                .arg(stats.reviewCount)
                .arg(stats.ignoredCount)
                .arg(stats.averageQuality, 0, 'f', 3));
    }

    void appendMaintenanceResult(const fsc::core::MaintenanceResult& result) {
        const QString state = result.ok ? "OK" : "FAILED";
        QString line = QString("%1 %2: %3")
                           .arg(state)
                           .arg(qs(result.action))
                           .arg(qs(result.message));
        if (!result.outputPath.empty()) {
            line += "\n" + qs(result.outputPath);
        }
        if (runtimeMaintenanceLog_ != nullptr) {
            runtimeMaintenanceLog_->append(line);
        }
        refreshRuntimeDatabaseInfo();
        statusBar()->showMessage(line.replace("\n", " "));
    }

    bool requireRuntimeDatabase() {
        if (database_) {
            return true;
        }
        showError(std::runtime_error("Open or create a database first."));
        return false;
    }

    void runRuntimeIntegrityCheck() {
        if (!requireRuntimeDatabase()) {
            return;
        }
        try {
            appendMaintenanceResult(database_->checkIntegrity());
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void runRuntimeBackup() {
        if (!requireRuntimeDatabase()) {
            return;
        }
        try {
            const auto source = database_->path();
            const auto suffix = source.extension().empty() ? ".fscdb" : source.extension().string();
            const auto defaultOutput = source.parent_path() / (source.stem().string() + "_backup" + suffix);
            const auto path = QFileDialog::getSaveFileName(
                this,
                "Save database backup",
                qs(defaultOutput.string()),
                "FSC Database (*.fscdb);;SQLite Database (*.sqlite *.db);;All Files (*)");
            if (path.isEmpty()) {
                return;
            }
            appendMaintenanceResult(database_->backupTo(pathFrom(path)));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void runRuntimeCheckpoint() {
        if (!requireRuntimeDatabase()) {
            return;
        }
        try {
            appendMaintenanceResult(database_->checkpointWal(true));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void runRuntimeVacuum() {
        if (!requireRuntimeDatabase()) {
            return;
        }
        if (QMessageBox::question(
                this,
                "FSC Studio Native",
                "VACUUM rewrites the database file and may take time on large libraries. Continue?") != QMessageBox::Yes) {
            return;
        }
        try {
            appendMaintenanceResult(database_->vacuum());
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void loadDenseMeshFace() {
        if (meshView_ == nullptr) {
            return;
        }
        if (!database_) {
            meshStatusLabel_->setText("Open a database first");
            meshView_->setMessage("No database");
            return;
        }
        try {
            const auto face = database_->loadFace(meshFaceIdSpin_->value());
            if (!face.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            const bool hasDenseMesh = !face->faceMesh3d.empty();
            const bool hasLandmarks = !face->landmarks3d.empty();
            if (!hasDenseMesh && !hasLandmarks) {
                meshStatusLabel_->setText("No cached 3D data");
                meshView_->setMessage("No cached dense mesh or 3D landmarks");
                return;
            }
            std::vector<std::vector<double>> points = hasDenseMesh ? face->faceMesh3d : face->landmarks3d;
            std::vector<std::vector<double>> overlay;
            if (hasDenseMesh && hasLandmarks && meshOverlayCheck_->isChecked()) {
                overlay = face->landmarks3d;
            }
            const QString source = hasDenseMesh ? "cached dense mesh" : "3D landmarks";
            meshStatusLabel_->setText(QString("Face %1: %2 point(s) from %3")
                                          .arg(face->id)
                                          .arg(points.size())
                                          .arg(source));
            meshView_->setData(
                std::move(points),
                std::move(overlay),
                QString("Face %1").arg(face->id));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void generateNativeMeshForSelectedFace() {
        if (!database_) {
            return;
        }
        try {
            const int faceId = meshFaceIdSpin_->value();
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            if (face->landmarks3d.empty()) {
                throw std::runtime_error("This face has no cached 3D landmarks to build a native mesh from.");
            }
            const auto mesh = fsc::mesh::buildSyntheticFaceMesh3d(face->landmarks3d);
            database_->updateFaceMesh3d(faceId, mesh);
            loadDenseMeshFace();
            statusBar()->showMessage(QString("Generated native mesh for face %1 (%2 points)").arg(faceId).arg(mesh.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void applyReviewFromControls() {
        applyReviewState(
            reviewStateCombo_->currentText().toStdString(),
            reviewIgnoredCombo_->currentIndex() == 1,
            reviewNotesEdit_->text().toStdString());
    }

    void applyReviewState(const std::string& state, bool ignored, const std::string& notes = {}) {
        if (!database_) {
            return;
        }
        try {
            const int faceId = selectedReviewFaceId();
            if (faceId <= 0) {
                throw std::runtime_error("Select a review row first.");
            }
            database_->updateFaceReview(faceId, state, ignored, notes.empty() ? reviewNotesEdit_->text().toStdString() : notes);
            reloadAll();
            statusBar()->showMessage("Review updated");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void suggestReviewPerson() {
        if (!database_) {
            return;
        }
        try {
            const int faceId = selectedReviewFaceId();
            if (faceId <= 0) {
                throw std::runtime_error("Select a review row first.");
            }
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            const auto result = fsc::core::identifyPerson(database_->loadIdentityProfiles(), face->embedding, selectedIdentityMode(), 5);
            reviewSuggestedPersonId_ = 0;
            if (result.candidates.empty()) {
                reviewSuggestionLabel_->setText("AI Suggested Person: unknown");
                return;
            }
            const auto& candidate = result.candidates.front();
            reviewSuggestedPersonId_ = candidate.profile.personId;
            reviewSuggestionLabel_->setText(
                QString("AI Suggested Person: %1 | %2 | score %3 | confidence %4% | evidence face %5")
                    .arg(qs(candidate.profile.personName))
                    .arg(qs(result.decision))
                    .arg(candidate.score, 0, 'f', 4)
                    .arg(candidate.confidence * 100.0, 0, 'f', 1)
                    .arg(candidate.evidenceFaceId));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void confirmReviewSuggestion() {
        if (!database_) {
            return;
        }
        try {
            const int faceId = selectedReviewFaceId();
            if (faceId <= 0) {
                throw std::runtime_error("Select a review row first.");
            }
            if (reviewSuggestedPersonId_ <= 0) {
                suggestReviewPerson();
            }
            if (reviewSuggestedPersonId_ <= 0) {
                throw std::runtime_error("No suggested person is available to confirm.");
            }
            database_->assignFaceToPerson(faceId, reviewSuggestedPersonId_);
            database_->updateFaceReview(faceId, "reviewed", false, "Confirmed native AI suggested person.");
            database_->rebuildIdentityProfiles();
            reloadAll();
            statusBar()->showMessage("Suggested person confirmed and profiles retrained");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void addPerson() {
        if (!database_) {
            return;
        }
        try {
            const auto name = personNameEdit_->text().trimmed();
            if (name.isEmpty()) {
                throw std::runtime_error("Person name is empty.");
            }
            const auto personId = database_->upsertPerson(name.toUtf8().constData());
            assignPersonSpin_->setValue(static_cast<int>(personId));
            personNameEdit_->clear();
            reloadAll();
            statusBar()->showMessage("Person saved");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void assignFace() {
        if (!database_) {
            return;
        }
        try {
            database_->assignFaceToPerson(assignFaceSpin_->value(), assignPersonSpin_->value());
            reloadAll();
            statusBar()->showMessage("Face assigned");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void trainProfiles() {
        if (!database_) {
            return;
        }
        try {
            const auto summary = database_->rebuildIdentityProfiles();
            reloadAll();
            statusBar()->showMessage(QString("Trained %1 profile(s), %2 weak").arg(summary.profilesBuilt).arg(summary.weakProfiles));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void refreshSearchFilterOptions() {
        if (searchPersonFilterCombo_ == nullptr || searchTagFilterCombo_ == nullptr) {
            return;
        }
        const auto currentPerson = searchPersonFilterCombo_->currentData().toString();
        const auto currentTag = searchTagFilterCombo_->currentData().toString();
        searchPersonFilterCombo_->blockSignals(true);
        searchTagFilterCombo_->blockSignals(true);
        searchPersonFilterCombo_->clear();
        searchTagFilterCombo_->clear();
        searchPersonFilterCombo_->addItem("All", "");
        searchTagFilterCombo_->addItem("All", "");
        if (database_) {
            try {
                for (const auto& person : database_->loadPeople()) {
                    searchPersonFilterCombo_->addItem(qs(person.name), qs(person.name));
                }
                for (const auto& tag : database_->loadTags()) {
                    searchTagFilterCombo_->addItem(qs(tag), qs(tag));
                }
            } catch (...) {
            }
        }
        const int personIndex = searchPersonFilterCombo_->findData(currentPerson);
        const int tagIndex = searchTagFilterCombo_->findData(currentTag);
        searchPersonFilterCombo_->setCurrentIndex(personIndex >= 0 ? personIndex : 0);
        searchTagFilterCombo_->setCurrentIndex(tagIndex >= 0 ? tagIndex : 0);
        searchPersonFilterCombo_->blockSignals(false);
        searchTagFilterCombo_->blockSignals(false);
    }

    void analyzeSearchImage() {
#ifdef FSC_ENABLE_ONNX
        try {
            if (searchImageEdit_ == nullptr || searchImageEdit_->text().trimmed().isEmpty()) {
                throw std::runtime_error("Select a query image first.");
            }
            const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
            fsc::vision::InsightFaceEngine engine(fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot), selectedRuntimeMode());
            const auto imagePath = pathFrom(searchImageEdit_->text());
            const auto image = fsc::vision::loadImageRgb(imagePath);
            searchQueryFaces_ = engine.analyze(image, 0.50f, 10);
            if (searchQueryFaces_.empty()) {
                throw std::runtime_error("No face detected in the query image.");
            }
            searchHits_.clear();
            lastSearchIdentityResult_ = {};
            currentSearchDatabaseFaceId_ = 0;
            if (searchTable_ != nullptr) {
                searchTable_->setRowCount(0);
            }
            if (searchIdentityTable_ != nullptr) {
                searchIdentityTable_->setRowCount(0);
            }
            if (searchResultPreviewLabel_ != nullptr) {
                searchResultPreviewLabel_->setText("Result preview");
                searchResultPreviewLabel_->setPixmap(QPixmap());
            }
            if (identityLabel_ != nullptr) {
                identityLabel_->setText("Identity: not searched");
            }
            searchFaceCombo_->blockSignals(true);
            searchFaceCombo_->clear();
            for (int index = 0; index < static_cast<int>(searchQueryFaces_.size()); ++index) {
                const auto& face = searchQueryFaces_[static_cast<size_t>(index)];
                searchFaceCombo_->addItem(
                    QString("Face %1 | det %2 | q %3")
                        .arg(index + 1)
                        .arg(face.detection.score, 0, 'f', 3)
                        .arg(face.qualityScore, 0, 'f', 3));
            }
            searchFaceCombo_->setCurrentIndex(0);
            searchFaceCombo_->blockSignals(false);
            searchQueryFaceIndex_ = 0;
            updateSearchQueryPreview();
            statusBar()->showMessage(QString("Analyzed query image: %1 face(s)").arg(searchQueryFaces_.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void updateSearchQueryPreview() {
        if (searchPreviewLabel_ == nullptr) {
            return;
        }
        QImage image(searchImageEdit_ == nullptr ? QString() : searchImageEdit_->text());
        if (image.isNull()) {
            searchPreviewLabel_->setText("Query image unavailable");
            searchPreviewLabel_->setPixmap(QPixmap());
            return;
        }
        QPixmap pixmap = QPixmap::fromImage(image).scaled(
            searchPreviewLabel_->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        if (!pixmap.isNull()) {
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing, true);
            const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, image.width()));
            const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, image.height()));
            for (int index = 0; index < static_cast<int>(searchQueryFaces_.size()); ++index) {
                const auto& box = searchQueryFaces_[static_cast<size_t>(index)].detection.box;
                const QRectF rect(
                    box.x1 * sx,
                    box.y1 * sy,
                    (box.x2 - box.x1) * sx,
                    (box.y2 - box.y1) * sy);
                painter.setPen(QPen(index == searchQueryFaceIndex_ ? QColor(0, 230, 70) : QColor(20, 180, 235), 2.0));
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(rect);
            }
        }
        searchPreviewLabel_->setPixmap(pixmap);
    }

    std::vector<float> currentSearchEmbedding(int64_t* databaseFaceId = nullptr) const {
        if (databaseFaceId != nullptr) {
            *databaseFaceId = 0;
        }
        if (!searchQueryFaces_.empty() && searchQueryFaceIndex_ >= 0 && searchQueryFaceIndex_ < static_cast<int>(searchQueryFaces_.size())) {
            return searchQueryFaces_[static_cast<size_t>(searchQueryFaceIndex_)].embedding;
        }
        const auto query = database_->loadFace(faceIdSpin_->value());
        if (!query.has_value()) {
            throw std::runtime_error("Face id not found.");
        }
        if (databaseFaceId != nullptr) {
            *databaseFaceId = query->id;
        }
        return query->embedding;
    }

    void runSearch() {
        if (!database_) {
            return;
        }
        try {
            if (searchPreviewTimer_ != nullptr) {
                searchPreviewTimer_->stop();
            }
            int64_t databaseFaceId = 0;
            const auto embedding = currentSearchEmbedding(&databaseFaceId);
            currentSearchDatabaseFaceId_ = databaseFaceId;
            const bool includeIgnored = searchIncludeIgnoredCheck_ != nullptr && searchIncludeIgnoredCheck_->isChecked();
            const double minQuality = searchMinQualitySpin_ != nullptr ? searchMinQualitySpin_->value() : 0.0;
            const QString personFilter = searchPersonFilterCombo_ != nullptr ? searchPersonFilterCombo_->currentData().toString() : QString();
            const QString tagFilter = searchTagFilterCombo_ != nullptr ? searchTagFilterCombo_->currentData().toString() : QString();
            auto records = database_->loadFaces(includeIgnored);
            records.erase(
                std::remove_if(records.begin(), records.end(), [minQuality, personFilter, tagFilter](const auto& record) {
                    if (record.qualityScore < minQuality) {
                        return true;
                    }
                    if (!personFilter.isEmpty() && qs(record.personName) != personFilter) {
                        return true;
                    }
                    if (!tagFilter.isEmpty()) {
                        auto tagText = qs(record.tagText);
                        tagText.replace(';', ',');
                        tagText.replace('|', ',');
                        const auto tags = tagText.split(',', Qt::SkipEmptyParts);
                        bool found = false;
                        for (const auto& tag : tags) {
                            if (tag.trimmed().compare(tagFilter, Qt::CaseInsensitive) == 0) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            return true;
                        }
                    }
                    return false;
                }),
                records.end());
            searchHits_ = fsc::core::searchFaces(
                records,
                embedding,
                topKSpin_->value(),
                searchThresholdSpin_ != nullptr ? searchThresholdSpin_->value() : -1.0,
                includeIgnored);
            searchTable_->setRowCount(0);
            for (auto it = searchHits_.begin(); it != searchHits_.end();) {
                if (databaseFaceId > 0 && it->record.id == databaseFaceId) {
                    it = searchHits_.erase(it);
                } else {
                    ++it;
                }
            }
            for (int index = 0; index < static_cast<int>(searchHits_.size()); ++index) {
                const auto& hit = searchHits_[static_cast<size_t>(index)];
                if (databaseFaceId > 0 && hit.record.id == databaseFaceId) {
                    continue;
                }
                const int row = searchTable_->rowCount();
                searchTable_->insertRow(row);
                searchTable_->setItem(row, 0, item(QString::number(index + 1)));
                searchTable_->setItem(row, 1, item(QString::number(hit.record.id)));
                searchTable_->setItem(row, 2, item(qs(hit.record.fileName)));
                searchTable_->setItem(row, 3, item(qs(hit.record.personName)));
                searchTable_->setItem(row, 4, item(qs(hit.record.tagText)));
                searchTable_->setItem(row, 5, numberItem(hit.cosine, 4));
                searchTable_->setItem(row, 6, numberItem(hit.similarityPercent(), 2));
                searchTable_->setItem(row, 7, numberItem(hit.record.qualityScore, 3));
            }
            populateSearchIdentityResult(fsc::core::identifyPerson(database_->loadIdentityProfiles(), embedding, selectedIdentityMode(), 5));
            searchTable_->resizeColumnsToContents();
            if (!searchHits_.empty()) {
                searchTable_->selectRow(0);
                startSearchPreviewAnimation();
            } else if (searchResultPreviewLabel_ != nullptr) {
                searchResultPreviewLabel_->setText("No results");
                searchResultPreviewLabel_->setPixmap(QPixmap());
            }
            statusBar()->showMessage(QString("Search complete: %1 result(s)").arg(searchHits_.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void runIdentify() {
        if (!database_) {
            return;
        }
        try {
            int64_t databaseFaceId = 0;
            const auto embedding = currentSearchEmbedding(&databaseFaceId);
            currentSearchDatabaseFaceId_ = databaseFaceId;
            const auto result = fsc::core::identifyPerson(database_->loadIdentityProfiles(), embedding, selectedIdentityMode(), 5);
            populateSearchIdentityResult(result);
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void populateSearchIdentityResult(fsc::core::IdentityResult result) {
        lastSearchIdentityResult_ = std::move(result);
        QString text = "Identity: " + qs(lastSearchIdentityResult_.decision);
        if (!lastSearchIdentityResult_.candidates.empty()) {
            const auto& candidate = lastSearchIdentityResult_.candidates.front();
            text += " | " + qs(candidate.profile.personName);
            text += " | score " + QString::number(candidate.score, 'f', 4);
            text += " | confidence " + QString::number(candidate.confidence * 100.0, 'f', 1) + "%";
        }
        if (identityLabel_ != nullptr) {
            identityLabel_->setText(text);
        }
        if (searchIdentityTable_ == nullptr) {
            return;
        }
        searchIdentityTable_->setRowCount(static_cast<int>(lastSearchIdentityResult_.candidates.size()));
        for (int row = 0; row < static_cast<int>(lastSearchIdentityResult_.candidates.size()); ++row) {
            const auto& candidate = lastSearchIdentityResult_.candidates[static_cast<size_t>(row)];
            searchIdentityTable_->setItem(row, 0, item(QString::number(row + 1)));
            searchIdentityTable_->setItem(row, 1, item(qs(candidate.profile.personName)));
            searchIdentityTable_->setItem(row, 2, item(row == 0 ? qs(lastSearchIdentityResult_.decision) : "candidate"));
            searchIdentityTable_->setItem(row, 3, numberItem(candidate.score, 4));
            searchIdentityTable_->setItem(row, 4, numberItem(candidate.confidence * 100.0, 1));
            searchIdentityTable_->setItem(row, 5, candidate.evidenceFaceId > 0 ? item(QString::number(candidate.evidenceFaceId)) : item("--"));
        }
        searchIdentityTable_->resizeColumnsToContents();
        if (!lastSearchIdentityResult_.candidates.empty()) {
            searchIdentityTable_->selectRow(0);
        }
    }

    void updateSearchResultPreviewForFace(int64_t faceId) {
        if (searchResultPreviewLabel_ == nullptr || !database_ || faceId <= 0) {
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                searchResultPreviewLabel_->setText("Face not found");
                searchResultPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            setDatabaseFacePreview(searchResultPreviewLabel_, *face, "Result preview");
        } catch (const std::exception& ex) {
            searchResultPreviewLabel_->setText(ex.what());
            searchResultPreviewLabel_->setPixmap(QPixmap());
        }
    }

    void updateSelectedSearchResultPreview() {
        if (searchTable_ == nullptr || searchTable_->selectionModel() == nullptr) {
            return;
        }
        if (searchPreviewTimer_ != nullptr && searchPreviewTimer_->isActive()) {
            searchPreviewTimer_->stop();
        }
        const auto selected = searchTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        if (row >= 0 && row < static_cast<int>(searchHits_.size())) {
            updateSearchResultPreviewForFace(searchHits_[static_cast<size_t>(row)].record.id);
        }
    }

    void startSearchPreviewAnimation() {
        if (searchPreviewTimer_ == nullptr || searchHits_.empty()) {
            if (!searchHits_.empty()) {
                updateSearchResultPreviewForFace(searchHits_.front().record.id);
            }
            return;
        }
        searchPreviewAnimationIndex_ = 0;
        const int previewCount = std::min<int>(static_cast<int>(searchHits_.size()), 24);
        if (previewCount <= 1) {
            updateSearchResultPreviewForFace(searchHits_.front().record.id);
            return;
        }
        updateSearchResultPreviewForFace(searchHits_.front().record.id);
        searchPreviewTimer_->start();
    }

    void advanceSearchPreviewAnimation() {
        if (searchPreviewTimer_ == nullptr || searchHits_.empty()) {
            return;
        }
        const int previewCount = std::min<int>(static_cast<int>(searchHits_.size()), 24);
        ++searchPreviewAnimationIndex_;
        if (searchPreviewAnimationIndex_ >= previewCount) {
            searchPreviewTimer_->stop();
            updateSearchResultPreviewForFace(searchHits_.front().record.id);
            if (searchTable_ != nullptr && searchTable_->rowCount() > 0) {
                searchTable_->selectRow(0);
            }
            statusBar()->showMessage(QString("Search preview stopped at best match: face %1").arg(searchHits_.front().record.id));
            return;
        }
        const auto& hit = searchHits_[static_cast<size_t>(searchPreviewAnimationIndex_)];
        updateSearchResultPreviewForFace(hit.record.id);
        statusBar()->showMessage(QString("Previewing result %1/%2: face %3")
                                     .arg(searchPreviewAnimationIndex_ + 1)
                                     .arg(previewCount)
                                     .arg(hit.record.id));
    }

    void updateSelectedSearchIdentityPreview() {
        if (searchIdentityTable_ == nullptr || searchIdentityTable_->selectionModel() == nullptr) {
            return;
        }
        const auto selected = searchIdentityTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        if (row < 0 || row >= static_cast<int>(lastSearchIdentityResult_.candidates.size())) {
            return;
        }
        const auto faceId = lastSearchIdentityResult_.candidates[static_cast<size_t>(row)].evidenceFaceId;
        if (faceId > 0) {
            updateSearchResultPreviewForFace(faceId);
        }
    }

    void assignSelectedSearchResult() {
        if (!database_ || searchTable_ == nullptr || searchAssignPersonEdit_ == nullptr) {
            return;
        }
        try {
            const auto selected = searchTable_->selectionModel() == nullptr ? QModelIndexList{} : searchTable_->selectionModel()->selectedRows();
            if (selected.empty()) {
                throw std::runtime_error("Select a search result first.");
            }
            const int row = selected.front().row();
            if (row < 0 || row >= static_cast<int>(searchHits_.size())) {
                throw std::runtime_error("Selected search row is out of range.");
            }
            const auto personName = searchAssignPersonEdit_->text().trimmed();
            if (personName.isEmpty()) {
                throw std::runtime_error("Enter a person name for the selected result.");
            }
            const auto faceId = searchHits_[static_cast<size_t>(row)].record.id;
            assignFaceToPersonName(faceId, personName);
            database_->rebuildIdentityProfiles();
            reloadAll();
            statusBar()->showMessage(QString("Assigned search result face %1 to %2").arg(faceId).arg(personName));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void confirmSearchIdentity() {
        if (!database_) {
            return;
        }
        try {
            if (currentSearchDatabaseFaceId_ <= 0) {
                throw std::runtime_error("Identity confirmation is available only when searching from a stored face id.");
            }
            if (lastSearchIdentityResult_.candidates.empty()) {
                throw std::runtime_error("Run Identify or Search before confirming identity.");
            }
            const auto& candidate = lastSearchIdentityResult_.candidates.front();
            if (candidate.profile.personId <= 0 || lastSearchIdentityResult_.decision == "unknown") {
                throw std::runtime_error("No confirmable identity suggestion is available.");
            }
            database_->assignFaceToPerson(currentSearchDatabaseFaceId_, candidate.profile.personId);
            database_->updateFaceReview(currentSearchDatabaseFaceId_, "reviewed", false, "Confirmed native Search identity suggestion.");
            database_->rebuildIdentityProfiles();
            reloadAll();
            statusBar()->showMessage(QString("Confirmed face %1 as %2").arg(currentSearchDatabaseFaceId_).arg(qs(candidate.profile.personName)));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    QString smoothedCameraName(int faceIndex, const QString& name) {
        auto& history = cameraVotesByFace_[faceIndex];
        history.push_back(name);
        while (history.size() > 6) {
            history.pop_front();
        }
        std::map<QString, std::pair<int, int>> counts;
        for (int index = 0; index < static_cast<int>(history.size()); ++index) {
            const auto& vote = history[static_cast<size_t>(index)];
            if (vote.isEmpty()) {
                continue;
            }
            auto& entry = counts[vote];
            ++entry.first;
            entry.second = index;
        }
        if (counts.empty()) {
            return name;
        }
        const auto best = std::max_element(counts.begin(), counts.end(), [](const auto& left, const auto& right) {
            if (left.second.first != right.second.first) {
                return left.second.first < right.second.first;
            }
            return left.second.second < right.second.second;
        });
        return best->second.first >= 2 ? best->first : name;
    }

    void refreshCameraDatabaseLabel() {
        if (cameraDatabaseLabel_ == nullptr) {
            return;
        }
        if (!database_) {
            cameraDatabaseLabel_->setText("Database: not opened");
            return;
        }
        const auto stats = database_->statistics();
        cameraDatabaseLabel_->setText(QString("Database: %1 | %2 faces | %3 identity profiles")
                                          .arg(qs(database_->path().string()))
                                          .arg(stats.faceCount)
                                          .arg(cameraIdentityProfiles_.size()));
    }

    void setCameraMatchPlaceholder(const QString& status) {
        if (cameraMatchStatusLabel_ != nullptr) {
            cameraMatchStatusLabel_->setText(status);
        }
        if (cameraMatchPreviewLabel_ != nullptr) {
            cameraMatchPreviewLabel_->setPixmap(QPixmap());
            cameraMatchPreviewLabel_->setText("Best match preview");
        }
    }

    void updateCameraMatchPreview(int64_t faceId, const QString& status) {
        if (cameraMatchStatusLabel_ != nullptr) {
            cameraMatchStatusLabel_->setText(status);
        }
        if (cameraMatchPreviewLabel_ == nullptr || !database_ || faceId <= 0) {
            setCameraMatchPlaceholder(status);
            return;
        }
        const auto face = database_->loadFace(faceId);
        if (!face.has_value()) {
            setCameraMatchPlaceholder(status);
            return;
        }
        QImage image(qs(face->sourcePath));
        if (image.isNull()) {
            setCameraMatchPlaceholder(status + " | preview unavailable");
            return;
        }
        QPixmap pixmap = QPixmap::fromImage(image).scaled(
            cameraMatchPreviewLabel_->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        if (!pixmap.isNull()) {
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing, true);
            const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, image.width()));
            const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, image.height()));
            if (face->bbox.size() >= 4) {
                const QRectF box(
                    QPointF(face->bbox[0] * sx, face->bbox[1] * sy),
                    QPointF(face->bbox[2] * sx, face->bbox[3] * sy));
                painter.setPen(QPen(QColor(0, 230, 70), 2.0));
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(box.normalized());
            }
            painter.setPen(QPen(QColor(20, 170, 220), 1.1));
            painter.setBrush(QColor(20, 210, 235));
            for (const auto& point : face->landmarks2d) {
                if (point.size() < 2) {
                    continue;
                }
                painter.drawEllipse(QPointF(point[0] * sx, point[1] * sy), 1.7, 1.7);
            }
        }
        cameraMatchPreviewLabel_->setPixmap(pixmap);
    }

#ifdef FSC_ENABLE_OPENCV
    void updateCameraPreviewPixmap(
        const cv::Mat& frame,
        const std::vector<fsc::vision::AnalyzedFace>* faces = nullptr,
        const std::set<int>* matchedFaces = nullptr) {
        if (cameraPreviewLabel_ == nullptr || frame.empty()) {
            return;
        }
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(image.copy()).scaled(
            cameraPreviewLabel_->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        if (!pixmap.isNull() && faces != nullptr && !faces->empty()) {
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing, true);
            const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, frame.cols));
            const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, frame.rows));
            for (int index = 0; index < static_cast<int>(faces->size()); ++index) {
                const auto& face = (*faces)[static_cast<size_t>(index)];
                const auto& box = face.detection.box;
                const bool matched = matchedFaces != nullptr && matchedFaces->find(index) != matchedFaces->end();
                const QColor color = matched ? QColor(0, 230, 70) : QColor(20, 180, 235);
                const QRectF rect(
                    box.x1 * sx,
                    box.y1 * sy,
                    (box.x2 - box.x1) * sx,
                    (box.y2 - box.y1) * sy);
                painter.setPen(QPen(color, matched ? 2.5 : 1.8));
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(rect.normalized());
                painter.setPen(QPen(color, 1.0));
                painter.drawText(rect.topLeft() + QPointF(4.0, -4.0), QString("#%1").arg(index + 1));
            }
        }
        cameraPreviewLabel_->setPixmap(pixmap);
    }
#endif

    void resetCameraEngine() {
#ifdef FSC_ENABLE_ONNX
        cameraEngine_.reset();
        cameraEngineKey_.clear();
#endif
    }

    void startCamera() {
#ifdef FSC_ENABLE_OPENCV
        try {
            stopCamera();
            if (!camera_.open(cameraIndexSpin_->value())) {
                throw std::runtime_error("Could not open camera.");
            }
            camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cameraVotesByFace_.clear();
            latestCameraFaces_.clear();
            latestCameraMatchedFaceIndexes_.clear();
            latestCameraFacesAt_ = {};
            refreshCameraDatabaseLabel();
            setCameraMatchPlaceholder("Match: waiting for frame");
            cameraFrameTimer_->start();
            cameraIdentifyTimer_->start();
            cameraStatusLabel_->setText("Camera running");
            statusBar()->showMessage("Camera started");
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with OpenCV camera support.");
#endif
    }

    void stopCamera() {
#ifdef FSC_ENABLE_OPENCV
        if (cameraFrameTimer_ != nullptr) {
            cameraFrameTimer_->stop();
        }
        if (cameraIdentifyTimer_ != nullptr) {
            cameraIdentifyTimer_->stop();
        }
        if (camera_.isOpened()) {
            camera_.release();
        }
        lastCameraFrame_.release();
        cameraVotesByFace_.clear();
        latestCameraFaces_.clear();
        latestCameraMatchedFaceIndexes_.clear();
        latestCameraFacesAt_ = {};
        setCameraMatchPlaceholder("Match: --");
        resetCameraEngine();
        if (cameraStatusLabel_ != nullptr) {
            cameraStatusLabel_->setText("Camera stopped");
        }
#endif
    }

    void captureCameraFrame() {
#ifdef FSC_ENABLE_OPENCV
        if (!camera_.isOpened()) {
            return;
        }
        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) {
            return;
        }
        lastCameraFrame_ = frame.clone();
        const auto now = std::chrono::steady_clock::now();
        const bool showRecentFaces = !latestCameraFaces_.empty() &&
            latestCameraFacesAt_ != std::chrono::steady_clock::time_point{} &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - latestCameraFacesAt_).count() <= 1000;
        updateCameraPreviewPixmap(
            frame,
            showRecentFaces ? &latestCameraFaces_ : nullptr,
            showRecentFaces ? &latestCameraMatchedFaceIndexes_ : nullptr);
#endif
    }

    void identifyCameraFrame() {
#if defined(FSC_ENABLE_OPENCV) && defined(FSC_ENABLE_ONNX)
        if (cameraAnalyzeBusy_) {
            return;
        }
        if (lastCameraFrame_.empty()) {
            cameraStatusLabel_->setText("No camera frame");
            return;
        }
        cameraAnalyzeBusy_ = true;
        try {
            const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
            const auto mode = selectedRuntimeMode();
            const std::string engineKey = modelRoot.string() + "|" + fsc::vision::toString(mode);
            if (!cameraEngine_ || cameraEngineKey_ != engineKey) {
                cameraEngine_ = std::make_unique<fsc::vision::InsightFaceEngine>(
                    fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot),
                    mode);
                cameraEngineKey_ = engineKey;
            }

            cv::Mat frameForAnalysis = lastCameraFrame_;
            double scaleX = 1.0;
            double scaleY = 1.0;
            const int processSize = cameraProcessSizeSpin_ != nullptr ? cameraProcessSizeSpin_->value() : 640;
            const int longEdge = std::max(lastCameraFrame_.cols, lastCameraFrame_.rows);
            if (processSize > 0 && longEdge > processSize) {
                const double scale = static_cast<double>(processSize) / static_cast<double>(longEdge);
                cv::resize(lastCameraFrame_, frameForAnalysis, cv::Size(), scale, scale, cv::INTER_AREA);
                scaleX = static_cast<double>(lastCameraFrame_.cols) / static_cast<double>(std::max(1, frameForAnalysis.cols));
                scaleY = static_cast<double>(lastCameraFrame_.rows) / static_cast<double>(std::max(1, frameForAnalysis.rows));
            }
            const auto image = rgbImageFromBgrMat(frameForAnalysis);
            auto faces = cameraEngine_->analyze(image, 0.50f, 10);
            if (scaleX != 1.0 || scaleY != 1.0) {
                scaleAnalyzedFaceCoordinates(faces, scaleX, scaleY);
            }
            if (faces.empty()) {
                cameraIdentityLabel_->setText("Identity: no face");
                cameraResultTable_->setRowCount(0);
                cameraVotesByFace_.clear();
                latestCameraFaces_.clear();
                latestCameraMatchedFaceIndexes_.clear();
                latestCameraFacesAt_ = {};
                setCameraMatchPlaceholder("Match: no face detected");
                cameraStatusLabel_->setText("No face detected");
                cameraAnalyzeBusy_ = false;
                return;
            }

            const std::vector<fsc::core::FaceRecord> storedFaces = database_ ? database_->loadFaces(false) : std::vector<fsc::core::FaceRecord>{};
            const int topK = cameraTopKSpin_ != nullptr ? cameraTopKSpin_->value() : 3;
            const double threshold = cameraThresholdSpin_ != nullptr ? cameraThresholdSpin_->value() : 0.35;
            std::set<int> matchedFaceIndexes;
            QString bestRawName = "unknown";
            QString bestStableName = "unknown";
            QString bestStatus = "Match: no database hit";
            int64_t bestPreviewFaceId = 0;
            double bestIdentityConfidence = -1.0;
            double bestHitCosine = -2.0;
            bool bestPreviewFromIdentity = false;

            cameraResultTable_->setRowCount(0);
            const auto addRow = [&](int faceIndex,
                                    const QString& identityName,
                                    const QString& decision,
                                    double confidence,
                                    int64_t evidenceFaceId,
                                    const fsc::core::SearchHit* hit) {
                const int row = cameraResultTable_->rowCount();
                cameraResultTable_->insertRow(row);
                cameraResultTable_->setItem(row, 0, item(QString("#%1").arg(faceIndex + 1)));
                cameraResultTable_->setItem(row, 1, item(identityName));
                cameraResultTable_->setItem(row, 2, item(decision));
                cameraResultTable_->setItem(row, 3, confidence >= 0.0 ? numberItem(confidence * 100.0, 1) : item("--"));
                cameraResultTable_->setItem(row, 4, evidenceFaceId > 0 ? item(QString::number(evidenceFaceId)) : item("--"));
                if (hit != nullptr) {
                    cameraResultTable_->setItem(row, 5, item(QString::number(hit->record.id)));
                    cameraResultTable_->setItem(row, 6, item(qs(hit->record.fileName)));
                    cameraResultTable_->setItem(row, 7, item(qs(hit->record.personName)));
                    cameraResultTable_->setItem(row, 8, numberItem(hit->cosine, 4));
                    cameraResultTable_->setItem(row, 9, numberItem(hit->similarityPercent(), 1));
                    cameraResultTable_->setItem(row, 10, numberItem(hit->record.qualityScore, 3));
                } else {
                    for (int column = 5; column < 11; ++column) {
                        cameraResultTable_->setItem(row, column, item("--"));
                    }
                }
            };

            for (int faceIndex = 0; faceIndex < static_cast<int>(faces.size()); ++faceIndex) {
                const auto& face = faces[static_cast<size_t>(faceIndex)];
                QString identityName = "--";
                QString decision = database_ ? "unknown" : "no database";
                double confidence = -1.0;
                int64_t evidenceFaceId = 0;

                if (database_) {
                    const auto identity = fsc::core::identifyPerson(cameraIdentityProfiles_, face.embedding, selectedIdentityMode(), 5);
                    decision = qs(identity.decision);
                    QString rawIdentityName;
                    if (!identity.candidates.empty()) {
                        const auto& candidate = identity.candidates.front();
                        rawIdentityName = qs(candidate.profile.personName);
                        identityName = rawIdentityName;
                        confidence = candidate.confidence;
                        evidenceFaceId = candidate.evidenceFaceId;
                        if (identity.decision != "unknown") {
                            const auto stableIdentityName = smoothedCameraName(faceIndex, rawIdentityName);
                            if (!stableIdentityName.isEmpty()) {
                                identityName = stableIdentityName;
                                if (stableIdentityName != rawIdentityName) {
                                    decision = "smoothed";
                                }
                            }
                            matchedFaceIndexes.insert(faceIndex);
                            if (candidate.confidence > bestIdentityConfidence ||
                                (!bestPreviewFromIdentity && candidate.evidenceFaceId > 0)) {
                                bestIdentityConfidence = candidate.confidence;
                                bestRawName = rawIdentityName;
                                bestStableName = identityName;
                                if (candidate.evidenceFaceId > 0) {
                                    bestPreviewFaceId = candidate.evidenceFaceId;
                                    bestPreviewFromIdentity = true;
                                    bestStatus = QString("Identity: %1 | %2 | confidence %3% | evidence %4")
                                                     .arg(identityName, decision)
                                                     .arg(candidate.confidence * 100.0, 0, 'f', 1)
                                                     .arg(candidate.evidenceFaceId);
                                }
                            }
                        }
                    }
                    if (identity.decision == "unknown") {
                        (void)smoothedCameraName(faceIndex, QString());
                    }
                }

                const auto hits = storedFaces.empty()
                    ? std::vector<fsc::core::SearchHit>{}
                    : fsc::core::searchFaces(storedFaces, face.embedding, topK, threshold, false);
                if (!hits.empty()) {
                    matchedFaceIndexes.insert(faceIndex);
                    if (!bestPreviewFromIdentity && (bestPreviewFaceId <= 0 || hits.front().cosine > bestHitCosine)) {
                        bestHitCosine = hits.front().cosine;
                        bestPreviewFaceId = hits.front().record.id;
                        bestStatus = QString("Nearest: face %1 | %2 | cosine %3 | similarity %4%")
                                         .arg(hits.front().record.id)
                                         .arg(qs(hits.front().record.personName).isEmpty() ? qs(hits.front().record.fileName) : qs(hits.front().record.personName))
                                         .arg(hits.front().cosine, 0, 'f', 4)
                                         .arg(hits.front().similarityPercent(), 0, 'f', 1);
                    }
                    for (const auto& hit : hits) {
                        addRow(faceIndex, identityName, decision, confidence, evidenceFaceId, &hit);
                    }
                } else {
                    addRow(faceIndex, identityName, decision, confidence, evidenceFaceId, nullptr);
                }
            }
            for (auto it = cameraVotesByFace_.begin(); it != cameraVotesByFace_.end();) {
                if (it->first >= static_cast<int>(faces.size())) {
                    it = cameraVotesByFace_.erase(it);
                } else {
                    ++it;
                }
            }

            latestCameraFaces_ = faces;
            latestCameraMatchedFaceIndexes_ = matchedFaceIndexes;
            latestCameraFacesAt_ = std::chrono::steady_clock::now();
            updateCameraPreviewPixmap(lastCameraFrame_, &latestCameraFaces_, &latestCameraMatchedFaceIndexes_);

            cameraIdentityLabel_->setText(QString("Identity: %1 face(s) | best %2 | stable %3")
                                              .arg(faces.size())
                                              .arg(bestRawName)
                                              .arg(bestStableName));
            if (bestPreviewFaceId > 0) {
                updateCameraMatchPreview(bestPreviewFaceId, bestStatus);
            } else {
                setCameraMatchPlaceholder(bestStatus);
            }
            cameraStatusLabel_->setText(QString("Detected %1 face(s), threshold %2, top %3, process %4")
                                            .arg(faces.size())
                                            .arg(threshold, 0, 'f', 3)
                                            .arg(topK)
                                            .arg(processSize));
            cameraResultTable_->resizeColumnsToContents();
        } catch (const std::exception& ex) {
            showError(ex);
        }
        cameraAnalyzeBusy_ = false;
#elif !defined(FSC_ENABLE_OPENCV)
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with OpenCV camera support.");
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void chooseImage(QLineEdit* target) {
        const auto path = QFileDialog::getOpenFileName(this, "Select image", {}, "Images (*.jpg *.jpeg *.png *.bmp *.ppm)");
        if (!path.isEmpty()) {
            target->setText(path);
        }
    }

    void selectCompareImage(char slot) {
        const auto path = QFileDialog::getOpenFileName(this, "Select image", {}, "Images (*.jpg *.jpeg *.png *.bmp *.ppm)");
        if (path.isEmpty()) {
            return;
        }
        auto* edit = slot == 'a' ? compareImageAEdit_ : compareImageBEdit_;
        auto* preview = slot == 'a' ? comparePreviewA_ : comparePreviewB_;
        auto* list = slot == 'a' ? compareFaceListA_ : compareFaceListB_;
        auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
        auto& selected = slot == 'a' ? selectedCompareA_ : selectedCompareB_;
        edit->setText(path);
        faces.clear();
        selected = 0;
        list->clear();
        preview->setImagePath(path);
        compareResultLabel_->setText("Cosine: --    Similarity: --");
        analyzeCompareImage(slot);
    }

    void analyzeCompareImage(char slot) {
#ifdef FSC_ENABLE_ONNX
        try {
            auto* edit = slot == 'a' ? compareImageAEdit_ : compareImageBEdit_;
            auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
            if (edit == nullptr || edit->text().isEmpty()) {
                throw std::runtime_error("Select both images first.");
            }
            const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
            fsc::vision::InsightFaceEngine engine(fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot), selectedRuntimeMode());
            const auto image = fsc::vision::loadImageRgb(pathFrom(edit->text()));
            faces = engine.analyze(image, 0.50f, 10);
            populateCompareFaces(slot);
            if (faces.empty()) {
                statusBar()->showMessage(QString("Image %1: no face detected").arg(slot == 'a' ? "A" : "B"));
                updateComparePreview(slot);
                return;
            }
            selectCompareFace(slot, 0);
            statusBar()->showMessage(QString("Image %1: detected %2 face(s)").arg(slot == 'a' ? "A" : "B").arg(faces.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void populateCompareFaces(char slot) {
        auto* list = slot == 'a' ? compareFaceListA_ : compareFaceListB_;
        const auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
        if (list == nullptr) {
            return;
        }
        updatingCompareLists_ = true;
        list->clear();
        for (int index = 0; index < static_cast<int>(faces.size()); ++index) {
            const auto& face = faces[static_cast<size_t>(index)];
            list->addItem(QString("Face %1: det %2, quality %3")
                              .arg(index + 1)
                              .arg(face.detection.score, 0, 'f', 3)
                              .arg(face.qualityScore, 0, 'f', 3));
        }
        updatingCompareLists_ = false;
    }

    void selectCompareFace(char slot, int index) {
        auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
        auto& selected = slot == 'a' ? selectedCompareA_ : selectedCompareB_;
        auto* list = slot == 'a' ? compareFaceListA_ : compareFaceListB_;
        if (faces.empty()) {
            return;
        }
        selected = std::clamp(index, 0, static_cast<int>(faces.size()) - 1);
        if (list != nullptr && list->currentRow() != selected) {
            updatingCompareLists_ = true;
            list->setCurrentRow(selected);
            updatingCompareLists_ = false;
        }
        updateComparePreview(slot);
    }

    void updateComparePreview(char slot) {
        auto* preview = slot == 'a' ? comparePreviewA_ : comparePreviewB_;
        const auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
        const int selected = slot == 'a' ? selectedCompareA_ : selectedCompareB_;
        if (preview != nullptr) {
            preview->setFaces(faces, selected);
        }
    }

    void toggleCompareFocus(char slot) {
        auto* preview = slot == 'a' ? comparePreviewA_ : comparePreviewB_;
        auto* button = slot == 'a' ? compareFocusAButton_ : compareFocusBButton_;
        if (preview == nullptr || button == nullptr) {
            return;
        }
        preview->setFocusOnFace(!preview->focusOnFace());
        button->setText(preview->focusOnFace() ? "Full Image" : "Focus on Face");
    }

    void compareImages() {
#ifdef FSC_ENABLE_ONNX
        try {
            if (compareFacesA_.empty()) {
                analyzeCompareImage('a');
            }
            if (compareFacesB_.empty()) {
                analyzeCompareImage('b');
            }
            if (compareFacesA_.empty() || compareFacesB_.empty()) {
                throw std::runtime_error("Select both images and wait for face detection first.");
            }
            selectedCompareA_ = std::clamp(selectedCompareA_, 0, static_cast<int>(compareFacesA_.size()) - 1);
            selectedCompareB_ = std::clamp(selectedCompareB_, 0, static_cast<int>(compareFacesB_.size()) - 1);
            const auto& faceA = compareFacesA_[static_cast<size_t>(selectedCompareA_)];
            const auto& faceB = compareFacesB_[static_cast<size_t>(selectedCompareB_)];
            const double cosine = fsc::core::dot(faceA.embedding, faceB.embedding);
            compareResultLabel_->setText(
                QString("Cosine: %1    Similarity: %2%    Quality A/B: %3/%4")
                    .arg(cosine, 0, 'f', 4)
                    .arg((cosine + 1.0) * 50.0, 0, 'f', 2)
                    .arg(faceA.qualityScore, 0, 'f', 3)
                    .arg(faceB.qualityScore, 0, 'f', 3));

            compareFaceTable_->setRowCount(2);
            const auto fill = [this](int row, const QString& name, const fsc::vision::AnalyzedFace& face) {
                compareFaceTable_->setItem(row, 0, item(name));
                compareFaceTable_->setItem(row, 1, numberItem(face.detection.score, 4));
                compareFaceTable_->setItem(row, 2, numberItem(face.qualityScore, 4));
                compareFaceTable_->setItem(row, 3, item(QString::number(face.landmarks2d.size())));
                compareFaceTable_->setItem(row, 4, item(QString::number(face.landmarks3d.size())));
            };
            fill(0, "A", faceA);
            fill(1, "B", faceB);
            compareFaceTable_->resizeColumnsToContents();
            updateComparePreview('a');
            updateComparePreview('b');
            statusBar()->showMessage("Images compared");
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void refreshClusters() {
        if (!database_) {
            return;
        }
        try {
            clusters_ = buildClusters(
                database_->loadFaces(clusterIncludeIgnoredCheck_ != nullptr && clusterIncludeIgnoredCheck_->isChecked()),
                clusterThresholdSpin_->value(),
                clusterMinSizeSpin_->value(),
                clusterMinQualitySpin_ != nullptr ? clusterMinQualitySpin_->value() : 0.0,
                clusterUnassignedOnlyCheck_ != nullptr && clusterUnassignedOnlyCheck_->isChecked(),
                clusterMaxFacesSpin_ != nullptr ? clusterMaxFacesSpin_->value() : 0);
            clusterTable_->setRowCount(static_cast<int>(clusters_.size()));
            for (int row = 0; row < static_cast<int>(clusters_.size()); ++row) {
                const auto& cluster = clusters_[static_cast<size_t>(row)];
                clusterTable_->setItem(row, 0, item(QString::number(row + 1)));
                clusterTable_->setItem(row, 1, item(QString::number(cluster.members.size())));
                clusterTable_->setItem(row, 2, numberItem(cluster.meanSimilarity, 4));
                clusterTable_->setItem(row, 3, numberItem(cluster.maxSimilarity, 4));
                clusterTable_->setItem(row, 4, numberItem(cluster.averageQuality, 3));
                clusterTable_->setItem(row, 5, item(joinClusterPeople(cluster.knownPeople)));
            }
            clusterTable_->resizeColumnsToContents();
            clusterMemberTable_->setRowCount(0);
            if (clusterPreviewLabel_ != nullptr) {
                clusterPreviewLabel_->setText("Select a cluster");
                clusterPreviewLabel_->setPixmap(QPixmap());
            }
            if (clusterSummaryLabel_ != nullptr) {
                clusterSummaryLabel_->setText(QString("%1 cluster(s) above threshold %2")
                                                  .arg(clusters_.size())
                                                  .arg(clusterThresholdSpin_->value(), 0, 'f', 3));
            }
            if (!clusters_.empty()) {
                clusterTable_->selectRow(0);
            }
            statusBar()->showMessage(QString("Built %1 cluster(s)").arg(clusters_.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    QString joinClusterPeople(const std::vector<std::string>& people) const {
        QString output;
        for (const auto& name : people) {
            if (!output.isEmpty()) {
                output += ", ";
            }
            output += qs(name);
        }
        return output;
    }

    void showSelectedClusterMembers() {
        const auto selected = clusterTable_->selectedItems();
        if (selected.empty()) {
            return;
        }
        const int index = selected.front()->row();
        if (index < 0 || index >= static_cast<int>(clusters_.size())) {
            return;
        }
        const auto& cluster = clusters_[static_cast<size_t>(index)];
        const auto& members = cluster.members;
        clusterMemberTable_->setRowCount(static_cast<int>(members.size()));
        for (int row = 0; row < static_cast<int>(members.size()); ++row) {
            const auto& member = members[static_cast<size_t>(row)];
            clusterMemberTable_->setItem(row, 0, item(QString::number(member.id)));
            clusterMemberTable_->setItem(row, 1, item(qs(member.fileName)));
            clusterMemberTable_->setItem(row, 2, item(qs(member.personName)));
            clusterMemberTable_->setItem(row, 3, item(qs(member.tagText)));
            clusterMemberTable_->setItem(row, 4, numberItem(member.qualityScore, 3));
            clusterMemberTable_->setItem(row, 5, item(qs(member.reviewState)));
        }
        clusterMemberTable_->resizeColumnsToContents();
        if (clusterPersonEdit_ != nullptr) {
            clusterPersonEdit_->setText(qs(cluster.suggestedName));
        }
        if (clusterTagsEdit_ != nullptr) {
            clusterTagsEdit_->setText("cluster-suggested");
        }
        if (cluster.representativeId > 0) {
            updateClusterPreviewForFace(cluster.representativeId);
        }
        if (!members.empty()) {
            clusterMemberTable_->selectRow(0);
        }
    }

    int selectedClusterIndex() const {
        if (clusterTable_ == nullptr || clusterTable_->selectionModel() == nullptr) {
            return -1;
        }
        const auto selected = clusterTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return -1;
        }
        const int index = selected.front().row();
        return index >= 0 && index < static_cast<int>(clusters_.size()) ? index : -1;
    }

    void showSelectedClusterMemberPreview() {
        const int clusterIndex = selectedClusterIndex();
        if (clusterIndex < 0 || clusterMemberTable_ == nullptr || clusterMemberTable_->selectionModel() == nullptr) {
            return;
        }
        const auto selected = clusterMemberTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        const auto& members = clusters_[static_cast<size_t>(clusterIndex)].members;
        if (row >= 0 && row < static_cast<int>(members.size())) {
            updateClusterPreviewForFace(members[static_cast<size_t>(row)].id);
        }
    }

    void updateClusterPreviewForFace(int64_t faceId) {
        if (clusterPreviewLabel_ == nullptr || !database_) {
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                clusterPreviewLabel_->setText("Face not found");
                clusterPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            setDatabaseFacePreview(clusterPreviewLabel_, *face, "No preview");
        } catch (const std::exception& ex) {
            clusterPreviewLabel_->setText(ex.what());
            clusterPreviewLabel_->setPixmap(QPixmap());
        }
    }

    void assignSelectedCluster() {
        if (!database_) {
            return;
        }
        try {
            const int clusterIndex = selectedClusterIndex();
            if (clusterIndex < 0) {
                throw std::runtime_error("Build and select a cluster first.");
            }
            const auto personName = clusterPersonEdit_ == nullptr ? QString() : clusterPersonEdit_->text().trimmed();
            if (personName.isEmpty()) {
                throw std::runtime_error("Enter a person name first.");
            }
            const QString tags = clusterTagsEdit_ == nullptr ? QString() : clusterTagsEdit_->text();
            const bool markReviewed = clusterMarkReviewedCheck_ == nullptr || clusterMarkReviewedCheck_->isChecked();
            int count = 0;
            for (const auto& member : clusters_[static_cast<size_t>(clusterIndex)].members) {
                assignFaceToPersonName(member.id, personName);
                if (!tags.trimmed().isEmpty()) {
                    database_->setFaceTags(member.id, tags.toUtf8().constData(), true);
                }
                if (markReviewed) {
                    database_->updateFaceReview(member.id, "reviewed", false, "Confirmed native cluster assignment.");
                }
                ++count;
            }
            database_->rebuildIdentityProfiles();
            reloadAll();
            refreshClusters();
            statusBar()->showMessage(QString("Assigned %1 face(s) to %2").arg(count).arg(personName));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void setLibraryImportPreview(const std::filesystem::path& imagePath) {
        if (libraryPreviewLabel_ == nullptr) {
            return;
        }
        libraryPreviewFaceId_ = 0;
        libraryFocusOnFace_ = false;
        if (libraryFocusButton_ != nullptr) {
            libraryFocusButton_->setText("Focus on Face");
        }
        if (libraryVisualTabs_ != nullptr) {
            libraryVisualTabs_->setCurrentIndex(0);
        }
        const QString displayPath = QString::fromStdWString(imagePath.wstring());
        QImage image(displayPath);
        if (image.isNull()) {
            libraryPreviewLabel_->setText("Image preview unavailable");
            libraryPreviewLabel_->setPixmap(QPixmap());
            return;
        }
        QSize target = libraryPreviewLabel_->size();
        if (target.width() < 32 || target.height() < 32) {
            target = QSize(420, 420);
        }
        const auto pixmap = QPixmap::fromImage(image).scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        libraryPreviewLabel_->setPixmap(pixmap);
    }

#ifdef FSC_ENABLE_ONNX
    int importImagePath(
        const std::filesystem::path& imagePath,
        fsc::vision::InsightFaceEngine& engine,
        bool clearLog) {
        if (importLog_ != nullptr && clearLog) {
            importLog_->setRowCount(0);
        }
        setLibraryImportPreview(imagePath);
        QApplication::processEvents();

        const auto imageHash = fsc::core::sha256File(imagePath);
        const bool duplicate = database_->imageHashExists(imageHash);
        const auto image = fsc::vision::loadImageRgb(imagePath);
        const auto faces = engine.analyze(image, 0.50f, 10);
        const double minQuality = libraryImportMinQualitySpin_ != nullptr ? libraryImportMinQualitySpin_->value() : 0.0;
        int inserted = 0;
        for (int index = 0; index < static_cast<int>(faces.size()); ++index) {
            const auto& face = faces[static_cast<size_t>(index)];
            QString insertedText = "skipped";
            if (face.qualityScore >= minQuality) {
                const auto id = database_->insertFace(insertRecordFromFace(imagePath, face, imageHash, duplicate));
                insertedText = QString::number(id);
                ++inserted;
            }
            if (importLog_ != nullptr) {
                const int row = importLog_->rowCount();
                importLog_->insertRow(row);
                importLog_->setItem(row, 0, item(insertedText));
                importLog_->setItem(row, 1, item(QString::number(index + 1)));
                importLog_->setItem(row, 2, numberItem(face.detection.score, 4));
                importLog_->setItem(row, 3, numberItem(face.qualityScore, 4));
                importLog_->setItem(row, 4, item(QString::number(face.landmarks2d.size())));
                importLog_->setItem(row, 5, item(QString::number(face.landmarks3d.size())));
            }
        }
        if (importLog_ != nullptr) {
            importLog_->resizeColumnsToContents();
        }
        appendLibraryActivity(QString("Imported %1 from %2%3")
                                  .arg(inserted)
                                  .arg(QString::fromStdWString(imagePath.filename().wstring()))
                                  .arg(duplicate ? " (duplicate source)" : ""));
        return inserted;
    }
#endif

    void importImage() {
#ifdef FSC_ENABLE_ONNX
        if (!database_) {
            return;
        }
        try {
            const auto modelRoot = pathFrom(modelRootEdit_->text());
            const auto imagePath = pathFrom(importImageEdit_->text());
            if (imagePath.empty()) {
                throw std::runtime_error("Select an image first.");
            }
            fsc::vision::InsightFaceEngine engine(fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot), selectedRuntimeMode());
            if (libraryProgressBar_ != nullptr) {
                libraryProgressBar_->setRange(0, 1);
                libraryProgressBar_->setValue(0);
            }
            const int inserted = importImagePath(imagePath, engine, true);
            if (libraryProgressBar_ != nullptr) {
                libraryProgressBar_->setValue(1);
            }
            reloadAll();
            statusBar()->showMessage(QString("Imported %1 face(s)").arg(inserted));
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void importFolder() {
#ifdef FSC_ENABLE_ONNX
        if (!database_) {
            return;
        }
        try {
            const QString folder = QFileDialog::getExistingDirectory(this, "Import image folder", importImageEdit_ == nullptr ? QString() : importImageEdit_->text());
            if (folder.isEmpty()) {
                return;
            }
            const QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.ppm"};
            std::vector<QString> files;
            QDirIterator iterator(folder, filters, QDir::Files, QDirIterator::Subdirectories);
            while (iterator.hasNext()) {
                const QString file = iterator.next();
                if (isSupportedImageFile(file)) {
                    files.push_back(file);
                }
            }
            if (files.empty()) {
                throw std::runtime_error("No supported image files found.");
            }
            if (importLog_ != nullptr) {
                importLog_->setRowCount(0);
            }
            if (libraryProgressBar_ != nullptr) {
                libraryProgressBar_->setRange(0, static_cast<int>(files.size()));
                libraryProgressBar_->setValue(0);
            }
            fsc::vision::InsightFaceEngine engine(
                fsc::vision::InsightFaceModelPaths::fromBuffaloL(pathFrom(modelRootEdit_->text())),
                selectedRuntimeMode());
            int insertedTotal = 0;
            int failed = 0;
            for (int index = 0; index < static_cast<int>(files.size()); ++index) {
                try {
                    insertedTotal += importImagePath(pathFrom(files[static_cast<size_t>(index)]), engine, false);
                } catch (const std::exception& ex) {
                    ++failed;
                    appendLibraryActivity(QString("Failed %1: %2")
                                              .arg(QFileInfo(files[static_cast<size_t>(index)]).fileName(), QString::fromUtf8(ex.what())));
                }
                if (libraryProgressBar_ != nullptr) {
                    libraryProgressBar_->setValue(index + 1);
                }
                QApplication::processEvents();
            }
            reloadAll();
            appendLibraryActivity(QString("Folder import complete: %1 face(s), %2 failed file(s)")
                                      .arg(insertedTotal)
                                      .arg(failed));
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void exportLibraryCsv() {
        if (!database_) {
            return;
        }
        try {
            const auto defaultPath = database_->path().parent_path() / (database_->path().stem().string() + "_faces.csv");
            const QString path = QFileDialog::getSaveFileName(
                this,
                "Export Library CSV",
                qs(defaultPath.string()),
                "CSV Files (*.csv);;All Files (*)");
            if (path.isEmpty()) {
                return;
            }
            const bool includeIgnored = libraryFilterIncludeIgnoredCheck_ == nullptr || libraryFilterIncludeIgnoredCheck_->isChecked();
            auto records = database_->loadFaces(includeIgnored);
            records.erase(
                std::remove_if(records.begin(), records.end(), [this](const auto& record) {
                    return !recordMatchesLibraryFilters(record);
                }),
                records.end());
            const int count = writeFacesCsv(records, pathFrom(path));
            appendLibraryActivity(QString("Exported %1 face(s) to %2").arg(count).arg(path));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void showError(const std::exception& ex) {
        QMessageBox::critical(this, "FSC Studio Native", ex.what());
        statusBar()->showMessage("Error");
    }

    QLabel* databaseLabel_ = nullptr;
    QListWidget* sidebar_ = nullptr;
    QLabel* languageLabel_ = nullptr;
    QComboBox* languageCombo_ = nullptr;
    QLabel* identityModeLabel_ = nullptr;
    QComboBox* identityModeCombo_ = nullptr;
    QLineEdit* databasePathEdit_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    std::vector<QString> tabKeys_;
    QLabel* formatLabel_ = nullptr;
    QLabel* modelLabel_ = nullptr;
    QLabel* metricLabel_ = nullptr;
    QLabel* facesLabel_ = nullptr;
    QLabel* peopleLabel_ = nullptr;
    QLabel* reviewLabel_ = nullptr;
    QLabel* qualityLabel_ = nullptr;
    QTableWidget* libraryTable_ = nullptr;
    QLabel* libraryPreviewLabel_ = nullptr;
    QTabWidget* libraryVisualTabs_ = nullptr;
    QPushButton* libraryFocusButton_ = nullptr;
    PointCloudWidget* libraryLandmarksView_ = nullptr;
    PointCloudWidget* libraryDenseMeshView_ = nullptr;
    QCheckBox* libraryMeshOverlayCheck_ = nullptr;
    QLabel* libraryMeshStatusLabel_ = nullptr;
    QLineEdit* libraryPersonEdit_ = nullptr;
    QLineEdit* libraryTagsEdit_ = nullptr;
    QComboBox* libraryReviewCombo_ = nullptr;
    QCheckBox* libraryIgnoredCheck_ = nullptr;
    QTextEdit* libraryNotesEdit_ = nullptr;
    QLineEdit* libraryBatchPersonEdit_ = nullptr;
    QLineEdit* libraryBatchTagsEdit_ = nullptr;
    QCheckBox* libraryBatchAppendTagsCheck_ = nullptr;
    QComboBox* libraryBatchReviewCombo_ = nullptr;
    QComboBox* libraryBatchIgnoredCombo_ = nullptr;
    QLineEdit* libraryBatchNotesEdit_ = nullptr;
    QDoubleSpinBox* libraryImportMinQualitySpin_ = nullptr;
    QLineEdit* libraryFilterTextEdit_ = nullptr;
    QComboBox* libraryFilterPersonCombo_ = nullptr;
    QComboBox* libraryFilterTagCombo_ = nullptr;
    QComboBox* libraryFilterReviewCombo_ = nullptr;
    QDoubleSpinBox* libraryFilterMinQualitySpin_ = nullptr;
    QCheckBox* libraryFilterIncludeIgnoredCheck_ = nullptr;
    QProgressBar* libraryProgressBar_ = nullptr;
    QTextEdit* libraryActivityLog_ = nullptr;
    int libraryPreviewFaceId_ = 0;
    bool libraryFocusOnFace_ = false;
    QTableWidget* peopleTable_ = nullptr;
    QLineEdit* personNameEdit_ = nullptr;
    QSpinBox* assignFaceSpin_ = nullptr;
    QSpinBox* assignPersonSpin_ = nullptr;
    QTableWidget* reviewTable_ = nullptr;
    QComboBox* reviewStateCombo_ = nullptr;
    QComboBox* reviewIgnoredCombo_ = nullptr;
    QLineEdit* reviewNotesEdit_ = nullptr;
    QLabel* reviewPreviewLabel_ = nullptr;
    QLabel* reviewSuggestionLabel_ = nullptr;
    int64_t reviewSuggestedPersonId_ = 0;
    QSpinBox* faceIdSpin_ = nullptr;
    QSpinBox* topKSpin_ = nullptr;
    QDoubleSpinBox* searchThresholdSpin_ = nullptr;
    QDoubleSpinBox* searchMinQualitySpin_ = nullptr;
    QCheckBox* searchIncludeIgnoredCheck_ = nullptr;
    QComboBox* searchPersonFilterCombo_ = nullptr;
    QComboBox* searchTagFilterCombo_ = nullptr;
    QLabel* identityLabel_ = nullptr;
    QTableWidget* searchIdentityTable_ = nullptr;
    QTableWidget* searchTable_ = nullptr;
    QLineEdit* searchImageEdit_ = nullptr;
    QComboBox* searchFaceCombo_ = nullptr;
    QLabel* searchPreviewLabel_ = nullptr;
    QLabel* searchResultPreviewLabel_ = nullptr;
    QLineEdit* searchAssignPersonEdit_ = nullptr;
    QTimer* searchPreviewTimer_ = nullptr;
    std::vector<fsc::vision::AnalyzedFace> searchQueryFaces_;
    std::vector<fsc::core::SearchHit> searchHits_;
    fsc::core::IdentityResult lastSearchIdentityResult_;
    int64_t currentSearchDatabaseFaceId_ = 0;
    int searchQueryFaceIndex_ = 0;
    int searchPreviewAnimationIndex_ = 0;
    QSpinBox* cameraIndexSpin_ = nullptr;
    QDoubleSpinBox* cameraThresholdSpin_ = nullptr;
    QSpinBox* cameraTopKSpin_ = nullptr;
    QSpinBox* cameraIntervalSpin_ = nullptr;
    QSpinBox* cameraProcessSizeSpin_ = nullptr;
    QCheckBox* cameraAutoCheck_ = nullptr;
    QLabel* cameraStatusLabel_ = nullptr;
    QLabel* cameraDatabaseLabel_ = nullptr;
    QLabel* cameraPreviewLabel_ = nullptr;
    QLabel* cameraMatchPreviewLabel_ = nullptr;
    QLabel* cameraMatchStatusLabel_ = nullptr;
    QLabel* cameraIdentityLabel_ = nullptr;
    QTableWidget* cameraResultTable_ = nullptr;
    QTimer* cameraFrameTimer_ = nullptr;
    QTimer* cameraIdentifyTimer_ = nullptr;
    QLineEdit* compareImageAEdit_ = nullptr;
    QLineEdit* compareImageBEdit_ = nullptr;
    QLabel* compareResultLabel_ = nullptr;
    QTableWidget* compareFaceTable_ = nullptr;
    FaceSelectionPreview* comparePreviewA_ = nullptr;
    FaceSelectionPreview* comparePreviewB_ = nullptr;
    QListWidget* compareFaceListA_ = nullptr;
    QListWidget* compareFaceListB_ = nullptr;
    QPushButton* compareFocusAButton_ = nullptr;
    QPushButton* compareFocusBButton_ = nullptr;
    std::vector<fsc::vision::AnalyzedFace> compareFacesA_;
    std::vector<fsc::vision::AnalyzedFace> compareFacesB_;
    int selectedCompareA_ = 0;
    int selectedCompareB_ = 0;
    bool updatingCompareLists_ = false;
    QDoubleSpinBox* clusterThresholdSpin_ = nullptr;
    QSpinBox* clusterMinSizeSpin_ = nullptr;
    QSpinBox* clusterMaxFacesSpin_ = nullptr;
    QDoubleSpinBox* clusterMinQualitySpin_ = nullptr;
    QCheckBox* clusterUnassignedOnlyCheck_ = nullptr;
    QCheckBox* clusterIncludeIgnoredCheck_ = nullptr;
    QTableWidget* clusterTable_ = nullptr;
    QTableWidget* clusterMemberTable_ = nullptr;
    QLabel* clusterPreviewLabel_ = nullptr;
    QLineEdit* clusterPersonEdit_ = nullptr;
    QLineEdit* clusterTagsEdit_ = nullptr;
    QCheckBox* clusterMarkReviewedCheck_ = nullptr;
    QLabel* clusterSummaryLabel_ = nullptr;
    QSpinBox* meshFaceIdSpin_ = nullptr;
    QCheckBox* meshOverlayCheck_ = nullptr;
    QLabel* meshStatusLabel_ = nullptr;
    PointCloudWidget* meshView_ = nullptr;
    QComboBox* runtimeModeCombo_ = nullptr;
    QLabel* runtimeBuildLabel_ = nullptr;
    QLabel* runtimeProviderLabel_ = nullptr;
    QLabel* runtimeNoteLabel_ = nullptr;
    QLabel* runtimeDatabasePathLabel_ = nullptr;
    QLabel* runtimeDatabaseStatsLabel_ = nullptr;
    QTextEdit* runtimeMaintenanceLog_ = nullptr;
    QLineEdit* modelRootEdit_ = nullptr;
    QLineEdit* importImageEdit_ = nullptr;
    QTableWidget* importLog_ = nullptr;
    std::vector<ClusterSummary> clusters_;
    std::vector<fsc::core::IdentityProfile> cameraIdentityProfiles_;
    std::map<int, std::deque<QString>> cameraVotesByFace_;
    std::vector<fsc::vision::AnalyzedFace> latestCameraFaces_;
    std::set<int> latestCameraMatchedFaceIndexes_;
    std::chrono::steady_clock::time_point latestCameraFacesAt_;
#ifdef FSC_ENABLE_OPENCV
    cv::VideoCapture camera_;
    cv::Mat lastCameraFrame_;
#endif
#ifdef FSC_ENABLE_ONNX
    std::unique_ptr<fsc::vision::InsightFaceEngine> cameraEngine_;
    std::string cameraEngineKey_;
#endif
    bool cameraAnalyzeBusy_ = false;
    std::unique_ptr<fsc::core::Database> database_;
};

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto stats = database.statistics();
            return stats.formatVersion.empty() ? 1 : 0;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--library-export-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const int count = writeFacesCsv(database.loadFaces(true), pathFrom(QString::fromLocal8Bit(argv[3])));
            return count > 0 && std::filesystem::exists(pathFrom(QString::fromLocal8Bit(argv[3]))) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--review-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            database.updateFaceReview(faceId, "reviewed", false, "qt-review-smoke");
            const auto face = database.loadFace(faceId);
            return face.has_value() && face->reviewState == "reviewed" && !face->ignored ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 3 && std::string(argv[1]) == "--cluster-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            (void)buildClusters(database.loadFaces(false), 0.62, 2);
            return 0;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--cluster-action-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto personName = std::string(argv[3]);
            auto clusters = buildClusters(database.loadFaces(false), 0.55, 2, 0.0, false, 5000);
            if (clusters.empty()) {
                return 2;
            }
            const auto personId = database.upsertPerson(personName);
            int count = 0;
            for (const auto& member : clusters.front().members) {
                database.assignFaceToPerson(member.id, personId);
                database.setFaceTags(member.id, "cluster-action-smoke", true);
                database.updateFaceReview(member.id, "reviewed", false, "native cluster action smoke");
                ++count;
            }
            database.rebuildIdentityProfiles();
            const auto face = database.loadFace(clusters.front().members.front().id);
            return count > 0 && face.has_value() && face->personName == personName &&
                    face->tagText.find("cluster-action-smoke") != std::string::npos &&
                    face->reviewState == "reviewed"
                ? 0
                : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--mesh-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value()) {
                return 1;
            }
            return !face->landmarks3d.empty() || !face->faceMesh3d.empty() ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--library-visual-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value() || face->landmarks3d.empty()) {
                return 1;
            }
            const auto mesh = face->faceMesh3d.empty()
                ? fsc::mesh::buildSyntheticFaceMesh3d(face->landmarks3d)
                : face->faceMesh3d;
            return !mesh.empty() ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--mesh-generate-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value() || face->landmarks3d.empty()) {
                return 1;
            }
            const auto mesh = fsc::mesh::buildSyntheticFaceMesh3d(face->landmarks3d);
            database.updateFaceMesh3d(faceId, mesh);
            const auto updated = database.loadFace(faceId);
            return updated.has_value() && !updated->faceMesh3d.empty() ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--metadata-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            database.setFaceTags(faceId, "native-smoke, parity", false);
            database.updateFaceReview(faceId, "open", false, "native metadata smoke");
            const auto face = database.loadFace(faceId);
            if (!face.has_value()) {
                return 1;
            }
            return face->tagText.find("native-smoke") != std::string::npos &&
                           face->notes == "native metadata smoke" &&
                           !face->ignored
                       ? 0
                       : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 6 && std::string(argv[1]) == "--search-action-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto queryFaceId = std::strtoll(argv[3], nullptr, 10);
            const auto assignFaceId = std::strtoll(argv[4], nullptr, 10);
            const auto personName = std::string(argv[5]);
            const auto query = database.loadFace(queryFaceId);
            if (!query.has_value()) {
                return 1;
            }
            const auto hits = fsc::core::searchFaces(database.loadFaces(false), query->embedding, 10, -1.0, false);
            if (hits.empty()) {
                return 1;
            }
            const auto personId = database.upsertPerson(personName);
            database.assignFaceToPerson(assignFaceId, personId);
            database.rebuildIdentityProfiles();
            const auto assigned = database.loadFace(assignFaceId);
            return assigned.has_value() && assigned->personName == personName ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--search-filter-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const std::string personName = "NativeSearchFilterSmoke";
            const std::string tagName = "native-filter-smoke";
            const auto personId = database.upsertPerson(personName);
            database.assignFaceToPerson(faceId, personId);
            database.setFaceTags(faceId, tagName, false);
            const auto tags = database.loadTags();
            const auto face = database.loadFace(faceId);
            if (!face.has_value()) {
                return 1;
            }
            const auto records = database.loadFaces(false);
            const bool personMatch = std::any_of(records.begin(), records.end(), [&](const auto& record) {
                return record.id == faceId && record.personName == personName && record.tagText.find(tagName) != std::string::npos;
            });
            const bool tagLoaded = std::find(tags.begin(), tags.end(), tagName) != tags.end();
            return personMatch && tagLoaded ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--maintenance-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto backupPath = pathFrom(QString::fromLocal8Bit(argv[3]));
            const auto integrity = database.checkIntegrity();
            const auto checkpoint = database.checkpointWal(true);
            const auto backup = database.backupTo(backupPath);
            const auto vacuum = database.vacuum();
            return integrity.ok && checkpoint.ok && backup.ok && vacuum.ok && std::filesystem::exists(backupPath) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 2 && std::string(argv[1]) == "--camera-smoke") {
#ifdef FSC_ENABLE_OPENCV
        return cv::getBuildInformation().empty() ? 1 : 0;
#else
        return 1;
#endif
    }
    if (argc >= 2 && std::string(argv[1]) == "--camera-open-smoke") {
#ifdef FSC_ENABLE_OPENCV
        const int cameraIndex = argc >= 3 ? std::atoi(argv[2]) : 0;
        cv::VideoCapture capture(cameraIndex);
        if (!capture.isOpened()) {
            return 2;
        }
        cv::Mat frame;
        capture >> frame;
        return frame.empty() ? 3 : 0;
#else
        return 1;
#endif
    }
#ifdef FSC_ENABLE_ONNX
    if (argc >= 5 && std::string(argv[1]) == "--camera-result-smoke") {
        try {
            const auto mode = argc >= 6 ? fsc::vision::parseRuntimeMode(argv[5]) : fsc::vision::RuntimeMode::Cpu;
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[3])));
            fsc::vision::InsightFaceEngine engine(
                fsc::vision::InsightFaceModelPaths::fromBuffaloL(pathFrom(QString::fromLocal8Bit(argv[2]))),
                mode);
            const auto image = fsc::vision::loadImageRgb(pathFrom(QString::fromLocal8Bit(argv[4])));
            const auto face = bestFace(engine.analyze(image, 0.50f, 10));
            const auto identity = fsc::core::identifyPerson(database.loadIdentityProfiles(), face.embedding, fsc::core::IdentityMode::Strict, 5);
            const auto hits = fsc::core::searchFaces(database.loadFaces(false), face.embedding, 3, 0.35, false);
            const bool identityOk = identity.decision == "unknown" || !identity.candidates.empty();
            return identityOk && !hits.empty() && std::isfinite(hits.front().cosine) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 5 && std::string(argv[1]) == "--compare-smoke") {
        try {
            const auto mode = argc >= 6 ? fsc::vision::parseRuntimeMode(argv[5]) : fsc::vision::RuntimeMode::Cpu;
            fsc::vision::InsightFaceEngine engine(
                fsc::vision::InsightFaceModelPaths::fromBuffaloL(pathFrom(QString::fromLocal8Bit(argv[2]))),
                mode);
            const auto imageA = fsc::vision::loadImageRgb(pathFrom(QString::fromLocal8Bit(argv[3])));
            const auto imageB = fsc::vision::loadImageRgb(pathFrom(QString::fromLocal8Bit(argv[4])));
            const auto faceA = bestFace(engine.analyze(imageA, 0.50f, 10));
            const auto faceB = bestFace(engine.analyze(imageB, 0.50f, 10));
            const double cosine = fsc::core::dot(faceA.embedding, faceB.embedding);
            return std::isfinite(cosine) && faceA.embedding.size() == faceB.embedding.size() ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
#endif

    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    if (argc > 1) {
        window.openDatabasePath(QString::fromLocal8Bit(argv[1]));
    }
    return app.exec();
}
