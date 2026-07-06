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
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
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
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
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
    double meanSimilarity = 0.0;
    double maxSimilarity = 0.0;
    double averageQuality = 0.0;
};

std::vector<ClusterSummary> buildClusters(std::vector<fsc::core::FaceRecord> records, double threshold, int minSize) {
    records.erase(
        std::remove_if(records.begin(), records.end(), [](const auto& record) {
            return record.embedding.empty();
        }),
        records.end());
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
        for (const int index : indexes) {
            const auto& record = records[static_cast<size_t>(index)];
            qualityTotal += record.qualityScore;
            cluster.members.push_back(record);
        }
        cluster.averageQuality = qualityTotal / static_cast<double>(cluster.members.size());
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
        form->addRow("Models", modelRow);
        form->addRow("Image", imageRow);
        leftLayout->addWidget(controls);

        auto* splitter = new QSplitter(Qt::Vertical, leftPanel);
        libraryTable_ = new QTableWidget(splitter);
        libraryTable_->setColumnCount(9);
        libraryTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Tags", "Quality", "Detection", "Review", "Ignored", "Source"});
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
        libraryFocusButton_ = new QPushButton("Focus on Face", visualPanel);
        libraryFocusButton_->setMaximumWidth(132);
        libraryPreviewLabel_ = new QLabel("Select a face", visualPanel);
        libraryPreviewLabel_->setAlignment(Qt::AlignCenter);
        libraryPreviewLabel_->setMinimumWidth(300);
        libraryPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        visualLayout->addWidget(libraryFocusButton_, 0, Qt::AlignLeft);
        visualLayout->addWidget(libraryPreviewLabel_, 1);
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
                updateLibraryPreview(faceId);
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
        connect(libraryFocusButton_, &QPushButton::clicked, this, [this] {
            libraryFocusOnFace_ = !libraryFocusOnFace_;
            if (libraryPreviewFaceId_ > 0) {
                updateLibraryPreview(libraryPreviewFaceId_);
            }
        });
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
        topKSpin_->setRange(1, 100);
        topKSpin_->setValue(10);
        topKSpin_->setPrefix("Top ");
        auto* searchButton = new QPushButton("Search", controls);
        auto* identifyButton = new QPushButton("Identify", controls);
        identityLabel_ = new QLabel("Identity: -", controls);
        controlsLayout->addWidget(faceIdSpin_);
        controlsLayout->addWidget(topKSpin_);
        controlsLayout->addWidget(searchButton);
        controlsLayout->addWidget(identifyButton);
        controlsLayout->addWidget(identityLabel_, 1);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        auto* leftPanel = new QWidget(splitter);
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        searchTable_ = new QTableWidget(leftPanel);
        searchTable_->setColumnCount(5);
        searchTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Cosine", "Similarity"});
        fitTable(searchTable_);
        leftLayout->addWidget(searchTable_, 1);
        searchPreviewLabel_ = new QLabel("Select or analyze a query image", splitter);
        searchPreviewLabel_->setAlignment(Qt::AlignCenter);
        searchPreviewLabel_->setMinimumWidth(320);
        searchPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        splitter->addWidget(leftPanel);
        splitter->addWidget(searchPreviewLabel_);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);

        addMainTab(page, "Search");
        connect(browseImageButton, &QPushButton::clicked, this, [this] { chooseImage(searchImageEdit_); });
        connect(analyzeImageButton, &QPushButton::clicked, this, [this] { analyzeSearchImage(); });
        connect(searchFaceCombo_, &QComboBox::currentIndexChanged, this, [this](int index) {
            searchQueryFaceIndex_ = index;
            updateSearchQueryPreview();
        });
        connect(searchButton, &QPushButton::clicked, this, [this] { runSearch(); });
        connect(identifyButton, &QPushButton::clicked, this, [this] { runIdentify(); });
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
        cameraAutoCheck_ = new QCheckBox("Auto Identify", controls);
        cameraAutoCheck_->setChecked(true);
        auto* startButton = new QPushButton("Start", controls);
        auto* stopButton = new QPushButton("Stop", controls);
        auto* identifyButton = new QPushButton("Identify Frame", controls);
        cameraStatusLabel_ = new QLabel("Camera stopped", controls);
        controlsLayout->addWidget(cameraIndexSpin_);
        controlsLayout->addWidget(cameraAutoCheck_);
        controlsLayout->addWidget(startButton);
        controlsLayout->addWidget(stopButton);
        controlsLayout->addWidget(identifyButton);
        controlsLayout->addWidget(cameraStatusLabel_, 1);
        layout->addWidget(controls);

        cameraPreviewLabel_ = new QLabel(page);
        cameraPreviewLabel_->setMinimumSize(640, 360);
        cameraPreviewLabel_->setAlignment(Qt::AlignCenter);
        cameraPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;");
        cameraPreviewLabel_->setText("Camera preview");
        layout->addWidget(cameraPreviewLabel_, 1);

        cameraIdentityLabel_ = new QLabel("Identity: -", page);
        layout->addWidget(cameraIdentityLabel_);
        cameraResultTable_ = new QTableWidget(page);
        cameraResultTable_->setColumnCount(5);
        cameraResultTable_->setHorizontalHeaderLabels({"Rank", "Person", "Decision", "Score", "Evidence"});
        fitTable(cameraResultTable_);
        layout->addWidget(cameraResultTable_, 1);

        cameraFrameTimer_ = new QTimer(this);
        cameraFrameTimer_->setInterval(33);
        cameraIdentifyTimer_ = new QTimer(this);
        cameraIdentifyTimer_->setInterval(1200);
        addMainTab(page, "Camera");

        connect(startButton, &QPushButton::clicked, this, [this] { startCamera(); });
        connect(stopButton, &QPushButton::clicked, this, [this] { stopCamera(); });
        connect(identifyButton, &QPushButton::clicked, this, [this] { identifyCameraFrame(); });
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
        auto* clusterButton = new QPushButton("Build Clusters", controls);
        controlsLayout->addWidget(new QLabel("Threshold", controls));
        controlsLayout->addWidget(clusterThresholdSpin_);
        controlsLayout->addWidget(new QLabel("Min Size", controls));
        controlsLayout->addWidget(clusterMinSizeSpin_);
        controlsLayout->addWidget(clusterButton);
        controlsLayout->addStretch(1);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(page);
        clusterTable_ = new QTableWidget(splitter);
        clusterTable_->setColumnCount(5);
        clusterTable_->setHorizontalHeaderLabels({"Cluster", "Faces", "Mean", "Max", "Avg Quality"});
        fitTable(clusterTable_);
        clusterMemberTable_ = new QTableWidget(splitter);
        clusterMemberTable_->setColumnCount(5);
        clusterMemberTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Quality", "Review"});
        fitTable(clusterMemberTable_);
        splitter->addWidget(clusterTable_);
        splitter->addWidget(clusterMemberTable_);
        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);
        addMainTab(page, "Clusters");

        connect(clusterButton, &QPushButton::clicked, this, [this] { refreshClusters(); });
        connect(clusterTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedClusterMembers(); });
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
        auto* layout = new QFormLayout(page);
        runtimeModeCombo_ = new QComboBox(page);
        runtimeModeCombo_->addItem("Auto", "auto");
        runtimeModeCombo_->addItem("CPU", "cpu");
        runtimeModeCombo_->addItem("DirectML", "directml");
        runtimeModeCombo_->setCurrentIndex(0);
        runtimeBuildLabel_ = new QLabel(page);
        runtimeProviderLabel_ = new QLabel(page);
        runtimeNoteLabel_ = new QLabel(page);
        runtimeNoteLabel_->setWordWrap(true);
        auto* refreshButton = new QPushButton("Refresh Runtime", page);
        layout->addRow("Mode", runtimeModeCombo_);
        layout->addRow("Build", runtimeBuildLabel_);
        layout->addRow("Provider", runtimeProviderLabel_);
        layout->addRow("Status", runtimeNoteLabel_);
        layout->addRow("", refreshButton);
        addMainTab(page, "Runtime");

        connect(refreshButton, &QPushButton::clicked, this, [this] { refreshRuntimeInfo(); });
        connect(runtimeModeCombo_, &QComboBox::currentTextChanged, this, [this] {
            resetCameraEngine();
            refreshRuntimeInfo();
        });
        refreshRuntimeInfo();
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
            showError(ex);
        }
    }

    void reloadAll() {
        if (!database_) {
            return;
        }
        loadOverview();
        loadLibrary();
        loadPeople();
        loadReview();
        cameraIdentityProfiles_ = database_->loadIdentityProfiles();
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
        const auto records = database_->loadFaces(true, 1000);
        libraryTable_->setRowCount(static_cast<int>(records.size()));
        for (int row = 0; row < static_cast<int>(records.size()); ++row) {
            const auto& record = records[static_cast<size_t>(row)];
            libraryTable_->setItem(row, 0, item(QString::number(record.id)));
            libraryTable_->setItem(row, 1, item(qs(record.fileName)));
            libraryTable_->setItem(row, 2, item(qs(record.personName)));
            libraryTable_->setItem(row, 3, item(qs(record.tagText)));
            libraryTable_->setItem(row, 4, numberItem(record.qualityScore, 3));
            libraryTable_->setItem(row, 5, numberItem(record.detectionScore, 3));
            libraryTable_->setItem(row, 6, item(qs(record.reviewState)));
            libraryTable_->setItem(row, 7, item(record.ignored ? "yes" : "no"));
            libraryTable_->setItem(row, 8, item(qs(record.sourcePath)));
        }
        libraryTable_->resizeColumnsToContents();
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
            int64_t databaseFaceId = 0;
            const auto embedding = currentSearchEmbedding(&databaseFaceId);
            const auto hits = fsc::core::searchFaces(database_->loadFaces(false), embedding, topKSpin_->value(), -1.0, false);
            searchTable_->setRowCount(0);
            for (const auto& hit : hits) {
                if (databaseFaceId > 0 && hit.record.id == databaseFaceId) {
                    continue;
                }
                const int row = searchTable_->rowCount();
                searchTable_->insertRow(row);
                searchTable_->setItem(row, 0, item(QString::number(hit.record.id)));
                searchTable_->setItem(row, 1, item(qs(hit.record.fileName)));
                searchTable_->setItem(row, 2, item(qs(hit.record.personName)));
                searchTable_->setItem(row, 3, numberItem(hit.cosine, 4));
                searchTable_->setItem(row, 4, numberItem(hit.similarityPercent(), 1));
            }
            searchTable_->resizeColumnsToContents();
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void runIdentify() {
        if (!database_) {
            return;
        }
        try {
            const auto embedding = currentSearchEmbedding();
            const auto result = fsc::core::identifyPerson(database_->loadIdentityProfiles(), embedding, selectedIdentityMode(), 5);
            QString text = "Identity: " + qs(result.decision);
            if (!result.candidates.empty()) {
                const auto& candidate = result.candidates.front();
                text += " | " + qs(candidate.profile.personName);
                text += " | score " + QString::number(candidate.score, 'f', 4);
                text += " | confidence " + QString::number(candidate.confidence * 100.0, 'f', 1) + "%";
            }
            identityLabel_->setText(text);
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    QString smoothedCameraName(const QString& name) {
        cameraVotes_.push_back(name);
        while (cameraVotes_.size() > 5) {
            cameraVotes_.pop_front();
        }
        std::map<QString, int> counts;
        for (const auto& vote : cameraVotes_) {
            ++counts[vote];
        }
        return std::max_element(counts.begin(), counts.end(), [](const auto& left, const auto& right) {
            return left.second < right.second;
        })->first;
    }

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
            cameraVotes_.clear();
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
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        cameraPreviewLabel_->setPixmap(QPixmap::fromImage(image.copy()).scaled(
            cameraPreviewLabel_->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
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

            const auto image = rgbImageFromBgrMat(lastCameraFrame_);
            const auto faces = cameraEngine_->analyze(image, 0.50f, 5);
            if (faces.empty()) {
                cameraIdentityLabel_->setText("Identity: no face");
                cameraResultTable_->setRowCount(0);
                cameraStatusLabel_->setText("No face detected");
                cameraAnalyzeBusy_ = false;
                return;
            }
            const auto& face = bestFace(faces);
            QString headline = QString("Identity: %1 face(s)").arg(faces.size());
            fsc::core::IdentityResult identity;
            if (database_) {
                identity = fsc::core::identifyPerson(cameraIdentityProfiles_, face.embedding, selectedIdentityMode(), 5);
                QString rawName = "unknown";
                if (!identity.candidates.empty() && identity.decision != "unknown") {
                    rawName = qs(identity.candidates.front().profile.personName);
                }
                const auto stableName = smoothedCameraName(rawName);
                headline += QString(" | %1 | stable %2").arg(qs(identity.decision), stableName);
            } else {
                headline += " | open database for identity";
            }
            cameraIdentityLabel_->setText(headline);
            cameraStatusLabel_->setText(QString("Detection %1, quality %2")
                                            .arg(face.detection.score, 0, 'f', 3)
                                            .arg(face.qualityScore, 0, 'f', 3));

            cameraResultTable_->setRowCount(static_cast<int>(identity.candidates.size()));
            for (int row = 0; row < static_cast<int>(identity.candidates.size()); ++row) {
                const auto& candidate = identity.candidates[static_cast<size_t>(row)];
                cameraResultTable_->setItem(row, 0, item(QString::number(row + 1)));
                cameraResultTable_->setItem(row, 1, item(qs(candidate.profile.personName)));
                cameraResultTable_->setItem(row, 2, item(qs(identity.decision)));
                cameraResultTable_->setItem(row, 3, numberItem(candidate.score, 4));
                cameraResultTable_->setItem(row, 4, item(QString::number(candidate.evidenceFaceId)));
            }
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
            clusters_ = buildClusters(database_->loadFaces(false), clusterThresholdSpin_->value(), clusterMinSizeSpin_->value());
            clusterTable_->setRowCount(static_cast<int>(clusters_.size()));
            for (int row = 0; row < static_cast<int>(clusters_.size()); ++row) {
                const auto& cluster = clusters_[static_cast<size_t>(row)];
                clusterTable_->setItem(row, 0, item(QString::number(row + 1)));
                clusterTable_->setItem(row, 1, item(QString::number(cluster.members.size())));
                clusterTable_->setItem(row, 2, numberItem(cluster.meanSimilarity, 4));
                clusterTable_->setItem(row, 3, numberItem(cluster.maxSimilarity, 4));
                clusterTable_->setItem(row, 4, numberItem(cluster.averageQuality, 3));
            }
            clusterTable_->resizeColumnsToContents();
            clusterMemberTable_->setRowCount(0);
            statusBar()->showMessage(QString("Built %1 cluster(s)").arg(clusters_.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
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
        const auto& members = clusters_[static_cast<size_t>(index)].members;
        clusterMemberTable_->setRowCount(static_cast<int>(members.size()));
        for (int row = 0; row < static_cast<int>(members.size()); ++row) {
            const auto& member = members[static_cast<size_t>(row)];
            clusterMemberTable_->setItem(row, 0, item(QString::number(member.id)));
            clusterMemberTable_->setItem(row, 1, item(qs(member.fileName)));
            clusterMemberTable_->setItem(row, 2, item(qs(member.personName)));
            clusterMemberTable_->setItem(row, 3, numberItem(member.qualityScore, 3));
            clusterMemberTable_->setItem(row, 4, item(qs(member.reviewState)));
        }
        clusterMemberTable_->resizeColumnsToContents();
    }

    void importImage() {
#ifdef FSC_ENABLE_ONNX
        if (!database_) {
            return;
        }
        try {
            const auto modelRoot = pathFrom(modelRootEdit_->text());
            const auto imagePath = pathFrom(importImageEdit_->text());
            const auto imageHash = fsc::core::sha256File(imagePath);
            const bool duplicate = database_->imageHashExists(imageHash);
            const auto image = fsc::vision::loadImageRgb(imagePath);
            fsc::vision::InsightFaceEngine engine(fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot), selectedRuntimeMode());
            const auto faces = engine.analyze(image, 0.50f, 10);
            importLog_->setRowCount(0);
            for (int index = 0; index < static_cast<int>(faces.size()); ++index) {
                const auto id = database_->insertFace(insertRecordFromFace(imagePath, faces[static_cast<size_t>(index)], imageHash, duplicate));
                const int row = importLog_->rowCount();
                importLog_->insertRow(row);
                importLog_->setItem(row, 0, item(QString::number(id)));
                importLog_->setItem(row, 1, item(QString::number(index)));
                importLog_->setItem(row, 2, numberItem(faces[static_cast<size_t>(index)].detection.score, 4));
                importLog_->setItem(row, 3, numberItem(faces[static_cast<size_t>(index)].qualityScore, 4));
                importLog_->setItem(row, 4, item(QString::number(faces[static_cast<size_t>(index)].landmarks2d.size())));
                importLog_->setItem(row, 5, item(QString::number(faces[static_cast<size_t>(index)].landmarks3d.size())));
            }
            importLog_->resizeColumnsToContents();
            reloadAll();
            statusBar()->showMessage(QString("Imported %1 face(s)%2").arg(faces.size()).arg(duplicate ? " as duplicate" : ""));
        } catch (const std::exception& ex) {
            showError(ex);
        }
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
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
    QPushButton* libraryFocusButton_ = nullptr;
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
    QLabel* identityLabel_ = nullptr;
    QTableWidget* searchTable_ = nullptr;
    QLineEdit* searchImageEdit_ = nullptr;
    QComboBox* searchFaceCombo_ = nullptr;
    QLabel* searchPreviewLabel_ = nullptr;
    std::vector<fsc::vision::AnalyzedFace> searchQueryFaces_;
    int searchQueryFaceIndex_ = 0;
    QSpinBox* cameraIndexSpin_ = nullptr;
    QCheckBox* cameraAutoCheck_ = nullptr;
    QLabel* cameraStatusLabel_ = nullptr;
    QLabel* cameraPreviewLabel_ = nullptr;
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
    QTableWidget* clusterTable_ = nullptr;
    QTableWidget* clusterMemberTable_ = nullptr;
    QSpinBox* meshFaceIdSpin_ = nullptr;
    QCheckBox* meshOverlayCheck_ = nullptr;
    QLabel* meshStatusLabel_ = nullptr;
    PointCloudWidget* meshView_ = nullptr;
    QComboBox* runtimeModeCombo_ = nullptr;
    QLabel* runtimeBuildLabel_ = nullptr;
    QLabel* runtimeProviderLabel_ = nullptr;
    QLabel* runtimeNoteLabel_ = nullptr;
    QLineEdit* modelRootEdit_ = nullptr;
    QLineEdit* importImageEdit_ = nullptr;
    QTableWidget* importLog_ = nullptr;
    std::vector<ClusterSummary> clusters_;
    std::vector<fsc::core::IdentityProfile> cameraIdentityProfiles_;
    std::deque<QString> cameraVotes_;
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
