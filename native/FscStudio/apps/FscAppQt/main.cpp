#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/core/VectorMath.hpp"
#ifdef FSC_ENABLE_ONNX
#include "fsc/legacy/LegacyDtb.hpp"
#endif
#include "fsc/mesh/FaceMesh.hpp"
#include "fsc/mesh/MediaPipeTopology.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <QApplication>
#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QFutureWatcher>
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
#include <QStringList>
#include <QTabBar>
#include <QTableWidget>
#include <QTabWidget>
#include <QTextEdit>
#include <QTextDocument>
#include <QTime>
#include <QTimer>
#include <QToolButton>
#include <QVariant>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef FSC_ENABLE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
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

std::string utf8FromPath(const std::filesystem::path& value) {
    const auto bytes = QString::fromStdWString(value.wstring()).toUtf8();
    return std::string(bytes.constData(), static_cast<size_t>(bytes.size()));
}

QImage previewImageFromRgb(const fsc::vision::RgbImage& image) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 3U) {
        return {};
    }
    QImage view(image.pixels.data(), image.width, image.height, image.width * 3, QImage::Format_RGB888);
    return view.copy();
}

QImage loadPreviewImage(const std::filesystem::path& path) {
    try {
        return previewImageFromRgb(fsc::vision::loadImageRgb(path));
    } catch (...) {
        return QImage(QString::fromStdWString(path.wstring()));
    }
}

QImage loadPreviewImage(const QString& path) {
    return loadPreviewImage(pathFrom(path));
}

QString defaultModelRoot() {
    const auto packaged = QApplication::applicationDirPath() + "/models/insightface/models";
    if (std::filesystem::exists(pathFrom(packaged))) {
        return packaged;
    }
    return "D:\\FSC\\model\\insightface\\models";
}

QString executableDirectory(const char* executablePath) {
#ifdef _WIN32
    std::vector<wchar_t> modulePath(32768);
    const DWORD length = GetModuleFileNameW(nullptr, modulePath.data(), static_cast<DWORD>(modulePath.size()));
    if (length > 0 && length < modulePath.size()) {
        return QFileInfo(QString::fromWCharArray(modulePath.data(), static_cast<qsizetype>(length))).absolutePath();
    }
#endif
    return QFileInfo(QString::fromLocal8Bit(executablePath)).absolutePath();
}

bool isAutomatedSmokeCommand(int argc, char** argv) {
    if (argc < 2 || argv[1] == nullptr) {
        return false;
    }
    const std::string command(argv[1]);
    return command == "--smoke" || command.ends_with("-smoke");
}

void configureDeployedQtRuntime(int argc, char** argv) {
    if (argc <= 0 || argv[0] == nullptr) {
        return;
    }
    const QString appDirectory = executableDirectory(argv[0]);
    if (QDir(appDirectory).exists("platforms")) {
        // qt.conf normally covers this, but keeping the application directory in
        // the global library paths also protects a copied portable installation.
        qputenv("QT_PLUGIN_PATH", QDir::toNativeSeparators(appDirectory).toLocal8Bit());
        QCoreApplication::addLibraryPath(appDirectory);
    }
    const bool requireWindowsSmoke =
        qEnvironmentVariable("FSC_QT_SMOKE_PLATFORM").compare("windows", Qt::CaseInsensitive) == 0;
    if (isAutomatedSmokeCommand(argc, argv) && !requireWindowsSmoke &&
        QFileInfo::exists(appDirectory + "/platforms/qminimal.dll")) {
        qputenv("QT_QPA_PLATFORM", "minimal");
    }
}

#ifdef FSC_ENABLE_ONNX
class SharedFaceAnalyzer final {
public:
    std::vector<fsc::vision::AnalyzedFace> analyze(
        const std::filesystem::path& modelRoot,
        fsc::vision::RuntimeMode runtimeMode,
        const std::filesystem::path& imagePath) {
        const auto image = fsc::vision::loadImageRgb(imagePath);
        return analyze(modelRoot, runtimeMode, image);
    }

    std::vector<fsc::vision::AnalyzedFace> analyze(
        const std::filesystem::path& modelRoot,
        fsc::vision::RuntimeMode runtimeMode,
        const fsc::vision::RgbImage& image) {
        std::lock_guard lock(mutex_);
        if (!engine_ || modelRoot_ != modelRoot || runtimeMode_ != runtimeMode) {
            engine_ = std::make_unique<fsc::vision::InsightFaceEngine>(
                fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot),
                runtimeMode);
            modelRoot_ = modelRoot;
            runtimeMode_ = runtimeMode;
        }
        return engine_->analyze(image, 0.50f, 10);
    }

private:
    std::mutex mutex_;
    std::filesystem::path modelRoot_;
    fsc::vision::RuntimeMode runtimeMode_ = fsc::vision::RuntimeMode::Auto;
    std::unique_ptr<fsc::vision::InsightFaceEngine> engine_;
};
#endif

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
    return suffix == "jpg" || suffix == "jpeg" || suffix == "png" || suffix == "bmp" ||
        suffix == "webp" || suffix == "tif" || suffix == "tiff" || suffix == "ppm";
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

QString legacyTranslatedText(const QString& key, const QString& language) {
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

using TranslationEntries = std::map<std::string, QString>;
using TranslationTable = std::map<std::string, TranslationEntries>;

const TranslationTable& uiTranslations() {
    static const TranslationTable translations = {
        {"zh", {
            {"Database", "数据库"}, {"Language", "语言"}, {"Identity Mode", "识别模式"},
            {"Strict", "严格"}, {"Balanced", "均衡"}, {"Broad", "宽松"},
            {"Overview", "概览"}, {"Library", "人脸库"}, {"People", "人物"}, {"Search", "搜索"},
            {"Camera", "摄像头"}, {"Review", "复核"}, {"Clusters", "聚类"}, {"Compare", "比对"},
            {"Runtime", "运行环境"}, {"Ready", "就绪"}, {"Browse", "浏览"}, {"Import Image", "导入图片"},
            {"Import Folder", "导入文件夹"}, {"Export CSV", "导出 CSV"}, {"Reload", "重新加载"},
            {"Filter", "筛选"}, {"Person", "人物"}, {"Tag", "标签"}, {"Include ignored", "包含已忽略"},
            {"Apply Filter", "应用筛选"}, {"Reset Filter", "重置筛选"}, {"Min quality", "最低质量"},
            {"Query", "查询"}, {"Open Database", "打开数据库"}, {"Use Library DB", "使用当前库"},
            {"Select Image", "选择图像"}, {"Top K", "返回数量"}, {"Threshold", "阈值"},
            {"Images", "图像"}, {"Image A", "图像 A"}, {"Image B", "图像 B"},
            {"Interval ms", "间隔毫秒"}, {"Process size", "处理尺寸"},
            {"Start Camera", "启动摄像头"}, {"Stop Camera", "停止摄像头"},
            {"Image", "图像"}, {"3D Landmarks", "3D 特征点"}, {"Points", "点云"}, {"Textured", "贴图"},
            {"Focus on Face", "聚焦于人脸"}, {"Full Image", "查看完整图像"}, {"View Full Image", "查看完整图像"},
            {"Generate Dense Mesh", "生成稠密网格"}, {"Save Metadata", "保存元数据"},
            {"Ignore in search", "在搜索中忽略"}, {"Append tags", "追加标签"},
            {"Apply to Selection", "应用到选中项"}, {"Selected", "选中"}, {"Batch", "批量"}, {"Activity", "活动"},
            {"Legacy", "旧版"}, {"Convert Legacy DTB", "转换旧 DTB"},
        }},
        {"ja", {
            {"Database", "データベース"}, {"Language", "言語"}, {"Identity Mode", "識別モード"},
            {"Strict", "厳格"}, {"Balanced", "標準"}, {"Broad", "広め"},
            {"Overview", "概要"}, {"Library", "顔ライブラリ"}, {"People", "人物"}, {"Search", "検索"},
            {"Camera", "カメラ"}, {"Review", "レビュー"}, {"Clusters", "クラスタ"}, {"Compare", "比較"},
            {"Runtime", "実行環境"}, {"Ready", "準備完了"}, {"Browse", "参照"}, {"Import Image", "画像を追加"},
            {"Import Folder", "フォルダーを追加"}, {"Export CSV", "CSVを書き出す"}, {"Reload", "再読み込み"},
            {"Filter", "フィルター"}, {"Person", "人物"}, {"Tag", "タグ"}, {"Include ignored", "無視を含める"},
            {"Apply Filter", "フィルターを適用"}, {"Reset Filter", "フィルターをリセット"}, {"Min quality", "最低品質"},
            {"Query", "検索画像"}, {"Open Database", "DBを開く"}, {"Use Library DB", "現在のDB"},
            {"Select Image", "画像選択"}, {"Top K", "件数"}, {"Threshold", "しきい値"},
            {"Images", "画像"}, {"Image A", "画像 A"}, {"Image B", "画像 B"},
            {"Interval ms", "間隔 ms"}, {"Process size", "処理サイズ"},
            {"Start Camera", "カメラ開始"}, {"Stop Camera", "カメラ停止"},
            {"Image", "画像"}, {"3D Landmarks", "3D ランドマーク"}, {"Points", "点"}, {"Textured", "テクスチャ"},
            {"Focus on Face", "顔にフォーカス"}, {"Full Image", "全体画像"}, {"View Full Image", "全体画像"},
            {"Generate Dense Mesh", "高密度メッシュを生成"}, {"Save Metadata", "メタデータを保存"},
            {"Ignore in search", "検索で無視"}, {"Append tags", "タグを追加"},
            {"Apply to Selection", "選択項目に適用"}, {"Selected", "選択"}, {"Batch", "一括"}, {"Activity", "操作履歴"},
            {"Legacy", "旧形式"}, {"Convert Legacy DTB", "旧DTB変換"},
        }},
        {"ko", {
            {"Database", "데이터베이스"}, {"Language", "언어"}, {"Identity Mode", "식별 모드"},
            {"Strict", "엄격"}, {"Balanced", "균형"}, {"Broad", "넓게"},
            {"Overview", "개요"}, {"Library", "얼굴 라이브러리"}, {"People", "인물"}, {"Search", "검색"},
            {"Camera", "카메라"}, {"Review", "검토"}, {"Clusters", "클러스터"}, {"Compare", "비교"},
            {"Runtime", "실행 환경"}, {"Ready", "준비됨"}, {"Browse", "찾아보기"}, {"Import Image", "이미지 가져오기"},
            {"Import Folder", "폴더 가져오기"}, {"Export CSV", "CSV 내보내기"}, {"Reload", "새로 고침"},
            {"Filter", "필터"}, {"Person", "인물"}, {"Tag", "태그"}, {"Include ignored", "무시 항목 포함"},
            {"Apply Filter", "필터 적용"}, {"Reset Filter", "필터 초기화"}, {"Min quality", "최소 품질"},
            {"Query", "검색 이미지"}, {"Open Database", "DB 열기"}, {"Use Library DB", "현재 DB 사용"},
            {"Select Image", "이미지 선택"}, {"Top K", "결과 수"}, {"Threshold", "임계값"},
            {"Images", "이미지"}, {"Image A", "이미지 A"}, {"Image B", "이미지 B"},
            {"Interval ms", "간격 ms"}, {"Process size", "처리 크기"},
            {"Start Camera", "카메라 시작"}, {"Stop Camera", "카메라 중지"},
            {"Image", "이미지"}, {"3D Landmarks", "3D 랜드마크"}, {"Points", "점"}, {"Textured", "텍스처"},
            {"Focus on Face", "얼굴에 초점"}, {"Full Image", "전체 이미지"}, {"View Full Image", "전체 이미지"},
            {"Generate Dense Mesh", "고밀도 메시 생성"}, {"Save Metadata", "메타데이터 저장"},
            {"Ignore in search", "검색에서 무시"}, {"Append tags", "태그 추가"},
            {"Apply to Selection", "선택 항목에 적용"}, {"Selected", "선택"}, {"Batch", "일괄"}, {"Activity", "작업 기록"},
            {"Legacy", "이전 형식"}, {"Convert Legacy DTB", "이전 DTB 변환"},
        }},
    };
    return translations;
}

QString translationKey(const QString& text) {
    for (const auto& [language, entries] : uiTranslations()) {
        (void)language;
        for (const auto& [key, value] : entries) {
            if (text == value) {
                return QString::fromUtf8(key.c_str());
            }
        }
    }
    return text;
}

QString translatedText(const QString& text, const QString& language) {
    const QString key = translationKey(text);
    if (language == "en") {
        return key;
    }
    const auto languageIt = uiTranslations().find(language.toStdString());
    if (languageIt == uiTranslations().end()) {
        return key;
    }
    const auto valueIt = languageIt->second.find(key.toStdString());
    return valueIt == languageIt->second.end() ? key : valueIt->second;
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
    record.fileName = utf8FromPath(imagePath.filename());
    record.sourcePath = utf8FromPath(imagePath);
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

class TexturedMeshWidget final : public QWidget {
public:
    enum class RenderMode { Points, Textured };

    explicit TexturedMeshWidget(QWidget* parent = nullptr)
        : QWidget(parent) {
        setMinimumSize(420, 320);
        setMouseTracking(true);
        setMessage("Select a face");
    }

    void setData(std::vector<std::vector<double>> points, std::vector<std::vector<double>> overlayPoints, QString message) {
        points_ = std::move(points);
        overlayPoints_ = std::move(overlayPoints);
        message_ = std::move(message);
        rebuildTextureTriangles();
        update();
    }

    void setTextureImage(const QImage& image) {
        texture_ = image.isNull() ? QImage{} : image.convertToFormat(QImage::Format_ARGB32);
        update();
    }

    void setRenderMode(RenderMode mode) {
        renderMode_ = mode;
        update();
    }

    void setView(double yaw, double pitch, double zoom = 1.0) {
        yaw_ = yaw;
        pitch_ = std::clamp(pitch, -1.5707, 1.5707);
        zoom_ = std::clamp(zoom, 0.5, 3.0);
        update();
    }

    void setMessage(QString message) {
        points_.clear();
        overlayPoints_.clear();
        textureTriangles_.clear();
        message_ = std::move(message);
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(17, 24, 39));
        painter.setRenderHint(QPainter::Antialiasing, true);
        if (points_.size() < 3) {
            painter.setPen(QColor(203, 213, 225));
            painter.drawText(rect(), Qt::AlignCenter, message_);
            return;
        }

        const Bounds bounds = computeBounds();
        const auto projected = projectRows(points_, bounds);
        bool textured = false;
        if (renderMode_ == RenderMode::Textured && !texture_.isNull() && points_.size() >= fsc::mesh::kMediaPipeFaceMeshPointCount) {
            textured = renderTexturedMesh(projected, bounds, painter);
        }
        if (textured) {
            drawOverlayLandmarks(painter, bounds);
            painter.setPen(QColor(219, 234, 254));
            painter.drawText(QPoint(14, 24), "textured face mesh");
        } else {
            drawPointMesh(painter, projected);
            painter.setPen(QColor(219, 234, 254));
            painter.drawText(QPoint(14, 24), message_.isEmpty() ? QString("3D mesh points") : message_);
        }
        painter.setPen(QColor(148, 163, 184));
        painter.drawText(QPoint(14, 46), "drag to rotate");
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            lastMouse_ = event->position();
            event->accept();
            return;
        }
        QWidget::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if ((event->buttons() & Qt::LeftButton) && !lastMouse_.isNull()) {
            const QPointF delta = event->position() - lastMouse_;
            yaw_ -= delta.x() * 0.01;
            pitch_ += delta.y() * 0.01;
            pitch_ = std::clamp(pitch_, -1.5707, 1.5707);
            lastMouse_ = event->position();
            update();
            event->accept();
            return;
        }
        QWidget::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            lastMouse_ = {};
            event->accept();
            return;
        }
        QWidget::mouseReleaseEvent(event);
    }

    void wheelEvent(QWheelEvent* event) override {
        const double steps = static_cast<double>(event->angleDelta().y()) / 120.0;
        zoom_ = std::clamp(zoom_ * (1.0 + steps * 0.08), 0.5, 3.0);
        update();
        event->accept();
    }

    void mouseDoubleClickEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            yaw_ = 0.0;
            pitch_ = -0.15;
            zoom_ = 1.0;
            update();
            event->accept();
            return;
        }
        QWidget::mouseDoubleClickEvent(event);
    }

private:
    struct Vec3 { double x = 0.0; double y = 0.0; double z = 0.0; };
    struct Bounds { Vec3 center; double span = 1.0; };
    struct Projected { QPointF screen; double depth = 0.0; Vec3 rotated; };

    static constexpr std::array<std::pair<int, int>, 63> kLandmarkEdges{{
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8},
        {8, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16},
        {17, 18}, {18, 19}, {19, 20}, {20, 21}, {22, 23}, {23, 24}, {24, 25}, {25, 26},
        {27, 28}, {28, 29}, {29, 30}, {31, 32}, {32, 33}, {33, 34}, {34, 35},
        {36, 37}, {37, 38}, {38, 39}, {39, 40}, {40, 41}, {41, 36},
        {42, 43}, {43, 44}, {44, 45}, {45, 46}, {46, 47}, {47, 42},
        {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53}, {53, 54}, {54, 55}, {55, 56},
        {56, 57}, {57, 58}, {58, 59}, {59, 48}, {60, 61}, {61, 62}, {62, 63}, {63, 64},
        {64, 65}, {65, 66}, {66, 67}, {67, 60},
    }};

    static bool valid(const std::vector<double>& point) {
        return point.size() >= 3 && std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
    }

    static double signedArea(const QPointF& a, const QPointF& b, const QPointF& c) {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    }

    Bounds computeBounds() const {
        Vec3 min{};
        Vec3 max{};
        bool first = true;
        for (const auto& point : points_) {
            if (!valid(point)) {
                continue;
            }
            const Vec3 value{point[0], point[1], point[2]};
            if (first) {
                min = max = value;
                first = false;
            } else {
                min.x = std::min(min.x, value.x); min.y = std::min(min.y, value.y); min.z = std::min(min.z, value.z);
                max.x = std::max(max.x, value.x); max.y = std::max(max.y, value.y); max.z = std::max(max.z, value.z);
            }
        }
        return {{(min.x + max.x) * 0.5, (min.y + max.y) * 0.5, (min.z + max.z) * 0.5},
                std::max({max.x - min.x, max.y - min.y, max.z - min.z, 1.0})};
    }

    Projected project(const Vec3& source, const Bounds& bounds) const {
        const double x = (source.x - bounds.center.x) / bounds.span;
        const double y = (source.y - bounds.center.y) / bounds.span;
        const double z = (source.z - bounds.center.z) / bounds.span;
        const double cosYaw = std::cos(yaw_);
        const double sinYaw = std::sin(yaw_);
        const double cosPitch = std::cos(pitch_);
        const double sinPitch = std::sin(pitch_);
        const double xYaw = x * cosYaw + z * sinYaw;
        const double zYaw = -x * sinYaw + z * cosYaw;
        const double yPitch = y * cosPitch - zYaw * sinPitch;
        const double zPitch = y * sinPitch + zYaw * cosPitch;
        const double scale = std::min(width(), height()) * 0.72 * zoom_;
        return {{width() * 0.5 + xYaw * scale, height() * 0.5 + yPitch * scale}, zPitch, {xYaw, yPitch, zPitch}};
    }

    std::vector<Projected> projectRows(const std::vector<std::vector<double>>& rows, const Bounds& bounds) const {
        std::vector<Projected> projected;
        projected.reserve(rows.size());
        for (const auto& point : rows) {
            projected.push_back(valid(point) ? project({point[0], point[1], point[2]}, bounds) : Projected{});
        }
        return projected;
    }

    static QRgb sampleTexture(const QImage& image, double x, double y) {
        const double clampedX = std::clamp(x, 0.0, static_cast<double>(std::max(0, image.width() - 1)));
        const double clampedY = std::clamp(y, 0.0, static_cast<double>(std::max(0, image.height() - 1)));
        const int x0 = static_cast<int>(std::floor(clampedX));
        const int y0 = static_cast<int>(std::floor(clampedY));
        const int x1 = std::min(image.width() - 1, x0 + 1);
        const int y1 = std::min(image.height() - 1, y0 + 1);
        const double tx = clampedX - x0;
        const double ty = clampedY - y0;
        const auto blend = [tx, ty](int a, int b, int c, int d) {
            const double top = a * (1.0 - tx) + b * tx;
            const double bottom = c * (1.0 - tx) + d * tx;
            return static_cast<int>(std::lround(top * (1.0 - ty) + bottom * ty));
        };
        const QRgb p00 = image.pixel(x0, y0);
        const QRgb p10 = image.pixel(x1, y0);
        const QRgb p01 = image.pixel(x0, y1);
        const QRgb p11 = image.pixel(x1, y1);
        return qRgb(blend(qRed(p00), qRed(p10), qRed(p01), qRed(p11)),
                    blend(qGreen(p00), qGreen(p10), qGreen(p01), qGreen(p11)),
                    blend(qBlue(p00), qBlue(p10), qBlue(p01), qBlue(p11)));
    }

    static bool backFacing(const Vec3& a, const Vec3& b, const Vec3& c) {
        const Vec3 u{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3 v{c.x - a.x, c.y - a.y, c.z - a.z};
        const Vec3 normal{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
        const double length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        return length > 1e-8 && normal.z / length > 0.10;
    }

    using TextureTriangle = std::array<std::uint16_t, 3>;

    struct DelaunayVertex {
        QPointF point;
        int meshIndex = -1;
    };

    struct DelaunayTriangle {
        int a = -1;
        int b = -1;
        int c = -1;
    };

    static std::array<std::uint16_t, 3> canonicalTriangle(TextureTriangle triangle) {
        std::sort(triangle.begin(), triangle.end());
        return triangle;
    }

    static double orientation(const QPointF& a, const QPointF& b, const QPointF& c) {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    }

    static bool pointInCircumcircle(
        const QPointF& point,
        const QPointF& a,
        const QPointF& b,
        const QPointF& c) {
        const double ax = a.x() - point.x();
        const double ay = a.y() - point.y();
        const double bx = b.x() - point.x();
        const double by = b.y() - point.y();
        const double cx = c.x() - point.x();
        const double cy = c.y() - point.y();
        const double determinant =
            (ax * ax + ay * ay) * (bx * cy - cx * by) -
            (bx * bx + by * by) * (ax * cy - cx * ay) +
            (cx * cx + cy * cy) * (ax * by - bx * ay);
        const double winding = orientation(a, b, c);
        if (std::abs(winding) <= 1e-9) {
            return false;
        }
        return winding > 0.0 ? determinant > 1e-7 : determinant < -1e-7;
    }

    bool hasUsableMeshIndex(int index) const {
        return index >= 0 && index < static_cast<int>(points_.size()) && valid(points_[index]);
    }

    TextureTriangle orientTextureTriangle(TextureTriangle triangle) const {
        if (!hasUsableMeshIndex(triangle[0]) || !hasUsableMeshIndex(triangle[1]) || !hasUsableMeshIndex(triangle[2])) {
            return triangle;
        }
        const auto& a = points_[triangle[0]];
        const auto& b = points_[triangle[1]];
        const auto& c = points_[triangle[2]];
        const Vec3 u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const Vec3 v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double normalZ = u.x * v.y - u.y * v.x;
        // The base MediaPipe topology faces the viewer with a negative Z normal
        // in this renderer. Keep generated eye triangles on the same side.
        if (normalZ > 0.0) {
            std::swap(triangle[1], triangle[2]);
        }
        return triangle;
    }

    std::vector<TextureTriangle> localDelaunayTriangles(const std::vector<int>& indexes) const {
        std::vector<DelaunayVertex> vertices;
        vertices.reserve(indexes.size() + 3);
        for (const int index : indexes) {
            if (!hasUsableMeshIndex(index)) {
                continue;
            }
            const QPointF source(points_[index][0], points_[index][1]);
            bool duplicate = false;
            for (const auto& existing : vertices) {
                const double dx = existing.point.x() - source.x();
                const double dy = existing.point.y() - source.y();
                if (dx * dx + dy * dy < 1e-6) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                vertices.push_back({source, index});
            }
        }
        if (vertices.size() < 3) {
            return {};
        }

        double minX = vertices.front().point.x();
        double maxX = minX;
        double minY = vertices.front().point.y();
        double maxY = minY;
        for (const auto& vertex : vertices) {
            minX = std::min(minX, vertex.point.x());
            maxX = std::max(maxX, vertex.point.x());
            minY = std::min(minY, vertex.point.y());
            maxY = std::max(maxY, vertex.point.y());
        }
        const double span = std::max(maxX - minX, maxY - minY);
        if (span < 2.0) {
            return {};
        }

        const double centerX = (minX + maxX) * 0.5;
        const double centerY = (minY + maxY) * 0.5;
        const int realVertexCount = static_cast<int>(vertices.size());
        const double extent = span * 32.0;
        vertices.push_back({QPointF(centerX - extent, centerY - extent), -1});
        vertices.push_back({QPointF(centerX, centerY + extent), -1});
        vertices.push_back({QPointF(centerX + extent, centerY - extent), -1});

        std::vector<DelaunayTriangle> triangles{{realVertexCount, realVertexCount + 1, realVertexCount + 2}};
        for (int pointIndex = 0; pointIndex < realVertexCount; ++pointIndex) {
            std::vector<int> invalid;
            std::map<std::pair<int, int>, int> edgeCounts;
            for (int triangleIndex = 0; triangleIndex < static_cast<int>(triangles.size()); ++triangleIndex) {
                const auto& triangle = triangles[triangleIndex];
                if (!pointInCircumcircle(
                        vertices[pointIndex].point,
                        vertices[triangle.a].point,
                        vertices[triangle.b].point,
                        vertices[triangle.c].point)) {
                    continue;
                }
                invalid.push_back(triangleIndex);
                for (const auto [start, end] : {
                         std::pair{triangle.a, triangle.b},
                         std::pair{triangle.b, triangle.c},
                         std::pair{triangle.c, triangle.a},
                     }) {
                    edgeCounts[{std::min(start, end), std::max(start, end)}] += 1;
                }
            }
            if (invalid.empty()) {
                continue;
            }
            std::sort(invalid.rbegin(), invalid.rend());
            for (const int triangleIndex : invalid) {
                triangles.erase(triangles.begin() + triangleIndex);
            }
            for (const auto& [edge, count] : edgeCounts) {
                if (count == 1) {
                    triangles.push_back({edge.first, edge.second, pointIndex});
                }
            }
        }

        std::vector<TextureTriangle> output;
        output.reserve(triangles.size());
        for (const auto& triangle : triangles) {
            if (triangle.a >= realVertexCount || triangle.b >= realVertexCount || triangle.c >= realVertexCount) {
                continue;
            }
            const TextureTriangle mapped{
                static_cast<std::uint16_t>(vertices[triangle.a].meshIndex),
                static_cast<std::uint16_t>(vertices[triangle.b].meshIndex),
                static_cast<std::uint16_t>(vertices[triangle.c].meshIndex),
            };
            const QPointF& a = vertices[triangle.a].point;
            const QPointF& b = vertices[triangle.b].point;
            const QPointF& c = vertices[triangle.c].point;
            if (std::abs(orientation(a, b, c)) * 0.5 >= 0.25) {
                output.push_back(orientTextureTriangle(mapped));
            }
        }
        return output;
    }

    void appendEyeRegionTriangles(
        const std::array<int, 16>& eyeLoop,
        const std::array<int, 5>& irisIndexes,
        std::set<TextureTriangle>& seen) {
        std::vector<int> indexes;
        indexes.reserve(eyeLoop.size() + irisIndexes.size());
        for (const int index : eyeLoop) {
            if (hasUsableMeshIndex(index)) {
                indexes.push_back(index);
            }
        }
        for (const int index : irisIndexes) {
            if (hasUsableMeshIndex(index)) {
                indexes.push_back(index);
            }
        }
        if (indexes.size() < 6) {
            return;
        }
        const auto append = [this, &seen, thisTriangles = &textureTriangles_](TextureTriangle triangle) {
            if (!hasUsableMeshIndex(triangle[0]) || !hasUsableMeshIndex(triangle[1]) || !hasUsableMeshIndex(triangle[2]) ||
                triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[0] == triangle[2]) {
                return;
            }
            const QPointF a(points_[triangle[0]][0], points_[triangle[0]][1]);
            const QPointF b(points_[triangle[1]][0], points_[triangle[1]][1]);
            const QPointF c(points_[triangle[2]][0], points_[triangle[2]][1]);
            if (std::abs(orientation(a, b, c)) * 0.5 < 0.25) {
                return;
            }
            triangle = orientTextureTriangle(triangle);
            if (seen.insert(canonicalTriangle(triangle)).second) {
                thisTriangles->push_back(triangle);
            }
        };

        if (std::all_of(irisIndexes.begin(), irisIndexes.end(), [this](int index) { return hasUsableMeshIndex(index); })) {
            for (size_t position = 1; position < irisIndexes.size(); ++position) {
                const size_t next = position == irisIndexes.size() - 1 ? 1 : position + 1;
                append({
                    static_cast<std::uint16_t>(irisIndexes[0]),
                    static_cast<std::uint16_t>(irisIndexes[position]),
                    static_cast<std::uint16_t>(irisIndexes[next]),
                });
            }
        }
        for (const auto& triangle : localDelaunayTriangles(indexes)) {
            append(triangle);
        }
    }

    void rebuildTextureTriangles() {
        textureTriangles_.clear();
        if (points_.size() < fsc::mesh::kMediaPipeFaceMeshPointCount) {
            return;
        }
        textureTriangles_.reserve(fsc::mesh::kMediaPipeFaceMeshTriangles.size() + 64);
        std::set<TextureTriangle> seen;
        for (const auto& triangle : fsc::mesh::kMediaPipeFaceMeshTriangles) {
            const TextureTriangle copied{triangle[0], triangle[1], triangle[2]};
            textureTriangles_.push_back(copied);
            seen.insert(canonicalTriangle(copied));
        }
        appendEyeRegionTriangles(
            {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246},
            {468, 469, 470, 471, 472},
            seen);
        appendEyeRegionTriangles(
            {263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466},
            {473, 474, 475, 476, 477},
            seen);
    }

    bool renderTexturedMesh(const std::vector<Projected>& projected, const Bounds&, QPainter& painter) const {
        const int canvasWidth = std::max(1, width());
        const int canvasHeight = std::max(1, height());
        QImage canvas(canvasWidth, canvasHeight, QImage::Format_ARGB32);
        canvas.fill(QColor(17, 24, 39));
        std::vector<float> zBuffer(static_cast<size_t>(canvasWidth) * canvasHeight, std::numeric_limits<float>::infinity());
        bool rendered = false;
        for (const auto& triangle : textureTriangles_) {
            const int ia = triangle[0]; const int ib = triangle[1]; const int ic = triangle[2];
            if (ic >= static_cast<int>(points_.size()) || !valid(points_[ia]) || !valid(points_[ib]) || !valid(points_[ic])) {
                continue;
            }
            const auto& a = projected[ia]; const auto& b = projected[ib]; const auto& c = projected[ic];
            const double area = signedArea(a.screen, b.screen, c.screen);
            if (std::abs(area) < 0.5) {
                continue;
            }
            const QPointF sourceA(points_[ia][0], points_[ia][1]);
            const QPointF sourceB(points_[ib][0], points_[ib][1]);
            const QPointF sourceC(points_[ic][0], points_[ic][1]);
            if (std::abs(signedArea(sourceA, sourceB, sourceC)) < 0.5) {
                continue;
            }
            const int minX = std::max(0, static_cast<int>(std::floor(std::min({a.screen.x(), b.screen.x(), c.screen.x()}))));
            const int maxX = std::min(canvasWidth - 1, static_cast<int>(std::ceil(std::max({a.screen.x(), b.screen.x(), c.screen.x()}))));
            const int minY = std::max(0, static_cast<int>(std::floor(std::min({a.screen.y(), b.screen.y(), c.screen.y()}))));
            const int maxY = std::min(canvasHeight - 1, static_cast<int>(std::ceil(std::max({a.screen.y(), b.screen.y(), c.screen.y()}))));
            if (maxX < minX || maxY < minY) {
                continue;
            }
            const bool isBackFacing = backFacing(a.rotated, b.rotated, c.rotated);
            const int shade = std::clamp(static_cast<int>(std::lround(30.0 + (a.depth + b.depth + c.depth) * 16.0 / 3.0)), 18, 42);
            for (int y = minY; y <= maxY; ++y) {
                auto* scanLine = reinterpret_cast<QRgb*>(canvas.scanLine(y));
                for (int x = minX; x <= maxX; ++x) {
                    const QPointF pixel(x + 0.5, y + 0.5);
                    const double w0 = signedArea(pixel, b.screen, c.screen) / area;
                    const double w1 = signedArea(pixel, c.screen, a.screen) / area;
                    const double w2 = 1.0 - w0 - w1;
                    if (w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4) {
                        continue;
                    }
                    const float depth = static_cast<float>(w0 * a.depth + w1 * b.depth + w2 * c.depth);
                    const size_t pixelIndex = static_cast<size_t>(y) * canvasWidth + x;
                    if (depth >= zBuffer[pixelIndex]) {
                        continue;
                    }
                    zBuffer[pixelIndex] = depth;
                    if (isBackFacing) {
                        scanLine[x] = qRgb(shade / 2, shade / 2 + 2, shade);
                    } else {
                        const double sourceX = w0 * sourceA.x() + w1 * sourceB.x() + w2 * sourceC.x();
                        const double sourceY = w0 * sourceA.y() + w1 * sourceB.y() + w2 * sourceC.y();
                        scanLine[x] = sampleTexture(texture_, sourceX, sourceY);
                    }
                    rendered = true;
                }
            }
        }
        if (rendered) {
            painter.drawImage(0, 0, canvas);
        }
        return rendered;
    }

    void drawPointMesh(QPainter& painter, const std::vector<Projected>& projected) const {
        std::vector<int> order;
        order.reserve(projected.size());
        for (int index = 0; index < static_cast<int>(projected.size()); ++index) {
            if (valid(points_[index])) order.push_back(index);
        }
        std::sort(order.begin(), order.end(), [&projected](int left, int right) { return projected[left].depth < projected[right].depth; });
        double minDepth = 0.0; double maxDepth = 1.0;
        if (!order.empty()) {
            minDepth = projected[order.front()].depth;
            maxDepth = projected[order.back()].depth;
        }
        const double span = std::max(1e-6, maxDepth - minDepth);
        for (const int index : order) {
            const double ratio = (projected[index].depth - minDepth) / span;
            const double radius = 2.2 + 2.0 * ratio;
            painter.setPen(QPen(QColor(15, 23, 42), 1));
            painter.setBrush(QColor::fromHsvF(0.55, 0.30 + 0.35 * ratio, 0.82 + 0.15 * ratio));
            painter.drawEllipse(projected[index].screen, radius, radius);
        }
    }

    void drawOverlayLandmarks(QPainter& painter, const Bounds& bounds) const {
        if (overlayPoints_.empty() || points_.size() < 3) {
            return;
        }
        std::vector<Vec3> mapped;
        mapped.reserve(overlayPoints_.size());
        for (const auto& overlay : overlayPoints_) {
            if (!valid(overlay)) {
                mapped.push_back({});
                continue;
            }
            std::array<std::pair<double, int>, 4> nearest{{
                {std::numeric_limits<double>::infinity(), -1}, {std::numeric_limits<double>::infinity(), -1},
                {std::numeric_limits<double>::infinity(), -1}, {std::numeric_limits<double>::infinity(), -1},
            }};
            for (int index = 0; index < static_cast<int>(points_.size()); ++index) {
                if (!valid(points_[index])) continue;
                const double dx = points_[index][0] - overlay[0];
                const double dy = points_[index][1] - overlay[1];
                const double distance = dx * dx + dy * dy;
                if (distance < nearest.back().first) {
                    nearest.back() = {distance, index};
                    std::sort(nearest.begin(), nearest.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
                }
            }
            double weightTotal = 0.0;
            double z = 0.0;
            for (const auto& [distance, index] : nearest) {
                if (index < 0) continue;
                const double weight = 1.0 / (distance + 1e-3);
                z += points_[index][2] * weight;
                weightTotal += weight;
            }
            mapped.push_back({overlay[0], overlay[1], weightTotal > 0.0 ? z / weightTotal : overlay[2]});
        }
        std::vector<Projected> projected;
        projected.reserve(mapped.size());
        for (const auto& point : mapped) projected.push_back(project(point, bounds));
        painter.setPen(QPen(QColor(251, 191, 36, 185), 1));
        for (const auto& [start, end] : kLandmarkEdges) {
            if (start < static_cast<int>(projected.size()) && end < static_cast<int>(projected.size())) {
                painter.drawLine(projected[start].screen, projected[end].screen);
            }
        }
        std::vector<int> order(projected.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&projected](int left, int right) { return projected[left].depth < projected[right].depth; });
        for (const int index : order) {
            painter.setPen(QPen(QColor(15, 23, 42, 190), 1));
            painter.setBrush(QColor(34, 211, 238, 220));
            painter.drawEllipse(projected[index].screen, 1.45, 1.45);
        }
    }

    std::vector<std::vector<double>> points_;
    std::vector<std::vector<double>> overlayPoints_;
    std::vector<TextureTriangle> textureTriangles_;
    QImage texture_;
    QString message_;
    RenderMode renderMode_ = RenderMode::Points;
    double yaw_ = 0.0;
    double pitch_ = -0.15;
    double zoom_ = 1.0;
    QPointF lastMouse_;
};

class FaceSelectionPreview final : public QLabel {
public:
    explicit FaceSelectionPreview(QWidget* parent = nullptr)
        : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(180, 180);
        setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        setText("Select image");
        setMouseTracking(true);
    }

    void setImagePath(QString path) {
        const bool changed = imagePath_ != path;
        imagePath_ = std::move(path);
        if (changed) {
            selectedIndex_ = 0;
            focusOnFace_ = false;
            faces_.clear();
        }
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
        QImage source = loadPreviewImage(imagePath_);
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
        setWindowTitle("FSC Studio");
        resize(1180, 760);
        buildUi();
    }

    void openDatabasePath(const QString& path) {
        openDatabase(path);
    }

#ifdef FSC_ENABLE_ONNX
    void startLibraryImportSmoke(
        const QString& databasePath,
        const QString& modelRoot,
        const QString& imagePath,
        const QString& runtimeMode) {
        openDatabasePath(databasePath);
        if (modelRootEdit_ != nullptr) {
            modelRootEdit_->setText(modelRoot);
        }
        if (runtimeModeCombo_ != nullptr) {
            const int modeIndex = runtimeModeCombo_->findData(runtimeMode.toLower());
            runtimeModeCombo_->setCurrentIndex(modeIndex >= 0 ? modeIndex : 1);
        }
        libraryImportSmokeStarted_ = true;
        startLibraryImport({imagePath});
    }

    [[nodiscard]] bool libraryImportSmokeFinished() const noexcept {
        return libraryImportSmokeStarted_ && !libraryImportActive_;
    }

    void startSearchQuerySmoke(const QString& modelRoot, const QString& imagePath, const QString& runtimeMode) {
        searchQuerySmokeMode_ = true;
        if (modelRootEdit_ != nullptr) {
            modelRootEdit_->setText(modelRoot);
        }
        if (runtimeModeCombo_ != nullptr) {
            const int modeIndex = runtimeModeCombo_->findData(runtimeMode.toLower());
            runtimeModeCombo_->setCurrentIndex(modeIndex >= 0 ? modeIndex : 1);
        }
        if (searchImageEdit_ != nullptr) {
            searchImageEdit_->setText(imagePath);
        }
        analyzeSearchImage();
    }

    [[nodiscard]] bool searchQuerySmokeFinished() const noexcept {
        return !searchQueryAnalysisActive_;
    }

    [[nodiscard]] int searchQuerySmokeFaceCount() const noexcept {
        return static_cast<int>(searchQueryFaces_.size());
    }

    void startCompareSmoke(
        const QString& modelRoot,
        const QString& imageA,
        const QString& imageB,
        const QString& runtimeMode) {
        if (modelRootEdit_ != nullptr) {
            modelRootEdit_->setText(modelRoot);
        }
        if (runtimeModeCombo_ != nullptr) {
            const int modeIndex = runtimeModeCombo_->findData(runtimeMode.toLower());
            runtimeModeCombo_->setCurrentIndex(modeIndex >= 0 ? modeIndex : 1);
        }
        compareImageAEdit_->setText(imageA);
        compareImageBEdit_->setText(imageB);
        comparePreviewA_->setImagePath(imageA);
        comparePreviewB_->setImagePath(imageB);
        compareFacesA_.clear();
        compareFacesB_.clear();
        analyzeCompareImage('a');
        analyzeCompareImage('b');
    }

    [[nodiscard]] bool compareSmokeFinished() const noexcept {
        return !compareAnalysisActiveA_ && !compareAnalysisActiveB_;
    }

    [[nodiscard]] bool compareSmokeReady() const noexcept {
        return !compareFacesA_.empty() && !compareFacesB_.empty() &&
            compareFacesA_.front().embedding.size() == compareFacesB_.front().embedding.size();
    }

#ifdef FSC_ENABLE_OPENCV
    void startCameraFrameSmoke(const QString& modelRoot, const QString& imagePath, const QString& runtimeMode) {
        if (modelRootEdit_ != nullptr) {
            modelRootEdit_->setText(modelRoot);
        }
        if (runtimeModeCombo_ != nullptr) {
            const int modeIndex = runtimeModeCombo_->findData(runtimeMode.toLower());
            runtimeModeCombo_->setCurrentIndex(modeIndex >= 0 ? modeIndex : 1);
        }
        const auto image = fsc::vision::loadImageRgb(pathFrom(imagePath));
        cv::Mat rgb(image.height, image.width, CV_8UC3, const_cast<std::uint8_t*>(image.pixels.data()));
        cv::cvtColor(rgb, lastCameraFrame_, cv::COLOR_RGB2BGR);
        ++cameraSessionGeneration_;
        cameraAnalyzeBusy_ = false;
        identifyCameraFrame();
    }

    [[nodiscard]] bool cameraFrameSmokeFinished() const noexcept {
        return !cameraAnalyzeBusy_;
    }

    [[nodiscard]] bool cameraFrameSmokeReady() const noexcept {
        return cameraResultTable_ != nullptr && cameraResultTable_->rowCount() > 0 && !latestCameraFaces_.empty();
    }

    void startCameraLiveSmoke(const QString& modelRoot, int cameraIndex, const QString& runtimeMode) {
        if (modelRootEdit_ != nullptr) {
            modelRootEdit_->setText(modelRoot);
        }
        if (runtimeModeCombo_ != nullptr) {
            const int modeIndex = runtimeModeCombo_->findData(runtimeMode.toLower());
            runtimeModeCombo_->setCurrentIndex(modeIndex >= 0 ? modeIndex : 1);
        }
        cameraIndexSpin_->setValue(cameraIndex);
        cameraCapturedFrameCount_ = 0;
        cameraCompletedAnalysisCount_ = 0;
        startCamera();
    }

    [[nodiscard]] bool cameraLiveSmokeReady() const noexcept {
        return cameraCapturedFrameCount_ >= 20 && cameraCompletedAnalysisCount_ >= 1;
    }

    void stopCameraLiveSmoke() {
        stopCamera();
    }
#endif
#endif

private:
    struct CameraResultActionRow {
        int faceIndex = -1;
        int64_t identityPersonId = 0;
        QString identityName;
        QString decision;
        int64_t evidenceFaceId = 0;
        int64_t hitFaceId = 0;
        double hitCosine = -2.0;
    };

    struct SearchQueryAnalysisResult {
        int generation = 0;
        QString imagePath;
        std::vector<fsc::vision::AnalyzedFace> faces;
        QString error;
    };

    struct CompareImageAnalysisResult {
        char slot = 'a';
        int generation = 0;
        QString imagePath;
        std::vector<fsc::vision::AnalyzedFace> faces;
        QString error;
    };

#if defined(FSC_ENABLE_OPENCV) && defined(FSC_ENABLE_ONNX)
    struct CameraFaceAnalysisResult {
        fsc::core::IdentityResult identity;
        std::vector<fsc::core::SearchHit> hits;
    };

    struct CameraFrameAnalysisResult {
        uint64_t token = 0;
        uint64_t session = 0;
        QString databasePath;
        cv::Mat frame;
        std::vector<fsc::vision::AnalyzedFace> faces;
        std::vector<CameraFaceAnalysisResult> matches;
        double threshold = 0.35;
        int topK = 3;
        int processSize = 640;
        QString error;
    };
#endif

#ifdef FSC_ENABLE_ONNX
    struct LibraryImportProgressEvent {
        int current = 0;
        int total = 0;
        QString imagePath;
        QString message;
        bool preview = false;
    };

    struct LibraryImportTaskState {
        std::mutex mutex;
        std::deque<LibraryImportProgressEvent> events;
    };

    struct LibraryImportSummary {
        uint64_t token = 0;
        QString databasePath;
        int imagesTotal = 0;
        int facesSaved = 0;
        int imagesWithoutFaces = 0;
        int failedImages = 0;
        int lowQualityFaces = 0;
        int duplicateImages = 0;
        double qualityTotal = 0.0;
        QString error;

        [[nodiscard]] double averageQuality() const noexcept {
            return facesSaved > 0 ? qualityTotal / static_cast<double>(facesSaved) : 0.0;
        }
    };
#endif

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

    void selectMainTab(const QString& key) {
        for (int index = 0; index < static_cast<int>(tabKeys_.size()); ++index) {
            if (tabKeys_[static_cast<size_t>(index)] == key && sidebar_ != nullptr) {
                sidebar_->setCurrentRow(index);
                return;
            }
        }
    }

    void applyTranslatedStaticText() {
        const auto translateText = [this](auto* widget) {
            const QString current = widget->text();
            if (!current.isEmpty()) {
                widget->setText(trUi(current));
            }
        };
        for (auto* widget : findChildren<QLabel*>()) {
            translateText(widget);
        }
        for (auto* widget : findChildren<QPushButton*>()) {
            translateText(widget);
        }
        for (auto* widget : findChildren<QToolButton*>()) {
            translateText(widget);
        }
        for (auto* widget : findChildren<QCheckBox*>()) {
            translateText(widget);
        }
        for (auto* widget : findChildren<QGroupBox*>()) {
            const QString current = widget->title();
            if (!current.isEmpty()) {
                widget->setTitle(trUi(current));
            }
        }
        for (auto* widget : findChildren<QTabWidget*>()) {
            for (int index = 0; index < widget->count(); ++index) {
                widget->setTabText(index, trUi(widget->tabText(index)));
            }
        }
        for (auto* widget : findChildren<QComboBox*>()) {
            if (widget == languageCombo_) {
                continue;
            }
            for (int index = 0; index < widget->count(); ++index) {
                widget->setItemText(index, trUi(widget->itemText(index)));
            }
        }
        for (auto* table : findChildren<QTableWidget*>()) {
            for (int index = 0; index < table->columnCount(); ++index) {
                if (auto* header = table->horizontalHeaderItem(index)) {
                    header->setText(trUi(header->text()));
                }
            }
        }
    }

    void applyLanguage() {
        setWindowTitle(trUi("FSC Studio"));
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
        applyTranslatedStaticText();
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
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(8);

        auto* workspace = new QGroupBox("Workspace", page);
        auto* workspaceLayout = new QGridLayout(workspace);
        overviewDatabasePathEdit_ = new QLineEdit(workspace);
        overviewDatabasePathEdit_->setReadOnly(true);
        overviewDatabasePathEdit_->setPlaceholderText("No database loaded");
        auto* newButton = new QPushButton("New Database", workspace);
        auto* openButton = new QPushButton("Open Database", workspace);
        auto* importButton = new QPushButton("Import Image", workspace);
        auto* folderButton = new QPushButton("Import Folder", workspace);
        auto* refreshButton = new QPushButton("Refresh", workspace);
        workspaceLayout->addWidget(new QLabel("Database", workspace), 0, 0);
        workspaceLayout->addWidget(overviewDatabasePathEdit_, 0, 1, 1, 5);
        workspaceLayout->addWidget(newButton, 1, 0);
        workspaceLayout->addWidget(openButton, 1, 1);
        workspaceLayout->addWidget(importButton, 1, 2);
        workspaceLayout->addWidget(folderButton, 1, 3);
        workspaceLayout->addWidget(refreshButton, 1, 4);
        layout->addWidget(workspace);

        auto* upper = new QHBoxLayout();
        auto* metricsBox = new QGroupBox("Database Metrics", page);
        auto* metricsLayout = new QVBoxLayout(metricsBox);
        overviewMetricsTable_ = new QTableWidget(0, 2, metricsBox);
        overviewMetricsTable_->setHorizontalHeaderLabels({"Metric", "Value"});
        fitTable(overviewMetricsTable_);
        metricsLayout->addWidget(overviewMetricsTable_);
        upper->addWidget(metricsBox, 1);

        auto* attentionBox = new QGroupBox("Attention", page);
        auto* attentionLayout = new QVBoxLayout(attentionBox);
        overviewAttentionTable_ = new QTableWidget(0, 2, attentionBox);
        overviewAttentionTable_->setHorizontalHeaderLabels({"Queue", "Count"});
        fitTable(overviewAttentionTable_);
        attentionLayout->addWidget(overviewAttentionTable_);
        auto* attentionActions = new QHBoxLayout();
        auto* reviewButton = new QPushButton("Review Queue", attentionBox);
        auto* peopleButton = new QPushButton("People", attentionBox);
        auto* searchButton = new QPushButton("Search", attentionBox);
        auto* clustersButton = new QPushButton("Clusters", attentionBox);
        attentionActions->addWidget(reviewButton);
        attentionActions->addWidget(peopleButton);
        attentionActions->addWidget(searchButton);
        attentionActions->addWidget(clustersButton);
        attentionLayout->addLayout(attentionActions);
        upper->addWidget(attentionBox, 1);
        layout->addLayout(upper, 1);

        auto* lower = new QHBoxLayout();
        auto* peopleBox = new QGroupBox("Top People", page);
        auto* peopleLayout = new QVBoxLayout(peopleBox);
        overviewPeopleTable_ = new QTableWidget(0, 3, peopleBox);
        overviewPeopleTable_->setHorizontalHeaderLabels({"Person", "Faces", "Review"});
        fitTable(overviewPeopleTable_);
        peopleLayout->addWidget(overviewPeopleTable_);
        lower->addWidget(peopleBox, 1);

        auto* tagsBox = new QGroupBox("Top Tags", page);
        auto* tagsLayout = new QVBoxLayout(tagsBox);
        overviewTagsTable_ = new QTableWidget(0, 2, tagsBox);
        overviewTagsTable_->setHorizontalHeaderLabels({"Tag", "Faces"});
        fitTable(overviewTagsTable_);
        tagsLayout->addWidget(overviewTagsTable_);
        lower->addWidget(tagsBox, 1);
        layout->addLayout(lower, 1);
        addMainTab(page, "Overview");

        connect(newButton, &QPushButton::clicked, this, [this] { createDatabase(); });
        connect(openButton, &QPushButton::clicked, this, [this] { chooseDatabase(); });
        connect(importButton, &QPushButton::clicked, this, [this] { importImage(); });
        connect(folderButton, &QPushButton::clicked, this, [this] { importFolder(); });
        connect(refreshButton, &QPushButton::clicked, this, [this] { reloadAll(); });
        connect(reviewButton, &QPushButton::clicked, this, [this] { selectMainTab("Review"); });
        connect(peopleButton, &QPushButton::clicked, this, [this] { selectMainTab("People"); });
        connect(searchButton, &QPushButton::clicked, this, [this] { selectMainTab("Search"); });
        connect(clustersButton, &QPushButton::clicked, this, [this] { selectMainTab("Clusters"); });
    }

    void buildLibraryTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* title = new QLabel("Library", page);
        title->setObjectName("PageTitle");
        layout->addWidget(title);
        auto* mainSplitter = new QSplitter(Qt::Horizontal, page);
        auto* leftPanel = new QWidget(mainSplitter);
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(0, 0, 0, 0);
        leftLayout->setSpacing(6);

        auto* controls = new QWidget(leftPanel);
        auto* form = new QFormLayout(controls);
        modelRootEdit_ = new QLineEdit(controls);
        modelRootEdit_->setText(defaultModelRoot());
        auto* modelButton = new QPushButton("Browse", controls);
        libraryImportImagesButton_ = new QPushButton("Add Images", controls);
        libraryImportImagesButton_->setObjectName("LibraryAddImages");
        libraryImportFolderButton_ = new QPushButton("Add Folder", controls);
        libraryImportFolderButton_->setObjectName("LibraryAddFolder");
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
        imageRowLayout->addWidget(libraryImportImagesButton_);
        imageRowLayout->addWidget(libraryImportFolderButton_);
        imageRowLayout->addWidget(reloadLibraryButton);
        imageRowLayout->addWidget(exportButton);
        imageRowLayout->addStretch(1);
        form->addRow("Models", modelRow);
        form->addRow("Import", imageRow);
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

        libraryTable_ = new QTableWidget(leftPanel);
        libraryTable_->setColumnCount(9);
        libraryTable_->setHorizontalHeaderLabels({"ID", "Name", "Person", "Tags", "Review", "Ignored", "Dupes", "Quality", "Source"});
        fitTable(libraryTable_);
        libraryTable_->setSelectionMode(QAbstractItemView::ExtendedSelection);
        leftLayout->addWidget(libraryTable_, 1);

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
        libraryMeshModeCombo_ = new QComboBox(denseControls);
        libraryMeshModeCombo_->addItem("Points", "points");
        libraryMeshModeCombo_->addItem("Textured", "textured");
        auto* generateLibraryMeshButton = new QPushButton("Generate Dense Mesh", denseControls);
        libraryMeshStatusLabel_ = new QLabel("Select a face", denseControls);
        denseControlsLayout->addWidget(libraryMeshOverlayCheck_);
        denseControlsLayout->addWidget(libraryMeshModeCombo_);
        denseControlsLayout->addWidget(generateLibraryMeshButton);
        denseControlsLayout->addWidget(libraryMeshStatusLabel_, 1);
        libraryDenseMeshView_ = new TexturedMeshWidget(denseTab);
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
        libraryActivityLog_->document()->setMaximumBlockCount(1000);
        activityLayout->addWidget(libraryProgressBar_);
        activityLayout->addWidget(libraryActivityLog_, 1);
        metadataTabs->addTab(activityTab, "Activity");
        visualLayout->addWidget(metadataTabs, 1);

        mainSplitter->addWidget(leftPanel);
        mainSplitter->addWidget(visualPanel);
        mainSplitter->setStretchFactor(0, 3);
        mainSplitter->setStretchFactor(1, 2);
        layout->addWidget(mainSplitter, 1);
        libraryImportProgressTimer_ = new QTimer(this);
        libraryImportProgressTimer_->setInterval(50);
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
                if (assignFaceSpin_ != nullptr) {
                    assignFaceSpin_->setValue(faceId);
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
        connect(libraryImportImagesButton_, &QPushButton::clicked, this, [this] { importImage(); });
        connect(libraryImportFolderButton_, &QPushButton::clicked, this, [this] { importFolder(); });
        connect(libraryImportProgressTimer_, &QTimer::timeout, this, [this] { drainLibraryImportProgress(); });
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
        connect(libraryMeshModeCombo_, &QComboBox::currentIndexChanged, this, [this] {
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
        auto* controlsLayout = new QGridLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        peopleDatabaseEdit_ = new QLineEdit(controls);
        peopleDatabaseEdit_->setReadOnly(true);
        peopleFilterEdit_ = new QLineEdit(controls);
        peopleFilterEdit_->setPlaceholderText("person name or notes");
        personNameEdit_ = new QLineEdit(controls);
        personNameEdit_->setPlaceholderText("New person name");
        auto* reloadPeopleButton = new QPushButton("Reload", controls);
        auto* trainButton = new QPushButton("Train Identity Profiles", controls);
        auto* addButton = new QPushButton("Add Person", controls);
        controlsLayout->addWidget(new QLabel("Database", controls), 0, 0);
        controlsLayout->addWidget(peopleDatabaseEdit_, 0, 1, 1, 4);
        controlsLayout->addWidget(reloadPeopleButton, 0, 5);
        controlsLayout->addWidget(trainButton, 0, 6);
        controlsLayout->addWidget(new QLabel("Filter", controls), 1, 0);
        controlsLayout->addWidget(peopleFilterEdit_, 1, 1, 1, 4);
        controlsLayout->addWidget(personNameEdit_, 1, 5);
        controlsLayout->addWidget(addButton, 1, 6);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        peopleTable_ = new QTableWidget(splitter);
        peopleTable_->setColumnCount(11);
        peopleTable_->setHorizontalHeaderLabels({"Name", "Faces", "Avg Q", "Rev", "Ign", "Identity", "Samples", "Exemplars", "Accept", "Health", "Scorer"});
        fitTable(peopleTable_);
        peopleTable_->setSelectionMode(QAbstractItemView::SingleSelection);
        peopleTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

        peopleMemberTable_ = new QTableWidget(splitter);
        peopleMemberTable_->setColumnCount(6);
        peopleMemberTable_->setHorizontalHeaderLabels({"ID", "Name", "Tags", "Quality", "Review", "Ignored"});
        fitTable(peopleMemberTable_);
        peopleMemberTable_->setSelectionMode(QAbstractItemView::SingleSelection);
        peopleMemberTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

        auto* rightPanel = new QWidget(splitter);
        auto* rightLayout = new QVBoxLayout(rightPanel);
        rightLayout->setContentsMargins(8, 0, 0, 0);
        peopleFocusButton_ = new QPushButton("Focus on Face", rightPanel);
        peopleFocusButton_->setMaximumWidth(132);
        peoplePreviewLabel_ = new QLabel("Select a person", rightPanel);
        peoplePreviewLabel_->setAlignment(Qt::AlignCenter);
        peoplePreviewLabel_->setMinimumWidth(300);
        peoplePreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        rightLayout->addWidget(peopleFocusButton_, 0, Qt::AlignLeft);
        rightLayout->addWidget(peoplePreviewLabel_, 1);

        auto* editor = new QGroupBox("Manage Person", rightPanel);
        auto* form = new QFormLayout(editor);
        peopleNameEdit_ = new QLineEdit(editor);
        peopleNotesEdit_ = new QTextEdit(editor);
        peopleNotesEdit_->setMinimumHeight(84);
        peopleMergeTargetCombo_ = new QComboBox(editor);
        assignFaceSpin_ = new QSpinBox(editor);
        assignFaceSpin_->setRange(1, 999999999);
        assignFaceSpin_->setPrefix("Face ");
        assignPersonSpin_ = new QSpinBox(editor);
        assignPersonSpin_->setRange(1, 999999999);
        assignPersonSpin_->setPrefix("Person ");
        auto* savePersonButton = new QPushButton("Save Name / Notes", editor);
        auto* mergeButton = new QPushButton("Merge Into Target", editor);
        auto* clearButton = new QPushButton("Clear Assignment", editor);
        auto* assignButton = new QPushButton("Assign Face", editor);
        auto* assignRow = new QWidget(editor);
        auto* assignRowLayout = new QHBoxLayout(assignRow);
        assignRowLayout->setContentsMargins(0, 0, 0, 0);
        assignRowLayout->addWidget(assignFaceSpin_);
        assignRowLayout->addWidget(assignPersonSpin_);
        assignRowLayout->addWidget(assignButton);
        form->addRow("Name", peopleNameEdit_);
        form->addRow("Notes", peopleNotesEdit_);
        form->addRow("Target", peopleMergeTargetCombo_);
        form->addRow("", savePersonButton);
        form->addRow("", mergeButton);
        form->addRow("", clearButton);
        form->addRow("Assign", assignRow);
        peopleSummaryLabel_ = new QLabel("No person selected", rightPanel);
        peopleProfileStatusLabel_ = new QLabel("Identity profile: not trained", rightPanel);
        peopleProfileStatusLabel_->setWordWrap(true);
        rightLayout->addWidget(editor);
        rightLayout->addWidget(peopleSummaryLabel_);
        rightLayout->addWidget(peopleProfileStatusLabel_);
        rightLayout->addStretch();

        splitter->addWidget(peopleTable_);
        splitter->addWidget(peopleMemberTable_);
        splitter->addWidget(rightPanel);
        splitter->setStretchFactor(0, 2);
        splitter->setStretchFactor(1, 2);
        splitter->setStretchFactor(2, 1);
        layout->addWidget(splitter, 1);
        addMainTab(page, "People");
        connect(peopleTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedPerson(); });
        connect(peopleMemberTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedPeopleMember(); });
        connect(peopleFilterEdit_, &QLineEdit::returnPressed, this, [this] { loadPeople(); });
        connect(reloadPeopleButton, &QPushButton::clicked, this, [this] { loadPeople(); });
        connect(addButton, &QPushButton::clicked, this, [this] { addPerson(); });
        connect(savePersonButton, &QPushButton::clicked, this, [this] { saveSelectedPerson(); });
        connect(mergeButton, &QPushButton::clicked, this, [this] { mergeSelectedPerson(); });
        connect(clearButton, &QPushButton::clicked, this, [this] { clearSelectedPerson(); });
        connect(assignButton, &QPushButton::clicked, this, [this] { assignFace(); });
        connect(trainButton, &QPushButton::clicked, this, [this] { trainProfiles(); });
        connect(peopleFocusButton_, &QPushButton::clicked, this, [this] {
            peopleFocusOnFace_ = !peopleFocusOnFace_;
            if (peoplePreviewFaceId_ > 0) {
                updatePeoplePreview(peoplePreviewFaceId_);
            }
        });
    }

    void buildReviewTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* controlsLayout = new QGridLayout(controls);
        controlsLayout->setContentsMargins(0, 0, 0, 0);
        reviewDatabaseEdit_ = new QLineEdit(controls);
        reviewDatabaseEdit_->setReadOnly(true);
        reviewFilterEdit_ = new QLineEdit(controls);
        reviewFilterEdit_->setPlaceholderText("name, path, person, tag, notes");
        reviewLimitSpin_ = new QSpinBox(controls);
        reviewLimitSpin_->setRange(10, 10000);
        reviewLimitSpin_->setValue(500);
        auto* reloadButton = new QPushButton("Reload", controls);
        auto* reviewedButton = new QPushButton("Mark Reviewed", controls);
        auto* ignoredButton = new QPushButton("Ignore / Restore", controls);
        auto* saveButton = new QPushButton("Save Metadata", controls);
        auto* resetButton = new QPushButton("Reset Filter", controls);
        controlsLayout->addWidget(new QLabel("Database", controls), 0, 0);
        controlsLayout->addWidget(reviewDatabaseEdit_, 0, 1, 1, 5);
        controlsLayout->addWidget(reloadButton, 0, 6);
        controlsLayout->addWidget(reviewedButton, 0, 7);
        controlsLayout->addWidget(ignoredButton, 0, 8);
        controlsLayout->addWidget(saveButton, 0, 9);
        controlsLayout->addWidget(new QLabel("Filter", controls), 1, 0);
        controlsLayout->addWidget(reviewFilterEdit_, 1, 1, 1, 4);
        controlsLayout->addWidget(new QLabel("Limit", controls), 1, 5);
        controlsLayout->addWidget(reviewLimitSpin_, 1, 6);
        controlsLayout->addWidget(resetButton, 1, 7);
        controlsLayout->setColumnStretch(4, 1);
        layout->addWidget(controls);

        auto* splitter = new QSplitter(Qt::Horizontal, page);
        reviewTable_ = new QTableWidget(splitter);
        reviewTable_->setColumnCount(8);
        reviewTable_->setHorizontalHeaderLabels({"ID", "Name", "Reason", "Person", "Tags", "Quality", "Dupes", "Notes"});
        fitTable(reviewTable_);
        reviewTable_->setSelectionMode(QAbstractItemView::SingleSelection);
        reviewTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
        auto* detailPanel = new QWidget(splitter);
        auto* detailLayout = new QVBoxLayout(detailPanel);
        detailLayout->setContentsMargins(8, 0, 0, 0);
        reviewFocusButton_ = new QPushButton("Focus on Face", detailPanel);
        reviewFocusButton_->setMaximumWidth(132);
        reviewPreviewLabel_ = new QLabel("Select a review item", detailPanel);
        reviewPreviewLabel_->setAlignment(Qt::AlignCenter);
        reviewPreviewLabel_->setMinimumWidth(320);
        reviewPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        auto* editor = new QGroupBox("Edit", detailPanel);
        auto* form = new QFormLayout(editor);
        reviewPersonEdit_ = new QLineEdit(editor);
        reviewTagsEdit_ = new QLineEdit(editor);
        reviewStateCombo_ = new QComboBox(editor);
        reviewStateCombo_->addItems({"open", "reviewed", "duplicate", "low_quality", "ignored"});
        reviewIgnoredCheck_ = new QCheckBox("Ignore in search", editor);
        reviewNotesEdit_ = new QTextEdit(editor);
        reviewNotesEdit_->setMinimumHeight(96);
        reviewSuggestionLabel_ = new QLabel("AI Suggested Person: not checked", editor);
        reviewSuggestionLabel_->setWordWrap(true);
        auto* confirmSuggestionButton = new QPushButton("Confirm AI Person", editor);
        auto* rejectSuggestionButton = new QPushButton("Reject AI Suggestion", editor);
        form->addRow("Person", reviewPersonEdit_);
        form->addRow("Tags", reviewTagsEdit_);
        form->addRow("Review", reviewStateCombo_);
        form->addRow("", reviewIgnoredCheck_);
        form->addRow("Notes", reviewNotesEdit_);
        form->addRow("AI Suggested Person", reviewSuggestionLabel_);
        form->addRow("", confirmSuggestionButton);
        form->addRow("", rejectSuggestionButton);
        detailLayout->addWidget(reviewFocusButton_, 0, Qt::AlignLeft);
        detailLayout->addWidget(reviewPreviewLabel_, 1);
        detailLayout->addWidget(editor);
        detailLayout->addStretch();
        splitter->addWidget(reviewTable_);
        splitter->addWidget(detailPanel);
        splitter->setStretchFactor(0, 3);
        splitter->setStretchFactor(1, 2);
        layout->addWidget(splitter, 1);
        addMainTab(page, "Review");

        connect(reviewTable_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedReviewRecord(); });
        connect(reviewFilterEdit_, &QLineEdit::returnPressed, this, [this] { loadReview(); });
        connect(reloadButton, &QPushButton::clicked, this, [this] { loadReview(); });
        connect(resetButton, &QPushButton::clicked, this, [this] {
            if (reviewFilterEdit_ != nullptr) {
                reviewFilterEdit_->clear();
            }
            if (reviewLimitSpin_ != nullptr) {
                reviewLimitSpin_->setValue(500);
            }
            loadReview();
        });
        connect(reviewFocusButton_, &QPushButton::clicked, this, [this] {
            reviewFocusOnFace_ = !reviewFocusOnFace_;
            if (currentReviewFaceId_ > 0) {
                updateReviewDetail(static_cast<int>(currentReviewFaceId_), false);
            }
        });
        connect(saveButton, &QPushButton::clicked, this, [this] { saveReviewMetadata(); });
        connect(reviewedButton, &QPushButton::clicked, this, [this] { applyReviewState("reviewed", false); });
        connect(ignoredButton, &QPushButton::clicked, this, [this] { toggleReviewIgnored(); });
        connect(confirmSuggestionButton, &QPushButton::clicked, this, [this] { confirmReviewSuggestion(); });
        connect(rejectSuggestionButton, &QPushButton::clicked, this, [this] { rejectReviewSuggestion(); });
    }

    void buildSearchTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* title = new QLabel("Search", page);
        title->setObjectName("PageTitle");
        layout->addWidget(title);

        auto* queryBox = new QGroupBox("Query", page);
        auto* queryLayout = new QGridLayout(queryBox);
        searchDatabasePathEdit_ = new QLineEdit(queryBox);
        searchDatabasePathEdit_->setObjectName("SearchDatabasePath");
        searchDatabasePathEdit_->setReadOnly(true);
        auto* openDatabaseButton = new QPushButton("Open Database", queryBox);
        openDatabaseButton->setObjectName("SearchOpenDatabase");
        auto* useLibraryDatabaseButton = new QPushButton("Use Library DB", queryBox);
        useLibraryDatabaseButton->setObjectName("SearchUseLibraryDatabase");

        searchImageEdit_ = new QLineEdit(queryBox);
        searchImageEdit_->setObjectName("SearchQueryImagePath");
        searchImageEdit_->setReadOnly(true);
        auto* selectImageButton = new QPushButton("Select Image", queryBox);
        selectImageButton->setObjectName("SearchSelectImage");
        auto* searchButton = new QPushButton("Search", queryBox);
        searchButton->setObjectName("SearchRun");

        topKSpin_ = new QSpinBox(queryBox);
        topKSpin_->setRange(1, 500);
        topKSpin_->setValue(30);
        searchThresholdSpin_ = new QDoubleSpinBox(queryBox);
        searchThresholdSpin_->setRange(-1.0, 1.0);
        searchThresholdSpin_->setDecimals(3);
        searchThresholdSpin_->setSingleStep(0.010);
        searchThresholdSpin_->setValue(-1.0);
        searchMinQualitySpin_ = new QDoubleSpinBox(queryBox);
        searchMinQualitySpin_->setRange(0.0, 1.0);
        searchMinQualitySpin_->setDecimals(3);
        searchMinQualitySpin_->setSingleStep(0.050);
        searchMinQualitySpin_->setValue(0.0);
        searchPersonFilterCombo_ = new QComboBox(queryBox);
        searchPersonFilterCombo_->setMinimumWidth(180);
        searchTagFilterCombo_ = new QComboBox(queryBox);
        searchTagFilterCombo_->setMinimumWidth(160);
        searchIncludeIgnoredCheck_ = new QCheckBox("Include ignored", queryBox);

        queryLayout->addWidget(new QLabel("Database", queryBox), 0, 0);
        queryLayout->addWidget(searchDatabasePathEdit_, 0, 1, 1, 3);
        queryLayout->addWidget(openDatabaseButton, 0, 4);
        queryLayout->addWidget(useLibraryDatabaseButton, 0, 5);
        queryLayout->addWidget(new QLabel("Image", queryBox), 1, 0);
        queryLayout->addWidget(searchImageEdit_, 1, 1, 1, 3);
        queryLayout->addWidget(selectImageButton, 1, 4);
        queryLayout->addWidget(searchButton, 1, 5);
        queryLayout->addWidget(new QLabel("Top K", queryBox), 2, 0);
        queryLayout->addWidget(topKSpin_, 2, 1);
        queryLayout->addWidget(new QLabel("Threshold", queryBox), 2, 2);
        queryLayout->addWidget(searchThresholdSpin_, 2, 3);
        queryLayout->addWidget(new QLabel("Min quality", queryBox), 2, 4);
        queryLayout->addWidget(searchMinQualitySpin_, 2, 5);
        queryLayout->addWidget(new QLabel("Person", queryBox), 3, 0);
        queryLayout->addWidget(searchPersonFilterCombo_, 3, 1);
        queryLayout->addWidget(new QLabel("Tag", queryBox), 3, 2);
        queryLayout->addWidget(searchTagFilterCombo_, 3, 3);
        queryLayout->addWidget(searchIncludeIgnoredCheck_, 3, 4, 1, 2);
        layout->addWidget(queryBox);

        auto* body = new QWidget(page);
        auto* bodyLayout = new QHBoxLayout(body);
        bodyLayout->setContentsMargins(0, 0, 0, 0);
        bodyLayout->setSpacing(12);

        auto* previewPanel = new QWidget(body);
        previewPanel->setMinimumWidth(300);
        previewPanel->setMaximumWidth(430);
        auto* previewLayout = new QVBoxLayout(previewPanel);
        previewLayout->setContentsMargins(0, 0, 0, 0);
        previewLayout->setSpacing(8);
        searchQueryFocusButton_ = new QToolButton(previewPanel);
        searchQueryFocusButton_->setObjectName("SearchQueryFocus");
        searchQueryFocusButton_->setText("Focus on Face");
        searchQueryFocusButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        searchQueryFocusButton_->setMaximumSize(118, 24);
        searchQueryFocusButton_->setEnabled(false);
        searchQueryPreview_ = new FaceSelectionPreview(previewPanel);
        searchQueryPreview_->setObjectName("SearchQueryPreview");
        searchFaceList_ = new QListWidget(previewPanel);
        searchFaceList_->setObjectName("SearchQueryFaceList");
        searchFaceList_->setMinimumHeight(68);
        searchFaceList_->setMaximumHeight(96);
        searchFaceList_->setSelectionMode(QAbstractItemView::SingleSelection);
        searchResultFocusButton_ = new QToolButton(previewPanel);
        searchResultFocusButton_->setObjectName("SearchResultFocus");
        searchResultFocusButton_->setText("Focus on Face");
        searchResultFocusButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        searchResultFocusButton_->setMaximumSize(118, 24);
        searchResultFocusButton_->setEnabled(false);
        searchResultPreviewLabel_ = new QLabel("Result", previewPanel);
        searchResultPreviewLabel_->setObjectName("SearchResultPreview");
        searchResultPreviewLabel_->setAlignment(Qt::AlignCenter);
        searchResultPreviewLabel_->setMinimumSize(180, 180);
        searchResultPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        previewLayout->addWidget(searchQueryFocusButton_, 0, Qt::AlignLeft);
        previewLayout->addWidget(searchQueryPreview_, 1);
        previewLayout->addWidget(searchFaceList_);
        previewLayout->addWidget(searchResultFocusButton_, 0, Qt::AlignLeft);
        previewLayout->addWidget(searchResultPreviewLabel_, 1);
        bodyLayout->addWidget(previewPanel, 0);

        auto* resultsPanel = new QWidget(body);
        auto* resultsLayout = new QVBoxLayout(resultsPanel);
        resultsLayout->setContentsMargins(0, 0, 0, 0);
        resultsLayout->setSpacing(8);
        identityLabel_ = new QLabel("Identity: not searched", resultsPanel);
        identityLabel_->setWordWrap(true);
        searchIdentityTable_ = new QTableWidget(resultsPanel);
        searchIdentityTable_->setColumnCount(6);
        searchIdentityTable_->setHorizontalHeaderLabels({"Rank", "Person", "Decision", "Score", "Confidence", "Evidence"});
        fitTable(searchIdentityTable_);
        searchIdentityTable_->setMaximumHeight(120);
        searchTable_ = new QTableWidget(resultsPanel);
        searchTable_->setColumnCount(8);
        searchTable_->setHorizontalHeaderLabels({"Rank", "ID", "Name", "Person", "Tags", "Cosine", "Similarity", "Quality"});
        fitTable(searchTable_);
        resultsLayout->addWidget(identityLabel_);
        resultsLayout->addWidget(searchIdentityTable_);
        resultsLayout->addWidget(searchTable_, 1);
        bodyLayout->addWidget(resultsPanel, 1);
        layout->addWidget(body, 1);

        searchPreviewTimer_ = new QTimer(this);
        searchPreviewTimer_->setInterval(70);
        addMainTab(page, "Search");
        connect(openDatabaseButton, &QPushButton::clicked, this, [this] { chooseDatabase(); });
        connect(useLibraryDatabaseButton, &QPushButton::clicked, this, [this] { syncSearchDatabasePath(); });
        connect(selectImageButton, &QPushButton::clicked, this, [this] { selectSearchQueryImage(); });
        connect(searchFaceList_, &QListWidget::currentRowChanged, this, [this](int index) {
            if (index < 0) {
                return;
            }
            searchQueryFaceIndex_ = index;
            updateSearchQueryPreview();
        });
        searchQueryPreview_->faceClicked = [this](int index) {
            if (searchFaceList_ != nullptr && index >= 0 && index < searchFaceList_->count()) {
                searchFaceList_->setCurrentRow(index);
            }
        };
        connect(searchQueryFocusButton_, &QToolButton::clicked, this, [this] { toggleSearchQueryFocus(); });
        connect(searchResultFocusButton_, &QToolButton::clicked, this, [this] { toggleSearchResultFocus(); });
        connect(searchButton, &QPushButton::clicked, this, [this] { runSearch(); });
        connect(searchTable_, &QTableWidget::itemSelectionChanged, this, [this] { updateSelectedSearchResultPreview(); });
        connect(searchIdentityTable_, &QTableWidget::itemSelectionChanged, this, [this] { updateSelectedSearchIdentityPreview(); });
        connect(searchPreviewTimer_, &QTimer::timeout, this, [this] { advanceSearchProgress(); });
        syncSearchDatabasePath();
    }

    void buildCameraTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* title = new QLabel("Camera", page);
        title->setObjectName("PageTitle");
        layout->addWidget(title);

        auto* controls = new QGroupBox("Camera", page);
        auto* controlsLayout = new QGridLayout(controls);
        cameraDatabasePathEdit_ = new QLineEdit(controls);
        cameraDatabasePathEdit_->setObjectName("CameraDatabasePath");
        cameraDatabasePathEdit_->setReadOnly(true);
        auto* openDatabaseButton = new QPushButton("Open Database", controls);
        openDatabaseButton->setObjectName("CameraOpenDatabase");
        auto* useLibraryButton = new QPushButton("Use Library DB", controls);
        useLibraryButton->setObjectName("CameraUseLibraryDatabase");

        cameraIndexSpin_ = new QSpinBox(controls);
        cameraIndexSpin_->setRange(0, 8);
        cameraThresholdSpin_ = new QDoubleSpinBox(controls);
        cameraThresholdSpin_->setRange(-1.0, 1.0);
        cameraThresholdSpin_->setDecimals(3);
        cameraThresholdSpin_->setSingleStep(0.010);
        cameraThresholdSpin_->setValue(0.350);
        cameraTopKSpin_ = new QSpinBox(controls);
        cameraTopKSpin_->setRange(1, 10);
        cameraTopKSpin_->setValue(3);
        cameraIntervalSpin_ = new QSpinBox(controls);
        cameraIntervalSpin_->setRange(100, 5000);
        cameraIntervalSpin_->setSingleStep(50);
        cameraIntervalSpin_->setValue(300);
        cameraProcessSizeSpin_ = new QSpinBox(controls);
        cameraProcessSizeSpin_->setRange(320, 1920);
        cameraProcessSizeSpin_->setSingleStep(80);
        cameraProcessSizeSpin_->setValue(640);
        cameraStartButton_ = new QPushButton("Start Camera", controls);
        cameraStartButton_->setObjectName("CameraStart");
        cameraStopButton_ = new QPushButton("Stop Camera", controls);
        cameraStopButton_->setObjectName("CameraStop");
        cameraStopButton_->setEnabled(false);

        controlsLayout->addWidget(new QLabel("Database", controls), 0, 0);
        controlsLayout->addWidget(cameraDatabasePathEdit_, 0, 1, 1, 5);
        controlsLayout->addWidget(openDatabaseButton, 0, 6);
        controlsLayout->addWidget(useLibraryButton, 0, 7);
        controlsLayout->addWidget(new QLabel("Camera", controls), 1, 0);
        controlsLayout->addWidget(cameraIndexSpin_, 1, 1);
        controlsLayout->addWidget(new QLabel("Threshold", controls), 1, 2);
        controlsLayout->addWidget(cameraThresholdSpin_, 1, 3);
        controlsLayout->addWidget(new QLabel("Interval ms", controls), 1, 4);
        controlsLayout->addWidget(cameraIntervalSpin_, 1, 5);
        controlsLayout->addWidget(new QLabel("Top K", controls), 1, 6);
        controlsLayout->addWidget(cameraTopKSpin_, 1, 7);
        controlsLayout->addWidget(new QLabel("Process size", controls), 2, 0);
        controlsLayout->addWidget(cameraProcessSizeSpin_, 2, 1);
        controlsLayout->addWidget(cameraStartButton_, 2, 2, 1, 2);
        controlsLayout->addWidget(cameraStopButton_, 2, 4, 1, 2);
        layout->addWidget(controls);

        auto* body = new QWidget(page);
        auto* bodyLayout = new QHBoxLayout(body);
        bodyLayout->setContentsMargins(0, 0, 0, 0);
        bodyLayout->setSpacing(12);
        cameraPreviewLabel_ = new QLabel(body);
        cameraPreviewLabel_->setObjectName("CameraPreview");
        cameraPreviewLabel_->setMinimumSize(360, 260);
        cameraPreviewLabel_->setAlignment(Qt::AlignCenter);
        cameraPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;");
        cameraPreviewLabel_->setText("Camera");
        bodyLayout->addWidget(cameraPreviewLabel_, 2);

        auto* resultsPanel = new QWidget(body);
        resultsPanel->setMinimumWidth(330);
        resultsPanel->setMaximumWidth(460);
        auto* resultsLayout = new QVBoxLayout(resultsPanel);
        resultsLayout->setContentsMargins(0, 0, 0, 0);
        resultsLayout->setSpacing(8);
        cameraMatchFocusButton_ = new QToolButton(resultsPanel);
        cameraMatchFocusButton_->setObjectName("CameraMatchFocus");
        cameraMatchFocusButton_->setText("Focus on Face");
        cameraMatchFocusButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        cameraMatchFocusButton_->setMaximumSize(118, 24);
        cameraMatchFocusButton_->setEnabled(false);
        cameraMatchPreviewLabel_ = new QLabel(resultsPanel);
        cameraMatchPreviewLabel_->setObjectName("CameraMatchPreview");
        cameraMatchPreviewLabel_->setMinimumSize(180, 180);
        cameraMatchPreviewLabel_->setAlignment(Qt::AlignCenter);
        cameraMatchPreviewLabel_->setStyleSheet("background:#0c1420;color:#dce8f5;border:1px solid #c8d5e6;");
        cameraMatchPreviewLabel_->setText("Match");
        resultsLayout->addWidget(cameraMatchFocusButton_, 0, Qt::AlignLeft);
        resultsLayout->addWidget(cameraMatchPreviewLabel_, 1);
        cameraMatchStatusLabel_ = new QLabel("No camera result", resultsPanel);
        cameraMatchStatusLabel_->setWordWrap(true);
        resultsLayout->addWidget(cameraMatchStatusLabel_);

        cameraResultTable_ = new QTableWidget(resultsPanel);
        cameraResultTable_->setObjectName("CameraResultTable");
        cameraResultTable_->setSelectionMode(QAbstractItemView::SingleSelection);
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
        resultsLayout->addWidget(cameraResultTable_, 1);
        bodyLayout->addWidget(resultsPanel, 0);
        layout->addWidget(body, 1);

        cameraFrameTimer_ = new QTimer(this);
        cameraFrameTimer_->setInterval(33);
        cameraIdentifyTimer_ = new QTimer(this);
        cameraIdentifyTimer_->setInterval(cameraIntervalSpin_->value());
        addMainTab(page, "Camera");

        connect(openDatabaseButton, &QPushButton::clicked, this, [this] { chooseDatabase(); });
        connect(useLibraryButton, &QPushButton::clicked, this, [this] { syncCameraDatabasePath(); });
        connect(cameraStartButton_, &QPushButton::clicked, this, [this] { startCamera(); });
        connect(cameraStopButton_, &QPushButton::clicked, this, [this] { stopCamera(); });
        connect(cameraMatchFocusButton_, &QToolButton::clicked, this, [this] { toggleCameraMatchFocus(); });
        connect(cameraIntervalSpin_, &QSpinBox::valueChanged, this, [this](int value) {
            if (cameraIdentifyTimer_ != nullptr) {
                cameraIdentifyTimer_->setInterval(value);
            }
        });
        connect(cameraFrameTimer_, &QTimer::timeout, this, [this] { captureCameraFrame(); });
        connect(cameraIdentifyTimer_, &QTimer::timeout, this, [this] { identifyCameraFrame(); });
        connect(cameraResultTable_, &QTableWidget::itemSelectionChanged, this, [this] { updateSelectedCameraResultPreview(); });
        syncCameraDatabasePath();
    }

    void buildCompareTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* title = new QLabel("Compare", page);
        title->setObjectName("PageTitle");
        layout->addWidget(title);

        auto* imagesBox = new QGroupBox("Images", page);
        auto* form = new QGridLayout(imagesBox);
        compareImageAEdit_ = new QLineEdit(imagesBox);
        compareImageBEdit_ = new QLineEdit(imagesBox);
        compareImageAEdit_->setObjectName("CompareImageAPath");
        compareImageBEdit_->setObjectName("CompareImageBPath");
        compareImageAEdit_->setReadOnly(true);
        compareImageBEdit_->setReadOnly(true);
        auto* browseA = new QPushButton("Image A", imagesBox);
        browseA->setObjectName("CompareSelectImageA");
        auto* browseB = new QPushButton("Image B", imagesBox);
        browseB->setObjectName("CompareSelectImageB");
        auto* compareButton = new QPushButton("Compare", imagesBox);
        compareButton->setObjectName("CompareRun");
        form->addWidget(new QLabel("A", imagesBox), 0, 0);
        form->addWidget(compareImageAEdit_, 0, 1);
        form->addWidget(browseA, 0, 2);
        form->addWidget(new QLabel("B", imagesBox), 1, 0);
        form->addWidget(compareImageBEdit_, 1, 1);
        form->addWidget(browseB, 1, 2);
        form->addWidget(compareButton, 0, 3, 2, 1);
        layout->addWidget(imagesBox);

        auto* previews = new QWidget(page);
        auto* previewsLayout = new QHBoxLayout(previews);
        previewsLayout->setContentsMargins(0, 0, 0, 0);
        previewsLayout->setSpacing(12);
        auto* panelA = new QWidget(previews);
        auto* panelALayout = new QVBoxLayout(panelA);
        panelALayout->setContentsMargins(0, 0, 0, 0);
        compareFocusAButton_ = new QToolButton(panelA);
        compareFocusAButton_->setObjectName("CompareFocusA");
        compareFocusAButton_->setText("Focus on Face");
        compareFocusAButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        compareFocusAButton_->setMaximumSize(118, 24);
        compareFocusAButton_->setEnabled(false);
        comparePreviewA_ = new FaceSelectionPreview(panelA);
        comparePreviewA_->setObjectName("ComparePreviewA");
        compareFaceListA_ = new QListWidget(panelA);
        compareFaceListA_->setObjectName("CompareFaceListA");
        compareFaceListA_->setMaximumHeight(110);
        panelALayout->addWidget(compareFocusAButton_, 0, Qt::AlignLeft);
        panelALayout->addWidget(comparePreviewA_, 1);
        panelALayout->addWidget(compareFaceListA_);
        auto* panelB = new QWidget(previews);
        auto* panelBLayout = new QVBoxLayout(panelB);
        panelBLayout->setContentsMargins(0, 0, 0, 0);
        compareFocusBButton_ = new QToolButton(panelB);
        compareFocusBButton_->setObjectName("CompareFocusB");
        compareFocusBButton_->setText("Focus on Face");
        compareFocusBButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        compareFocusBButton_->setMaximumSize(118, 24);
        compareFocusBButton_->setEnabled(false);
        comparePreviewB_ = new FaceSelectionPreview(panelB);
        comparePreviewB_->setObjectName("ComparePreviewB");
        compareFaceListB_ = new QListWidget(panelB);
        compareFaceListB_->setObjectName("CompareFaceListB");
        compareFaceListB_->setMaximumHeight(110);
        panelBLayout->addWidget(compareFocusBButton_, 0, Qt::AlignLeft);
        panelBLayout->addWidget(comparePreviewB_, 1);
        panelBLayout->addWidget(compareFaceListB_);
        previewsLayout->addWidget(panelA);
        previewsLayout->addWidget(panelB);
        layout->addWidget(previews, 1);

        compareResultLabel_ = new QLabel("Cosine: --    Similarity: --", page);
        compareResultLabel_->setObjectName("Metric");
        compareResultLabel_->setAlignment(Qt::AlignCenter);
        layout->addWidget(compareResultLabel_);
        addMainTab(page, "Compare");

        comparePreviewA_->faceClicked = [this](int index) { selectCompareFace('a', index); };
        comparePreviewB_->faceClicked = [this](int index) { selectCompareFace('b', index); };
        connect(browseA, &QPushButton::clicked, this, [this] { selectCompareImage('a'); });
        connect(browseB, &QPushButton::clicked, this, [this] { selectCompareImage('b'); });
        connect(compareFocusAButton_, &QToolButton::clicked, this, [this] { toggleCompareFocus('a'); });
        connect(compareFocusBButton_, &QToolButton::clicked, this, [this] { toggleCompareFocus('b'); });
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

        auto* legacyBox = new QGroupBox("Legacy", page);
        auto* legacyLayout = new QVBoxLayout(legacyBox);
        auto* legacyInfo = new QLabel(
            "Convert a trusted legacy .dtb database by re-analyzing its embedded RGB images. "
            "The converted database keeps local preview files and does not require Python at runtime.",
            legacyBox);
        legacyInfo->setWordWrap(true);
        runtimeLegacyConvertButton_ = new QPushButton("Convert Legacy DTB", legacyBox);
        runtimeLegacyProgressBar_ = new QProgressBar(legacyBox);
        runtimeLegacyProgressBar_->setRange(0, 1);
        runtimeLegacyProgressBar_->setValue(0);
        runtimeLegacyProgressBar_->setTextVisible(true);
        legacyLayout->addWidget(legacyInfo);
        legacyLayout->addWidget(runtimeLegacyConvertButton_, 0, Qt::AlignLeft);
        legacyLayout->addWidget(runtimeLegacyProgressBar_);
        layout->addWidget(legacyBox);
        layout->addStretch(1);
        addMainTab(page, "Runtime");

        connect(refreshButton, &QPushButton::clicked, this, [this] { refreshRuntimeInfo(); });
        connect(refreshDatabaseButton, &QPushButton::clicked, this, [this] { refreshRuntimeDatabaseInfo(); });
        connect(integrityButton, &QPushButton::clicked, this, [this] { runRuntimeIntegrityCheck(); });
        connect(backupButton, &QPushButton::clicked, this, [this] { runRuntimeBackup(); });
        connect(checkpointButton, &QPushButton::clicked, this, [this] { runRuntimeCheckpoint(); });
        connect(vacuumButton, &QPushButton::clicked, this, [this] { runRuntimeVacuum(); });
        connect(runtimeLegacyConvertButton_, &QPushButton::clicked, this, [this] { runRuntimeLegacyConversion(); });
        connect(runtimeModeCombo_, &QComboBox::currentTextChanged, this, [this] {
            refreshRuntimeInfo();
        });
        refreshRuntimeInfo();
        refreshRuntimeDatabaseInfo();
    }

    void chooseDatabase() {
        const auto path = QFileDialog::getOpenFileName(
            this,
            "Open or convert database",
            {},
            "FSC Database (*.fscdb);;Legacy dlib database (*.dtb);;SQLite Database (*.sqlite *.db);;All Files (*)");
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
        if (QFileInfo(path).suffix().compare("dtb", Qt::CaseInsensitive) == 0) {
            selectMainTab("Runtime");
            runRuntimeLegacyConversion(path);
            return;
        }
        try {
            database_ = std::make_unique<fsc::core::Database>(pathFrom(path));
            databasePathEdit_->setText(path);
            reloadAll();
            statusBar()->showMessage("Opened " + path);
        } catch (const std::exception& ex) {
            database_.reset();
            cameraIdentityProfiles_.clear();
            cameraIdentityProfilesSnapshot_ = std::make_shared<const std::vector<fsc::core::IdentityProfile>>();
            cameraStoredFaces_ = std::make_shared<const std::vector<fsc::core::FaceRecord>>();
            syncCameraDatabasePath();
            syncSearchDatabasePath();
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
        cameraIdentityProfilesSnapshot_ = std::make_shared<const std::vector<fsc::core::IdentityProfile>>(
            cameraIdentityProfiles_);
        cameraStoredFaces_ = std::make_shared<const std::vector<fsc::core::FaceRecord>>(
            database_->loadFaces(false));
        syncSearchDatabasePath();
        refreshRuntimeDatabaseInfo();
        syncCameraDatabasePath();
    }

    void loadOverview() {
        const auto stats = database_->statistics();
        if (overviewDatabasePathEdit_ != nullptr) {
            overviewDatabasePathEdit_->setText(qs(database_->path().string()));
        }
        const auto setRows = [](QTableWidget* table, const std::vector<std::pair<QString, QString>>& rows) {
            if (table == nullptr) {
                return;
            }
            table->setRowCount(static_cast<int>(rows.size()));
            for (int index = 0; index < static_cast<int>(rows.size()); ++index) {
                table->setItem(index, 0, item(rows[static_cast<size_t>(index)].first));
                table->setItem(index, 1, item(rows[static_cast<size_t>(index)].second));
            }
        };
        setRows(overviewMetricsTable_, {
            {"Faces", QString::number(stats.faceCount)},
            {"People", QString::number(stats.peopleCount)},
            {"Tags", QString::number(stats.tagCount)},
            {"Average Quality", QString::number(stats.averageQuality, 'f', 3)},
            {"Model", qs(stats.modelName)},
            {"Format", QString("v%1").arg(qs(stats.formatVersion))},
            {"Metric", qs(stats.metric)},
        });
        setRows(overviewAttentionTable_, {
            {"Needs review", QString::number(stats.reviewCount)},
            {"Ignored", QString::number(stats.ignoredCount)},
            {"Duplicate image groups", QString::number(stats.duplicateImageGroupCount)},
        });

        if (overviewPeopleTable_ != nullptr) {
            const auto people = database_->loadPeople();
            const int rowCount = std::min(10, static_cast<int>(people.size()));
            overviewPeopleTable_->setRowCount(rowCount);
            for (int row = 0; row < rowCount; ++row) {
                const auto& person = people[static_cast<size_t>(row)];
                overviewPeopleTable_->setItem(row, 0, item(qs(person.name)));
                overviewPeopleTable_->setItem(row, 1, item(QString::number(person.faceCount)));
                overviewPeopleTable_->setItem(row, 2, item(QString::number(person.reviewCount)));
            }
        }
        if (overviewTagsTable_ != nullptr) {
            const auto tags = database_->loadTagSummaries(12);
            overviewTagsTable_->setRowCount(static_cast<int>(tags.size()));
            for (int row = 0; row < static_cast<int>(tags.size()); ++row) {
                const auto& tag = tags[static_cast<size_t>(row)];
                overviewTagsTable_->setItem(row, 0, item(qs(tag.name)));
                overviewTagsTable_->setItem(row, 1, item(QString::number(tag.faceCount)));
            }
        }
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

    void setDatabaseFacePreview(QLabel* label, const fsc::core::FaceRecord& face, const QString& fallbackText, bool focusOnFace = false) {
        if (label == nullptr) {
            return;
        }
        QImage image = loadPreviewImage(pathFrom(qs(face.sourcePath)));
        if (image.isNull()) {
            label->setText(fallbackText);
            label->setPixmap(QPixmap());
            return;
        }
        QRectF bbox;
        if (face.bbox.size() >= 4) {
            bbox = QRectF(QPointF(face.bbox[0], face.bbox[1]), QPointF(face.bbox[2], face.bbox[3])).normalized();
        }
        QPointF offset(0.0, 0.0);
        QImage view = image;
        if (focusOnFace && bbox.isValid() && bbox.width() > 1.0 && bbox.height() > 1.0) {
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
            label->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        if (!pixmap.isNull()) {
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing, true);
            const double sx = static_cast<double>(pixmap.width()) / static_cast<double>(std::max(1, view.width()));
            const double sy = static_cast<double>(pixmap.height()) / static_cast<double>(std::max(1, view.height()));
            if (bbox.isValid() && bbox.width() > 1.0 && bbox.height() > 1.0) {
                const QRectF box(
                    (bbox.left() - offset.x()) * sx,
                    (bbox.top() - offset.y()) * sy,
                    bbox.width() * sx,
                    bbox.height() * sy);
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
                const QPointF drawPoint((point[0] - offset.x()) * sx, (point[1] - offset.y()) * sy);
                if (drawPoint.x() >= 0.0 && drawPoint.y() >= 0.0 && drawPoint.x() <= pixmap.width() && drawPoint.y() <= pixmap.height()) {
                    painter.drawEllipse(drawPoint, 1.7, 1.7);
                }
            }
        }
        label->setPixmap(pixmap);
    }

    void updateLibraryPreview(int faceId) {
        if (libraryPreviewLabel_ == nullptr || !database_) {
            return;
        }
        libraryPreviewFaceId_ = faceId;
        if (libraryFocusButton_ != nullptr) {
            libraryFocusButton_->setEnabled(faceId > 0);
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                libraryPreviewLabel_->setText("Face not found");
                libraryPreviewLabel_->setPixmap(QPixmap());
                return;
            }
            QImage image = loadPreviewImage(pathFrom(qs(face->sourcePath)));
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
            libraryFocusButton_->setText(libraryFocusOnFace_ ? trUi("Full Image") : trUi("Focus on Face"));
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

            const bool hasDenseMesh = fsc::mesh::isMediaPipeFaceMesh(face->faceMesh3d);
            const bool hasLandmarks = !face->landmarks3d.empty();
            if (!hasDenseMesh) {
                const QString detail = face->faceMesh3d.empty()
                    ? "No cached MediaPipe dense mesh. Select Generate Mesh to analyze the source image."
                    : QString("Cached dense mesh is incompatible (%1 points; expected 478). Select Generate Mesh to repair it.")
                          .arg(face->faceMesh3d.size());
                libraryDenseMeshView_->setMessage(detail);
                if (libraryMeshStatusLabel_ != nullptr) {
                    libraryMeshStatusLabel_->setText(detail);
                }
                return;
            }

            std::vector<std::vector<double>> meshPoints = face->faceMesh3d;
            const bool textured = libraryMeshModeCombo_ != nullptr && libraryMeshModeCombo_->currentData().toString() == "textured";
            std::vector<std::vector<double>> overlay;
            if (textured && hasLandmarks && libraryMeshOverlayCheck_ != nullptr && libraryMeshOverlayCheck_->isChecked()) {
                overlay = face->landmarks3d;
            }
            const QString source = "cached MediaPipe dense mesh";
            if (libraryMeshStatusLabel_ != nullptr) {
                libraryMeshStatusLabel_->setText(QString("Face %1: %2 point(s) from %3")
                                                     .arg(face->id)
                                                     .arg(meshPoints.size())
                                                     .arg(source));
            }
            libraryDenseMeshView_->setTextureImage(loadPreviewImage(pathFrom(qs(face->sourcePath))));
            libraryDenseMeshView_->setRenderMode(textured ? TexturedMeshWidget::RenderMode::Textured : TexturedMeshWidget::RenderMode::Points);
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

    std::vector<std::vector<double>> buildMediaPipeMeshForFace(const fsc::core::FaceRecord& face) {
        if (face.sourcePath.empty()) {
            throw std::runtime_error("This face has no source image for MediaPipe dense-mesh analysis.");
        }
        const std::filesystem::path sourcePath = face.sourcePath;
        if (!std::filesystem::is_regular_file(sourcePath)) {
            throw std::runtime_error("The source image for this face is no longer available: " + sourcePath.string());
        }
        const auto modelPath = fsc::mesh::defaultMediaPipeFaceLandmarkerModelPath();
        if (!mediaPipeFaceLandmarker_ || mediaPipeFaceLandmarkerModelPath_ != modelPath) {
            fsc::mesh::MediaPipeFaceLandmarkerOptions options;
            options.modelAssetPath = modelPath;
            mediaPipeFaceLandmarker_ = std::make_unique<fsc::mesh::MediaPipeFaceLandmarker>(std::move(options));
            mediaPipeFaceLandmarkerModelPath_ = modelPath;
        }
        const auto image = fsc::vision::loadImageRgb(sourcePath);
        return fsc::mesh::selectBestMediaPipeFaceMesh(mediaPipeFaceLandmarker_->detect(image), face.bbox);
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
            const auto mesh = buildMediaPipeMeshForFace(*face);
            database_->updateFaceMesh3d(libraryPreviewFaceId_, mesh);
            updateLibrary3dPreview(libraryPreviewFaceId_);
            statusBar()->showMessage(QString("Generated MediaPipe dense mesh for face %1 (%2 points)").arg(libraryPreviewFaceId_).arg(mesh.size()));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void loadPeople() {
        if (!database_ || peopleTable_ == nullptr) {
            return;
        }
        if (peopleDatabaseEdit_ != nullptr) {
            peopleDatabaseEdit_->setText(qs(database_->path().string()));
        }
        const QString filter = peopleFilterEdit_ == nullptr ? QString() : peopleFilterEdit_->text().trimmed();
        peopleRows_ = database_->loadPeople();
        peopleRows_.erase(
            std::remove_if(peopleRows_.begin(), peopleRows_.end(), [filter](const auto& person) {
                if (filter.isEmpty()) {
                    return false;
                }
                return !qs(person.name).contains(filter, Qt::CaseInsensitive) &&
                    !qs(person.notes).contains(filter, Qt::CaseInsensitive);
            }),
            peopleRows_.end());
        peopleTable_->setRowCount(static_cast<int>(peopleRows_.size()));
        for (int row = 0; row < static_cast<int>(peopleRows_.size()); ++row) {
            const auto& person = peopleRows_[static_cast<size_t>(row)];
            peopleTable_->setItem(row, 0, item(qs(person.name)));
            peopleTable_->setItem(row, 1, item(QString::number(person.faceCount)));
            peopleTable_->setItem(row, 2, numberItem(person.averageQuality, 3));
            peopleTable_->setItem(row, 3, item(QString::number(person.reviewCount)));
            peopleTable_->setItem(row, 4, item(QString::number(person.ignoredCount)));
            peopleTable_->setItem(row, 5, item(person.identityStatus.empty() ? "not trained" : qs(person.identityStatus)));
            peopleTable_->setItem(row, 6, item(person.identitySampleCount > 0 ? QString::number(person.identitySampleCount) : ""));
            peopleTable_->setItem(row, 7, item(person.identityExemplarCount > 0 ? QString::number(person.identityExemplarCount) : ""));
            peopleTable_->setItem(row, 8, item(person.identityAcceptThreshold > 0.0 ? QString::number(person.identityAcceptThreshold, 'f', 3) : ""));
            peopleTable_->setItem(row, 9, item(qs(person.identityHealth)));
            peopleTable_->setItem(row, 10, item(qs(person.identityScoringModelVersion)));
        }
        peopleTable_->resizeColumnsToContents();
        refreshPeopleMergeTargets(0);
        if (peopleSummaryLabel_ != nullptr) {
            peopleSummaryLabel_->setText(QString("%1 person(s)").arg(peopleRows_.size()));
        }
        if (peopleMemberTable_ != nullptr) {
            peopleMemberTable_->setRowCount(0);
        }
        peopleMembers_.clear();
        currentPersonId_ = 0;
        peoplePreviewFaceId_ = 0;
        if (peoplePreviewLabel_ != nullptr) {
            peoplePreviewLabel_->setText(peopleRows_.empty() ? "No people" : "Select a person");
            peoplePreviewLabel_->setPixmap(QPixmap());
        }
        if (!peopleRows_.empty()) {
            peopleTable_->selectRow(0);
        }
    }

    void refreshPeopleMergeTargets(int64_t excludePersonId) {
        if (peopleMergeTargetCombo_ == nullptr) {
            return;
        }
        const auto current = peopleMergeTargetCombo_->currentData().toLongLong();
        peopleMergeTargetCombo_->blockSignals(true);
        peopleMergeTargetCombo_->clear();
        peopleMergeTargetCombo_->addItem("Select target", 0);
        for (const auto& person : peopleRows_) {
            if (person.id == excludePersonId) {
                continue;
            }
            peopleMergeTargetCombo_->addItem(QString("%1 (%2)").arg(qs(person.name)).arg(person.faceCount), QVariant::fromValue<qlonglong>(person.id));
        }
        const int index = peopleMergeTargetCombo_->findData(QVariant::fromValue<qlonglong>(current));
        peopleMergeTargetCombo_->setCurrentIndex(index >= 0 ? index : 0);
        peopleMergeTargetCombo_->blockSignals(false);
    }

    void showSelectedPerson() {
        if (peopleTable_ == nullptr || peopleTable_->selectionModel() == nullptr) {
            return;
        }
        const auto selected = peopleTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        if (row < 0 || row >= static_cast<int>(peopleRows_.size())) {
            return;
        }
        const auto& person = peopleRows_[static_cast<size_t>(row)];
        currentPersonId_ = person.id;
        if (assignPersonSpin_ != nullptr) {
            assignPersonSpin_->setValue(static_cast<int>(person.id));
        }
        if (peopleNameEdit_ != nullptr) {
            peopleNameEdit_->setText(qs(person.name));
        }
        if (peopleNotesEdit_ != nullptr) {
            peopleNotesEdit_->setPlainText(qs(person.notes));
        }
        refreshPeopleMergeTargets(person.id);
        loadPeopleMembers(person.id);
        if (person.representativeFaceId > 0) {
            updatePeoplePreview(person.representativeFaceId);
        } else if (peoplePreviewLabel_ != nullptr) {
            peoplePreviewLabel_->setText("No faces");
            peoplePreviewLabel_->setPixmap(QPixmap());
        }
        if (peopleSummaryLabel_ != nullptr) {
            peopleSummaryLabel_->setText(
                QString("%1: %2 face(s), %3 review item(s), avg quality %4")
                    .arg(qs(person.name))
                    .arg(person.faceCount)
                    .arg(person.reviewCount)
                    .arg(person.averageQuality, 0, 'f', 3));
        }
        if (peopleProfileStatusLabel_ != nullptr) {
            if (!person.identityStatus.empty()) {
                peopleProfileStatusLabel_->setText(
                    QString("Identity profile: %1, samples %2, exemplars %3, accept %4, health %5, scorer %6")
                        .arg(qs(person.identityStatus))
                        .arg(person.identitySampleCount)
                        .arg(person.identityExemplarCount)
                        .arg(person.identityAcceptThreshold, 0, 'f', 3)
                        .arg(person.identityHealth.empty() ? "unknown" : qs(person.identityHealth))
                        .arg(person.identityScoringModelVersion.empty() ? "unknown" : qs(person.identityScoringModelVersion)));
            } else {
                peopleProfileStatusLabel_->setText("Identity profile: not trained");
            }
        }
    }

    void loadPeopleMembers(int64_t personId) {
        if (!database_ || peopleMemberTable_ == nullptr || personId <= 0) {
            return;
        }
        peopleMembers_ = database_->loadFacesForPerson(personId, true);
        peopleMemberTable_->setRowCount(static_cast<int>(peopleMembers_.size()));
        for (int row = 0; row < static_cast<int>(peopleMembers_.size()); ++row) {
            const auto& record = peopleMembers_[static_cast<size_t>(row)];
            peopleMemberTable_->setItem(row, 0, item(QString::number(record.id)));
            peopleMemberTable_->setItem(row, 1, item(qs(record.fileName)));
            peopleMemberTable_->setItem(row, 2, item(qs(record.tagText)));
            peopleMemberTable_->setItem(row, 3, numberItem(record.qualityScore, 3));
            peopleMemberTable_->setItem(row, 4, item(qs(record.reviewState)));
            peopleMemberTable_->setItem(row, 5, item(record.ignored ? "yes" : ""));
        }
        peopleMemberTable_->resizeColumnsToContents();
    }

    void showSelectedPeopleMember() {
        if (peopleMemberTable_ == nullptr || peopleMemberTable_->selectionModel() == nullptr) {
            return;
        }
        const auto selected = peopleMemberTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        if (row < 0 || row >= static_cast<int>(peopleMembers_.size())) {
            return;
        }
        const auto faceId = peopleMembers_[static_cast<size_t>(row)].id;
        if (assignFaceSpin_ != nullptr) {
            assignFaceSpin_->setValue(static_cast<int>(faceId));
        }
        updatePeoplePreview(faceId);
    }

    void updatePeoplePreview(int64_t faceId) {
        peoplePreviewFaceId_ = faceId;
        if (peopleFocusButton_ != nullptr) {
            peopleFocusButton_->setText(peopleFocusOnFace_ ? "Full Image" : "Focus on Face");
        }
        if (!database_ || peoplePreviewLabel_ == nullptr || faceId <= 0) {
            return;
        }
        try {
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                peoplePreviewLabel_->setText("Face not found");
                peoplePreviewLabel_->setPixmap(QPixmap());
                return;
            }
            setDatabaseFacePreview(peoplePreviewLabel_, *face, "No preview", peopleFocusOnFace_);
        } catch (const std::exception& ex) {
            peoplePreviewLabel_->setText(ex.what());
            peoplePreviewLabel_->setPixmap(QPixmap());
        }
    }

    void loadReview() {
        if (!database_ || reviewTable_ == nullptr) {
            return;
        }
        if (reviewDatabaseEdit_ != nullptr) {
            reviewDatabaseEdit_->setText(qs(database_->path().string()));
        }
        const QString textFilter = reviewFilterEdit_ == nullptr ? QString() : reviewFilterEdit_->text().trimmed();
        const int limit = reviewLimitSpin_ == nullptr ? 500 : reviewLimitSpin_->value();
        auto records = database_->loadFaces(true);
        records.erase(
            std::remove_if(records.begin(), records.end(), [textFilter](const auto& record) {
                if (!recordNeedsReview(record)) {
                    return true;
                }
                if (textFilter.isEmpty()) {
                    return false;
                }
                const QString haystack = QString("%1\n%2\n%3\n%4\n%5")
                                             .arg(qs(record.fileName), qs(record.sourcePath), qs(record.personName), qs(record.tagText), qs(record.notes));
                return !haystack.contains(textFilter, Qt::CaseInsensitive);
            }),
            records.end());
        if (limit > 0 && records.size() > static_cast<size_t>(limit)) {
            records.resize(static_cast<size_t>(limit));
        }
        reviewRows_ = std::move(records);
        reviewTable_->setRowCount(0);
        for (const auto& record : reviewRows_) {
            const int row = reviewTable_->rowCount();
            reviewTable_->insertRow(row);
            reviewTable_->setItem(row, 0, item(QString::number(record.id)));
            reviewTable_->setItem(row, 1, item(qs(record.fileName)));
            reviewTable_->setItem(row, 2, item(reviewReason(record)));
            reviewTable_->setItem(row, 3, item(qs(record.personName)));
            reviewTable_->setItem(row, 4, item(qs(record.tagText)));
            reviewTable_->setItem(row, 5, numberItem(record.qualityScore, 3));
            reviewTable_->setItem(row, 6, item(record.duplicateCount > 1 ? QString::number(record.duplicateCount) : ""));
            reviewTable_->setItem(row, 7, item(qs(record.notes)));
        }
        reviewTable_->resizeColumnsToContents();
        currentReviewFaceId_ = 0;
        reviewSuggestedPersonId_ = 0;
        reviewSuggestedPersonName_.clear();
        if (reviewPreviewLabel_ != nullptr) {
            reviewPreviewLabel_->setText(reviewRows_.empty() ? "No review items" : "Select a review item");
            reviewPreviewLabel_->setPixmap(QPixmap());
        }
        if (reviewSuggestionLabel_ != nullptr) {
            reviewSuggestionLabel_->setText("AI Suggested Person: not checked");
        }
        statusBar()->showMessage(QString("Review queue: %1 item(s)").arg(reviewRows_.size()));
    }

    static bool recordNeedsReview(const fsc::core::FaceRecord& record) {
        return record.ignored ||
            record.reviewState != "reviewed" ||
            record.personId <= 0 ||
            record.duplicateCount > 1 ||
            record.qualityScore < 0.45;
    }

    static QString reviewReason(const fsc::core::FaceRecord& record) {
        QStringList reasons;
        if (record.ignored) {
            reasons << "ignored";
        }
        if (record.reviewState != "reviewed") {
            reasons << qs(record.reviewState);
        }
        if (record.personId <= 0) {
            reasons << "unassigned";
        }
        if (record.duplicateCount > 1) {
            reasons << "duplicate";
        }
        if (record.qualityScore < 0.45) {
            reasons << "low quality";
        }
        reasons.removeDuplicates();
        return reasons.empty() ? "review" : reasons.join(", ");
    }

    int selectedReviewFaceId() const {
        if (reviewTable_ == nullptr || reviewTable_->selectionModel() == nullptr) {
            return 0;
        }
        const auto selected = reviewTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return 0;
        }
        const int row = selected.front().row();
        const auto* idItem = reviewTable_->item(row, 0);
        return idItem == nullptr ? 0 : idItem->text().toInt();
    }

    void showSelectedReviewRecord() {
        if (reviewTable_ == nullptr || reviewTable_->selectionModel() == nullptr) {
            return;
        }
        const auto selected = reviewTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return;
        }
        const int row = selected.front().row();
        if (row < 0 || row >= static_cast<int>(reviewRows_.size())) {
            return;
        }
        const auto& record = reviewRows_[static_cast<size_t>(row)];
        currentReviewFaceId_ = record.id;
        if (reviewPersonEdit_ != nullptr) {
            reviewPersonEdit_->setText(qs(record.personName));
        }
        if (reviewTagsEdit_ != nullptr) {
            reviewTagsEdit_->setText(qs(record.tagText));
        }
        if (reviewStateCombo_ != nullptr) {
            reviewStateCombo_->setCurrentText(qs(record.reviewState));
        }
        if (reviewIgnoredCheck_ != nullptr) {
            reviewIgnoredCheck_->setChecked(record.ignored);
        }
        if (reviewNotesEdit_ != nullptr) {
            reviewNotesEdit_->setPlainText(qs(record.notes));
        }
        updateReviewDetail(static_cast<int>(record.id), true);
    }

    void updateReviewDetail(int faceId, bool refreshSuggestion = true) {
        if (refreshSuggestion) {
            reviewSuggestedPersonId_ = 0;
            reviewSuggestedPersonName_.clear();
        }
        if (refreshSuggestion && reviewSuggestionLabel_ != nullptr) {
            reviewSuggestionLabel_->setText("AI Suggested Person: not checked");
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
            if (reviewFocusButton_ != nullptr) {
                reviewFocusButton_->setText(reviewFocusOnFace_ ? "Full Image" : "Focus on Face");
            }
            setDatabaseFacePreview(reviewPreviewLabel_, *face, "No preview", reviewFocusOnFace_);
            if (refreshSuggestion) {
                updateReviewSuggestion(*face);
            }
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

    void runRuntimeLegacyConversion(QString source = {}) {
#ifdef FSC_ENABLE_ONNX
        if (source.isEmpty()) {
            source = QFileDialog::getOpenFileName(
                this,
                "Select legacy database",
                {},
                "Legacy dlib database (*.dtb);;All Files (*)");
        }
        if (source.isEmpty()) {
            return;
        }
        const auto sourcePath = pathFrom(source);
        const auto defaultOutput = sourcePath.parent_path() /
            (sourcePath.stem().string() + "_insightface.fscdb");
        const auto output = QFileDialog::getSaveFileName(
            this,
            "Save converted database",
            qs(defaultOutput.string()),
            "FSC Database (*.fscdb);;All Files (*)");
        if (output.isEmpty()) {
            return;
        }

        if (runtimeLegacyConvertButton_ != nullptr) {
            runtimeLegacyConvertButton_->setEnabled(false);
        }
        if (runtimeLegacyProgressBar_ != nullptr) {
            runtimeLegacyProgressBar_->setRange(0, 1);
            runtimeLegacyProgressBar_->setValue(0);
        }
        try {
            fsc::legacy::LegacyConversionOptions options;
            options.models = fsc::vision::InsightFaceModelPaths::fromBuffaloL(pathFrom(defaultModelRoot()));
            options.runtimeMode = selectedRuntimeMode();
            options.progress = [this](const std::string& message, int current, int total) {
                if (runtimeLegacyProgressBar_ != nullptr) {
                    runtimeLegacyProgressBar_->setRange(0, std::max(1, total));
                    runtimeLegacyProgressBar_->setValue(current);
                }
                if (runtimeMaintenanceLog_ != nullptr &&
                    (total <= 50 || current == total || current % 10 == 0 || message.find("skipped") != std::string::npos)) {
                    runtimeMaintenanceLog_->append(QString("[%1/%2] %3").arg(current).arg(total).arg(qs(message)));
                }
                statusBar()->showMessage(QString("Converting legacy DTB: %1/%2").arg(current).arg(total));
                QApplication::processEvents();
            };
            const auto summary = fsc::legacy::convertLegacyDtb(sourcePath, pathFrom(output), options);
            if (runtimeMaintenanceLog_ != nullptr) {
                runtimeMaintenanceLog_->append(
                    QString("Converted legacy DTB: saved %1, skipped %2, total %3\n%4")
                        .arg(summary.facesSaved)
                        .arg(summary.skippedRows)
                        .arg(summary.rowsTotal)
                        .arg(qs(summary.outputPath.string())));
            }
            openDatabase(qs(summary.outputPath.string()));
            statusBar()->showMessage("Legacy DTB conversion complete");
        } catch (const std::exception& ex) {
            showError(ex);
        }
        if (runtimeLegacyConvertButton_ != nullptr) {
            runtimeLegacyConvertButton_->setEnabled(true);
        }
#else
        showError(std::runtime_error("This build does not include ONNX Runtime, so legacy DTB conversion is unavailable."));
#endif
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
            const std::string effectiveNotes = notes.empty() && reviewNotesEdit_ != nullptr
                ? std::string(reviewNotesEdit_->toPlainText().toUtf8().constData())
                : notes;
            database_->updateFaceReview(faceId, state, ignored, effectiveNotes);
            reloadAll();
            statusBar()->showMessage("Review updated");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void saveReviewMetadata() {
        if (!database_) {
            return;
        }
        try {
            const int faceId = selectedReviewFaceId();
            if (faceId <= 0) {
                throw std::runtime_error("Select a review row first.");
            }
            assignFaceToPersonName(faceId, reviewPersonEdit_ == nullptr ? QString() : reviewPersonEdit_->text());
            database_->setFaceTags(faceId, reviewTagsEdit_ == nullptr ? std::string() : reviewTagsEdit_->text().toUtf8().constData(), false);
            database_->updateFaceReview(
                faceId,
                reviewStateCombo_ == nullptr ? "open" : reviewStateCombo_->currentText().toStdString(),
                reviewIgnoredCheck_ != nullptr && reviewIgnoredCheck_->isChecked(),
                reviewNotesEdit_ == nullptr ? std::string() : std::string(reviewNotesEdit_->toPlainText().toUtf8().constData()));
            reloadAll();
            statusBar()->showMessage("Review metadata saved");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void toggleReviewIgnored() {
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
            const bool ignored = !face->ignored;
            database_->updateFaceReview(faceId, ignored ? "ignored" : "open", ignored, face->notes);
            reloadAll();
            statusBar()->showMessage(ignored ? "Review item ignored" : "Review item restored");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void updateReviewSuggestion(const fsc::core::FaceRecord& face) {
        reviewSuggestedPersonId_ = 0;
        reviewSuggestedPersonName_.clear();
        if (reviewSuggestionLabel_ == nullptr || !database_) {
            return;
        }
        try {
            const auto result = fsc::core::identifyPerson(database_->loadIdentityProfiles(), face.embedding, selectedIdentityMode(), 3);
            if (result.candidates.empty() || result.decision == "unknown") {
                reviewSuggestionLabel_->setText("AI Suggested Person: unknown");
                return;
            }
            const auto& candidate = result.candidates.front();
            reviewSuggestedPersonId_ = candidate.profile.personId;
            reviewSuggestedPersonName_ = candidate.profile.personName;
            reviewSuggestionLabel_->setText(
                QString("%1 | %2 | score %3 | confidence %4% | margin %5 | evidence face %6")
                    .arg(qs(candidate.profile.personName))
                    .arg(qs(result.decision))
                    .arg(candidate.score, 0, 'f', 4)
                    .arg(candidate.confidence * 100.0, 0, 'f', 1)
                    .arg(candidate.margin, 0, 'f', 4)
                    .arg(candidate.evidenceFaceId));
        } catch (const std::exception& ex) {
            reviewSuggestionLabel_->setText(QString("AI Suggested Person: %1").arg(QString::fromUtf8(ex.what())));
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
            updateReviewSuggestion(*face);
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

    void rejectReviewSuggestion() {
        if (!database_) {
            return;
        }
        try {
            const int faceId = selectedReviewFaceId();
            if (faceId <= 0) {
                throw std::runtime_error("Select a review row first.");
            }
            if (reviewSuggestedPersonName_.empty()) {
                throw std::runtime_error("No AI suggested person is available.");
            }
            const auto face = database_->loadFace(faceId);
            if (!face.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            std::string notes = reviewNotesEdit_ == nullptr
                ? face->notes
                : std::string(reviewNotesEdit_->toPlainText().toUtf8().constData());
            const std::string rejection = "AI suggestion rejected: " + reviewSuggestedPersonName_;
            if (notes.find(rejection) == std::string::npos) {
                if (!notes.empty()) {
                    notes += "\n";
                }
                notes += rejection;
            }
            database_->updateFaceReview(faceId, face->reviewState, face->ignored, notes);
            if (reviewNotesEdit_ != nullptr) {
                reviewNotesEdit_->setPlainText(qs(notes));
            }
            if (reviewSuggestionLabel_ != nullptr) {
                reviewSuggestionLabel_->setText("Rejected: " + qs(reviewSuggestedPersonName_));
            }
            statusBar()->showMessage(QString("Rejected AI suggestion for face %1").arg(faceId));
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

    void saveSelectedPerson() {
        if (!database_ || currentPersonId_ <= 0) {
            showError(std::runtime_error("Select a person first."));
            return;
        }
        try {
            const std::string name = peopleNameEdit_ == nullptr
                ? std::string()
                : std::string(peopleNameEdit_->text().trimmed().toUtf8().constData());
            const std::string notes = peopleNotesEdit_ == nullptr
                ? std::string()
                : std::string(peopleNotesEdit_->toPlainText().toUtf8().constData());
            database_->renamePerson(currentPersonId_, name, notes);
            reloadAll();
            statusBar()->showMessage("Person updated");
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void mergeSelectedPerson() {
        if (!database_ || currentPersonId_ <= 0) {
            showError(std::runtime_error("Select a person first."));
            return;
        }
        try {
            const int64_t targetId = peopleMergeTargetCombo_ == nullptr ? 0 : peopleMergeTargetCombo_->currentData().toLongLong();
            if (targetId <= 0) {
                throw std::runtime_error("Select a merge target first.");
            }
            const int moved = database_->mergePeople(currentPersonId_, targetId);
            reloadAll();
            statusBar()->showMessage(QString("Merged %1 face(s)").arg(moved));
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void clearSelectedPerson() {
        if (!database_ || currentPersonId_ <= 0) {
            showError(std::runtime_error("Select a person first."));
            return;
        }
        const QString name = peopleNameEdit_ == nullptr ? QString("this person") : peopleNameEdit_->text();
        if (QMessageBox::question(
                this,
                "FSC Studio Native",
                QString("Clear all assignments for %1 and delete this person?").arg(name)) != QMessageBox::Yes) {
            return;
        }
        try {
            const int cleared = database_->clearPersonAssignment(currentPersonId_, true);
            reloadAll();
            statusBar()->showMessage(QString("Cleared %1 face assignment(s)").arg(cleared));
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

    void syncSearchDatabasePath() {
        if (searchDatabasePathEdit_ == nullptr) {
            return;
        }
        searchDatabasePathEdit_->setText(
            database_ ? QString::fromStdWString(database_->path().wstring()) : QString());
        refreshSearchFilterOptions();
    }

    void selectSearchQueryImage() {
        const auto path = QFileDialog::getOpenFileName(
            this,
            "Select query image",
            {},
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.ppm);;All Files (*)");
        if (path.isEmpty()) {
            return;
        }
        searchImageEdit_->setText(path);
        analyzeSearchImage();
    }

    void analyzeSearchImage() {
#ifdef FSC_ENABLE_ONNX
        if (searchImageEdit_ == nullptr || searchImageEdit_->text().trimmed().isEmpty()) {
            showError(std::runtime_error("Select a query image first."));
            return;
        }
        const QString imagePathText = searchImageEdit_->text();
        const auto imagePath = pathFrom(imagePathText);
        const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
        const auto runtimeMode = selectedRuntimeMode();
        const auto analyzer = sharedFaceAnalyzer_;
        const int generation = ++searchQueryGeneration_;
        searchQueryAnalysisActive_ = true;

        resetSearchProgress();
        searchQueryFaces_.clear();
        searchQueryFaceIndex_ = 0;
        searchHits_.clear();
        lastSearchIdentityResult_ = {};
        searchResultPreviewFaceId_ = 0;
        searchResultFocusOnFace_ = false;
        if (searchFaceList_ != nullptr) {
            searchFaceList_->clear();
        }
        if (searchQueryPreview_ != nullptr) {
            searchQueryPreview_->setImagePath(imagePathText);
            searchQueryPreview_->setFaces({}, 0);
        }
        if (searchQueryFocusButton_ != nullptr) {
            searchQueryFocusButton_->setText(trUi("Focus on Face"));
            searchQueryFocusButton_->setEnabled(false);
        }
        if (searchResultFocusButton_ != nullptr) {
            searchResultFocusButton_->setText(trUi("Focus on Face"));
            searchResultFocusButton_->setEnabled(false);
        }
        if (searchTable_ != nullptr) {
            searchTable_->setRowCount(0);
        }
        if (searchIdentityTable_ != nullptr) {
            searchIdentityTable_->setRowCount(0);
        }
        if (searchResultPreviewLabel_ != nullptr) {
            searchResultPreviewLabel_->setText("Result");
            searchResultPreviewLabel_->setPixmap(QPixmap());
        }
        if (identityLabel_ != nullptr) {
            identityLabel_->setText("Identity: not searched");
        }
        statusBar()->showMessage("Detecting query faces...");

        auto* watcher = new QFutureWatcher<SearchQueryAnalysisResult>(this);
        connect(watcher, &QFutureWatcher<SearchQueryAnalysisResult>::finished, this, [this, watcher] {
            const auto result = watcher->result();
            watcher->deleteLater();
            finishSearchImageAnalysis(result);
        });
        watcher->setFuture(QtConcurrent::run([generation, imagePathText, imagePath, modelRoot, runtimeMode, analyzer] {
            SearchQueryAnalysisResult result;
            result.generation = generation;
            result.imagePath = imagePathText;
            try {
                result.faces = analyzer->analyze(modelRoot, runtimeMode, imagePath);
                if (result.faces.empty()) {
                    result.error = "No face detected in the query image.";
                }
            } catch (const std::exception& ex) {
                result.error = QString::fromUtf8(ex.what());
            }
            return result;
        }));
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void finishSearchImageAnalysis(const SearchQueryAnalysisResult& result) {
        if (result.generation != searchQueryGeneration_ || searchImageEdit_ == nullptr ||
            result.imagePath != searchImageEdit_->text()) {
            return;
        }
        searchQueryAnalysisActive_ = false;
        if (!result.error.isEmpty()) {
            statusBar()->showMessage(result.error);
            if (!searchQuerySmokeMode_) {
                showError(std::runtime_error(result.error.toUtf8().constData()));
            }
            return;
        }
        searchQueryFaces_ = result.faces;
        if (searchFaceList_ != nullptr) {
            searchFaceList_->blockSignals(true);
            searchFaceList_->clear();
            for (int index = 0; index < static_cast<int>(searchQueryFaces_.size()); ++index) {
                const auto& face = searchQueryFaces_[static_cast<size_t>(index)];
                searchFaceList_->addItem(
                    QString("Face %1: det %2, quality %3")
                        .arg(index + 1)
                        .arg(face.detection.score, 0, 'f', 3)
                        .arg(face.qualityScore, 0, 'f', 3));
            }
            searchFaceList_->blockSignals(false);
            searchFaceList_->setCurrentRow(0);
        }
        searchQueryFaceIndex_ = 0;
        updateSearchQueryPreview();
        statusBar()->showMessage(
            QString("Detected %1 face(s). Select a face, then search.").arg(searchQueryFaces_.size()));
    }

    void updateSearchQueryPreview() {
        if (searchQueryPreview_ == nullptr) {
            return;
        }
        if (searchImageEdit_ != nullptr) {
            searchQueryPreview_->setImagePath(searchImageEdit_->text());
        }
        searchQueryPreview_->setFaces(searchQueryFaces_, searchQueryFaceIndex_);
        if (searchQueryFocusButton_ != nullptr) {
            searchQueryFocusButton_->setText(searchQueryPreview_->focusOnFace() ? trUi("View Full Image") : trUi("Focus on Face"));
            searchQueryFocusButton_->setEnabled(!searchQueryFaces_.empty());
        }
    }

    void toggleSearchQueryFocus() {
        if (searchQueryPreview_ == nullptr || searchQueryFaces_.empty()) {
            return;
        }
        searchQueryPreview_->setFocusOnFace(!searchQueryPreview_->focusOnFace());
        if (searchQueryFocusButton_ != nullptr) {
            searchQueryFocusButton_->setText(searchQueryPreview_->focusOnFace() ? trUi("View Full Image") : trUi("Focus on Face"));
        }
    }

    void toggleSearchResultFocus() {
        if (searchResultPreviewFaceId_ <= 0) {
            return;
        }
        searchResultFocusOnFace_ = !searchResultFocusOnFace_;
        if (searchResultFocusButton_ != nullptr) {
            searchResultFocusButton_->setText(searchResultFocusOnFace_ ? trUi("View Full Image") : trUi("Focus on Face"));
        }
        updateSearchResultPreviewForFace(searchResultPreviewFaceId_);
    }

    std::vector<float> currentSearchEmbedding() const {
        if (!searchQueryFaces_.empty() && searchQueryFaceIndex_ >= 0 && searchQueryFaceIndex_ < static_cast<int>(searchQueryFaces_.size())) {
            return searchQueryFaces_[static_cast<size_t>(searchQueryFaceIndex_)].embedding;
        }
        throw std::runtime_error("Select a query image and wait for face detection first.");
    }

    void runSearch() {
        if (!database_) {
            return;
        }
        try {
            resetSearchProgress();
            const auto embedding = currentSearchEmbedding();
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
            searchHits_.clear();
            searchTable_->setRowCount(0);
            searchIdentityTable_->setRowCount(0);
            lastSearchIdentityResult_ = {};
            if (identityLabel_ != nullptr) {
                identityLabel_->setText("Identity: searching...");
            }
            if (searchResultPreviewLabel_ != nullptr) {
                searchResultPreviewLabel_->setText("Comparing database faces...");
                searchResultPreviewLabel_->setPixmap(QPixmap());
            }
            searchResultPreviewFaceId_ = 0;
            searchResultFocusOnFace_ = false;
            if (searchResultFocusButton_ != nullptr) {
                searchResultFocusButton_->setText(trUi("Focus on Face"));
                searchResultFocusButton_->setEnabled(false);
            }
            beginSearchProgress(std::move(records), embedding);
        } catch (const std::exception& ex) {
            showError(ex);
        }
    }

    void resetSearchProgress() {
        if (searchPreviewTimer_ != nullptr) {
            searchPreviewTimer_->stop();
        }
        searchProgressActive_ = false;
        searchProgressRecords_.clear();
        searchProgressHits_.clear();
        searchProgressQuery_.clear();
        searchProgressCursor_ = 0;
        searchProgressPreviewLastMs_ = -1;
    }

    void beginSearchProgress(std::vector<fsc::core::FaceRecord> records, const std::vector<float>& embedding) {
        searchProgressRecords_ = std::move(records);
        searchProgressQuery_ = fsc::core::normalize(embedding);
        searchProgressTopK_ = topKSpin_ != nullptr ? topKSpin_->value() : 30;
        searchProgressThreshold_ = searchThresholdSpin_ != nullptr ? searchThresholdSpin_->value() : -1.0;
        searchProgressCursor_ = 0;
        searchProgressHits_.clear();
        searchProgressPreviewLastMs_ = -1;
        searchProgressClock_.start();
        searchProgressActive_ = true;
        if (searchProgressRecords_.empty()) {
            finishSearchProgress();
            return;
        }
        statusBar()->showMessage(QString("Comparing 0/%1 database faces").arg(searchProgressRecords_.size()));
        if (searchPreviewTimer_ != nullptr) {
            // Computation yields after a short time slice. Rendering is separately throttled.
            searchPreviewTimer_->setInterval(0);
            searchPreviewTimer_->start();
        }
    }

    void showSearchProgressPreview(const fsc::core::FaceRecord& record, size_t current, size_t total) {
        updateSearchResultPreviewForFace(record.id);
        searchProgressPreviewLastMs_ = searchProgressClock_.elapsed();
        statusBar()->showMessage(QString("Comparing %1/%2: %3")
                                     .arg(current)
                                     .arg(total)
                                     .arg(qs(record.fileName)));
    }

    void advanceSearchProgress() {
        if (!searchProgressActive_) {
            return;
        }
        const size_t total = searchProgressRecords_.size();
        if (searchProgressCursor_ == 0 && total > 0) {
            showSearchProgressPreview(searchProgressRecords_.front(), 1, total);
        }

        QElapsedTimer workSlice;
        workSlice.start();
        size_t processed = 0;
        while (searchProgressCursor_ < total && (processed < 64 || workSlice.elapsed() < 6)) {
            const auto& record = searchProgressRecords_[searchProgressCursor_];
            if (record.embedding.size() == searchProgressQuery_.size()) {
                const double score = fsc::core::dot(record.embedding, searchProgressQuery_);
                if (score >= searchProgressThreshold_) {
                    searchProgressHits_.push_back({record, score});
                }
            }
            ++searchProgressCursor_;
            ++processed;
        }

        if (searchProgressCursor_ == 0) {
            finishSearchProgress();
            return;
        }
        const bool completed = searchProgressCursor_ >= total;
        const qint64 now = searchProgressClock_.elapsed();
        if (completed || searchProgressPreviewLastMs_ < 0 || now - searchProgressPreviewLastMs_ >= 80) {
            showSearchProgressPreview(searchProgressRecords_[searchProgressCursor_ - 1], searchProgressCursor_, total);
        }
        if (completed) {
            finishSearchProgress();
        }
    }

    void finishSearchProgress() {
        if (searchPreviewTimer_ != nullptr) {
            searchPreviewTimer_->stop();
            searchPreviewTimer_->setInterval(70);
        }
        searchProgressActive_ = false;
        std::sort(searchProgressHits_.begin(), searchProgressHits_.end(), [](const auto& left, const auto& right) {
            return left.cosine > right.cosine;
        });
        if (searchProgressTopK_ > 0 && searchProgressHits_.size() > static_cast<size_t>(searchProgressTopK_)) {
            searchProgressHits_.resize(static_cast<size_t>(searchProgressTopK_));
        }
        searchHits_ = std::move(searchProgressHits_);
        const auto completedQuery = searchProgressQuery_;
        searchProgressRecords_.clear();
        searchProgressQuery_.clear();

        searchTable_->setRowCount(static_cast<int>(searchHits_.size()));
        for (int row = 0; row < static_cast<int>(searchHits_.size()); ++row) {
            const auto& hit = searchHits_[static_cast<size_t>(row)];
            searchTable_->setItem(row, 0, item(QString::number(row + 1)));
            searchTable_->setItem(row, 1, item(QString::number(hit.record.id)));
            searchTable_->setItem(row, 2, item(qs(hit.record.fileName)));
            searchTable_->setItem(row, 3, item(qs(hit.record.personName)));
            searchTable_->setItem(row, 4, item(qs(hit.record.tagText)));
            searchTable_->setItem(row, 5, numberItem(hit.cosine, 4));
            searchTable_->setItem(row, 6, numberItem(hit.similarityPercent(), 2));
            searchTable_->setItem(row, 7, numberItem(hit.record.qualityScore, 3));
        }
        populateSearchIdentityResult(fsc::core::identifyPerson(
            database_->loadIdentityProfiles(),
            completedQuery,
            selectedIdentityMode(),
            5));
        searchTable_->resizeColumnsToContents();
        if (!searchHits_.empty()) {
            searchTable_->selectRow(0);
            updateSearchResultPreviewForFace(searchHits_.front().record.id);
            statusBar()->showMessage(QString("Search complete: %1 result(s). Best match: face %2")
                                         .arg(searchHits_.size())
                                         .arg(searchHits_.front().record.id));
        } else {
            searchResultPreviewFaceId_ = 0;
            if (searchResultPreviewLabel_ != nullptr) {
                searchResultPreviewLabel_->setText("No results");
                searchResultPreviewLabel_->setPixmap(QPixmap());
            }
            if (searchResultFocusButton_ != nullptr) {
                searchResultFocusButton_->setEnabled(false);
            }
            statusBar()->showMessage("Search complete: 0 result(s)");
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
            const auto face = database_->loadFacePreview(faceId);
            if (!face.has_value()) {
                searchResultPreviewLabel_->setText("Face not found");
                searchResultPreviewLabel_->setPixmap(QPixmap());
                searchResultPreviewFaceId_ = 0;
                if (searchResultFocusButton_ != nullptr) {
                    searchResultFocusButton_->setEnabled(false);
                }
                return;
            }
            searchResultPreviewFaceId_ = faceId;
            setDatabaseFacePreview(searchResultPreviewLabel_, *face, "Result preview", searchResultFocusOnFace_);
            if (searchResultFocusButton_ != nullptr) {
                searchResultFocusButton_->setText(
                    searchResultFocusOnFace_ ? trUi("View Full Image") : trUi("Focus on Face"));
                searchResultFocusButton_->setEnabled(true);
            }
        } catch (const std::exception& ex) {
            searchResultPreviewLabel_->setText(ex.what());
            searchResultPreviewLabel_->setPixmap(QPixmap());
            searchResultPreviewFaceId_ = 0;
            if (searchResultFocusButton_ != nullptr) {
                searchResultFocusButton_->setEnabled(false);
            }
        }
    }

    void updateSelectedSearchResultPreview() {
        if (searchTable_ == nullptr || searchTable_->selectionModel() == nullptr) {
            return;
        }
        if (searchProgressActive_) {
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

    std::optional<CameraResultActionRow> selectedCameraResultRow() const {
        if (cameraResultTable_ == nullptr || cameraResultTable_->selectionModel() == nullptr) {
            return std::nullopt;
        }
        const auto selected = cameraResultTable_->selectionModel()->selectedRows();
        if (selected.empty()) {
            return std::nullopt;
        }
        const int row = selected.front().row();
        if (row < 0 || row >= static_cast<int>(cameraResultRows_.size())) {
            return std::nullopt;
        }
        return cameraResultRows_[static_cast<size_t>(row)];
    }

    int64_t cameraActionFaceId(const CameraResultActionRow& row) const {
        return row.hitFaceId > 0 ? row.hitFaceId : row.evidenceFaceId;
    }

    bool usefulCameraPersonName(const QString& value) const {
        const auto text = value.trimmed();
        return !text.isEmpty() && text != "--" && text != "unknown" && text != "no database";
    }

    void updateSelectedCameraResultPreview() {
        const auto row = selectedCameraResultRow();
        if (!row.has_value()) {
            return;
        }
        const int64_t previewFaceId = cameraActionFaceId(*row);
        if (previewFaceId <= 0) {
            setCameraMatchPlaceholder("Match: selected row has no stored database face");
            return;
        }
        QString status = QString("Selected: camera face #%1 | database face %2")
                             .arg(row->faceIndex + 1)
                             .arg(previewFaceId);
        if (row->identityPersonId > 0 && usefulCameraPersonName(row->identityName)) {
            status += QString(" | identity %1 (%2)").arg(row->identityName, row->decision);
        }
        if (row->hitFaceId > 0 && row->hitCosine > -1.5) {
            status += QString(" | cosine %1").arg(row->hitCosine, 0, 'f', 4);
        }
        updateCameraMatchPreview(previewFaceId, status);
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

    void syncCameraDatabasePath() {
        if (cameraDatabasePathEdit_ == nullptr) {
            return;
        }
        cameraDatabasePathEdit_->setText(
            database_ ? QString::fromStdWString(database_->path().wstring()) : QString());
    }

    void setCameraMatchPlaceholder(const QString& status) {
        cameraMatchPreviewFaceId_ = 0;
        cameraMatchFocusOnFace_ = false;
        if (cameraMatchStatusLabel_ != nullptr) {
            cameraMatchStatusLabel_->setText(status);
        }
        if (cameraMatchFocusButton_ != nullptr) {
            cameraMatchFocusButton_->setText(trUi("Focus on Face"));
            cameraMatchFocusButton_->setEnabled(false);
        }
        if (cameraMatchPreviewLabel_ != nullptr) {
            cameraMatchPreviewLabel_->setPixmap(QPixmap());
            cameraMatchPreviewLabel_->setText("Match");
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
        const auto face = database_->loadFacePreview(faceId);
        if (!face.has_value()) {
            setCameraMatchPlaceholder(status);
            return;
        }
        cameraMatchPreviewFaceId_ = faceId;
        setDatabaseFacePreview(cameraMatchPreviewLabel_, *face, "Match", cameraMatchFocusOnFace_);
        if (cameraMatchFocusButton_ != nullptr) {
            cameraMatchFocusButton_->setText(
                cameraMatchFocusOnFace_ ? trUi("View Full Image") : trUi("Focus on Face"));
            cameraMatchFocusButton_->setEnabled(true);
        }
    }

    void toggleCameraMatchFocus() {
        if (cameraMatchPreviewFaceId_ <= 0) {
            return;
        }
        cameraMatchFocusOnFace_ = !cameraMatchFocusOnFace_;
        updateCameraMatchPreview(cameraMatchPreviewFaceId_, cameraMatchStatusLabel_->text());
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

    void startCamera() {
#ifdef FSC_ENABLE_OPENCV
        try {
            stopCamera();
            const int cameraIndex = cameraIndexSpin_->value();
            if (!camera_.open(cameraIndex, cv::CAP_DSHOW) && !camera_.open(cameraIndex)) {
                throw std::runtime_error("Could not open camera " + std::to_string(cameraIndex) + ".");
            }
            camera_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            camera_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            ++cameraSessionGeneration_;
            cameraAnalyzeBusy_ = false;
            cameraVotesByFace_.clear();
            cameraResultRows_.clear();
            if (cameraResultTable_ != nullptr) {
                cameraResultTable_->setRowCount(0);
            }
            latestCameraFaces_.clear();
            latestCameraMatchedFaceIndexes_.clear();
            latestCameraFacesAt_ = {};
            syncCameraDatabasePath();
            setCameraMatchPlaceholder("Match: waiting for frame");
            cameraFrameTimer_->start();
            cameraIdentifyTimer_->start();
            cameraStartButton_->setEnabled(false);
            cameraStopButton_->setEnabled(true);
            cameraIndexSpin_->setEnabled(false);
            statusBar()->showMessage(QString("Camera %1 started.").arg(cameraIndex));
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
        ++cameraSessionGeneration_;
        ++cameraAnalysisToken_;
        cameraAnalyzeBusy_ = false;
        lastCameraFrame_.release();
        cameraVotesByFace_.clear();
        cameraResultRows_.clear();
        if (cameraResultTable_ != nullptr) {
            cameraResultTable_->setRowCount(0);
        }
        latestCameraFaces_.clear();
        latestCameraMatchedFaceIndexes_.clear();
        latestCameraFacesAt_ = {};
        setCameraMatchPlaceholder("No camera result");
        if (cameraPreviewLabel_ != nullptr) {
            cameraPreviewLabel_->setPixmap(QPixmap());
            cameraPreviewLabel_->setText("Camera stopped");
        }
        if (cameraStartButton_ != nullptr) {
            cameraStartButton_->setEnabled(true);
        }
        if (cameraStopButton_ != nullptr) {
            cameraStopButton_->setEnabled(false);
        }
        if (cameraIndexSpin_ != nullptr) {
            cameraIndexSpin_->setEnabled(true);
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
        ++cameraCapturedFrameCount_;
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
            statusBar()->showMessage("No camera frame");
            return;
        }
        if (!database_ || !cameraStoredFaces_ || cameraStoredFaces_->empty()) {
            statusBar()->showMessage("Open a database before using camera recognition.");
            return;
        }

        cameraAnalyzeBusy_ = true;
        const uint64_t token = ++cameraAnalysisToken_;
        const uint64_t session = cameraSessionGeneration_;
        const QString databasePath = QString::fromStdWString(database_->path().wstring());
        const cv::Mat frame = lastCameraFrame_.clone();
        const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
        const auto runtimeMode = selectedRuntimeMode();
        const auto identityMode = selectedIdentityMode();
        const int processSize = cameraProcessSizeSpin_ != nullptr ? cameraProcessSizeSpin_->value() : 640;
        const int topK = cameraTopKSpin_ != nullptr ? cameraTopKSpin_->value() : 3;
        const double threshold = cameraThresholdSpin_ != nullptr ? cameraThresholdSpin_->value() : 0.35;
        const auto analyzer = cameraFaceAnalyzer_;
        const auto storedFaces = cameraStoredFaces_;
        const auto identityProfiles = cameraIdentityProfilesSnapshot_;

        auto* watcher = new QFutureWatcher<CameraFrameAnalysisResult>(this);
        connect(watcher, &QFutureWatcher<CameraFrameAnalysisResult>::finished, this, [this, watcher] {
            auto result = watcher->result();
            watcher->deleteLater();
            finishCameraFrameAnalysis(std::move(result));
        });
        watcher->setFuture(QtConcurrent::run(
            [token,
             session,
             databasePath,
             frame,
             modelRoot,
             runtimeMode,
             identityMode,
             processSize,
             topK,
             threshold,
             analyzer,
             storedFaces,
             identityProfiles]() mutable {
                CameraFrameAnalysisResult result;
                result.token = token;
                result.session = session;
                result.databasePath = databasePath;
                result.frame = frame;
                result.processSize = processSize;
                result.topK = topK;
                result.threshold = threshold;
                try {
                    cv::Mat recognitionFrame = frame;
                    double scaleX = 1.0;
                    double scaleY = 1.0;
                    const int longEdge = std::max(frame.cols, frame.rows);
                    if (processSize > 0 && longEdge > processSize) {
                        const double scale = static_cast<double>(processSize) / static_cast<double>(longEdge);
                        cv::resize(frame, recognitionFrame, cv::Size(), scale, scale, cv::INTER_AREA);
                        scaleX = static_cast<double>(frame.cols) / static_cast<double>(std::max(1, recognitionFrame.cols));
                        scaleY = static_cast<double>(frame.rows) / static_cast<double>(std::max(1, recognitionFrame.rows));
                    }
                    result.faces = analyzer->analyze(modelRoot, runtimeMode, rgbImageFromBgrMat(recognitionFrame));
                    if (scaleX != 1.0 || scaleY != 1.0) {
                        scaleAnalyzedFaceCoordinates(result.faces, scaleX, scaleY);
                    }
                    result.matches.reserve(result.faces.size());
                    for (const auto& face : result.faces) {
                        CameraFaceAnalysisResult match;
                        if (identityProfiles && !identityProfiles->empty()) {
                            match.identity = fsc::core::identifyPerson(*identityProfiles, face.embedding, identityMode, 3);
                        }
                        if (storedFaces && !storedFaces->empty()) {
                            match.hits = fsc::core::searchFaces(*storedFaces, face.embedding, topK, threshold, false);
                        }
                        result.matches.push_back(std::move(match));
                    }
                } catch (const std::exception& ex) {
                    result.error = QString::fromUtf8(ex.what());
                }
                return result;
            }));
#elif !defined(FSC_ENABLE_OPENCV)
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with OpenCV camera support.");
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

#if defined(FSC_ENABLE_OPENCV) && defined(FSC_ENABLE_ONNX)
    void finishCameraFrameAnalysis(CameraFrameAnalysisResult result) {
        if (result.token != cameraAnalysisToken_) {
            return;
        }
        cameraAnalyzeBusy_ = false;
        if (result.session != cameraSessionGeneration_) {
            return;
        }
        ++cameraCompletedAnalysisCount_;
        const QString currentDatabase = database_ ? QString::fromStdWString(database_->path().wstring()) : QString();
        if (result.databasePath != currentDatabase) {
            return;
        }
        if (!result.error.isEmpty()) {
            statusBar()->showMessage("Camera recognition error: " + result.error);
            return;
        }

        cameraResultTable_->blockSignals(true);
        cameraResultTable_->setRowCount(0);
        cameraResultRows_.clear();
        if (result.faces.empty()) {
            cameraVotesByFace_.clear();
            latestCameraFaces_.clear();
            latestCameraMatchedFaceIndexes_.clear();
            latestCameraFacesAt_ = {};
            setCameraMatchPlaceholder("No face detected in the current camera frame.");
            cameraResultTable_->blockSignals(false);
            statusBar()->showMessage("No face detected in the current camera frame.");
            return;
        }

        std::set<int> matchedFaceIndexes;
        int bestIdentityPriority = -1;
        double bestIdentityScore = -2.0;
        int bestIdentityRow = -1;
        int bestIdentityFaceIndex = -1;
        const fsc::core::IdentityCandidate* bestIdentityCandidate = nullptr;
        QString bestIdentityDisplayName;
        QString bestIdentityDecision;
        const fsc::core::SearchHit* bestHit = nullptr;
        int bestHitRow = -1;
        int bestHitFaceIndex = -1;

        const auto addRow = [&](int faceIndex,
                                const QString& identityName,
                                const QString& decision,
                                double confidence,
                                int64_t identityPersonId,
                                int64_t evidenceFaceId,
                                const fsc::core::SearchHit* hit) {
            const int row = cameraResultTable_->rowCount();
            cameraResultTable_->insertRow(row);
            cameraResultTable_->setItem(row, 0, item(QString::number(faceIndex + 1)));
            cameraResultTable_->setItem(row, 1, item(identityName));
            cameraResultTable_->setItem(row, 2, item(decision));
            cameraResultTable_->setItem(
                row,
                3,
                confidence >= 0.0 ? item(QString::number(confidence * 100.0, 'f', 1) + "%") : item(""));
            cameraResultTable_->setItem(row, 4, evidenceFaceId > 0 ? item(QString::number(evidenceFaceId)) : item(""));
            if (hit != nullptr) {
                cameraResultTable_->setItem(row, 5, item(QString::number(hit->record.id)));
                cameraResultTable_->setItem(row, 6, item(qs(hit->record.fileName)));
                cameraResultTable_->setItem(row, 7, item(qs(hit->record.personName)));
                cameraResultTable_->setItem(row, 8, numberItem(hit->cosine, 4));
                cameraResultTable_->setItem(row, 9, numberItem(hit->similarityPercent(), 2));
                cameraResultTable_->setItem(row, 10, numberItem(hit->record.qualityScore, 3));
            } else {
                for (int column = 5; column < 11; ++column) {
                    cameraResultTable_->setItem(row, column, item(""));
                }
            }
            CameraResultActionRow actionRow;
            actionRow.faceIndex = faceIndex;
            actionRow.identityPersonId = identityPersonId;
            actionRow.identityName = identityName;
            actionRow.decision = decision;
            actionRow.evidenceFaceId = evidenceFaceId;
            if (hit != nullptr) {
                actionRow.hitFaceId = hit->record.id;
                actionRow.hitCosine = hit->cosine;
            }
            cameraResultRows_.push_back(std::move(actionRow));
            return row;
        };

        for (int faceIndex = 0; faceIndex < static_cast<int>(result.faces.size()); ++faceIndex) {
            const auto& match = result.matches[static_cast<size_t>(faceIndex)];
            const auto& identity = match.identity;
            QString decision = qs(identity.decision);
            QString identityName;
            QString rawIdentityName;
            double confidence = -1.0;
            int64_t evidenceFaceId = 0;
            int64_t identityPersonId = 0;
            const fsc::core::IdentityCandidate* candidate = nullptr;
            if (!identity.candidates.empty()) {
                candidate = &identity.candidates.front();
                rawIdentityName = qs(candidate->profile.personName);
                confidence = candidate->confidence;
                evidenceFaceId = candidate->evidenceFaceId;
                identityPersonId = candidate->profile.personId;
            }
            if ((identity.decision == "confirmed" || identity.decision == "review") && candidate != nullptr) {
                identityName = smoothedCameraName(faceIndex, rawIdentityName);
                if (identityName.isEmpty()) {
                    identityName = rawIdentityName;
                } else if (identityName != rawIdentityName) {
                    decision = "smoothed";
                }
                matchedFaceIndexes.insert(faceIndex);
            } else {
                (void)smoothedCameraName(faceIndex, QString());
            }

            int firstRow = -1;
            if (!match.hits.empty()) {
                matchedFaceIndexes.insert(faceIndex);
                for (const auto& hit : match.hits) {
                    const int row = addRow(
                        faceIndex,
                        identityName,
                        decision,
                        confidence,
                        identityPersonId,
                        evidenceFaceId,
                        &hit);
                    if (firstRow < 0) {
                        firstRow = row;
                    }
                    if (bestHit == nullptr || hit.cosine > bestHit->cosine) {
                        bestHit = &hit;
                        bestHitRow = row;
                        bestHitFaceIndex = faceIndex;
                    }
                }
            } else {
                firstRow = addRow(
                    faceIndex,
                    identityName,
                    decision,
                    confidence,
                    identityPersonId,
                    evidenceFaceId,
                    nullptr);
            }

            if ((identity.decision == "confirmed" || identity.decision == "review") && candidate != nullptr) {
                const int priority = identity.decision == "confirmed" ? 2 : 1;
                if (priority > bestIdentityPriority ||
                    (priority == bestIdentityPriority && candidate->score > bestIdentityScore)) {
                    bestIdentityPriority = priority;
                    bestIdentityScore = candidate->score;
                    bestIdentityRow = firstRow;
                    bestIdentityFaceIndex = faceIndex;
                    bestIdentityCandidate = candidate;
                    bestIdentityDisplayName = identityName;
                    bestIdentityDecision = qs(identity.decision);
                }
            }
        }

        for (auto it = cameraVotesByFace_.begin(); it != cameraVotesByFace_.end();) {
            if (it->first >= static_cast<int>(result.faces.size())) {
                it = cameraVotesByFace_.erase(it);
            } else {
                ++it;
            }
        }
        latestCameraFaces_ = result.faces;
        latestCameraMatchedFaceIndexes_ = matchedFaceIndexes;
        latestCameraFacesAt_ = std::chrono::steady_clock::now();
        updateCameraPreviewPixmap(result.frame, &latestCameraFaces_, &latestCameraMatchedFaceIndexes_);

        QString status;
        int selectedRow = -1;
        if (bestIdentityCandidate != nullptr) {
            status = QString("Face %1: identity %2 (%3, confidence %4%, score %5)")
                         .arg(bestIdentityFaceIndex + 1)
                         .arg(bestIdentityDisplayName)
                         .arg(bestIdentityDecision)
                         .arg(bestIdentityCandidate->confidence * 100.0, 0, 'f', 1)
                         .arg(bestIdentityCandidate->score, 0, 'f', 4);
            if (bestIdentityCandidate->evidenceFaceId > 0) {
                updateCameraMatchPreview(bestIdentityCandidate->evidenceFaceId, status);
            } else {
                setCameraMatchPlaceholder(status);
            }
            selectedRow = bestIdentityRow;
        } else if (bestHit != nullptr) {
            status = QString("Face %1: best match %2, %3%")
                         .arg(bestHitFaceIndex + 1)
                         .arg(qs(bestHit->record.fileName))
                         .arg(bestHit->similarityPercent(), 0, 'f', 2);
            updateCameraMatchPreview(bestHit->record.id, status);
            selectedRow = bestHitRow;
        } else {
            status = QString("Detected %1 face(s), no database match above threshold.").arg(result.faces.size());
            setCameraMatchPlaceholder(status);
        }
        if (selectedRow >= 0) {
            cameraResultTable_->selectRow(selectedRow);
        }
        cameraResultTable_->blockSignals(false);
        cameraResultTable_->resizeColumnsToContents();
        statusBar()->showMessage(status);
    }
#endif

    void chooseImage(QLineEdit* target) {
        const auto path = QFileDialog::getOpenFileName(this, "Select image", {}, "Images (*.jpg *.jpeg *.png *.bmp *.ppm)");
        if (!path.isEmpty()) {
            target->setText(path);
        }
    }

    void selectCompareImage(char slot) {
        const auto path = QFileDialog::getOpenFileName(
            this,
            "Select image",
            {},
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.ppm);;All Files (*)");
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
        preview->setFaces({}, 0);
        preview->setFocusOnFace(false);
        auto* focusButton = slot == 'a' ? compareFocusAButton_ : compareFocusBButton_;
        if (focusButton != nullptr) {
            focusButton->setText(trUi("Focus on Face"));
            focusButton->setEnabled(false);
        }
        compareResultLabel_->setText("Cosine: --    Similarity: --");
        analyzeCompareImage(slot);
    }

    void analyzeCompareImage(char slot) {
#ifdef FSC_ENABLE_ONNX
        auto* edit = slot == 'a' ? compareImageAEdit_ : compareImageBEdit_;
        if (edit == nullptr || edit->text().isEmpty()) {
            showError(std::runtime_error("Select an image first."));
            return;
        }
        const int generation = slot == 'a' ? ++compareGenerationA_ : ++compareGenerationB_;
        if (slot == 'a') {
            compareAnalysisActiveA_ = true;
        } else {
            compareAnalysisActiveB_ = true;
        }
        const QString imagePathText = edit->text();
        const auto imagePath = pathFrom(imagePathText);
        const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
        const auto runtimeMode = selectedRuntimeMode();
        const auto analyzer = sharedFaceAnalyzer_;
        statusBar()->showMessage(QString("Detecting faces in image %1...").arg(slot == 'a' ? "A" : "B"));

        auto* watcher = new QFutureWatcher<CompareImageAnalysisResult>(this);
        connect(watcher, &QFutureWatcher<CompareImageAnalysisResult>::finished, this, [this, watcher] {
            auto result = watcher->result();
            watcher->deleteLater();
            finishCompareImageAnalysis(std::move(result));
        });
        watcher->setFuture(QtConcurrent::run(
            [slot, generation, imagePathText, imagePath, modelRoot, runtimeMode, analyzer] {
                CompareImageAnalysisResult result;
                result.slot = slot;
                result.generation = generation;
                result.imagePath = imagePathText;
                try {
                    result.faces = analyzer->analyze(modelRoot, runtimeMode, imagePath);
                } catch (const std::exception& ex) {
                    result.error = QString::fromUtf8(ex.what());
                }
                return result;
            }));
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void finishCompareImageAnalysis(CompareImageAnalysisResult result) {
        auto* edit = result.slot == 'a' ? compareImageAEdit_ : compareImageBEdit_;
        const int currentGeneration = result.slot == 'a' ? compareGenerationA_ : compareGenerationB_;
        if (edit == nullptr || result.generation != currentGeneration || result.imagePath != edit->text()) {
            return;
        }
        if (result.slot == 'a') {
            compareAnalysisActiveA_ = false;
        } else {
            compareAnalysisActiveB_ = false;
        }
        if (!result.error.isEmpty()) {
            statusBar()->showMessage(result.error);
            showError(std::runtime_error(result.error.toUtf8().constData()));
            return;
        }
        auto& faces = result.slot == 'a' ? compareFacesA_ : compareFacesB_;
        faces = std::move(result.faces);
        populateCompareFaces(result.slot);
        auto* focusButton = result.slot == 'a' ? compareFocusAButton_ : compareFocusBButton_;
        if (focusButton != nullptr) {
            focusButton->setEnabled(!faces.empty());
        }
        if (faces.empty()) {
            updateComparePreview(result.slot);
        } else {
            selectCompareFace(result.slot, 0);
        }
        statusBar()->showMessage(
            QString("Image %1: detected %2 face(s).")
                .arg(result.slot == 'a' ? "A" : "B")
                .arg(faces.size()));
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
        const auto& faces = slot == 'a' ? compareFacesA_ : compareFacesB_;
        if (faces.empty()) {
            return;
        }
        preview->setFocusOnFace(!preview->focusOnFace());
        button->setText(preview->focusOnFace() ? trUi("View Full Image") : trUi("Focus on Face"));
    }

    void compareImages() {
#ifdef FSC_ENABLE_ONNX
        try {
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
            libraryFocusButton_->setText(trUi("Focus on Face"));
            libraryFocusButton_->setEnabled(false);
        }
        if (libraryVisualTabs_ != nullptr) {
            libraryVisualTabs_->setCurrentIndex(0);
        }
        QImage image = loadPreviewImage(imagePath);
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
    bool ensureLibraryDatabaseForImport() {
        if (database_) {
            return true;
        }
        createDatabase();
        return database_ != nullptr;
    }

    void startLibraryImport(std::vector<QString> files) {
        if (!database_ || libraryImportActive_) {
            if (libraryImportActive_) {
                statusBar()->showMessage("An image import is already running.");
            }
            return;
        }
        files.erase(
            std::remove_if(files.begin(), files.end(), [](const QString& path) {
                return path.trimmed().isEmpty() || !isSupportedImageFile(path);
            }),
            files.end());
        std::sort(files.begin(), files.end(), [](const QString& left, const QString& right) {
            return QString::compare(left, right, Qt::CaseInsensitive) < 0;
        });
        files.erase(
            std::unique(files.begin(), files.end(), [](const QString& left, const QString& right) {
                return QString::compare(left, right, Qt::CaseInsensitive) == 0;
            }),
            files.end());
        if (files.empty()) {
            showError(std::runtime_error("No supported image files were selected."));
            return;
        }

        const uint64_t token = ++libraryImportToken_;
        const QString databasePath = QString::fromStdWString(database_->path().wstring());
        const auto modelRoot = pathFrom(modelRootEdit_ != nullptr ? modelRootEdit_->text() : defaultModelRoot());
        const auto runtimeMode = selectedRuntimeMode();
        const double minQuality = libraryImportMinQualitySpin_ != nullptr ? libraryImportMinQualitySpin_->value() : 0.0;
        const auto analyzer = sharedFaceAnalyzer_;
        const auto state = std::make_shared<LibraryImportTaskState>();
        libraryImportTaskState_ = state;
        libraryImportActive_ = true;
        if (libraryImportImagesButton_ != nullptr) {
            libraryImportImagesButton_->setEnabled(false);
        }
        if (libraryImportFolderButton_ != nullptr) {
            libraryImportFolderButton_->setEnabled(false);
        }
        if (libraryProgressBar_ != nullptr) {
            libraryProgressBar_->setRange(0, static_cast<int>(files.size()));
            libraryProgressBar_->setValue(0);
        }
        appendLibraryActivity(QString("Importing %1 image file(s)...").arg(files.size()));
        if (libraryImportProgressTimer_ != nullptr) {
            libraryImportProgressTimer_->start();
        }

        auto* watcher = new QFutureWatcher<LibraryImportSummary>(this);
        connect(watcher, &QFutureWatcher<LibraryImportSummary>::finished, this, [this, watcher] {
            auto summary = watcher->result();
            watcher->deleteLater();
            finishLibraryImport(std::move(summary));
        });
        watcher->setFuture(QtConcurrent::run(
            [token, databasePath, files = std::move(files), modelRoot, runtimeMode, minQuality, analyzer, state]() mutable {
                LibraryImportSummary summary;
                summary.token = token;
                summary.databasePath = databasePath;
                summary.imagesTotal = static_cast<int>(files.size());
                const auto push = [state](LibraryImportProgressEvent event) {
                    std::lock_guard lock(state->mutex);
                    state->events.push_back(std::move(event));
                };

                struct PendingFace {
                    fsc::core::FaceInsertRecord record;
                    QString displayName;
                    double quality = 0.0;
                    bool duplicate = false;
                    int imageIndex = 0;
                };

                try {
                    fsc::core::Database workerDatabase(pathFrom(databasePath));
                    std::set<std::string> seenHashes;
                    std::vector<PendingFace> pending;
                    pending.reserve(64);

                    const auto flushPending = [&] {
                        if (pending.empty()) {
                            return;
                        }
                        std::vector<fsc::core::FaceInsertRecord> records;
                        records.reserve(pending.size());
                        for (const auto& item : pending) {
                            records.push_back(item.record);
                        }
                        const auto ids = workerDatabase.insertFaces(records);
                        for (size_t index = 0; index < ids.size(); ++index) {
                            const auto& saved = pending[index];
                            ++summary.facesSaved;
                            summary.qualityTotal += saved.quality;
                            push({
                                saved.imageIndex,
                                summary.imagesTotal,
                                {},
                                QString("%1: saved face %2%3")
                                    .arg(saved.displayName)
                                    .arg(ids[index])
                                    .arg(saved.duplicate ? " duplicate" : ""),
                                false,
                            });
                        }
                        pending.clear();
                    };

                    for (int index = 0; index < summary.imagesTotal; ++index) {
                        const QString imagePathText = files[static_cast<size_t>(index)];
                        const std::filesystem::path imagePath = pathFrom(imagePathText);
                        const QString fileName = QFileInfo(imagePathText).fileName();
                        push({
                            index + 1,
                            summary.imagesTotal,
                            imagePathText,
                            QString("Analyzing %1").arg(fileName),
                            true,
                        });

                        try {
                            const std::string imageHash = fsc::core::sha256File(imagePath);
                            const bool duplicate = seenHashes.contains(imageHash) || workerDatabase.imageHashExists(imageHash);
                            if (duplicate) {
                                ++summary.duplicateImages;
                            }
                            seenHashes.insert(imageHash);
                            const auto faces = analyzer->analyze(modelRoot, runtimeMode, imagePath);
                            if (faces.empty()) {
                                ++summary.imagesWithoutFaces;
                                push({index + 1, summary.imagesTotal, {}, QString("%1: no faces").arg(fileName), false});
                                continue;
                            }

                            for (int faceIndex = 0; faceIndex < static_cast<int>(faces.size()); ++faceIndex) {
                                const auto& face = faces[static_cast<size_t>(faceIndex)];
                                const QString displayName = faces.size() > 1
                                    ? QString("%1 #%2").arg(fileName).arg(faceIndex + 1)
                                    : fileName;
                                if (face.qualityScore < minQuality) {
                                    ++summary.lowQualityFaces;
                                    push({
                                        index + 1,
                                        summary.imagesTotal,
                                        {},
                                        QString("%1: skipped low quality (%2)").arg(displayName).arg(face.qualityScore, 0, 'f', 3),
                                        false,
                                    });
                                    continue;
                                }
                                auto record = insertRecordFromFace(imagePath, face, imageHash, duplicate);
                                record.fileName = displayName.toUtf8().constData();
                                if (duplicate) {
                                    record.notes = "Same source image hash already exists in this database or import batch.";
                                }
                                pending.push_back({std::move(record), displayName, face.qualityScore, duplicate, index + 1});
                            }
                            if (pending.size() >= 50) {
                                flushPending();
                            }
                            push({
                                index + 1,
                                summary.imagesTotal,
                                {},
                                QString("Processed %1 face(s) from %2").arg(faces.size()).arg(fileName),
                                false,
                            });
                        } catch (const std::exception& ex) {
                            ++summary.failedImages;
                            push({
                                index + 1,
                                summary.imagesTotal,
                                {},
                                QString("%1: failed (%2)").arg(fileName, QString::fromUtf8(ex.what())),
                                false,
                            });
                        }
                    }
                    flushPending();
                } catch (const std::exception& ex) {
                    summary.error = QString::fromUtf8(ex.what());
                }
                return summary;
            }));
    }

    void drainLibraryImportProgress() {
        const auto state = libraryImportTaskState_;
        if (!state) {
            return;
        }
        std::deque<LibraryImportProgressEvent> events;
        {
            std::lock_guard lock(state->mutex);
            events.swap(state->events);
        }
        std::optional<LibraryImportProgressEvent> latestPreview;
        for (const auto& event : events) {
            if (libraryProgressBar_ != nullptr) {
                libraryProgressBar_->setRange(0, std::max(1, event.total));
                libraryProgressBar_->setValue(std::max(libraryProgressBar_->value(), event.current));
            }
            if (!event.message.isEmpty()) {
                appendLibraryActivity(event.message);
            }
            if (event.preview) {
                latestPreview = event;
            }
        }
        if (latestPreview.has_value() && !latestPreview->imagePath.isEmpty()) {
            setLibraryImportPreview(pathFrom(latestPreview->imagePath));
            statusBar()->showMessage(QString("%1 (%2/%3)")
                                         .arg(latestPreview->message)
                                         .arg(latestPreview->current)
                                         .arg(latestPreview->total));
        }
    }

    void finishLibraryImport(LibraryImportSummary summary) {
        if (summary.token != libraryImportToken_) {
            return;
        }
        drainLibraryImportProgress();
        libraryImportActive_ = false;
        if (libraryImportProgressTimer_ != nullptr) {
            libraryImportProgressTimer_->stop();
        }
        if (libraryImportImagesButton_ != nullptr) {
            libraryImportImagesButton_->setEnabled(true);
        }
        if (libraryImportFolderButton_ != nullptr) {
            libraryImportFolderButton_->setEnabled(true);
        }
        if (!summary.error.isEmpty()) {
            appendLibraryActivity(QString("Import stopped: %1").arg(summary.error));
            showError(std::runtime_error(summary.error.toUtf8().constData()));
        }
        appendLibraryActivity(QString("Done: %1 face(s), %2 no-face image(s), %3 failed image(s), "
                                      "%4 low-quality face(s) skipped, %5 duplicate image(s), avg quality %6.")
                                  .arg(summary.facesSaved)
                                  .arg(summary.imagesWithoutFaces)
                                  .arg(summary.failedImages)
                                  .arg(summary.lowQualityFaces)
                                  .arg(summary.duplicateImages)
                                  .arg(summary.averageQuality(), 0, 'f', 3));
        if (database_ && QString::fromStdWString(database_->path().wstring()) == summary.databasePath) {
            reloadAll();
        }
        libraryImportTaskState_.reset();
    }
#endif

    void importImage() {
#ifdef FSC_ENABLE_ONNX
        if (!ensureLibraryDatabaseForImport()) {
            return;
        }
        const QStringList selected = QFileDialog::getOpenFileNames(
            this,
            "Add images",
            {},
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.ppm);;All Files (*)");
        if (selected.isEmpty()) {
            return;
        }
        std::vector<QString> files;
        files.reserve(static_cast<size_t>(selected.size()));
        for (const auto& path : selected) {
            files.push_back(path);
        }
        startLibraryImport(std::move(files));
#else
        QMessageBox::information(this, "FSC Studio Native", "This build was not compiled with ONNX Runtime.");
#endif
    }

    void importFolder() {
#ifdef FSC_ENABLE_ONNX
        if (!ensureLibraryDatabaseForImport()) {
            return;
        }
        const QString folder = QFileDialog::getExistingDirectory(this, "Add folder recursively");
        if (folder.isEmpty()) {
            return;
        }
        std::vector<QString> files;
        QDirIterator iterator(folder, QDir::Files, QDirIterator::Subdirectories);
        while (iterator.hasNext()) {
            const QString file = iterator.next();
            if (isSupportedImageFile(file)) {
                files.push_back(file);
            }
        }
        if (files.empty()) {
            showError(std::runtime_error("No supported image files were found in the selected folder."));
            return;
        }
        appendLibraryActivity(QString("Importing %1 image file(s) from %2").arg(files.size()).arg(folder));
        startLibraryImport(std::move(files));
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
    QLineEdit* overviewDatabasePathEdit_ = nullptr;
    QTableWidget* overviewMetricsTable_ = nullptr;
    QTableWidget* overviewAttentionTable_ = nullptr;
    QTableWidget* overviewPeopleTable_ = nullptr;
    QTableWidget* overviewTagsTable_ = nullptr;
    QTableWidget* libraryTable_ = nullptr;
    QLabel* libraryPreviewLabel_ = nullptr;
    QTabWidget* libraryVisualTabs_ = nullptr;
    QPushButton* libraryFocusButton_ = nullptr;
    PointCloudWidget* libraryLandmarksView_ = nullptr;
    TexturedMeshWidget* libraryDenseMeshView_ = nullptr;
    QCheckBox* libraryMeshOverlayCheck_ = nullptr;
    QComboBox* libraryMeshModeCombo_ = nullptr;
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
    QPushButton* libraryImportImagesButton_ = nullptr;
    QPushButton* libraryImportFolderButton_ = nullptr;
    QTimer* libraryImportProgressTimer_ = nullptr;
#ifdef FSC_ENABLE_ONNX
    std::shared_ptr<LibraryImportTaskState> libraryImportTaskState_;
    uint64_t libraryImportToken_ = 0;
    bool libraryImportActive_ = false;
    bool libraryImportSmokeStarted_ = false;
#endif
    int libraryPreviewFaceId_ = 0;
    bool libraryFocusOnFace_ = false;
    QTableWidget* peopleTable_ = nullptr;
    QLineEdit* peopleDatabaseEdit_ = nullptr;
    QLineEdit* peopleFilterEdit_ = nullptr;
    QTableWidget* peopleMemberTable_ = nullptr;
    QLabel* peoplePreviewLabel_ = nullptr;
    QPushButton* peopleFocusButton_ = nullptr;
    QLineEdit* peopleNameEdit_ = nullptr;
    QTextEdit* peopleNotesEdit_ = nullptr;
    QComboBox* peopleMergeTargetCombo_ = nullptr;
    QLabel* peopleSummaryLabel_ = nullptr;
    QLabel* peopleProfileStatusLabel_ = nullptr;
    QLineEdit* personNameEdit_ = nullptr;
    QSpinBox* assignFaceSpin_ = nullptr;
    QSpinBox* assignPersonSpin_ = nullptr;
    std::vector<fsc::core::PersonSummary> peopleRows_;
    std::vector<fsc::core::FaceRecord> peopleMembers_;
    int64_t currentPersonId_ = 0;
    int64_t peoplePreviewFaceId_ = 0;
    bool peopleFocusOnFace_ = false;
    QTableWidget* reviewTable_ = nullptr;
    QLineEdit* reviewDatabaseEdit_ = nullptr;
    QLineEdit* reviewFilterEdit_ = nullptr;
    QSpinBox* reviewLimitSpin_ = nullptr;
    QLineEdit* reviewPersonEdit_ = nullptr;
    QLineEdit* reviewTagsEdit_ = nullptr;
    QComboBox* reviewStateCombo_ = nullptr;
    QCheckBox* reviewIgnoredCheck_ = nullptr;
    QTextEdit* reviewNotesEdit_ = nullptr;
    QLabel* reviewPreviewLabel_ = nullptr;
    QPushButton* reviewFocusButton_ = nullptr;
    QLabel* reviewSuggestionLabel_ = nullptr;
    int64_t reviewSuggestedPersonId_ = 0;
    std::string reviewSuggestedPersonName_;
    std::vector<fsc::core::FaceRecord> reviewRows_;
    int64_t currentReviewFaceId_ = 0;
    bool reviewFocusOnFace_ = false;
    QSpinBox* topKSpin_ = nullptr;
    QDoubleSpinBox* searchThresholdSpin_ = nullptr;
    QDoubleSpinBox* searchMinQualitySpin_ = nullptr;
    QCheckBox* searchIncludeIgnoredCheck_ = nullptr;
    QComboBox* searchPersonFilterCombo_ = nullptr;
    QComboBox* searchTagFilterCombo_ = nullptr;
    QLabel* identityLabel_ = nullptr;
    QTableWidget* searchIdentityTable_ = nullptr;
    QTableWidget* searchTable_ = nullptr;
    QLineEdit* searchDatabasePathEdit_ = nullptr;
    QLineEdit* searchImageEdit_ = nullptr;
    QListWidget* searchFaceList_ = nullptr;
    FaceSelectionPreview* searchQueryPreview_ = nullptr;
    QToolButton* searchQueryFocusButton_ = nullptr;
    QLabel* searchResultPreviewLabel_ = nullptr;
    QToolButton* searchResultFocusButton_ = nullptr;
    QTimer* searchPreviewTimer_ = nullptr;
    std::vector<fsc::vision::AnalyzedFace> searchQueryFaces_;
    std::vector<fsc::core::SearchHit> searchHits_;
    std::vector<fsc::core::FaceRecord> searchProgressRecords_;
    std::vector<fsc::core::SearchHit> searchProgressHits_;
    std::vector<float> searchProgressQuery_;
    QElapsedTimer searchProgressClock_;
    fsc::core::IdentityResult lastSearchIdentityResult_;
    int64_t searchResultPreviewFaceId_ = 0;
    int searchQueryFaceIndex_ = 0;
    int searchQueryGeneration_ = 0;
    int searchProgressTopK_ = 30;
    double searchProgressThreshold_ = -1.0;
    size_t searchProgressCursor_ = 0;
    qint64 searchProgressPreviewLastMs_ = -1;
    bool searchProgressActive_ = false;
    bool searchResultFocusOnFace_ = false;
    bool searchQueryAnalysisActive_ = false;
    bool searchQuerySmokeMode_ = false;
    QSpinBox* cameraIndexSpin_ = nullptr;
    QDoubleSpinBox* cameraThresholdSpin_ = nullptr;
    QSpinBox* cameraTopKSpin_ = nullptr;
    QSpinBox* cameraIntervalSpin_ = nullptr;
    QSpinBox* cameraProcessSizeSpin_ = nullptr;
    QLineEdit* cameraDatabasePathEdit_ = nullptr;
    QPushButton* cameraStartButton_ = nullptr;
    QPushButton* cameraStopButton_ = nullptr;
    QLabel* cameraPreviewLabel_ = nullptr;
    QLabel* cameraMatchPreviewLabel_ = nullptr;
    QLabel* cameraMatchStatusLabel_ = nullptr;
    QToolButton* cameraMatchFocusButton_ = nullptr;
    QTableWidget* cameraResultTable_ = nullptr;
    QTimer* cameraFrameTimer_ = nullptr;
    QTimer* cameraIdentifyTimer_ = nullptr;
    QLineEdit* compareImageAEdit_ = nullptr;
    QLineEdit* compareImageBEdit_ = nullptr;
    QLabel* compareResultLabel_ = nullptr;
    FaceSelectionPreview* comparePreviewA_ = nullptr;
    FaceSelectionPreview* comparePreviewB_ = nullptr;
    QListWidget* compareFaceListA_ = nullptr;
    QListWidget* compareFaceListB_ = nullptr;
    QToolButton* compareFocusAButton_ = nullptr;
    QToolButton* compareFocusBButton_ = nullptr;
    std::vector<fsc::vision::AnalyzedFace> compareFacesA_;
    std::vector<fsc::vision::AnalyzedFace> compareFacesB_;
    int selectedCompareA_ = 0;
    int selectedCompareB_ = 0;
    int compareGenerationA_ = 0;
    int compareGenerationB_ = 0;
    bool updatingCompareLists_ = false;
    bool compareAnalysisActiveA_ = false;
    bool compareAnalysisActiveB_ = false;
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
    QComboBox* runtimeModeCombo_ = nullptr;
    QLabel* runtimeBuildLabel_ = nullptr;
    QLabel* runtimeProviderLabel_ = nullptr;
    QLabel* runtimeNoteLabel_ = nullptr;
    QLabel* runtimeDatabasePathLabel_ = nullptr;
    QLabel* runtimeDatabaseStatsLabel_ = nullptr;
    QTextEdit* runtimeMaintenanceLog_ = nullptr;
    QPushButton* runtimeLegacyConvertButton_ = nullptr;
    QProgressBar* runtimeLegacyProgressBar_ = nullptr;
    QLineEdit* modelRootEdit_ = nullptr;
    std::vector<ClusterSummary> clusters_;
    std::vector<fsc::core::IdentityProfile> cameraIdentityProfiles_;
    std::shared_ptr<const std::vector<fsc::core::IdentityProfile>> cameraIdentityProfilesSnapshot_ =
        std::make_shared<const std::vector<fsc::core::IdentityProfile>>();
    std::shared_ptr<const std::vector<fsc::core::FaceRecord>> cameraStoredFaces_ =
        std::make_shared<const std::vector<fsc::core::FaceRecord>>();
    std::vector<CameraResultActionRow> cameraResultRows_;
    std::map<int, std::deque<QString>> cameraVotesByFace_;
    std::vector<fsc::vision::AnalyzedFace> latestCameraFaces_;
    std::set<int> latestCameraMatchedFaceIndexes_;
    std::chrono::steady_clock::time_point latestCameraFacesAt_;
    int64_t cameraMatchPreviewFaceId_ = 0;
    uint64_t cameraSessionGeneration_ = 0;
    uint64_t cameraAnalysisToken_ = 0;
    int cameraCapturedFrameCount_ = 0;
    int cameraCompletedAnalysisCount_ = 0;
    bool cameraMatchFocusOnFace_ = false;
    std::unique_ptr<fsc::mesh::MediaPipeFaceLandmarker> mediaPipeFaceLandmarker_;
    std::filesystem::path mediaPipeFaceLandmarkerModelPath_;
#ifdef FSC_ENABLE_OPENCV
    cv::VideoCapture camera_;
    cv::Mat lastCameraFrame_;
#endif
#ifdef FSC_ENABLE_ONNX
    std::shared_ptr<SharedFaceAnalyzer> sharedFaceAnalyzer_ = std::make_shared<SharedFaceAnalyzer>();
    std::shared_ptr<SharedFaceAnalyzer> cameraFaceAnalyzer_ = std::make_shared<SharedFaceAnalyzer>();
#endif
    bool cameraAnalyzeBusy_ = false;
    std::unique_ptr<fsc::core::Database> database_;
};

} // namespace

int main(int argc, char** argv) {
    configureDeployedQtRuntime(argc, argv);
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
    if (argc >= 4 && std::string(argv[1]) == "--review-action-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto personId = database.upsertPerson("NativeReviewSmoke", "review smoke");
            database.assignFaceToPerson(faceId, personId);
            database.setFaceTags(faceId, "native-review-smoke", false);
            database.updateFaceReview(faceId, "open", false, "native review action smoke");
            const auto face = database.loadFace(faceId);
            if (!face.has_value()) {
                return 1;
            }
            return face->personName == "NativeReviewSmoke" &&
                    face->tagText.find("native-review-smoke") != std::string::npos &&
                    face->reviewState == "open" &&
                    !face->ignored &&
                    face->notes == "native review action smoke" &&
                    face->duplicateCount >= 0
                ? 0
                : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--people-action-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto sourceId = database.upsertPerson("NativePeopleSmokeSource", "source notes");
            const auto targetId = database.upsertPerson("NativePeopleSmokeTarget", "target notes");
            database.assignFaceToPerson(faceId, sourceId);
            database.renamePerson(sourceId, "NativePeopleSmokeRenamed", "renamed notes");
            const auto people = database.loadPeople();
            const bool renamed = std::any_of(people.begin(), people.end(), [sourceId](const auto& person) {
                return person.id == sourceId && person.name == "NativePeopleSmokeRenamed" && person.notes == "renamed notes";
            });
            const int moved = database.mergePeople(sourceId, targetId);
            const auto assigned = database.loadFace(faceId);
            const auto members = database.loadFacesForPerson(targetId, true);
            const int cleared = database.clearPersonAssignment(targetId, true);
            const auto clearedFace = database.loadFace(faceId);
            return renamed && moved >= 1 && assigned.has_value() && assigned->personId == targetId &&
                    !members.empty() && cleared >= 1 && clearedFace.has_value() && clearedFace->personId == 0
                ? 0
                : 1;
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
            return fsc::mesh::isMediaPipeFaceMesh(face->faceMesh3d) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 4 && std::string(argv[1]) == "--mesh-generate-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value() || face->sourcePath.empty() || !std::filesystem::is_regular_file(face->sourcePath)) {
                return 1;
            }
            fsc::mesh::MediaPipeFaceLandmarker landmarker;
            const auto mesh = fsc::mesh::selectBestMediaPipeFaceMesh(
                landmarker.detect(fsc::vision::loadImageRgb(face->sourcePath)),
                face->bbox);
            database.updateFaceMesh3d(faceId, mesh);
            const auto updated = database.loadFace(faceId);
            return updated.has_value() && fsc::mesh::isMediaPipeFaceMesh(updated->faceMesh3d) ? 0 : 1;
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
    if (argc >= 5 && std::string(argv[1]) == "--camera-action-smoke") {
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const std::string personName = argv[4];
            const auto personId = database.upsertPerson(personName);
            database.assignFaceToPerson(faceId, personId);
            database.updateFaceReview(faceId, "reviewed", false, "native camera action smoke");
            database.rebuildIdentityProfiles();
            const auto face = database.loadFace(faceId);
            return face.has_value() && face->personName == personName && face->reviewState == "reviewed" && !face->ignored ? 0 : 1;
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

#ifdef FSC_ENABLE_ONNX
#ifdef FSC_ENABLE_OPENCV
    if (argc >= 5 && std::string(argv[1]) == "--camera-live-smoke") {
        QApplication uiApp(argc, argv);
        MainWindow window;
        const QString runtimeMode = argc >= 6 ? QString::fromLocal8Bit(argv[5]) : QString("cpu");
        window.openDatabasePath(QString::fromLocal8Bit(argv[2]));
        window.startCameraLiveSmoke(
            QString::fromLocal8Bit(argv[3]),
            std::atoi(argv[4]),
            runtimeMode);
        int outcome = 5;
        QTimer::singleShot(8000, &uiApp, [&] {
            outcome = window.cameraLiveSmokeReady() ? 0 : 4;
            window.stopCameraLiveSmoke();
            uiApp.quit();
        });
        uiApp.exec();
        return outcome;
    }

    if (argc >= 5 && std::string(argv[1]) == "--camera-ui-smoke") {
        QApplication uiApp(argc, argv);
        MainWindow window;
        const QString runtimeMode = argc >= 6 ? QString::fromLocal8Bit(argv[5]) : QString("cpu");
        window.openDatabasePath(QString::fromLocal8Bit(argv[2]));
        window.startCameraFrameSmoke(
            QString::fromLocal8Bit(argv[3]),
            QString::fromLocal8Bit(argv[4]),
            runtimeMode);
        int outcome = 5;
        QTimer poll;
        QObject::connect(&poll, &QTimer::timeout, &uiApp, [&] {
            if (window.cameraFrameSmokeFinished()) {
                outcome = window.cameraFrameSmokeReady() ? 0 : 4;
                uiApp.quit();
            }
        });
        poll.start(20);
        QTimer::singleShot(120000, &uiApp, [&] {
            outcome = 6;
            uiApp.quit();
        });
        uiApp.exec();
        return outcome;
    }
#endif

    if (argc >= 5 && std::string(argv[1]) == "--library-import-ui-smoke") {
        QApplication uiApp(argc, argv);
        const QString databasePath = QString::fromLocal8Bit(argv[2]);
        try {
            fsc::core::Database::createEmpty(pathFrom(databasePath), true);
        } catch (...) {
            return 2;
        }
        MainWindow window;
        const QString runtimeMode = argc >= 6 ? QString::fromLocal8Bit(argv[5]) : QString("cpu");
        window.startLibraryImportSmoke(
            databasePath,
            QString::fromLocal8Bit(argv[3]),
            QString::fromLocal8Bit(argv[4]),
            runtimeMode);
        int outcome = 5;
        QTimer poll;
        QObject::connect(&poll, &QTimer::timeout, &uiApp, [&] {
            if (!window.libraryImportSmokeFinished()) {
                return;
            }
            try {
                fsc::core::Database database(pathFrom(databasePath));
                outcome = database.statistics().faceCount > 0 ? 0 : 4;
            } catch (...) {
                outcome = 7;
            }
            uiApp.quit();
        });
        poll.start(20);
        QTimer::singleShot(120000, &uiApp, [&] {
            outcome = 6;
            uiApp.quit();
        });
        uiApp.exec();
        return outcome;
    }

    if (argc >= 5 && std::string(argv[1]) == "--compare-ui-smoke") {
        QApplication uiApp(argc, argv);
        MainWindow window;
        const QString runtimeMode = argc >= 6 ? QString::fromLocal8Bit(argv[5]) : QString("cpu");
        window.startCompareSmoke(
            QString::fromLocal8Bit(argv[2]),
            QString::fromLocal8Bit(argv[3]),
            QString::fromLocal8Bit(argv[4]),
            runtimeMode);
        int outcome = 5;
        QTimer poll;
        QObject::connect(&poll, &QTimer::timeout, &uiApp, [&] {
            if (window.compareSmokeFinished()) {
                outcome = window.compareSmokeReady() ? 0 : 4;
                uiApp.quit();
            }
        });
        poll.start(20);
        QTimer::singleShot(120000, &uiApp, [&] {
            outcome = 6;
            uiApp.quit();
        });
        uiApp.exec();
        return outcome;
    }

    if (argc >= 4 && std::string(argv[1]) == "--search-query-ui-smoke") {
        QApplication uiApp(argc, argv);
        MainWindow window;
        const QString runtimeMode = argc >= 5 ? QString::fromLocal8Bit(argv[4]) : QString("cpu");
        window.startSearchQuerySmoke(
            QString::fromLocal8Bit(argv[2]),
            QString::fromLocal8Bit(argv[3]),
            runtimeMode);
        int outcome = 5;
        QTimer poll;
        QObject::connect(&poll, &QTimer::timeout, &uiApp, [&] {
            if (window.searchQuerySmokeFinished()) {
                outcome = window.searchQuerySmokeFaceCount() > 0 ? 0 : 4;
                uiApp.quit();
            }
        });
        poll.start(20);
        QTimer::singleShot(120000, &uiApp, [&] {
            outcome = 6;
            uiApp.quit();
        });
        uiApp.exec();
        return outcome;
    }
#endif

    if (argc >= 2 && std::string(argv[1]) == "--ui-language-smoke") {
        const QString language = argc >= 3 ? QString::fromLocal8Bit(argv[2]) : QString("zh");
        QApplication uiApp(argc, argv);
        MainWindow window;
        QComboBox* languageSelector = nullptr;
        for (auto* combo : window.findChildren<QComboBox*>()) {
            if (combo->findData("en") >= 0 && combo->findData("zh") >= 0 &&
                combo->findData("ja") >= 0 && combo->findData("ko") >= 0) {
                languageSelector = combo;
                break;
            }
        }
        if (languageSelector == nullptr) {
            return 2;
        }
        const int languageIndex = languageSelector->findData(language);
        if (languageIndex < 0) {
            return 3;
        }
        languageSelector->setCurrentIndex(languageIndex);
        bool legacyActionPresent = false;
        for (auto* button : window.findChildren<QPushButton*>()) {
            if (translationKey(button->text()) == "Convert Legacy DTB") {
                legacyActionPresent = true;
                break;
            }
        }
        const bool searchControlsPresent =
            window.findChild<QLineEdit*>("SearchDatabasePath") != nullptr &&
            window.findChild<QPushButton*>("SearchOpenDatabase") != nullptr &&
            window.findChild<QPushButton*>("SearchUseLibraryDatabase") != nullptr &&
            window.findChild<QPushButton*>("SearchSelectImage") != nullptr &&
            window.findChild<QWidget*>("SearchQueryPreview") != nullptr &&
            window.findChild<QListWidget*>("SearchQueryFaceList") != nullptr &&
            window.findChild<QToolButton*>("SearchQueryFocus") != nullptr &&
            window.findChild<QToolButton*>("SearchResultFocus") != nullptr;
        const bool compareControlsPresent =
            window.findChild<QLineEdit*>("CompareImageAPath") != nullptr &&
            window.findChild<QLineEdit*>("CompareImageBPath") != nullptr &&
            window.findChild<QPushButton*>("CompareSelectImageA") != nullptr &&
            window.findChild<QPushButton*>("CompareSelectImageB") != nullptr &&
            window.findChild<QPushButton*>("CompareRun") != nullptr &&
            window.findChild<QWidget*>("ComparePreviewA") != nullptr &&
            window.findChild<QWidget*>("ComparePreviewB") != nullptr &&
            window.findChild<QListWidget*>("CompareFaceListA") != nullptr &&
            window.findChild<QListWidget*>("CompareFaceListB") != nullptr &&
            window.findChild<QToolButton*>("CompareFocusA") != nullptr &&
            window.findChild<QToolButton*>("CompareFocusB") != nullptr;
        const bool cameraControlsPresent =
            window.findChild<QLineEdit*>("CameraDatabasePath") != nullptr &&
            window.findChild<QPushButton*>("CameraOpenDatabase") != nullptr &&
            window.findChild<QPushButton*>("CameraUseLibraryDatabase") != nullptr &&
            window.findChild<QPushButton*>("CameraStart") != nullptr &&
            window.findChild<QPushButton*>("CameraStop") != nullptr &&
            window.findChild<QWidget*>("CameraPreview") != nullptr &&
            window.findChild<QWidget*>("CameraMatchPreview") != nullptr &&
            window.findChild<QToolButton*>("CameraMatchFocus") != nullptr &&
            window.findChild<QTableWidget*>("CameraResultTable") != nullptr;
        for (auto* list : window.findChildren<QListWidget*>()) {
            if (list->count() == 9 && list->item(0) != nullptr &&
                list->item(0)->text() == translatedText("Overview", language) &&
                legacyActionPresent && searchControlsPresent && compareControlsPresent && cameraControlsPresent) {
                return 0;
            }
        }
        return 4;
    }

    if (argc >= 3 && std::string(argv[1]) == "--overview-smoke") {
        QApplication uiApp(argc, argv);
        MainWindow window;
        try {
            window.openDatabasePath(QString::fromLocal8Bit(argv[2]));
        } catch (...) {
            return 2;
        }
        const auto hasRows = [&window](const QString& firstHeader, const QString& secondHeader) {
            for (auto* table : window.findChildren<QTableWidget*>()) {
                if (table->columnCount() >= 2 && table->rowCount() > 0 &&
                    table->horizontalHeaderItem(0) != nullptr && table->horizontalHeaderItem(1) != nullptr &&
                    table->horizontalHeaderItem(0)->text() == firstHeader &&
                    table->horizontalHeaderItem(1)->text() == secondHeader) {
                    return true;
                }
            }
            return false;
        };
        return hasRows("Metric", "Value") && hasRows("Queue", "Count") ? 0 : 3;
    }

    if (argc >= 5 && std::string(argv[1]) == "--mesh-render-smoke") {
        QApplication renderApp(argc, argv);
        try {
            fsc::core::Database database(pathFrom(QString::fromLocal8Bit(argv[2])));
            const auto faceId = std::strtoll(argv[3], nullptr, 10);
            const auto face = database.loadFace(faceId);
            if (!face.has_value() || !fsc::mesh::isMediaPipeFaceMesh(face->faceMesh3d)) {
                return 2;
            }
            const QImage source = loadPreviewImage(pathFrom(qs(face->sourcePath)));
            if (source.isNull()) {
                return 3;
            }

            TexturedMeshWidget preview;
            preview.resize(640, 480);
            preview.setData(face->faceMesh3d, face->landmarks3d, "native mesh render smoke");
            preview.setTextureImage(source);
            preview.setRenderMode(TexturedMeshWidget::RenderMode::Textured);
            if (argc >= 7) {
                preview.setView(std::strtod(argv[5], nullptr), std::strtod(argv[6], nullptr));
            }

            QImage rendered(preview.size(), QImage::Format_ARGB32);
            rendered.fill(QColor(17, 24, 39));
            QPainter painter(&rendered);
            preview.render(&painter);
            painter.end();

            int changedPixels = 0;
            for (int y = 0; y < rendered.height(); ++y) {
                const auto* row = reinterpret_cast<const QRgb*>(rendered.constScanLine(y));
                for (int x = 0; x < rendered.width(); ++x) {
                    if (row[x] != qRgb(17, 24, 39)) {
                        ++changedPixels;
                    }
                }
            }
            const auto outputPath = pathFrom(QString::fromLocal8Bit(argv[4]));
            if (!outputPath.parent_path().empty()) {
                std::filesystem::create_directories(outputPath.parent_path());
            }
            if (changedPixels <= (rendered.width() * rendered.height()) / 20) {
                return 4;
            }
            return rendered.save(QString::fromStdWString(outputPath.wstring())) ? 0 : 5;
        } catch (...) {
            return 9;
        }
    }

    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    if (argc > 1) {
        window.openDatabasePath(QString::fromLocal8Bit(argv[1]));
    }
    return app.exec();
}
