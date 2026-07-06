#include "fsc/core/Database.hpp"
#include "fsc/core/FileHash.hpp"
#include "fsc/core/IdentityGallery.hpp"
#include "fsc/core/Search.hpp"
#include "fsc/vision/Image.hpp"
#include "fsc/vision/InsightFaceEngine.hpp"
#include "fsc/vision/ModelPaths.hpp"

#include <QApplication>
#include <QAbstractItemView>
#include <QFileDialog>
#include <QFormLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QStyle>
#include <QTableWidget>
#include <QTabWidget>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
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

QTableWidgetItem* item(const QString& value) {
    auto* output = new QTableWidgetItem(value);
    output->setFlags(output->flags() & ~Qt::ItemIsEditable);
    return output;
}

QTableWidgetItem* numberItem(double value, int decimals = 4) {
    return item(QString::number(value, 'f', decimals));
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
    void buildUi() {
        auto* root = new QWidget(this);
        auto* rootLayout = new QVBoxLayout(root);
        rootLayout->setContentsMargins(10, 10, 10, 10);
        rootLayout->setSpacing(8);

        auto* toolbar = new QWidget(root);
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

        toolbarLayout->addWidget(new QLabel("Database", toolbar));
        toolbarLayout->addWidget(databasePathEdit_, 1);
        toolbarLayout->addWidget(openButton);
        toolbarLayout->addWidget(createButton);
        toolbarLayout->addWidget(refreshButton);
        rootLayout->addWidget(toolbar);

        tabs_ = new QTabWidget(root);
        rootLayout->addWidget(tabs_, 1);
        buildOverviewTab();
        buildLibraryTab();
        buildPeopleTab();
        buildSearchTab();
        buildImportTab();

        setCentralWidget(root);
        statusBar()->showMessage("Ready");

        connect(openButton, &QToolButton::clicked, this, [this] { chooseDatabase(); });
        connect(createButton, &QToolButton::clicked, this, [this] { createDatabase(); });
        connect(refreshButton, &QToolButton::clicked, this, [this] { reloadAll(); });
        connect(databasePathEdit_, &QLineEdit::returnPressed, this, [this] { openDatabase(databasePathEdit_->text()); });
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
        tabs_->addTab(page, "Overview");
    }

    void buildLibraryTab() {
        libraryTable_ = new QTableWidget(tabs_);
        libraryTable_->setColumnCount(8);
        libraryTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Quality", "Detection", "Review", "Ignored", "Source"});
        fitTable(libraryTable_);
        tabs_->addTab(libraryTable_, "Library");
    }

    void buildPeopleTab() {
        peopleTable_ = new QTableWidget(tabs_);
        peopleTable_->setColumnCount(8);
        peopleTable_->setHorizontalHeaderLabels({"ID", "Name", "Faces", "Avg Quality", "Identity", "Samples", "Exemplars", "Health"});
        fitTable(peopleTable_);
        tabs_->addTab(peopleTable_, "People");
    }

    void buildSearchTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

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

        searchTable_ = new QTableWidget(page);
        searchTable_->setColumnCount(5);
        searchTable_->setHorizontalHeaderLabels({"ID", "File", "Person", "Cosine", "Similarity"});
        fitTable(searchTable_);
        layout->addWidget(searchTable_, 1);

        tabs_->addTab(page, "Search");
        connect(searchButton, &QPushButton::clicked, this, [this] { runSearch(); });
        connect(identifyButton, &QPushButton::clicked, this, [this] { runIdentify(); });
    }

    void buildImportTab() {
        auto* page = new QWidget(tabs_);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* controls = new QWidget(page);
        auto* form = new QFormLayout(controls);
        modelRootEdit_ = new QLineEdit(controls);
        modelRootEdit_->setText("D:\\FSC\\model\\insightface\\models");
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
        form->addRow("Models", modelRow);
        form->addRow("Image", imageRow);
        form->addRow("", importButton);
        layout->addWidget(controls);

        importLog_ = new QTableWidget(page);
        importLog_->setColumnCount(6);
        importLog_->setHorizontalHeaderLabels({"Inserted ID", "Face", "Detection", "Quality", "2D", "3D"});
        fitTable(importLog_);
        layout->addWidget(importLog_, 1);

        tabs_->addTab(page, "Import");
        connect(modelButton, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getExistingDirectory(this, "Select model root", modelRootEdit_->text());
            if (!path.isEmpty()) {
                modelRootEdit_->setText(path);
            }
        });
        connect(imageButton, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getOpenFileName(this, "Select image", {}, "Images (*.jpg *.jpeg *.png *.bmp *.ppm)");
            if (!path.isEmpty()) {
                importImageEdit_->setText(path);
            }
        });
        connect(importButton, &QPushButton::clicked, this, [this] { importImage(); });
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
            libraryTable_->setItem(row, 3, numberItem(record.qualityScore, 3));
            libraryTable_->setItem(row, 4, numberItem(record.detectionScore, 3));
            libraryTable_->setItem(row, 5, item(qs(record.reviewState)));
            libraryTable_->setItem(row, 6, item(record.ignored ? "yes" : "no"));
            libraryTable_->setItem(row, 7, item(qs(record.sourcePath)));
        }
        libraryTable_->resizeColumnsToContents();
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

    void runSearch() {
        if (!database_) {
            return;
        }
        try {
            const auto query = database_->loadFace(faceIdSpin_->value());
            if (!query.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            const auto hits = fsc::core::searchFaces(database_->loadFaces(false), query->embedding, topKSpin_->value(), -1.0, false);
            searchTable_->setRowCount(0);
            for (const auto& hit : hits) {
                if (hit.record.id == query->id) {
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
            const auto query = database_->loadFace(faceIdSpin_->value());
            if (!query.has_value()) {
                throw std::runtime_error("Face id not found.");
            }
            const auto result = fsc::core::identifyPerson(database_->loadIdentityProfiles(), query->embedding, fsc::core::IdentityMode::Strict, 5);
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
            fsc::vision::InsightFaceEngine engine(fsc::vision::InsightFaceModelPaths::fromBuffaloL(modelRoot), fsc::vision::RuntimeMode::Cpu);
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

    QLineEdit* databasePathEdit_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    QLabel* formatLabel_ = nullptr;
    QLabel* modelLabel_ = nullptr;
    QLabel* metricLabel_ = nullptr;
    QLabel* facesLabel_ = nullptr;
    QLabel* peopleLabel_ = nullptr;
    QLabel* reviewLabel_ = nullptr;
    QLabel* qualityLabel_ = nullptr;
    QTableWidget* libraryTable_ = nullptr;
    QTableWidget* peopleTable_ = nullptr;
    QSpinBox* faceIdSpin_ = nullptr;
    QSpinBox* topKSpin_ = nullptr;
    QLabel* identityLabel_ = nullptr;
    QTableWidget* searchTable_ = nullptr;
    QLineEdit* modelRootEdit_ = nullptr;
    QLineEdit* importImageEdit_ = nullptr;
    QTableWidget* importLog_ = nullptr;
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

    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    if (argc > 1) {
        window.openDatabasePath(QString::fromLocal8Bit(argv[1]));
    }
    return app.exec();
}
