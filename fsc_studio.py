from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image, ImageDraw
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fsc_face_database import DEFAULT_EXTENSION, FaceRecord, LegacyDatabaseError, normalize_database_path
from fsc_face_engine import (
    AnalyzedFace,
    cosine_similarity,
    get_engine,
    preview_png_to_pil,
    read_image_bgr,
    render_face_overlay,
    render_focused_faces_overlay,
    render_faces_overlay,
    render_plain_preview,
    similarity_percent,
)
from fsc_studio_services import (
    ImportSummary,
    LegacyConversionSummary,
    MaintenanceResult,
    PrimaryFace,
    SearchHit,
    FaceCluster,
    PersonSummary,
    TagSummary,
    analyze_faces,
    analyze_primary_face,
    assign_faces_to_person,
    assign_person,
    backup_database,
    build_face_clusters,
    check_database_integrity,
    checkpoint_database,
    clear_person_assignment,
    collect_image_paths,
    compare_images,
    convert_legacy_dtb_to_database,
    create_database,
    default_legacy_conversion_output_path,
    export_faces_csv,
    import_images_to_database,
    load_database_statistics,
    load_people,
    load_person_summaries,
    load_preview,
    load_records,
    load_review_queue,
    load_tag_summaries,
    load_tags,
    merge_people,
    rename_person,
    search_database,
    search_database_progressive,
    set_tags,
    update_faces_metadata,
    update_review,
    vacuum_database,
)


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All files (*.*)"
DB_FILTER = f"FSC database (*{DEFAULT_EXTENSION});;Legacy dlib database (*.dtb);;All files (*.*)"
PREVIEW_SIZE = 360
PREVIEW_MIN_SIZE = 180

I18N: dict[str, dict[str, str]] = {
    "zh": {
        "Overview": "概览",
        "Library": "人脸库",
        "People": "人物",
        "Search": "搜索",
        "Camera": "摄像头",
        "Review": "复核",
        "Clusters": "聚类",
        "Compare": "比对",
        "Runtime": "运行",
        "Workspace": "工作区",
        "Database": "数据库",
        "New Database": "新建库",
        "Open / Convert": "打开/转换",
        "Import Images": "导入图像",
        "Import Folder": "导入文件夹",
        "Refresh": "刷新",
        "Database Metrics": "数据库指标",
        "Attention": "待处理",
        "Review Queue": "复核队列",
        "Top People": "主要人物",
        "Top Tags": "主要标签",
        "File": "文件",
        "New": "新建",
        "Open": "打开",
        "Add Images": "添加图像",
        "Add Folder": "添加文件夹",
        "Reload": "重载",
        "Export CSV": "导出 CSV",
        "Filter": "过滤",
        "Person": "人物",
        "Tag": "标签",
        "Include ignored": "包含忽略项",
        "Apply Filter": "应用过滤",
        "Reset Filter": "重置过滤",
        "Min quality": "最低质量",
        "Selected Face": "选中人脸",
        "Save Metadata": "保存元数据",
        "Batch Selected": "批量处理",
        "Append tags": "追加标签",
        "Apply to Selection": "应用到选中项",
        "Selected": "选中",
        "Batch": "批量",
        "Activity": "活动",
        "Query": "查询",
        "Open Database": "打开数据库",
        "Use Library DB": "使用当前库",
        "Select Image": "选择图像",
        "Image": "图像",
        "Top K": "返回数量",
        "Threshold": "阈值",
        "Interval ms": "间隔毫秒",
        "Process size": "处理尺寸",
        "Start Camera": "启动摄像头",
        "Stop Camera": "停止摄像头",
        "Match": "匹配结果",
        "Result": "结果",
        "Images": "图像",
        "Image A": "图像 A",
        "Image B": "图像 B",
        "Engine": "引擎",
        "Load Runtime": "加载运行时",
        "Current Database": "当前数据库",
        "Refresh Database Stats": "刷新数据库统计",
        "Maintenance": "维护",
        "Check Integrity": "完整性检查",
        "Backup DB": "备份数据库",
        "Checkpoint WAL": "Checkpoint WAL",
        "VACUUM": "VACUUM",
        "Legacy": "旧版",
        "Convert Legacy DTB": "转换旧 DTB",
        "Focus on Face": "聚焦于人脸",
        "View Full Image": "查看完整图像",
        "Language": "语言",
    },
    "ja": {
        "Overview": "概要",
        "Library": "ライブラリ",
        "People": "人物",
        "Search": "検索",
        "Camera": "カメラ",
        "Review": "レビュー",
        "Clusters": "クラスタ",
        "Compare": "比較",
        "Runtime": "実行環境",
        "Workspace": "ワークスペース",
        "Database": "データベース",
        "New Database": "新規DB",
        "Open / Convert": "開く/変換",
        "Import Images": "画像を追加",
        "Import Folder": "フォルダ追加",
        "Refresh": "更新",
        "Database Metrics": "DB指標",
        "Attention": "要対応",
        "Review Queue": "レビュー待ち",
        "Top People": "主要人物",
        "Top Tags": "主要タグ",
        "File": "ファイル",
        "New": "新規",
        "Open": "開く",
        "Add Images": "画像追加",
        "Add Folder": "フォルダ追加",
        "Reload": "再読込",
        "Export CSV": "CSV出力",
        "Filter": "フィルタ",
        "Person": "人物",
        "Tag": "タグ",
        "Include ignored": "無視を含む",
        "Apply Filter": "適用",
        "Reset Filter": "リセット",
        "Min quality": "最低品質",
        "Selected Face": "選択顔",
        "Save Metadata": "保存",
        "Batch Selected": "一括処理",
        "Append tags": "タグ追加",
        "Apply to Selection": "選択に適用",
        "Selected": "選択",
        "Batch": "一括",
        "Activity": "ログ",
        "Query": "検索画像",
        "Open Database": "DBを開く",
        "Use Library DB": "現在のDB",
        "Select Image": "画像選択",
        "Image": "画像",
        "Top K": "件数",
        "Threshold": "しきい値",
        "Interval ms": "間隔 ms",
        "Process size": "処理サイズ",
        "Start Camera": "カメラ開始",
        "Stop Camera": "カメラ停止",
        "Match": "一致結果",
        "Result": "結果",
        "Images": "画像",
        "Image A": "画像 A",
        "Image B": "画像 B",
        "Engine": "エンジン",
        "Load Runtime": "読込",
        "Current Database": "現在のDB",
        "Refresh Database Stats": "統計更新",
        "Maintenance": "保守",
        "Check Integrity": "整合性確認",
        "Backup DB": "DBバックアップ",
        "Checkpoint WAL": "WAL保存",
        "VACUUM": "VACUUM",
        "Legacy": "旧形式",
        "Convert Legacy DTB": "旧DTB変換",
        "Focus on Face": "顔にフォーカス",
        "View Full Image": "全体画像",
        "Language": "言語",
    },
    "ko": {
        "Overview": "개요",
        "Library": "라이브러리",
        "People": "인물",
        "Search": "검색",
        "Camera": "카메라",
        "Review": "검토",
        "Clusters": "클러스터",
        "Compare": "비교",
        "Runtime": "런타임",
        "Workspace": "작업공간",
        "Database": "데이터베이스",
        "New Database": "새 DB",
        "Open / Convert": "열기/변환",
        "Import Images": "이미지 가져오기",
        "Import Folder": "폴더 가져오기",
        "Refresh": "새로고침",
        "Database Metrics": "DB 지표",
        "Attention": "확인 필요",
        "Review Queue": "검토 대기",
        "Top People": "주요 인물",
        "Top Tags": "주요 태그",
        "File": "파일",
        "New": "새로 만들기",
        "Open": "열기",
        "Add Images": "이미지 추가",
        "Add Folder": "폴더 추가",
        "Reload": "다시 로드",
        "Export CSV": "CSV 내보내기",
        "Filter": "필터",
        "Person": "인물",
        "Tag": "태그",
        "Include ignored": "무시 항목 포함",
        "Apply Filter": "필터 적용",
        "Reset Filter": "필터 초기화",
        "Min quality": "최소 품질",
        "Selected Face": "선택한 얼굴",
        "Save Metadata": "저장",
        "Batch Selected": "일괄 처리",
        "Append tags": "태그 추가",
        "Apply to Selection": "선택 항목 적용",
        "Selected": "선택",
        "Batch": "일괄",
        "Activity": "작업 로그",
        "Query": "검색 이미지",
        "Open Database": "DB 열기",
        "Use Library DB": "현재 DB 사용",
        "Select Image": "이미지 선택",
        "Image": "이미지",
        "Top K": "결과 수",
        "Threshold": "임계값",
        "Interval ms": "간격 ms",
        "Process size": "처리 크기",
        "Start Camera": "카메라 시작",
        "Stop Camera": "카메라 중지",
        "Match": "일치 결과",
        "Result": "결과",
        "Images": "이미지",
        "Image A": "이미지 A",
        "Image B": "이미지 B",
        "Engine": "엔진",
        "Load Runtime": "런타임 로드",
        "Current Database": "현재 DB",
        "Refresh Database Stats": "통계 새로고침",
        "Maintenance": "유지관리",
        "Check Integrity": "무결성 검사",
        "Backup DB": "DB 백업",
        "Checkpoint WAL": "WAL 체크포인트",
        "VACUUM": "VACUUM",
        "Legacy": "이전 형식",
        "Convert Legacy DTB": "이전 DTB 변환",
        "Focus on Face": "얼굴에 초점",
        "View Full Image": "전체 이미지",
        "Language": "언어",
    },
}


def i18n_key(text: str) -> str:
    for mapping in I18N.values():
        for key, value in mapping.items():
            if text == value:
                return key
    return text


def tr_text(text: str, language: str) -> str:
    if language == "en":
        return i18n_key(text)
    return I18N.get(language, {}).get(i18n_key(text), text)


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(str, int, int)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable, *args, progress_arg: bool = False, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.progress_arg = progress_arg
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            if self.progress_arg:
                kwargs["progress"] = self.signals.progress.emit
            result = self.fn(*self.args, **kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


def pil_to_pixmap(image: Image.Image, max_size: int = PREVIEW_SIZE) -> QPixmap:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width > 0 and height > 0:
        scale = max_size / max(width, height)
        rgb = rgb.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def pil_to_fitted_pixmap(image: Image.Image, max_width: int, max_height: int) -> QPixmap:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width > 0 and height > 0:
        scale = min(max_width / width, max_height / height)
        rgb = rgb.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


FOCUS_BUTTON_TEXT = "Focus on Face"
FULL_IMAGE_BUTTON_TEXT = "View Full Image"
CURRENT_LANGUAGE = "en"


def iter_xy_points(points: object) -> list[tuple[float, float]]:
    if not points:
        return []
    result: list[tuple[float, float]] = []
    for point in points:
        try:
            values = list(point)
        except TypeError:
            continue
        if len(values) >= 2:
            result.append((float(values[0]), float(values[1])))
    return result


def crop_focus_image(
    image: Image.Image,
    bbox: list | tuple,
    *,
    kps: object = None,
    landmarks: object = None,
    margin: float = 0.75,
    max_size: int = PREVIEW_SIZE,
) -> Image.Image | None:
    if len(bbox) < 4:
        return None
    width, height = image.size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    left = max(0, int(round(x1 - face_width * margin)))
    top = max(0, int(round(y1 - face_height * margin)))
    right = min(width, int(round(x2 + face_width * margin)))
    bottom = min(height, int(round(y2 + face_height * margin)))
    if right <= left or bottom <= top:
        return None
    focused = image.crop((left, top, right, bottom)).copy()
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    scale = max_size / max(crop_width, crop_height)
    focused = focused.resize(
        (max(1, int(round(crop_width * scale))), max(1, int(round(crop_height * scale)))),
        Image.Resampling.LANCZOS,
    )
    scale_x = focused.width / crop_width
    scale_y = focused.height / crop_height

    def map_point(x: float, y: float) -> tuple[float, float]:
        return (x - left) * scale_x, (y - top) * scale_y

    draw = ImageDraw.Draw(focused)
    rx1, ry1 = map_point(x1, y1)
    rx2, ry2 = map_point(x2, y2)
    draw.rectangle((rx1, ry1, rx2, ry2), outline=(0, 255, 0), width=2)
    for x, y in iter_xy_points(landmarks):
        if left <= x <= right and top <= y <= bottom:
            px, py = map_point(x, y)
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(0, 180, 255))
    keypoint_colors = [
        (0, 128, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 0),
    ]
    for index, (x, y) in enumerate(iter_xy_points(kps)):
        if left <= x <= right and top <= y <= bottom:
            px, py = map_point(x, y)
            color = keypoint_colors[index % len(keypoint_colors)]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    return focused


def record_image_from_source(record: FaceRecord, *, focused: bool = False) -> Image.Image | None:
    if not record.source_path:
        return None
    source = Path(record.source_path)
    if not source.exists():
        return None
    try:
        with Image.open(source) as image:
            full = image.convert("RGB")
    except Exception:
        return None
    if focused:
        return crop_focus_image(full, record.bbox, kps=record.kps, landmarks=record.landmarks)
    draw = ImageDraw.Draw(full)
    if record.bbox and len(record.bbox) >= 4:
        x1, y1, x2, y2 = [float(value) for value in record.bbox[:4]]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=4)
    return full


def bgr_frame_to_pil(image_bgr) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_camera_faces(
    image_bgr,
    faces: list[AnalyzedFace],
    matched_face_indexes: set[int] | None = None,
) -> Image.Image:
    image = bgr_frame_to_pil(image_bgr)
    draw = ImageDraw.Draw(image)
    matched_face_indexes = matched_face_indexes or set()
    for index, face in enumerate(faces):
        if len(face.bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
        matched = index in matched_face_indexes
        color = (0, 255, 0) if matched else (255, 180, 0)
        width = 4 if matched else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{index + 1}"
        if matched:
            label = f"{label} match"
        draw.rectangle((x1, max(0, y1 - 22), x1 + max(28, 10 * len(label)), y1), fill=color)
        draw.text((x1 + 6, max(0, y1 - 19)), label, fill=(0, 0, 0))
    return image


def resize_camera_frame_for_recognition(image_bgr, max_long_edge: int):
    height, width = image_bgr.shape[:2]
    max_long_edge = max(1, int(max_long_edge))
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image_bgr, 1.0, 1.0
    scale = max_long_edge / long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / new_height


def scale_camera_faces_to_frame(
    faces: list[AnalyzedFace],
    image_bgr,
    scale_x: float,
    scale_y: float,
) -> list[AnalyzedFace]:
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return [
            AnalyzedFace(
                image_bgr=image_bgr,
                embedding=face.embedding,
                bbox=face.bbox,
                kps=face.kps,
                landmarks=face.landmarks,
                det_score=face.det_score,
                quality_score=face.quality_score,
                quality_details=face.quality_details,
                source_path=face.source_path,
                file_name=face.file_name,
            )
            for face in faces
        ]

    def scale_points(points: list[list[float]] | None) -> list[list[float]] | None:
        if points is None:
            return None
        scaled: list[list[float]] = []
        for point in points:
            if len(point) >= 2:
                scaled.append([float(point[0]) * scale_x, float(point[1]) * scale_y])
        return scaled

    scaled_faces: list[AnalyzedFace] = []
    for face in faces:
        bbox = [
            float(face.bbox[0]) * scale_x,
            float(face.bbox[1]) * scale_y,
            float(face.bbox[2]) * scale_x,
            float(face.bbox[3]) * scale_y,
        ] if len(face.bbox) >= 4 else []
        scaled_faces.append(
            AnalyzedFace(
                image_bgr=image_bgr,
                embedding=face.embedding,
                bbox=bbox,
                kps=scale_points(face.kps) or [],
                landmarks=scale_points(face.landmarks),
                det_score=face.det_score,
                quality_score=face.quality_score,
                quality_details=face.quality_details,
                source_path=face.source_path,
                file_name=face.file_name,
            )
        )
    return scaled_faces


class FocusablePreviewLabel(QLabel):
    focus_changed = Signal(bool)

    def __init__(self, text: str = "No image") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(PREVIEW_MIN_SIZE, PREVIEW_MIN_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("Preview")
        self.base_image: Image.Image | None = None
        self.focus_image: Image.Image | None = None
        self.empty_text = text
        self.focus_mode = False
        self.focus_button: QPushButton | None = None
        self._display_source_image: Image.Image | None = None
        self._display_text = text

    def set_preview_image(
        self,
        image: Image.Image | None,
        text: str = "No image",
        *,
        focus_image: Image.Image | None = None,
    ) -> None:
        self.base_image = image.copy() if image is not None else None
        self.focus_image = focus_image.copy() if focus_image is not None else None
        self.empty_text = text
        if not self.can_focus():
            self.focus_mode = False
        self.refresh_preview()
        self.sync_focus_button()

    def refresh_preview(self) -> None:
        image = self.focus_image if self.focus_mode and self.focus_image is not None else self.base_image
        self.display_image(image, self.empty_text)

    def display_image(self, image: Image.Image | None, text: str = "No image") -> None:
        self._display_source_image = image.copy() if image is not None else None
        self._display_text = text
        self.render_display_image()

    def render_display_image(self) -> None:
        self.setPixmap(QPixmap())
        self.setText("")
        if self._display_source_image is None:
            self.setText(self._display_text)
            return
        rect = self.contentsRect()
        available_width = max(1, rect.width() - 4)
        available_height = max(1, rect.height() - 4)
        if available_width <= 1 or available_height <= 1:
            available_width = PREVIEW_SIZE
            available_height = PREVIEW_SIZE
        self.setPixmap(pil_to_fitted_pixmap(self._display_source_image, available_width, available_height))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._display_source_image is not None:
            self.render_display_image()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(PREVIEW_SIZE + 16, PREVIEW_SIZE + 16)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(PREVIEW_MIN_SIZE, PREVIEW_MIN_SIZE)

    def can_focus(self) -> bool:
        return self.focus_image is not None

    def set_focus_mode(self, enabled: bool) -> None:
        next_mode = bool(enabled) and self.can_focus()
        if self.focus_mode == next_mode:
            self.sync_focus_button()
            return
        self.focus_mode = next_mode
        self.refresh_preview()
        self.sync_focus_button()
        self.focus_changed.emit(self.focus_mode)

    def toggle_focus(self) -> None:
        self.set_focus_mode(not self.focus_mode)

    def attach_focus_button(self, button: QPushButton) -> None:
        self.focus_button = button
        button.clicked.connect(self.toggle_focus)
        self.sync_focus_button()

    def sync_focus_button(self) -> None:
        if self.focus_button is None:
            return
        self.focus_button.setEnabled(self.can_focus())
        base_text = FULL_IMAGE_BUTTON_TEXT if self.focus_mode else FOCUS_BUTTON_TEXT
        text = tr_text(base_text, CURRENT_LANGUAGE)
        self.focus_button.setText(text)
        self.focus_button.setToolTip(text)


def set_image(label: QLabel, image: Image.Image | None, text: str = "No image") -> None:
    if isinstance(label, FocusablePreviewLabel):
        label.set_preview_image(image, text)
        return
    label.setPixmap(QPixmap())
    label.setText("")
    if image is None:
        label.setText(text)
        return
    label.setPixmap(pil_to_pixmap(image))


def set_record_preview(
    label: QLabel,
    record: FaceRecord,
    database_path: str = "",
) -> None:
    preview = record.preview_png
    if preview is None and database_path:
        try:
            preview = load_preview(database_path, record.id)
        except Exception:
            preview = None
    image = preview_png_to_pil(preview) if preview else record_image_from_source(record, focused=False)
    focus_image = record_image_from_source(record, focused=True)
    if isinstance(label, FocusablePreviewLabel):
        label.set_preview_image(image, "No preview", focus_image=focus_image)
    else:
        set_image(label, image, "No preview")


def plain_preview_from_path(path: str) -> Image.Image:
    return render_plain_preview(read_image_bgr(path), PREVIEW_SIZE)


def make_preview_label(text: str = "No image") -> FocusablePreviewLabel:
    return FocusablePreviewLabel(text)


class FacePreviewLabel(FocusablePreviewLabel):
    face_selected = Signal(int)

    def __init__(self, text: str = "No image") -> None:
        super().__init__(text)
        self.faces: list[AnalyzedFace] = []
        self.selected_index = 0
        self.original_size: tuple[int, int] = (0, 0)

    def set_plain_image(self, image: Image.Image | None, text: str = "No image") -> None:
        self.faces = []
        self.original_size = image.size if image is not None else (0, 0)
        self.base_image = image.copy() if image is not None else None
        self.focus_image = None
        self.empty_text = text
        self.focus_mode = False
        FocusablePreviewLabel.refresh_preview(self)
        self.sync_focus_button()

    def set_faces(self, faces: list[AnalyzedFace], selected_index: int = 0) -> None:
        self.faces = list(faces)
        if not self.faces:
            self.original_size = (0, 0)
            self.focus_mode = False
            self.display_image(None, "No face")
            self.sync_focus_button()
            return
        self.selected_index = max(0, min(int(selected_index), len(self.faces) - 1))
        height, width = self.faces[0].image_bgr.shape[:2]
        self.original_size = (int(width), int(height))
        self.refresh_preview()
        self.sync_focus_button()

    def can_focus(self) -> bool:
        return bool(self.faces)

    def refresh_preview(self) -> None:
        if not self.faces:
            self.display_image(None, "No face")
            return
        if self.focus_mode:
            image = render_focused_faces_overlay(self.faces, self.selected_index, PREVIEW_SIZE)
        else:
            image = render_faces_overlay(self.faces, self.selected_index, PREVIEW_SIZE)
        self.display_image(image, "No face")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.focus_mode or not self.faces or self.pixmap() is None:
            super().mousePressEvent(event)
            return
        pixmap = self.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        offset_x = (self.width() - pixmap_width) / 2.0
        offset_y = (self.height() - pixmap_height) / 2.0
        pos = event.position()
        x = float(pos.x()) - offset_x
        y = float(pos.y()) - offset_y
        if x < 0 or y < 0 or x > pixmap_width or y > pixmap_height:
            return
        original_width, original_height = self.original_size
        if original_width <= 0 or original_height <= 0:
            return
        image_x = x * original_width / max(1, pixmap_width)
        image_y = y * original_height / max(1, pixmap_height)
        matches: list[tuple[float, int]] = []
        for index, face in enumerate(self.faces):
            x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                matches.append(((x2 - x1) * (y2 - y1), index))
        if matches:
            _, index = min(matches)
            self.face_selected.emit(index)


def make_face_preview_label(text: str = "No image") -> FacePreviewLabel:
    return FacePreviewLabel(text)


def make_focus_button(label: FocusablePreviewLabel) -> QPushButton:
    button = QPushButton(FOCUS_BUTTON_TEXT)
    button.setObjectName("FocusButton")
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setFixedHeight(24)
    button.setMaximumWidth(118)
    button.setMinimumWidth(76)
    label.attach_focus_button(button)
    return button


def make_focus_overlay(label: FocusablePreviewLabel, button: QPushButton) -> QWidget:
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout = QGridLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(label, 0, 0)
    layout.addWidget(button, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.setRowStretch(0, 1)
    layout.setColumnStretch(0, 1)
    return container


def table_item(value: object) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class StudioPage(QWidget):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__()
        self.window = window

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "FSC Studio", message)

    def offer_legacy_conversion(self, file_path: str) -> None:
        answer = QMessageBox.question(
            self,
            "Convert legacy database",
            "Legacy .dtb databases cannot be searched directly in the modern engine. "
            "Convert this file to an InsightFace .fscdb database now?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.window.start_legacy_conversion(file_path)

    def run_task(
        self,
        status: str,
        fn: Callable,
        on_result: Callable[[object], None],
        *args,
        progress_arg: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
        **kwargs,
    ) -> None:
        self.window.run_task(
            status,
            fn,
            on_result,
            *args,
            progress_arg=progress_arg,
            on_progress=on_progress,
            on_error=self.show_error,
            **kwargs,
        )


class OverviewPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Overview")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        actions = QGroupBox("Workspace")
        action_grid = QGridLayout(actions)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        self.db_path.setPlaceholderText("No database loaded")
        new_btn = QPushButton("New Database")
        open_btn = QPushButton("Open / Convert")
        import_btn = QPushButton("Import Images")
        folder_btn = QPushButton("Import Folder")
        refresh_btn = QPushButton("Refresh")
        new_btn.clicked.connect(self.window.library_page.create_database)
        open_btn.clicked.connect(self.window.library_page.open_database)
        import_btn.clicked.connect(self.window.library_page.import_images)
        folder_btn.clicked.connect(self.window.library_page.import_folder)
        refresh_btn.clicked.connect(self.reload_summary)
        action_grid.addWidget(QLabel("Database"), 0, 0)
        action_grid.addWidget(self.db_path, 0, 1, 1, 5)
        action_grid.addWidget(new_btn, 1, 0)
        action_grid.addWidget(open_btn, 1, 1)
        action_grid.addWidget(import_btn, 1, 2)
        action_grid.addWidget(folder_btn, 1, 3)
        action_grid.addWidget(refresh_btn, 1, 4)
        layout.addWidget(actions)

        body = QHBoxLayout()

        metrics_box = QGroupBox("Database Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)
        body.addWidget(metrics_box, 1)

        attention_box = QGroupBox("Attention")
        attention_layout = QVBoxLayout(attention_box)
        self.attention_table = QTableWidget(0, 2)
        self.attention_table.setHorizontalHeaderLabels(["Queue", "Count"])
        self.attention_table.verticalHeader().setVisible(False)
        self.attention_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.attention_table.horizontalHeader().setStretchLastSection(True)
        attention_layout.addWidget(self.attention_table)
        review_btn = QPushButton("Review Queue")
        people_btn = QPushButton("People")
        search_btn = QPushButton("Search")
        cluster_btn = QPushButton("Clusters")
        review_btn.clicked.connect(lambda: self.window.show_page("Review"))
        people_btn.clicked.connect(lambda: self.window.show_page("People"))
        search_btn.clicked.connect(lambda: self.window.show_page("Search"))
        cluster_btn.clicked.connect(lambda: self.window.show_page("Clusters"))
        action_row = QHBoxLayout()
        action_row.addWidget(review_btn)
        action_row.addWidget(people_btn)
        action_row.addWidget(search_btn)
        action_row.addWidget(cluster_btn)
        attention_layout.addLayout(action_row)
        body.addWidget(attention_box, 1)

        layout.addLayout(body, 1)

        lists = QHBoxLayout()
        people_box = QGroupBox("Top People")
        people_layout = QVBoxLayout(people_box)
        self.people_table = QTableWidget(0, 3)
        self.people_table.setHorizontalHeaderLabels(["Person", "Faces", "Review"])
        self.people_table.verticalHeader().setVisible(False)
        self.people_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.people_table.horizontalHeader().setStretchLastSection(True)
        people_layout.addWidget(self.people_table)
        lists.addWidget(people_box, 1)

        tags_box = QGroupBox("Top Tags")
        tags_layout = QVBoxLayout(tags_box)
        self.tags_table = QTableWidget(0, 2)
        self.tags_table.setHorizontalHeaderLabels(["Tag", "Faces"])
        self.tags_table.verticalHeader().setVisible(False)
        self.tags_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.tags_table.horizontalHeader().setStretchLastSection(True)
        tags_layout.addWidget(self.tags_table)
        lists.addWidget(tags_box, 1)
        layout.addLayout(lists, 1)

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)
        self.reload_summary()

    def reload_summary(self) -> None:
        path = self.window.current_database
        if not path:
            self.db_path.clear()
            self.set_metric_rows([("Status", "No database loaded")])
            self.set_attention_rows([])
            self.set_people_rows([])
            self.set_tag_rows([])
            return
        try:
            stats = load_database_statistics(path)
            people = load_person_summaries(path, limit=10)
            tags = load_tag_summaries(path, limit=12)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.db_path.setText(path)
        self.set_metric_rows(
            [
                ("Faces", stats["face_count"]),
                ("People", stats["people_count"]),
                ("Tags", stats["tag_count"]),
                ("Average Quality", f"{stats['avg_quality']:.3f}"),
                ("Quality Range", f"{stats['min_quality']:.3f} - {stats['max_quality']:.3f}"),
                ("Model", stats.get("model_name", "")),
                ("Format", f"v{stats['format_version']}"),
            ]
        )
        self.set_attention_rows(
            [
                ("Needs review", stats["review_count"]),
                ("Ignored", stats["ignored_count"]),
                ("Duplicate image groups", stats["duplicate_hash_count"]),
            ]
        )
        self.set_people_rows(people)
        self.set_tag_rows(tags)

    def set_metric_rows(self, rows: list[tuple[str, object]]) -> None:
        self.metrics_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.metrics_table.setItem(row, 0, table_item(name))
            self.metrics_table.setItem(row, 1, table_item(value))

    def set_attention_rows(self, rows: list[tuple[str, object]]) -> None:
        self.attention_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.attention_table.setItem(row, 0, table_item(name))
            self.attention_table.setItem(row, 1, table_item(value))

    def set_people_rows(self, people: list[PersonSummary]) -> None:
        self.people_table.setRowCount(len(people))
        for row, person in enumerate(people):
            self.people_table.setItem(row, 0, table_item(person.name))
            self.people_table.setItem(row, 1, table_item(person.face_count))
            self.people_table.setItem(row, 2, table_item(person.review_count))

    def set_tag_rows(self, tags: list[TagSummary]) -> None:
        self.tags_table.setRowCount(len(tags))
        for row, tag in enumerate(tags):
            self.tags_table.setItem(row, 0, table_item(tag.name))
            self.tags_table.setItem(row, 1, table_item(tag.face_count))


class LibraryPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.records = []
        self.selected_record_id: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Library")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        db_box = QGroupBox("Database")
        db_layout = QGridLayout(db_box)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        create_btn = QPushButton("New")
        open_btn = QPushButton("Open")
        import_btn = QPushButton("Add Images")
        import_folder_btn = QPushButton("Add Folder")
        reload_btn = QPushButton("Reload")
        export_btn = QPushButton("Export CSV")
        create_btn.clicked.connect(self.create_database)
        open_btn.clicked.connect(self.open_database)
        import_btn.clicked.connect(self.import_images)
        import_folder_btn.clicked.connect(self.import_folder)
        reload_btn.clicked.connect(self.reload_records)
        export_btn.clicked.connect(self.export_csv)
        db_layout.addWidget(QLabel("File"), 0, 0)
        db_layout.addWidget(self.db_path, 0, 1, 1, 3)
        db_layout.addWidget(create_btn, 0, 4)
        db_layout.addWidget(open_btn, 0, 5)
        db_layout.addWidget(import_btn, 0, 6)
        db_layout.addWidget(import_folder_btn, 0, 7)
        db_layout.addWidget(reload_btn, 0, 8)
        db_layout.addWidget(export_btn, 0, 9)
        layout.addWidget(db_box)

        body = QHBoxLayout()
        self.min_quality = QDoubleSpinBox()
        self.min_quality.setRange(0.0, 1.0)
        self.min_quality.setSingleStep(0.05)
        self.min_quality.setDecimals(3)
        self.min_quality.setValue(0.0)
        db_layout.addWidget(QLabel("Min quality"), 1, 0)
        db_layout.addWidget(self.min_quality, 1, 1)

        self.stats_label = QLabel("No database loaded")
        db_layout.addWidget(self.stats_label, 1, 2, 1, 8)

        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("name, path, person, tag, notes")
        self.filter_person = QComboBox()
        self.filter_tag = QComboBox()
        self.filter_review = QComboBox()
        self.filter_review.addItem("All", "")
        for state in ["open", "reviewed", "duplicate", "low_quality", "ignored"]:
            self.filter_review.addItem(state, state)
        self.filter_min_quality = QDoubleSpinBox()
        self.filter_min_quality.setRange(0.0, 1.0)
        self.filter_min_quality.setSingleStep(0.05)
        self.filter_min_quality.setDecimals(3)
        self.filter_min_quality.setValue(0.0)
        self.filter_include_ignored = QCheckBox("Include ignored")
        self.filter_include_ignored.setChecked(True)
        apply_filter = QPushButton("Apply Filter")
        reset_filter = QPushButton("Reset Filter")
        apply_filter.clicked.connect(self.reload_records)
        reset_filter.clicked.connect(self.reset_filters)
        self.filter_text.returnPressed.connect(self.reload_records)
        db_layout.addWidget(QLabel("Filter"), 2, 0)
        db_layout.addWidget(self.filter_text, 2, 1, 1, 2)
        db_layout.addWidget(QLabel("Person"), 2, 3)
        db_layout.addWidget(self.filter_person, 2, 4)
        db_layout.addWidget(QLabel("Tag"), 2, 5)
        db_layout.addWidget(self.filter_tag, 2, 6)
        db_layout.addWidget(self.filter_include_ignored, 2, 7)
        db_layout.addWidget(apply_filter, 2, 8)
        db_layout.addWidget(reset_filter, 2, 9)
        db_layout.addWidget(QLabel("Review"), 3, 0)
        db_layout.addWidget(self.filter_review, 3, 1)
        db_layout.addWidget(QLabel("Min quality"), 3, 2)
        db_layout.addWidget(self.filter_min_quality, 3, 3)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Name", "Person", "Tags", "Review", "Ignored", "Dupes", "Quality", "Source"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self.show_selected_record)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(2, 130)
        self.table.setColumnWidth(3, 160)
        self.table.setColumnWidth(4, 95)
        self.table.setColumnWidth(5, 95)
        self.table.setColumnWidth(6, 75)
        self.table.setColumnWidth(7, 80)
        self.table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.table, 2)

        right_panel = QWidget()
        right_panel.setMinimumWidth(430)
        right_panel.setMaximumWidth(500)
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        self.preview = make_preview_label("Select a face")
        self.preview.setMinimumSize(320, 320)
        self.preview_focus = make_focus_button(self.preview)
        metadata_box = QGroupBox("Selected Face")
        metadata_form = QFormLayout(metadata_box)
        self.person_edit = QLineEdit()
        self.tags_edit = QLineEdit()
        self.review_combo = QComboBox()
        self.review_combo.addItems(["open", "reviewed", "duplicate", "low_quality", "ignored"])
        self.ignored_check = QCheckBox("Ignore in search")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(70)
        save_metadata = QPushButton("Save Metadata")
        save_metadata.clicked.connect(self.save_selected_metadata)
        metadata_form.addRow("Person", self.person_edit)
        metadata_form.addRow("Tags", self.tags_edit)
        metadata_form.addRow("Review", self.review_combo)
        metadata_form.addRow("", self.ignored_check)
        metadata_form.addRow("Notes", self.notes_edit)
        metadata_form.addRow("", save_metadata)
        batch_box = QGroupBox("Batch Selected")
        batch_form = QFormLayout(batch_box)
        self.batch_person_edit = QLineEdit()
        self.batch_tags_edit = QLineEdit()
        self.batch_append_tags = QCheckBox("Append tags")
        self.batch_review_combo = QComboBox()
        self.batch_review_combo.addItems(["No change", "open", "reviewed", "duplicate", "low_quality", "ignored"])
        self.batch_ignored_combo = QComboBox()
        self.batch_ignored_combo.addItems(["No change", "Ignore", "Restore"])
        self.batch_notes_edit = QLineEdit()
        self.batch_notes_edit.setPlaceholderText("leave blank for no change")
        apply_batch = QPushButton("Apply to Selection")
        apply_batch.clicked.connect(self.apply_batch_metadata)
        batch_form.addRow("Person", self.batch_person_edit)
        batch_form.addRow("Tags", self.batch_tags_edit)
        batch_form.addRow("", self.batch_append_tags)
        batch_form.addRow("Review", self.batch_review_combo)
        batch_form.addRow("Ignored", self.batch_ignored_combo)
        batch_form.addRow("Notes", self.batch_notes_edit)
        batch_form.addRow("", apply_batch)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        activity_box = QWidget()
        activity_layout = QVBoxLayout(activity_box)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        activity_layout.addWidget(self.progress)
        activity_layout.addWidget(self.log, 1)
        tabs = QTabWidget()
        tabs.addTab(metadata_box, "Selected")
        tabs.addTab(batch_box, "Batch")
        tabs.addTab(activity_box, "Activity")
        right.addWidget(make_focus_overlay(self.preview, self.preview_focus))
        right.addWidget(tabs, 1)
        body.addWidget(right_panel, 0)
        layout.addLayout(body, 1)

    def create_database(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Create database", "", DB_FILTER)
        if not file_path:
            return
        file_path = normalize_database_path(file_path)
        self.progress.setRange(0, 0)
        self.run_task("Creating database...", create_database, self.on_database_created, file_path)

    def on_database_created(self, result: object) -> None:
        path = str(result)
        self.window.set_current_database(path)
        self.db_path.setText(path)
        self.log.append(f"Created {path}")
        self.reload_records()

    def open_database(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open database", "", DB_FILTER)
        if not file_path:
            return
        if file_path.lower().endswith(".dtb"):
            self.offer_legacy_conversion(file_path)
            return
        self.window.set_current_database(file_path)
        self.db_path.setText(file_path)
        self.reload_records()

    def import_images(self) -> None:
        if not self.window.current_database:
            file_path, _ = QFileDialog.getSaveFileName(self, "Create database", "", DB_FILTER)
            if not file_path:
                return
            self.window.set_current_database(normalize_database_path(file_path))
            self.db_path.setText(self.window.current_database)

        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", IMAGE_FILTER)
        if not files:
            return
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.run_task(
            "Importing images...",
            import_images_to_database,
            self.on_import_complete,
            self.window.current_database,
            files,
            min_quality=self.min_quality.value(),
            progress_arg=True,
            on_progress=self.on_import_progress,
        )

    def import_folder(self) -> None:
        if not self.window.current_database:
            file_path, _ = QFileDialog.getSaveFileName(self, "Create database", "", DB_FILTER)
            if not file_path:
                return
            self.window.set_current_database(normalize_database_path(file_path))
            self.db_path.setText(self.window.current_database)

        folder = QFileDialog.getExistingDirectory(self, "Add folder recursively")
        if not folder:
            return
        files = collect_image_paths([folder], recursive=True)
        if not files:
            self.show_error("No supported image files were found in the selected folder.")
            return
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.log.append(f"Importing {len(files)} image file(s) from {folder}")
        self.run_task(
            "Importing folder...",
            import_images_to_database,
            self.on_import_complete,
            self.window.current_database,
            files,
            min_quality=self.min_quality.value(),
            progress_arg=True,
            on_progress=self.on_import_progress,
        )

    def on_import_progress(self, message: str, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        if message.startswith("IMAGE_PREVIEW|"):
            _, path, text = message.split("|", 2)
            try:
                set_image(self.preview, plain_preview_from_path(path), "No preview")
            except Exception:
                pass
            self.log.append(text)
            self.window.set_status(f"{text} ({current}/{total})")
            return
        self.log.append(message)

    def on_import_complete(self, result: object) -> None:
        summary = result
        if isinstance(summary, ImportSummary):
            self.log.append(
                f"Done: {summary.faces_saved} face(s), "
                f"{summary.images_without_faces} no-face image(s), "
                f"{summary.failed_images} failed image(s), "
                f"{summary.low_quality_faces} low-quality face(s) skipped, "
                f"{summary.duplicate_images} duplicate image(s), "
                f"avg quality {summary.average_quality:.3f}."
            )
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.review_page.reload_records()

    def reload_records(self) -> None:
        path = self.window.current_database
        if not path:
            self.table.setRowCount(0)
            set_image(self.preview, None, "No database")
            return
        try:
            records, metadata = load_records(
                path,
                include_ignored=self.filter_include_ignored.isChecked(),
                person_filter=str(self.filter_person.currentData() or ""),
                tag_filter=str(self.filter_tag.currentData() or ""),
                review_filter=str(self.filter_review.currentData() or ""),
                min_quality=self.filter_min_quality.value(),
                text_filter=self.filter_text.text().strip(),
            )
        except LegacyDatabaseError as exc:
            self.show_error(str(exc))
            return
        except Exception as exc:
            self.show_error(str(exc))
            return

        self.records = records
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            self.table.setItem(row, 0, table_item(record.id))
            self.table.setItem(row, 1, table_item(record.file_name))
            self.table.setItem(row, 2, table_item(record.person_name))
            self.table.setItem(row, 3, table_item(record.tag_text))
            self.table.setItem(row, 4, table_item(record.review_state))
            self.table.setItem(row, 5, table_item("yes" if record.ignored else ""))
            self.table.setItem(row, 6, table_item(record.duplicate_count if record.duplicate_count > 1 else ""))
            self.table.setItem(row, 7, table_item(f"{record.quality_score:.3f}"))
            self.table.setItem(row, 8, table_item(record.source_path))
        stats = load_database_statistics(path)
        self.stats_label.setText(
            "Faces: {face_count}    Avg quality: {avg_quality:.3f}    "
            "People: {people_count}    Tags: {tag_count}    Review: {review_count}    "
            "Ignored: {ignored_count}    Format: v{format_version}".format(**stats)
        )
        self.log.append(
            f"Loaded {len(records)} face(s). Model: {metadata.get('model_name', 'unknown')}"
        )
        self.refresh_filter_options()
        self.window.search_page.refresh_filter_options()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

    def show_selected_record(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.records):
            return
        record = self.records[row]
        self.selected_record_id = record.id
        set_record_preview(self.preview, record, self.window.current_database)
        self.person_edit.setText(record.person_name)
        self.tags_edit.setText(record.tag_text)
        index = self.review_combo.findText(record.review_state)
        self.review_combo.setCurrentIndex(index if index >= 0 else 0)
        self.ignored_check.setChecked(record.ignored)
        self.notes_edit.setPlainText(record.notes)

    def save_selected_metadata(self) -> None:
        if not self.window.current_database or self.selected_record_id is None:
            self.show_error("Select a face first.")
            return
        try:
            assign_person(self.window.current_database, self.selected_record_id, self.person_edit.text())
            set_tags(self.window.current_database, self.selected_record_id, self.tags_edit.text())
            update_review(
                self.window.current_database,
                self.selected_record_id,
                ignored=self.ignored_check.isChecked(),
                review_state=self.review_combo.currentText(),
                notes=self.notes_edit.toPlainText(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.log.append(f"Updated face {self.selected_record_id}")
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.review_page.reload_records()

    def apply_batch_metadata(self) -> None:
        if not self.window.current_database:
            self.show_error("Open or create a database first.")
            return
        face_ids = self.selected_face_ids()
        if not face_ids:
            self.show_error("Select one or more faces first.")
            return
        person_name = self.batch_person_edit.text().strip()
        tag_text = self.batch_tags_edit.text().strip()
        review_state = self.batch_review_combo.currentText()
        ignored_text = self.batch_ignored_combo.currentText()
        notes = self.batch_notes_edit.text()
        ignored = None
        if ignored_text == "Ignore":
            ignored = True
        elif ignored_text == "Restore":
            ignored = False
        try:
            changed = update_faces_metadata(
                self.window.current_database,
                face_ids,
                person_name=person_name if person_name else None,
                tag_text=tag_text if tag_text else None,
                append_tags=self.batch_append_tags.isChecked(),
                ignored=ignored,
                review_state=None if review_state == "No change" else review_state,
                notes=notes if notes else None,
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.log.append(f"Batch updated {changed} selected face(s).")
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.search_page.refresh_filter_options()
        self.window.review_page.reload_records()

    def selected_face_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.table.selectionModel().selectedRows():
            row = index.row()
            if 0 <= row < len(self.records):
                ids.append(int(self.records[row].id))
        return ids

    def export_csv(self) -> None:
        if not self.window.current_database:
            self.show_error("Open or create a database first.")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "Export library CSV", "", "CSV (*.csv);;All files (*.*)")
        if not output_path:
            return
        try:
            result = export_faces_csv(self.window.current_database, output_path)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.log.append(f"Exported {result}")

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)
        self.refresh_filter_options()

    def refresh_filter_options(self) -> None:
        path = self.window.current_database
        person_value = self.filter_person.currentData()
        tag_value = self.filter_tag.currentData()
        self.filter_person.clear()
        self.filter_tag.clear()
        self.filter_person.addItem("All", "")
        self.filter_tag.addItem("All", "")
        if path:
            try:
                for person in load_people(path):
                    self.filter_person.addItem(person.name, person.name)
                for tag in load_tags(path):
                    self.filter_tag.addItem(tag.name, tag.name)
            except Exception:
                pass
        self._restore_combo_value(self.filter_person, person_value)
        self._restore_combo_value(self.filter_tag, tag_value)

    def reset_filters(self) -> None:
        self.filter_text.clear()
        self.filter_review.setCurrentIndex(0)
        self.filter_min_quality.setValue(0.0)
        self.filter_include_ignored.setChecked(True)
        self.filter_person.setCurrentIndex(0)
        self.filter_tag.setCurrentIndex(0)
        self.reload_records()

    def _restore_combo_value(self, combo: QComboBox, value: object) -> None:
        if value in (None, ""):
            combo.setCurrentIndex(0)
            return
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)


class PeoplePage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.people: list[PersonSummary] = []
        self.members = []
        self.current_person: PersonSummary | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("People")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("People")
        grid = QGridLayout(controls)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("person name or notes")
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_people)
        self.filter_text.returnPressed.connect(self.reload_people)
        grid.addWidget(QLabel("Database"), 0, 0)
        grid.addWidget(self.db_path, 0, 1, 1, 4)
        grid.addWidget(reload_btn, 0, 5)
        grid.addWidget(QLabel("Filter"), 1, 0)
        grid.addWidget(self.filter_text, 1, 1, 1, 5)
        layout.addWidget(controls)

        body = QHBoxLayout()
        self.people_table = QTableWidget(0, 5)
        self.people_table.setHorizontalHeaderLabels(["Name", "Faces", "Avg Q", "Rev", "Ign"])
        self.people_table.verticalHeader().setVisible(False)
        self.people_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.people_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.people_table.itemSelectionChanged.connect(self.show_selected_person)
        self.people_table.setColumnWidth(0, 160)
        self.people_table.setColumnWidth(1, 60)
        self.people_table.setColumnWidth(2, 65)
        self.people_table.setColumnWidth(3, 45)
        self.people_table.setColumnWidth(4, 45)
        self.people_table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.people_table, 1)

        self.member_table = QTableWidget(0, 6)
        self.member_table.setHorizontalHeaderLabels(["ID", "Name", "Tags", "Quality", "Review", "Ignored"])
        self.member_table.verticalHeader().setVisible(False)
        self.member_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.member_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.member_table.itemSelectionChanged.connect(self.show_selected_member)
        self.member_table.setColumnWidth(0, 70)
        self.member_table.setColumnWidth(1, 180)
        self.member_table.setColumnWidth(2, 150)
        self.member_table.setColumnWidth(3, 80)
        self.member_table.setColumnWidth(4, 90)
        self.member_table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.member_table, 1)

        right = QVBoxLayout()
        self.preview = make_preview_label("Select a person")
        self.preview_focus = make_focus_button(self.preview)
        editor = QGroupBox("Manage Person")
        form = QFormLayout(editor)
        self.name_edit = QLineEdit()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(90)
        self.merge_target = QComboBox()
        save_btn = QPushButton("Save Name / Notes")
        merge_btn = QPushButton("Merge Into Target")
        clear_btn = QPushButton("Clear Assignment")
        save_btn.clicked.connect(self.save_person)
        merge_btn.clicked.connect(self.merge_selected_person)
        clear_btn.clicked.connect(self.clear_selected_person)
        form.addRow("Name", self.name_edit)
        form.addRow("Notes", self.notes_edit)
        form.addRow("Target", self.merge_target)
        form.addRow("", save_btn)
        form.addRow("", merge_btn)
        form.addRow("", clear_btn)
        self.summary = QLabel("No person selected")
        right.addWidget(make_focus_overlay(self.preview, self.preview_focus))
        right.addWidget(editor)
        right.addWidget(self.summary)
        right.addStretch()
        body.addLayout(right, 1)
        layout.addLayout(body, 1)

    def reload_people(self) -> None:
        if not self.window.current_database:
            self.people_table.setRowCount(0)
            self.member_table.setRowCount(0)
            set_image(self.preview, None, "No database")
            return
        try:
            self.people = load_person_summaries(
                self.window.current_database,
                text_filter=self.filter_text.text().strip(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.people_table.setRowCount(len(self.people))
        for row, person in enumerate(self.people):
            self.people_table.setItem(row, 0, table_item(person.name))
            self.people_table.setItem(row, 1, table_item(person.face_count))
            self.people_table.setItem(row, 2, table_item(f"{person.average_quality:.3f}"))
            self.people_table.setItem(row, 3, table_item(person.review_count))
            self.people_table.setItem(row, 4, table_item(person.ignored_count))
        self.refresh_merge_targets()
        self.member_table.setRowCount(0)
        self.current_person = None
        self.summary.setText(f"{len(self.people)} person(s)")
        if self.people:
            self.people_table.selectRow(0)

    def show_selected_person(self) -> None:
        selected = self.people_table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.people):
            return
        person = self.people[row]
        self.current_person = person
        self.name_edit.setText(person.name)
        self.notes_edit.setPlainText(person.notes)
        self.refresh_merge_targets(exclude_person_id=person.id)
        self.load_members(person)
        if person.representative_face_id:
            self.show_preview(person.representative_face_id)
        else:
            set_image(self.preview, None, "No faces")
        self.summary.setText(
            f"{person.name}: {person.face_count} face(s), {person.review_count} review item(s), "
            f"avg quality {person.average_quality:.3f}"
        )

    def load_members(self, person: PersonSummary) -> None:
        if not self.window.current_database:
            return
        try:
            records, _ = load_records(
                self.window.current_database,
                include_ignored=True,
                person_filter=person.name,
                include_preview=False,
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.members = records
        self.member_table.setRowCount(len(records))
        for row, record in enumerate(records):
            self.member_table.setItem(row, 0, table_item(record.id))
            self.member_table.setItem(row, 1, table_item(record.file_name))
            self.member_table.setItem(row, 2, table_item(record.tag_text))
            self.member_table.setItem(row, 3, table_item(f"{record.quality_score:.3f}"))
            self.member_table.setItem(row, 4, table_item(record.review_state))
            self.member_table.setItem(row, 5, table_item("yes" if record.ignored else ""))

    def show_selected_member(self) -> None:
        selected = self.member_table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.members):
            return
        set_record_preview(self.preview, self.members[row], self.window.current_database)

    def show_preview(self, face_id: int) -> None:
        if not self.window.current_database:
            return
        record = next((item for item in self.members if item.id == face_id), None)
        if record:
            set_record_preview(self.preview, record, self.window.current_database)
            return
        try:
            preview = load_preview(self.window.current_database, face_id)
        except Exception:
            preview = None
        set_image(self.preview, preview_png_to_pil(preview) if preview else None, "No preview")

    def save_person(self) -> None:
        if not self.window.current_database or not self.current_person:
            self.show_error("Select a person first.")
            return
        try:
            rename_person(
                self.window.current_database,
                self.current_person.id,
                self.name_edit.text(),
                self.notes_edit.toPlainText(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.after_people_changed("Person updated.")

    def merge_selected_person(self) -> None:
        if not self.window.current_database or not self.current_person:
            self.show_error("Select a person first.")
            return
        target_id = int(self.merge_target.currentData() or 0)
        if not target_id:
            self.show_error("Select a merge target first.")
            return
        try:
            moved = merge_people(self.window.current_database, self.current_person.id, target_id)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.after_people_changed(f"Merged {moved} face(s).")

    def clear_selected_person(self) -> None:
        if not self.window.current_database or not self.current_person:
            self.show_error("Select a person first.")
            return
        answer = QMessageBox.question(
            self,
            "FSC Studio",
            f"Clear all assignments for {self.current_person.name} and delete this person?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            cleared = clear_person_assignment(self.window.current_database, self.current_person.id)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.after_people_changed(f"Cleared {cleared} face assignment(s).")

    def after_people_changed(self, status: str) -> None:
        self.window.set_status(status)
        self.reload_people()
        self.window.overview_page.reload_summary()
        self.window.library_page.reload_records()
        self.window.search_page.refresh_filter_options()
        self.window.review_page.reload_records()

    def refresh_merge_targets(self, exclude_person_id: int | None = None) -> None:
        current_value = self.merge_target.currentData()
        self.merge_target.clear()
        self.merge_target.addItem("Select target", 0)
        for person in self.people:
            if exclude_person_id is not None and person.id == exclude_person_id:
                continue
            self.merge_target.addItem(f"{person.name} ({person.face_count})", person.id)
        if current_value:
            index = self.merge_target.findData(current_value)
            self.merge_target.setCurrentIndex(index if index >= 0 else 0)

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)


class SearchPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.query_path = ""
        self.query_face: PrimaryFace | None = None
        self.query_faces: list[AnalyzedFace] = []
        self.selected_query_index = 0
        self.current_search_database = ""
        self._updating_query_faces = False
        self.query_generation = 0
        self.search_generation = 0
        self.active_search_generation = 0
        self.hits: list[SearchHit] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("Query")
        grid = QGridLayout(controls)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        self.query_edit = QLineEdit()
        self.query_edit.setReadOnly(True)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 500)
        self.top_k.setValue(30)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1.0, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(3)
        self.threshold.setValue(-1.0)
        self.min_quality = QDoubleSpinBox()
        self.min_quality.setRange(0.0, 1.0)
        self.min_quality.setSingleStep(0.05)
        self.min_quality.setDecimals(3)
        self.min_quality.setValue(0.0)
        self.person_filter = QComboBox()
        self.tag_filter = QComboBox()
        self.include_ignored = QCheckBox("Include ignored")
        select_query = QPushButton("Select Image")
        open_db = QPushButton("Open Database")
        use_current = QPushButton("Use Library DB")
        search_btn = QPushButton("Search")
        select_query.clicked.connect(self.select_query)
        open_db.clicked.connect(self.open_database)
        use_current.clicked.connect(self.use_current_database)
        search_btn.clicked.connect(self.search)
        grid.addWidget(QLabel("Database"), 0, 0)
        grid.addWidget(self.db_path, 0, 1, 1, 3)
        grid.addWidget(open_db, 0, 4)
        grid.addWidget(use_current, 0, 5)
        grid.addWidget(QLabel("Image"), 1, 0)
        grid.addWidget(self.query_edit, 1, 1, 1, 3)
        grid.addWidget(select_query, 1, 4)
        grid.addWidget(search_btn, 1, 5)
        grid.addWidget(QLabel("Top K"), 2, 0)
        grid.addWidget(self.top_k, 2, 1)
        grid.addWidget(QLabel("Threshold"), 2, 2)
        grid.addWidget(self.threshold, 2, 3)
        grid.addWidget(QLabel("Min quality"), 2, 4)
        grid.addWidget(self.min_quality, 2, 5)
        grid.addWidget(QLabel("Person"), 3, 0)
        grid.addWidget(self.person_filter, 3, 1)
        grid.addWidget(QLabel("Tag"), 3, 2)
        grid.addWidget(self.tag_filter, 3, 3)
        grid.addWidget(self.include_ignored, 3, 4, 1, 2)
        layout.addWidget(controls)

        body = QHBoxLayout()
        body.setSpacing(12)
        preview_panel = QWidget()
        preview_panel.setMinimumWidth(300)
        preview_panel.setMaximumWidth(430)
        preview_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        previews = QVBoxLayout(preview_panel)
        previews.setContentsMargins(0, 0, 0, 0)
        previews.setSpacing(8)
        self.query_preview = make_face_preview_label("Query")
        self.query_preview.face_selected.connect(self.select_query_face)
        self.query_focus = make_focus_button(self.query_preview)
        self.query_face_list = QListWidget()
        self.query_face_list.setMinimumHeight(68)
        self.query_face_list.setMaximumHeight(96)
        self.query_face_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.query_face_list.itemSelectionChanged.connect(self.on_query_face_list_changed)
        self.result_preview = make_preview_label("Result")
        self.result_focus = make_focus_button(self.result_preview)
        previews.addWidget(make_focus_overlay(self.query_preview, self.query_focus), 1)
        previews.addWidget(self.query_face_list)
        previews.addWidget(make_focus_overlay(self.result_preview, self.result_focus), 1)
        body.addWidget(preview_panel, 0)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "ID", "Name", "Person", "Tags", "Cosine", "Similarity", "Quality"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_selected_result)
        body.addWidget(self.table, 1)
        layout.addLayout(body, 1)

    def open_database(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open database", "", DB_FILTER)
        if not file_path:
            return
        if file_path.lower().endswith(".dtb"):
            self.offer_legacy_conversion(file_path)
            return
        self.window.set_current_database(file_path)
        self.db_path.setText(file_path)
        self.refresh_filter_options()

    def use_current_database(self) -> None:
        self.db_path.setText(self.window.current_database)
        self.refresh_filter_options()

    def select_query(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select query image", "", IMAGE_FILTER)
        if not file_path:
            return
        self.query_generation += 1
        generation = self.query_generation
        self.query_path = file_path
        self.query_edit.setText(file_path)
        self.query_faces = []
        self.selected_query_index = 0
        self.query_face_list.clear()
        self.query_face = None
        self.table.setRowCount(0)
        self.hits = []
        set_image(self.result_preview, None, "Result")
        try:
            self.query_preview.set_plain_image(plain_preview_from_path(file_path))
        except Exception as exc:
            self.show_error(str(exc))
            return
        def task() -> dict[str, object]:
            return {"generation": generation, "path": file_path, "faces": analyze_faces(file_path)}

        self.run_task("Detecting query faces...", task, self.on_query_faces_ready)

    def on_query_faces_ready(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        generation = int(result.get("generation", 0))
        if generation != self.query_generation:
            return
        self.query_faces = list(result.get("faces", []))
        self._updating_query_faces = True
        self.query_face_list.clear()
        for index, face in enumerate(self.query_faces, start=1):
            self.query_face_list.addItem(
                f"Face {index}: det {face.det_score:.3f}, quality {face.quality_score:.3f}"
            )
        self._updating_query_faces = False
        self.select_query_face(0)
        self.window.set_status(f"Detected {len(self.query_faces)} face(s). Select a face, then search.")

    def on_query_face_list_changed(self) -> None:
        if self._updating_query_faces:
            return
        row = self.query_face_list.currentRow()
        if row >= 0:
            self.select_query_face(row)

    def select_query_face(self, index: int) -> None:
        if not self.query_faces:
            return
        self.selected_query_index = max(0, min(int(index), len(self.query_faces) - 1))
        self.query_preview.set_faces(self.query_faces, self.selected_query_index)
        if self.query_face_list.currentRow() != self.selected_query_index:
            self._updating_query_faces = True
            self.query_face_list.setCurrentRow(self.selected_query_index)
            self._updating_query_faces = False

    def search(self) -> None:
        database_path = self.db_path.text() or self.window.current_database
        if not database_path:
            self.show_error("Open or create a database first.")
            return
        if not self.query_faces:
            self.show_error("Select a query image and wait for face detection first.")
            return

        selected_face = self.query_faces[self.selected_query_index]
        detected_count = len(self.query_faces)
        self.search_generation += 1
        generation = self.search_generation
        self.active_search_generation = generation
        self.hits = []
        self.table.setRowCount(0)

        def task(progress=None) -> dict[str, object]:
            primary = PrimaryFace(face=selected_face, detected_count=detected_count)
            def tagged_progress(message: str, current: int, total: int) -> None:
                if progress:
                    progress(f"SEARCH_PROGRESS|{generation}|{message}", current, total)

            hits, metadata = search_database_progressive(
                database_path,
                primary.face.embedding,
                top_k=self.top_k.value(),
                threshold=self.threshold.value(),
                person_filter=str(self.person_filter.currentData() or ""),
                tag_filter=str(self.tag_filter.currentData() or ""),
                min_quality=self.min_quality.value(),
                include_ignored=self.include_ignored.isChecked(),
                progress=tagged_progress,
            )
            return {"generation": generation, "primary": primary, "hits": hits, "metadata": metadata}

        self.current_search_database = database_path
        self.run_task(
            "Searching...",
            task,
            self.on_search_complete,
            progress_arg=True,
            on_progress=self.on_search_progress,
        )

    def on_search_progress(self, message: str, current: int, total: int) -> None:
        if not message.startswith("SEARCH_PROGRESS|"):
            self.window.set_status(f"{message} ({current}/{total})")
            return
        parts = message.split("|", 3)
        if len(parts) < 4:
            return
        try:
            generation = int(parts[1])
        except ValueError:
            return
        if generation != self.active_search_generation:
            return
        message = parts[3]
        if message.startswith("FACE_PREVIEW|"):
            parts = message.split("|", 2)
            face_id = int(parts[1])
            name = parts[2] if len(parts) > 2 else str(face_id)
            try:
                preview = load_preview(self.current_search_database, face_id)
                set_image(self.result_preview, preview_png_to_pil(preview) if preview else None, "No preview")
            except Exception:
                pass
            self.window.set_status(f"Comparing {current}/{total}: {name}")
            return
        self.window.set_status(f"{message} ({current}/{total})")

    def on_search_complete(self, result: object) -> None:
        data = result
        if not isinstance(data, dict):
            return
        generation = int(data.get("generation", 0))
        if generation != self.active_search_generation:
            return
        self.active_search_generation = 0
        self.query_face = data["primary"]
        self.hits = list(data["hits"])
        self.query_preview.set_faces(self.query_faces, self.selected_query_index)
        self.table.setRowCount(len(self.hits))
        for row, hit in enumerate(self.hits):
            self.table.setItem(row, 0, table_item(row + 1))
            self.table.setItem(row, 1, table_item(hit.record.id))
            self.table.setItem(row, 2, table_item(hit.record.file_name))
            self.table.setItem(row, 3, table_item(hit.record.person_name))
            self.table.setItem(row, 4, table_item(hit.record.tag_text))
            self.table.setItem(row, 5, table_item(f"{hit.cosine:.4f}"))
            self.table.setItem(row, 6, table_item(f"{hit.similarity:.2f}%"))
            self.table.setItem(row, 7, table_item(f"{hit.record.quality_score:.3f}"))
        if self.hits:
            self.table.selectRow(0)
            self.show_hit_preview(0)
        else:
            set_image(self.result_preview, None, "No results")
        self.window.set_status(
            f"Search complete: {len(self.hits)} result(s). "
            f"Query quality {self.query_face.face.quality_score:.3f}"
        )

    def show_selected_result(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        self.show_hit_preview(row)

    def show_hit_preview(self, row: int) -> None:
        if row >= len(self.hits):
            return
        record = self.hits[row].record
        database_path = self.db_path.text() or self.window.current_database
        set_record_preview(self.result_preview, record, database_path)

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)
        self.refresh_filter_options()

    def refresh_filter_options(self) -> None:
        database_path = self.db_path.text() or self.window.current_database
        self.person_filter.clear()
        self.tag_filter.clear()
        self.person_filter.addItem("All", "")
        self.tag_filter.addItem("All", "")
        if not database_path:
            return
        try:
            for person in load_people(database_path):
                self.person_filter.addItem(person.name, person.name)
            for tag in load_tags(database_path):
                self.tag_filter.addItem(tag.name, tag.name)
        except Exception:
            return


class ReviewPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.records = []
        self.selected_record_id: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Review")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("Queue")
        grid = QGridLayout(controls)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        reload_btn = QPushButton("Reload")
        mark_reviewed_btn = QPushButton("Mark Reviewed")
        ignore_btn = QPushButton("Ignore / Restore")
        save_btn = QPushButton("Save Metadata")
        reload_btn.clicked.connect(self.reload_records)
        mark_reviewed_btn.clicked.connect(self.mark_reviewed)
        ignore_btn.clicked.connect(self.toggle_ignored)
        save_btn.clicked.connect(self.save_selected_metadata)
        grid.addWidget(QLabel("Database"), 0, 0)
        grid.addWidget(self.db_path, 0, 1, 1, 5)
        grid.addWidget(reload_btn, 0, 6)
        grid.addWidget(mark_reviewed_btn, 0, 7)
        grid.addWidget(ignore_btn, 0, 8)
        grid.addWidget(save_btn, 0, 9)
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("name, path, person, tag, notes")
        self.limit = QSpinBox()
        self.limit.setRange(10, 10000)
        self.limit.setValue(500)
        reset_filter = QPushButton("Reset Filter")
        reset_filter.clicked.connect(self.reset_filters)
        self.filter_text.returnPressed.connect(self.reload_records)
        grid.addWidget(QLabel("Filter"), 1, 0)
        grid.addWidget(self.filter_text, 1, 1, 1, 4)
        grid.addWidget(QLabel("Limit"), 1, 5)
        grid.addWidget(self.limit, 1, 6)
        grid.addWidget(reset_filter, 1, 7)
        layout.addWidget(controls)

        body = QHBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Name", "Reason", "Person", "Tags", "Quality", "Dupes", "Notes"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.show_selected_record)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(2, 180)
        self.table.setColumnWidth(3, 130)
        self.table.setColumnWidth(4, 160)
        self.table.setColumnWidth(5, 80)
        self.table.setColumnWidth(6, 60)
        self.table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.table, 2)

        right = QVBoxLayout()
        self.preview = make_preview_label("Select a review item")
        self.preview_focus = make_focus_button(self.preview)
        editor = QGroupBox("Edit")
        form = QFormLayout(editor)
        self.person_edit = QLineEdit()
        self.tags_edit = QLineEdit()
        self.review_combo = QComboBox()
        self.review_combo.addItems(["open", "reviewed", "duplicate", "low_quality", "ignored"])
        self.ignored_check = QCheckBox("Ignore in search")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(100)
        form.addRow("Person", self.person_edit)
        form.addRow("Tags", self.tags_edit)
        form.addRow("Review", self.review_combo)
        form.addRow("", self.ignored_check)
        form.addRow("Notes", self.notes_edit)
        right.addWidget(make_focus_overlay(self.preview, self.preview_focus))
        right.addWidget(editor)
        right.addStretch()
        body.addLayout(right, 1)
        layout.addLayout(body, 1)

    def reload_records(self) -> None:
        path = self.window.current_database
        if not path:
            self.table.setRowCount(0)
            set_image(self.preview, None, "No database")
            return
        try:
            records, _ = load_review_queue(
                path,
                limit=self.limit.value(),
                text_filter=self.filter_text.text().strip(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return

        self.records = records
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            self.table.setItem(row, 0, table_item(record.id))
            self.table.setItem(row, 1, table_item(record.file_name))
            self.table.setItem(row, 2, table_item(self.review_reason(record)))
            self.table.setItem(row, 3, table_item(record.person_name))
            self.table.setItem(row, 4, table_item(record.tag_text))
            self.table.setItem(row, 5, table_item(f"{record.quality_score:.3f}"))
            self.table.setItem(row, 6, table_item(record.duplicate_count if record.duplicate_count > 1 else ""))
            self.table.setItem(row, 7, table_item(record.notes))
        self.window.set_status(f"Review queue: {len(records)} item(s).")

    def show_selected_record(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.records):
            return
        record = self.records[row]
        self.selected_record_id = record.id
        set_record_preview(self.preview, record, self.window.current_database)
        self.person_edit.setText(record.person_name)
        self.tags_edit.setText(record.tag_text)
        index = self.review_combo.findText(record.review_state)
        self.review_combo.setCurrentIndex(index if index >= 0 else 0)
        self.ignored_check.setChecked(record.ignored)
        self.notes_edit.setPlainText(record.notes)

    def save_selected_metadata(self) -> None:
        if not self.window.current_database or self.selected_record_id is None:
            self.show_error("Select a review item first.")
            return
        try:
            assign_person(self.window.current_database, self.selected_record_id, self.person_edit.text())
            set_tags(self.window.current_database, self.selected_record_id, self.tags_edit.text())
            update_review(
                self.window.current_database,
                self.selected_record_id,
                ignored=self.ignored_check.isChecked(),
                review_state=self.review_combo.currentText(),
                notes=self.notes_edit.toPlainText(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.library_page.reload_records()

    def mark_reviewed(self) -> None:
        if not self.window.current_database or self.selected_record_id is None:
            self.show_error("Select a review item first.")
            return
        try:
            update_review(self.window.current_database, self.selected_record_id, review_state="reviewed")
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.library_page.reload_records()

    def toggle_ignored(self) -> None:
        record = self.current_record()
        if not record or not self.window.current_database:
            self.show_error("Select a review item first.")
            return
        ignored = not record.ignored
        state = "ignored" if ignored else "open"
        try:
            update_review(self.window.current_database, record.id, ignored=ignored, review_state=state)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.reload_records()
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.library_page.reload_records()

    def current_record(self):
        if self.selected_record_id is None:
            return None
        for record in self.records:
            if record.id == self.selected_record_id:
                return record
        return None

    def review_reason(self, record) -> str:
        reasons = []
        if record.ignored:
            reasons.append("ignored")
        if record.review_state != "reviewed":
            reasons.append(record.review_state)
        if not record.person_name:
            reasons.append("unassigned")
        if record.duplicate_count > 1:
            reasons.append("duplicate")
        if record.quality_score < 0.45:
            reasons.append("low quality")
        return ", ".join(dict.fromkeys(reasons)) or "review"

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)

    def reset_filters(self) -> None:
        self.filter_text.clear()
        self.limit.setValue(500)
        self.reload_records()


class ClustersPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.clusters: list[FaceCluster] = []
        self.current_cluster: FaceCluster | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Clusters")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("Similarity Groups")
        grid = QGridLayout(controls)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 0.95)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(3)
        self.threshold.setValue(0.55)
        self.min_size = QSpinBox()
        self.min_size.setRange(2, 50)
        self.min_size.setValue(2)
        self.max_faces = QSpinBox()
        self.max_faces.setRange(100, 100000)
        self.max_faces.setValue(5000)
        self.min_quality = QDoubleSpinBox()
        self.min_quality.setRange(0.0, 1.0)
        self.min_quality.setSingleStep(0.05)
        self.min_quality.setDecimals(3)
        self.min_quality.setValue(0.0)
        self.unassigned_only = QCheckBox("Unassigned only")
        self.include_ignored = QCheckBox("Include ignored")
        build_btn = QPushButton("Build Clusters")
        build_btn.clicked.connect(self.build_clusters)
        grid.addWidget(QLabel("Database"), 0, 0)
        grid.addWidget(self.db_path, 0, 1, 1, 5)
        grid.addWidget(build_btn, 0, 6)
        grid.addWidget(QLabel("Threshold"), 1, 0)
        grid.addWidget(self.threshold, 1, 1)
        grid.addWidget(QLabel("Min size"), 1, 2)
        grid.addWidget(self.min_size, 1, 3)
        grid.addWidget(QLabel("Max faces"), 1, 4)
        grid.addWidget(self.max_faces, 1, 5)
        grid.addWidget(QLabel("Min quality"), 2, 0)
        grid.addWidget(self.min_quality, 2, 1)
        grid.addWidget(self.unassigned_only, 2, 2, 1, 2)
        grid.addWidget(self.include_ignored, 2, 4, 1, 2)
        layout.addWidget(controls)

        body = QHBoxLayout()
        self.cluster_table = QTableWidget(0, 6)
        self.cluster_table.setHorizontalHeaderLabels(["Cluster", "Faces", "Mean", "Max", "Avg Quality", "Known People"])
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.cluster_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.cluster_table.itemSelectionChanged.connect(self.show_selected_cluster)
        self.cluster_table.setColumnWidth(0, 70)
        self.cluster_table.setColumnWidth(1, 70)
        self.cluster_table.setColumnWidth(2, 80)
        self.cluster_table.setColumnWidth(3, 80)
        self.cluster_table.setColumnWidth(4, 95)
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.cluster_table, 1)

        self.member_table = QTableWidget(0, 6)
        self.member_table.setHorizontalHeaderLabels(["ID", "Name", "Person", "Tags", "Quality", "Review"])
        self.member_table.verticalHeader().setVisible(False)
        self.member_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.member_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.member_table.itemSelectionChanged.connect(self.show_selected_member)
        self.member_table.setColumnWidth(0, 70)
        self.member_table.setColumnWidth(1, 170)
        self.member_table.setColumnWidth(2, 130)
        self.member_table.setColumnWidth(3, 150)
        self.member_table.setColumnWidth(4, 80)
        self.member_table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(self.member_table, 1)

        right = QVBoxLayout()
        self.preview = make_preview_label("Select a cluster")
        self.preview_focus = make_focus_button(self.preview)
        assign_box = QGroupBox("Batch Assign")
        form = QFormLayout(assign_box)
        self.person_edit = QLineEdit()
        self.tags_edit = QLineEdit()
        self.mark_reviewed = QCheckBox("Mark reviewed")
        self.mark_reviewed.setChecked(True)
        assign_btn = QPushButton("Assign Cluster")
        assign_btn.clicked.connect(self.assign_cluster)
        form.addRow("Person", self.person_edit)
        form.addRow("Tags", self.tags_edit)
        form.addRow("", self.mark_reviewed)
        form.addRow("", assign_btn)
        self.summary = QLabel("No clusters built")
        right.addWidget(make_focus_overlay(self.preview, self.preview_focus))
        right.addWidget(assign_box)
        right.addWidget(self.summary)
        right.addStretch()
        body.addLayout(right, 1)
        layout.addLayout(body, 1)

    def build_clusters(self) -> None:
        if not self.window.current_database:
            self.show_error("Open or create a database first.")
            return
        self.run_task(
            "Building face clusters...",
            build_face_clusters,
            self.on_clusters_ready,
            self.window.current_database,
            threshold=self.threshold.value(),
            min_cluster_size=self.min_size.value(),
            include_ignored=self.include_ignored.isChecked(),
            unassigned_only=self.unassigned_only.isChecked(),
            min_quality=self.min_quality.value(),
            max_faces=self.max_faces.value(),
        )

    def on_clusters_ready(self, result: object) -> None:
        if not isinstance(result, tuple):
            return
        clusters, _ = result
        self.clusters = list(clusters)
        self.current_cluster = None
        self.cluster_table.setRowCount(len(self.clusters))
        for row, cluster in enumerate(self.clusters):
            self.cluster_table.setItem(row, 0, table_item(cluster.cluster_id))
            self.cluster_table.setItem(row, 1, table_item(cluster.size))
            self.cluster_table.setItem(row, 2, table_item(f"{cluster.mean_similarity:.4f}"))
            self.cluster_table.setItem(row, 3, table_item(f"{cluster.max_similarity:.4f}"))
            self.cluster_table.setItem(row, 4, table_item(f"{cluster.average_quality:.3f}"))
            self.cluster_table.setItem(row, 5, table_item(", ".join(cluster.existing_people)))
        self.member_table.setRowCount(0)
        set_image(self.preview, None, "Select a cluster")
        self.summary.setText(f"{len(self.clusters)} cluster(s) above threshold {self.threshold.value():.3f}")
        if self.clusters:
            self.cluster_table.selectRow(0)
        self.window.set_status(self.summary.text())

    def show_selected_cluster(self) -> None:
        selected = self.cluster_table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self.clusters):
            return
        cluster = self.clusters[row]
        self.current_cluster = cluster
        self.member_table.setRowCount(cluster.size)
        for member_row, record in enumerate(cluster.records):
            self.member_table.setItem(member_row, 0, table_item(record.id))
            self.member_table.setItem(member_row, 1, table_item(record.file_name))
            self.member_table.setItem(member_row, 2, table_item(record.person_name))
            self.member_table.setItem(member_row, 3, table_item(record.tag_text))
            self.member_table.setItem(member_row, 4, table_item(f"{record.quality_score:.3f}"))
            self.member_table.setItem(member_row, 5, table_item(record.review_state))
        self.person_edit.setText(cluster.suggested_name)
        self.tags_edit.setText("cluster-suggested")
        self.show_member_preview(cluster.representative_id)
        if cluster.size:
            self.member_table.selectRow(0)

    def show_selected_member(self) -> None:
        selected = self.member_table.selectionModel().selectedRows()
        if not selected or not self.current_cluster:
            return
        row = selected[0].row()
        if row >= len(self.current_cluster.records):
            return
        self.show_member_preview(self.current_cluster.records[row].id)

    def show_member_preview(self, face_id: int) -> None:
        if not self.window.current_database:
            return
        record = None
        if self.current_cluster:
            record = next((item for item in self.current_cluster.records if item.id == face_id), None)
        if record:
            set_record_preview(self.preview, record, self.window.current_database)
            return
        try:
            preview = load_preview(self.window.current_database, face_id)
        except Exception:
            preview = None
        set_image(self.preview, preview_png_to_pil(preview) if preview else None, "No preview")

    def assign_cluster(self) -> None:
        if not self.window.current_database or not self.current_cluster:
            self.show_error("Build and select a cluster first.")
            return
        person_name = self.person_edit.text().strip()
        if not person_name:
            self.show_error("Enter a person name first.")
            return
        try:
            count = assign_faces_to_person(
                self.window.current_database,
                self.current_cluster.face_ids,
                person_name,
                tag_text=self.tags_edit.text(),
                mark_reviewed=self.mark_reviewed.isChecked(),
            )
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.window.set_status(f"Assigned {count} face(s) to {person_name}.")
        self.window.overview_page.reload_summary()
        self.window.people_page.reload_people()
        self.window.library_page.reload_records()
        self.window.review_page.reload_records()
        self.window.search_page.refresh_filter_options()
        self.build_clusters()

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)


class ComparePage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.path_a = ""
        self.path_b = ""
        self.faces_a: list[AnalyzedFace] = []
        self.faces_b: list[AnalyzedFace] = []
        self.selected_a = 0
        self.selected_b = 0
        self._updating_compare_lists = False
        self.compare_generations = {"a": 0, "b": 0}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Compare")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("Images")
        form = QGridLayout(controls)
        self.edit_a = QLineEdit()
        self.edit_b = QLineEdit()
        self.edit_a.setReadOnly(True)
        self.edit_b.setReadOnly(True)
        btn_a = QPushButton("Image A")
        btn_b = QPushButton("Image B")
        compare_btn = QPushButton("Compare")
        btn_a.clicked.connect(lambda: self.select_image("a"))
        btn_b.clicked.connect(lambda: self.select_image("b"))
        compare_btn.clicked.connect(self.compare)
        form.addWidget(QLabel("A"), 0, 0)
        form.addWidget(self.edit_a, 0, 1)
        form.addWidget(btn_a, 0, 2)
        form.addWidget(QLabel("B"), 1, 0)
        form.addWidget(self.edit_b, 1, 1)
        form.addWidget(btn_b, 1, 2)
        form.addWidget(compare_btn, 0, 3, 2, 1)
        layout.addWidget(controls)

        previews = QHBoxLayout()
        panel_a = QVBoxLayout()
        panel_b = QVBoxLayout()
        self.preview_a = make_face_preview_label("Image A")
        self.preview_b = make_face_preview_label("Image B")
        self.preview_a.face_selected.connect(lambda index: self.select_face("a", index))
        self.preview_b.face_selected.connect(lambda index: self.select_face("b", index))
        self.focus_a = make_focus_button(self.preview_a)
        self.focus_b = make_focus_button(self.preview_b)
        self.face_list_a = QListWidget()
        self.face_list_b = QListWidget()
        self.face_list_a.setMaximumHeight(110)
        self.face_list_b.setMaximumHeight(110)
        self.face_list_a.itemSelectionChanged.connect(lambda: self.on_face_list_changed("a"))
        self.face_list_b.itemSelectionChanged.connect(lambda: self.on_face_list_changed("b"))
        panel_a.addWidget(make_focus_overlay(self.preview_a, self.focus_a))
        panel_a.addWidget(self.face_list_a)
        panel_b.addWidget(make_focus_overlay(self.preview_b, self.focus_b))
        panel_b.addWidget(self.face_list_b)
        previews.addLayout(panel_a)
        previews.addLayout(panel_b)
        layout.addLayout(previews, 1)

        self.result = QLabel("Cosine: --    Similarity: --")
        self.result.setObjectName("Metric")
        self.result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result)

    def select_image(self, slot: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", IMAGE_FILTER)
        if not file_path:
            return
        self.compare_generations[slot] += 1
        generation = self.compare_generations[slot]
        try:
            preview = plain_preview_from_path(file_path)
        except Exception as exc:
            self.show_error(str(exc))
            return
        if slot == "a":
            self.path_a = file_path
            self.edit_a.setText(file_path)
            self.faces_a = []
            self.face_list_a.clear()
            self.preview_a.set_plain_image(preview)
            self.selected_a = 0
        else:
            self.path_b = file_path
            self.edit_b.setText(file_path)
            self.faces_b = []
            self.face_list_b.clear()
            self.preview_b.set_plain_image(preview)
            self.selected_b = 0
        self.result.setText("Cosine: --    Similarity: --")

        def task() -> dict[str, object]:
            return {
                "slot": slot,
                "generation": generation,
                "path": file_path,
                "faces": analyze_faces(file_path),
            }

        self.run_task(
            f"Detecting faces in image {slot.upper()}...",
            task,
            self.on_compare_faces_ready,
        )

    def on_compare_faces_ready(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        slot = str(result.get("slot", ""))
        generation = int(result.get("generation", 0))
        if slot not in self.compare_generations or generation != self.compare_generations[slot]:
            return
        faces = list(result.get("faces", []))
        if slot == "a":
            self.faces_a = faces
            self.populate_compare_faces("a", faces)
            self.select_face("a", 0)
        else:
            self.faces_b = faces
            self.populate_compare_faces("b", faces)
            self.select_face("b", 0)
        self.window.set_status(f"Image {slot.upper()}: detected {len(faces)} face(s).")

    def populate_compare_faces(self, slot: str, faces: list[AnalyzedFace]) -> None:
        face_list = self.face_list_a if slot == "a" else self.face_list_b
        self._updating_compare_lists = True
        face_list.clear()
        for index, face in enumerate(faces, start=1):
            face_list.addItem(f"Face {index}: det {face.det_score:.3f}, quality {face.quality_score:.3f}")
        self._updating_compare_lists = False

    def on_face_list_changed(self, slot: str) -> None:
        if self._updating_compare_lists:
            return
        face_list = self.face_list_a if slot == "a" else self.face_list_b
        row = face_list.currentRow()
        if row >= 0:
            self.select_face(slot, row)

    def select_face(self, slot: str, index: int) -> None:
        faces = self.faces_a if slot == "a" else self.faces_b
        if not faces:
            return
        selected = max(0, min(int(index), len(faces) - 1))
        if slot == "a":
            self.selected_a = selected
            self.preview_a.set_faces(self.faces_a, selected)
            face_list = self.face_list_a
        else:
            self.selected_b = selected
            self.preview_b.set_faces(self.faces_b, selected)
            face_list = self.face_list_b
        if face_list.currentRow() != selected:
            self._updating_compare_lists = True
            face_list.setCurrentRow(selected)
            self._updating_compare_lists = False

    def compare(self) -> None:
        if not self.faces_a or not self.faces_b:
            self.show_error("Select both images and wait for face detection first.")
            return
        face_a = self.faces_a[self.selected_a]
        face_b = self.faces_b[self.selected_b]
        cosine = cosine_similarity(face_a.embedding, face_b.embedding)
        self.preview_a.set_faces(self.faces_a, self.selected_a)
        self.preview_b.set_faces(self.faces_b, self.selected_b)
        self.result.setText(
            f"Cosine: {cosine:.4f}    Similarity: {similarity_percent(cosine):.2f}%    "
            f"Quality A/B: {face_a.quality_score:.3f}/{face_b.quality_score:.3f}"
        )
        self.window.set_status("Comparison complete.")


class CameraPage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        self.capture: cv2.VideoCapture | None = None
        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(33)
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.processing_frame = False
        self.last_process_at = 0.0
        self.latest_faces: list[AnalyzedFace] = []
        self.latest_matched_face_indexes: set[int] = set()
        self.latest_faces_at = 0.0
        self.latest_database = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Camera")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        controls = QGroupBox("Camera")
        grid = QGridLayout(controls)
        self.db_path = QLineEdit()
        self.db_path.setReadOnly(True)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        self.camera_index.setValue(0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1.0, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(3)
        self.threshold.setValue(0.35)
        self.interval_ms = QSpinBox()
        self.interval_ms.setRange(100, 5000)
        self.interval_ms.setSingleStep(50)
        self.interval_ms.setValue(300)
        self.process_size = QSpinBox()
        self.process_size.setRange(320, 1920)
        self.process_size.setSingleStep(80)
        self.process_size.setValue(640)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 10)
        self.top_k.setValue(3)
        open_db = QPushButton("Open Database")
        use_current = QPushButton("Use Library DB")
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setEnabled(False)
        open_db.clicked.connect(self.open_database)
        use_current.clicked.connect(self.use_current_database)
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        grid.addWidget(QLabel("Database"), 0, 0)
        grid.addWidget(self.db_path, 0, 1, 1, 5)
        grid.addWidget(open_db, 0, 6)
        grid.addWidget(use_current, 0, 7)
        grid.addWidget(QLabel("Camera"), 1, 0)
        grid.addWidget(self.camera_index, 1, 1)
        grid.addWidget(QLabel("Threshold"), 1, 2)
        grid.addWidget(self.threshold, 1, 3)
        grid.addWidget(QLabel("Interval ms"), 1, 4)
        grid.addWidget(self.interval_ms, 1, 5)
        grid.addWidget(QLabel("Top K"), 1, 6)
        grid.addWidget(self.top_k, 1, 7)
        grid.addWidget(QLabel("Process size"), 2, 0)
        grid.addWidget(self.process_size, 2, 1)
        grid.addWidget(self.start_btn, 2, 2, 1, 2)
        grid.addWidget(self.stop_btn, 2, 4, 1, 2)
        layout.addWidget(controls)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.camera_preview = make_preview_label("Camera")
        self.camera_preview.setMinimumSize(360, 260)
        body.addWidget(self.camera_preview, 2)

        right_panel = QWidget()
        right_panel.setMinimumWidth(330)
        right_panel.setMaximumWidth(460)
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)
        self.match_preview = make_preview_label("Match")
        self.match_focus = make_focus_button(self.match_preview)
        self.match_status = QLabel("No camera result")
        self.match_status.setWordWrap(True)
        self.match_table = QTableWidget(0, 7)
        self.match_table.setHorizontalHeaderLabels(
            ["Face", "ID", "Name", "Person", "Cosine", "Similarity", "Quality"]
        )
        self.match_table.verticalHeader().setVisible(False)
        self.match_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.match_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.match_table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(make_focus_overlay(self.match_preview, self.match_focus), 1)
        right.addWidget(self.match_status)
        right.addWidget(self.match_table, 1)
        body.addWidget(right_panel, 0)
        layout.addLayout(body, 1)

    def open_database(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open database", "", DB_FILTER)
        if not file_path:
            return
        if file_path.lower().endswith(".dtb"):
            self.offer_legacy_conversion(file_path)
            return
        self.window.set_current_database(file_path)
        self.db_path.setText(file_path)

    def use_current_database(self) -> None:
        self.db_path.setText(self.window.current_database)

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database)

    def start_camera(self) -> None:
        if self.capture is not None:
            return
        index = int(self.camera_index.value())
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            self.show_error(f"Could not open camera {index}.")
            return
        self.capture = capture
        self.latest_faces = []
        self.latest_matched_face_indexes = set()
        self.processing_frame = False
        self.last_process_at = 0.0
        self.camera_timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_index.setEnabled(False)
        self.window.set_status(f"Camera {index} started.")

    def stop_camera(self) -> None:
        self.camera_timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.processing_frame = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_index.setEnabled(True)
        set_image(self.camera_preview, None, "Camera stopped")
        self.window.set_status("Camera stopped.")

    def update_camera_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.window.set_status("Camera frame read failed.")
            return

        now = time.monotonic()
        show_boxes = self.latest_faces and now - self.latest_faces_at <= 1.0
        if show_boxes:
            image = draw_camera_faces(frame, self.latest_faces, self.latest_matched_face_indexes)
        else:
            image = bgr_frame_to_pil(frame)
        self.camera_preview.set_preview_image(image, "Camera")

        database_path = self.db_path.text() or self.window.current_database
        if not database_path or self.processing_frame:
            return
        if now - self.last_process_at < self.interval_ms.value() / 1000.0:
            return
        self.last_process_at = now
        self.processing_frame = True
        self.latest_database = database_path
        threshold = self.threshold.value()
        top_k = self.top_k.value()
        process_size = self.process_size.value()
        frame_for_processing = frame.copy()

        def task() -> dict[str, object]:
            recognition_frame, scale_x, scale_y = resize_camera_frame_for_recognition(
                frame_for_processing,
                process_size,
            )
            recognized_faces = get_engine(prefer_gpu=True).extract_faces_from_bgr(recognition_frame, "camera")
            faces = scale_camera_faces_to_frame(recognized_faces, frame_for_processing, scale_x, scale_y)
            rows: list[dict[str, object]] = []
            for face_index, face in enumerate(faces):
                hits, _ = search_database(
                    database_path,
                    face.embedding,
                    top_k=top_k,
                    threshold=threshold,
                    include_ignored=False,
                )
                for rank, hit in enumerate(hits, start=1):
                    rows.append({"face_index": face_index, "rank": rank, "hit": hit})
            return {"database": database_path, "frame": frame_for_processing, "faces": faces, "rows": rows}

        self.window.run_task(
            "Recognizing camera frame...",
            task,
            self.on_camera_result,
            on_error=self.on_camera_error,
        )

    def on_camera_error(self, message: str) -> None:
        self.processing_frame = False
        self.window.set_status(f"Camera recognition error: {message}")

    def on_camera_result(self, result: object) -> None:
        self.processing_frame = False
        if not isinstance(result, dict):
            return
        database_path = str(result.get("database", ""))
        if database_path != (self.db_path.text() or self.window.current_database):
            return
        frame = result.get("frame")
        faces = list(result.get("faces", []))
        rows = list(result.get("rows", []))
        self.latest_faces = faces
        self.latest_matched_face_indexes = {int(row["face_index"]) for row in rows if row.get("hit") is not None}
        self.latest_faces_at = time.monotonic()
        if frame is not None:
            self.camera_preview.set_preview_image(
                draw_camera_faces(frame, faces, self.latest_matched_face_indexes),
                "Camera",
            )
        self.update_match_results(database_path, faces, rows)

    def update_match_results(self, database_path: str, faces: list[AnalyzedFace], rows: list[dict[str, object]]) -> None:
        rows.sort(key=lambda row: getattr(row.get("hit"), "cosine", -2.0), reverse=True)
        self.match_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            hit = row.get("hit")
            if not isinstance(hit, SearchHit):
                continue
            self.match_table.setItem(row_index, 0, table_item(int(row["face_index"]) + 1))
            self.match_table.setItem(row_index, 1, table_item(hit.record.id))
            self.match_table.setItem(row_index, 2, table_item(hit.record.file_name))
            self.match_table.setItem(row_index, 3, table_item(hit.record.person_name))
            self.match_table.setItem(row_index, 4, table_item(f"{hit.cosine:.4f}"))
            self.match_table.setItem(row_index, 5, table_item(f"{hit.similarity:.2f}%"))
            self.match_table.setItem(row_index, 6, table_item(f"{hit.record.quality_score:.3f}"))

        if rows:
            best = rows[0]["hit"]
            if isinstance(best, SearchHit):
                set_record_preview(self.match_preview, best.record, database_path)
                self.match_table.selectRow(0)
                self.match_status.setText(
                    f"Face {int(rows[0]['face_index']) + 1}: best match {best.record.file_name}, "
                    f"{best.similarity:.2f}%"
                )
                self.window.set_status(self.match_status.text())
                return

        set_image(self.match_preview, None, "No match")
        if faces:
            self.match_status.setText(f"Detected {len(faces)} face(s), no database match above threshold.")
        else:
            self.match_status.setText("No face detected in the current camera frame.")
        self.window.set_status(self.match_status.text())


class RuntimePage(StudioPage):
    def __init__(self, window: "FscStudioWindow") -> None:
        super().__init__(window)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Runtime")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        box = QGroupBox("Engine")
        form = QFormLayout(box)
        self.status = QLabel("Not loaded")
        self.providers = QLabel("--")
        self.model_root = QLabel(str(Path("model") / "insightface"))
        load_btn = QPushButton("Load Runtime")
        load_btn.clicked.connect(self.load_runtime)
        form.addRow("Status", self.status)
        form.addRow("Providers", self.providers)
        form.addRow("Model root", self.model_root)
        form.addRow("", load_btn)
        layout.addWidget(box)

        db_box = QGroupBox("Current Database")
        db_form = QFormLayout(db_box)
        self.db_path = QLabel("--")
        self.db_stats = QLabel("--")
        refresh_db = QPushButton("Refresh Database Stats")
        refresh_db.clicked.connect(self.refresh_database)
        db_form.addRow("Path", self.db_path)
        db_form.addRow("Stats", self.db_stats)
        db_form.addRow("", refresh_db)
        layout.addWidget(db_box)

        maintenance = QGroupBox("Maintenance")
        maintenance_layout = QGridLayout(maintenance)
        integrity_btn = QPushButton("Check Integrity")
        backup_btn = QPushButton("Backup DB")
        checkpoint_btn = QPushButton("Checkpoint WAL")
        vacuum_btn = QPushButton("VACUUM")
        integrity_btn.clicked.connect(self.run_integrity_check)
        backup_btn.clicked.connect(self.run_backup)
        checkpoint_btn.clicked.connect(self.run_checkpoint)
        vacuum_btn.clicked.connect(self.run_vacuum)
        self.maintenance_log = QTextEdit()
        self.maintenance_log.setReadOnly(True)
        self.maintenance_log.setMinimumHeight(110)
        self.maintenance_log.setPlaceholderText("Runtime operation results")
        maintenance_layout.addWidget(integrity_btn, 0, 0)
        maintenance_layout.addWidget(backup_btn, 0, 1)
        maintenance_layout.addWidget(checkpoint_btn, 0, 2)
        maintenance_layout.addWidget(vacuum_btn, 0, 3)
        maintenance_layout.addWidget(self.maintenance_log, 1, 0, 1, 4)
        layout.addWidget(maintenance)

        legacy = QGroupBox("Legacy")
        legacy_layout = QVBoxLayout(legacy)
        legacy_layout.addWidget(QLabel("Legacy .dtb files are not searched directly. Convert or rebuild them as .fscdb."))
        convert_legacy_btn = QPushButton("Convert Legacy DTB")
        convert_legacy_btn.clicked.connect(self.run_legacy_conversion)
        legacy_layout.addWidget(convert_legacy_btn)
        layout.addWidget(legacy)
        layout.addStretch()

    def load_runtime(self) -> None:
        self.run_task("Loading runtime...", lambda: get_engine(prefer_gpu=True).info, self.on_runtime_loaded)

    def on_runtime_loaded(self, result: object) -> None:
        info = result
        self.status.setText(info.status_text)
        self.providers.setText(", ".join(info.providers))
        self.model_root.setText(info.model_root)
        self.window.set_status(info.status_text)

    def refresh_database(self) -> None:
        if not self.window.current_database:
            self.db_path.setText("--")
            self.db_stats.setText("No database loaded")
            return
        try:
            stats = load_database_statistics(self.window.current_database)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.db_path.setText(self.window.current_database)
        self.db_stats.setText(
            "v{format_version} | faces {face_count} | people {people_count} | tags {tag_count} | "
            "review {review_count} | ignored {ignored_count} | avg quality {avg_quality:.3f}".format(**stats)
        )

    def sync_database_path(self) -> None:
        self.db_path.setText(self.window.current_database or "--")
        if self.window.current_database:
            self.refresh_database()

    def current_database_or_error(self) -> str:
        if not self.window.current_database:
            self.show_error("Open or create a database first.")
            return ""
        return self.window.current_database

    def run_integrity_check(self) -> None:
        path = self.current_database_or_error()
        if not path:
            return
        self.run_task("Checking database integrity...", check_database_integrity, self.on_maintenance_result, path)

    def run_backup(self) -> None:
        path = self.current_database_or_error()
        if not path:
            return
        source = Path(path)
        default_output = source.with_name(f"{source.stem}_backup{source.suffix or DEFAULT_EXTENSION}")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save database backup",
            str(default_output),
            f"FSC database (*{DEFAULT_EXTENSION});;All files (*.*)",
        )
        if not output_path:
            return
        self.run_task("Backing up database...", backup_database, self.on_maintenance_result, path, output_path)

    def run_checkpoint(self) -> None:
        path = self.current_database_or_error()
        if not path:
            return
        self.run_task("Checkpointing WAL...", checkpoint_database, self.on_maintenance_result, path)

    def run_vacuum(self) -> None:
        path = self.current_database_or_error()
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "FSC Studio",
            "VACUUM rewrites the database file and may take time on large libraries. Continue?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.run_task("Running VACUUM...", vacuum_database, self.on_maintenance_result, path)

    def run_legacy_conversion(self) -> None:
        source_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select legacy database",
            "",
            "Legacy dlib database (*.dtb);;All files (*.*)",
        )
        if not source_path:
            return
        self.start_legacy_conversion(source_path)

    def start_legacy_conversion(self, source_path: str) -> None:
        default_output = default_legacy_conversion_output_path(source_path)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save converted database",
            default_output,
            f"FSC database (*{DEFAULT_EXTENSION});;All files (*.*)",
        )
        if not output_path:
            return
        self.maintenance_log.append(f"Converting legacy database:\n{source_path}\n{output_path}")
        self.run_task(
            "Converting legacy .dtb...",
            convert_legacy_dtb_to_database,
            self.on_legacy_conversion_result,
            source_path,
            output_path,
            progress_arg=True,
            on_progress=self.on_legacy_conversion_progress,
        )

    def on_legacy_conversion_progress(self, message: str, current: int, total: int) -> None:
        self.window.set_status(f"{message} ({current}/{total})")
        if total <= 50 or current == total or current % 10 == 0 or "skipped" in message:
            self.maintenance_log.append(f"[{current}/{total}] {message}")

    def on_legacy_conversion_result(self, result: object) -> None:
        if not isinstance(result, LegacyConversionSummary):
            self.maintenance_log.append(str(result))
            return
        line = (
            f"Converted legacy DTB: saved {result.faces_saved}, skipped {result.skipped_rows}, "
            f"total {result.rows_total}\n{result.output_path}"
        )
        self.maintenance_log.append(line)
        self.window.set_current_database(result.output_path)

    def on_maintenance_result(self, result: object) -> None:
        if not isinstance(result, MaintenanceResult):
            self.maintenance_log.append(str(result))
            return
        state = "OK" if result.ok else "FAILED"
        line = f"{state} {result.action}: {result.message}"
        if result.output_path:
            line = f"{line}\n{result.output_path}"
        self.maintenance_log.append(line)
        self.refresh_database()
        self.window.set_status(line.replace("\n", " "))


class FscStudioWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_database = ""
        self.thread_pool = QThreadPool.globalInstance()
        self.setWindowTitle("FSC Studio")
        self.resize(1320, 820)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar_panel = QWidget()
        sidebar_panel.setObjectName("SidebarPanel")
        sidebar_panel.setFixedWidth(190)
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.page_names = ["Overview", "Library", "People", "Search", "Camera", "Review", "Clusters", "Compare", "Runtime"]
        for name in self.page_names:
            QListWidgetItem(name, self.sidebar)
        sidebar_layout.addWidget(self.sidebar, 1)
        language_box = QWidget()
        language_box.setObjectName("SidebarPanel")
        language_layout = QVBoxLayout(language_box)
        language_layout.setContentsMargins(10, 8, 10, 12)
        self.language_label = QLabel("Language")
        self.language_label.setObjectName("SidebarLabel")
        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("日本語", "ja")
        self.language_combo.addItem("한국어", "ko")
        self.language_combo.currentIndexChanged.connect(self.apply_language)
        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.language_combo)
        sidebar_layout.addWidget(language_box)
        outer.addWidget(sidebar_panel)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        self.stack = QStackedWidget()
        self.library_page = LibraryPage(self)
        self.overview_page = OverviewPage(self)
        self.people_page = PeoplePage(self)
        self.search_page = SearchPage(self)
        self.camera_page = CameraPage(self)
        self.review_page = ReviewPage(self)
        self.clusters_page = ClustersPage(self)
        self.compare_page = ComparePage(self)
        self.runtime_page = RuntimePage(self)
        for page in [
            self.overview_page,
            self.library_page,
            self.people_page,
            self.search_page,
            self.camera_page,
            self.review_page,
            self.clusters_page,
            self.compare_page,
            self.runtime_page,
        ]:
            self.stack.addWidget(page)
        content.addWidget(self.stack, 1)

        self.status = QLabel("Ready")
        self.status.setObjectName("StatusBar")
        content.addWidget(self.status)
        outer.addLayout(content, 1)

        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)
        self.apply_style()
        self.apply_language()

    def set_current_database(self, path: str) -> None:
        self.current_database = path
        self.overview_page.sync_database_path()
        self.library_page.sync_database_path()
        self.people_page.sync_database_path()
        self.search_page.sync_database_path()
        self.camera_page.sync_database_path()
        self.review_page.sync_database_path()
        self.clusters_page.sync_database_path()
        self.runtime_page.sync_database_path()
        self.set_status(f"Database: {path}")

    def show_page(self, name: str) -> None:
        for row, page_name in enumerate(self.page_names):
            if page_name == name:
                self.sidebar.setCurrentRow(row)
                return

    def start_legacy_conversion(self, source_path: str) -> None:
        self.sidebar.setCurrentRow(self.stack.indexOf(self.runtime_page))
        self.runtime_page.start_legacy_conversion(source_path)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.camera_page.stop_camera()
        super().closeEvent(event)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def run_task(
        self,
        status: str,
        fn: Callable,
        on_result: Callable[[object], None],
        *args,
        progress_arg: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        self.set_status(status)
        worker = Worker(fn, *args, progress_arg=progress_arg, **kwargs)
        worker.signals.result.connect(on_result)
        worker.signals.progress.connect(on_progress or self.on_progress)
        worker.signals.error.connect(on_error or self.on_worker_error)
        self.thread_pool.start(worker)

    def on_progress(self, message: str, current: int, total: int) -> None:
        self.set_status(f"{message} ({current}/{total})")

    def on_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "FSC Studio", message)
        self.set_status(f"Error: {message}")

    def apply_language(self) -> None:
        global CURRENT_LANGUAGE
        language = str(self.language_combo.currentData() or "en")
        CURRENT_LANGUAGE = language
        self.setWindowTitle(tr_text("FSC Studio", language))
        self.language_label.setText(tr_text("Language", language))
        for row, name in enumerate(self.page_names):
            item = self.sidebar.item(row)
            if item:
                item.setText(tr_text(name, language))
        for widget_class in (QLabel, QPushButton, QCheckBox):
            for widget in self.findChildren(widget_class):
                if widget is self.language_label:
                    continue
                text = widget.text()
                if text:
                    widget.setText(tr_text(text, language))
        for group_box in self.findChildren(QGroupBox):
            title = group_box.title()
            if title:
                group_box.setTitle(tr_text(title, language))
        for tabs in self.findChildren(QTabWidget):
            for index in range(tabs.count()):
                tabs.setTabText(index, tr_text(tabs.tabText(index), language))
        for table in self.findChildren(QTableWidget):
            for column in range(table.columnCount()):
                item = table.horizontalHeaderItem(column)
                if item:
                    item.setText(tr_text(item.text(), language))
        for preview in self.findChildren(FocusablePreviewLabel):
            preview.sync_focus_button()

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #f6f7f9;
                color: #1f2933;
                font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
                font-size: 10.5pt;
            }
            #Sidebar {
                background: #20242b;
                color: #eef2f6;
                border: none;
                padding: 14px 8px;
            }
            #SidebarPanel {
                background: #20242b;
            }
            #SidebarLabel {
                background: #20242b;
                color: #cfd8e3;
                font-weight: 650;
            }
            #Sidebar::item {
                min-height: 42px;
                padding-left: 14px;
                border-radius: 6px;
            }
            #Sidebar::item:selected {
                background: #2f7d7e;
                color: white;
            }
            #PageTitle {
                font-size: 22pt;
                font-weight: 650;
                color: #111827;
                padding-bottom: 4px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 8px;
                margin-top: 12px;
                padding: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #2f7d7e;
                font-weight: 650;
            }
            QLineEdit, QTextEdit, QTableWidget, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cfd7e3;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #2f7d7e;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #cfd7e3;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background: #2f7d7e;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #256b6d;
            }
            QPushButton:pressed {
                background: #1f5c5d;
            }
            #FocusButton {
                background: rgba(255, 255, 255, 218);
                color: #1f2933;
                border: 1px solid #b7c2cf;
                border-radius: 5px;
                padding: 2px 6px;
                font-size: 8.5pt;
                font-weight: 600;
            }
            #FocusButton:hover {
                background: rgba(240, 247, 247, 235);
                border-color: #2f7d7e;
            }
            #FocusButton:disabled {
                background: rgba(255, 255, 255, 150);
                color: #8892a0;
                border-color: #d5dce5;
            }
            #Preview {
                background: #ffffff;
                border: 1px solid #cfd7e3;
                border-radius: 8px;
                color: #6b7280;
            }
            #Metric {
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 8px;
                padding: 16px;
                font-size: 18pt;
                font-weight: 650;
                color: #7a4b00;
            }
            #StatusBar {
                background: #e9edf3;
                border-top: 1px solid #d8dee8;
                padding: 8px 14px;
                color: #374151;
            }
            QHeaderView::section {
                background: #e9edf3;
                border: none;
                border-right: 1px solid #d8dee8;
                padding: 7px;
                font-weight: 650;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = FscStudioWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
