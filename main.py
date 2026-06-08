import sys
import cv2
import numpy as np
import time
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QMenuBar, QMenu, QFileDialog,
                             QSlider, QGroupBox, QListWidget, QPushButton, QMessageBox,
                             QSizePolicy, QTabWidget, QStatusBar, QFrame, QLineEdit,
                             QCheckBox, QScrollArea, QStackedWidget, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import (QImage, QPixmap, QAction, QPainter, QColor, QPen, 
                         QBrush, QPainterPath, QPalette)

from image_processor import ImageProcessor
from image_compression import CompressionSimulator
from ml_module import HumanDetector

class InteractiveCanvas(QWidget):
    mode_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_before = None
        self.image_after = None
        self.pixmap_before = None
        self.pixmap_after = None
        self.view_mode = "split"
        self.split_ratio = 0.5
        self.is_dragging = False
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.grid_brush = self.create_grid_pattern()

    def create_grid_pattern(self):
        pix = QPixmap(20, 20)
        pix.fill(QColor("#18181b"))
        painter = QPainter(pix)
        painter.fillRect(0, 0, 10, 10, QColor("#222226"))
        painter.fillRect(10, 10, 10, 10, QColor("#222226"))
        painter.end()
        return QBrush(pix)

    def setImages(self, img_before, img_after):
        self.image_before = img_before
        self.image_after = img_after
        self.pixmap_before = self.convert_cv_to_pixmap(img_before)
        self.pixmap_after = self.convert_cv_to_pixmap(img_after)
        self.update()

    def setViewMode(self, mode):
        self.view_mode = mode
        self.update()
        self.mode_changed.emit(mode)

    def convert_cv_to_pixmap(self, img):
        if img is None:
            return None
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, self.grid_brush)

        if self.pixmap_after is None:
            painter.setPen(QColor("#71717a"))
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                             "Please import an image using the 'Open New Photo 📂' button to begin")
            return
            
        if self.view_mode == "split" and self.pixmap_before is not None:
            self.draw_split_view(painter, w, h)
        elif self.view_mode == "side_by_side" and self.pixmap_before is not None:
            self.draw_side_by_side_view(painter, w, h)
        else: # "single"
            self.draw_single_view(painter, w, h)

    def get_image_draw_rect(self, pixmap, container_w, container_h):
        pw, ph = pixmap.width(), pixmap.height()
        scale = min(container_w / pw, container_h / ph)
        dw = int(pw * scale)
        dh = int(ph * scale)
        dx = (container_w - dw) // 2
        dy = (container_h - dh) // 2
        return QRect(dx, dy, dw, dh)

    def draw_single_view(self, painter, w, h):
        rect = self.get_image_draw_rect(self.pixmap_after, w, h)
        painter.drawPixmap(rect, self.pixmap_after)
        self.draw_text_overlay(painter, rect.left() + 10, rect.top() + 20, "PROCESSED", "#3b82f6")

    def draw_side_by_side_view(self, painter, w, h):
        half_w = w // 2
        rect_left = self.get_image_draw_rect(self.pixmap_before, half_w, h)
        painter.drawPixmap(rect_left, self.pixmap_before)
        self.draw_text_overlay(painter, rect_left.left() + 10, rect_left.top() + 20, "BEFORE", "#94a3b8")
        rect_right = self.get_image_draw_rect(self.pixmap_after, half_w, h)
        rect_right.translate(half_w, 0)
        painter.drawPixmap(rect_right, self.pixmap_after)
        self.draw_text_overlay(painter, rect_right.left() + 10, rect_right.top() + 20, "AFTER", "#3b82f6")
        painter.setPen(QPen(QColor("#3f3f46"), 1))
        painter.drawLine(half_w, 0, half_w, h)

    def draw_split_view(self, painter, w, h):
        rect = self.get_image_draw_rect(self.pixmap_after, w, h)
        split_abs_x = int(rect.left() + rect.width() * self.split_ratio)
        split_abs_x = max(rect.left(), min(rect.right(), split_abs_x))
        painter.save()
        clip_rect_left = QRect(rect.left(), rect.top(), split_abs_x - rect.left(), rect.height())
        painter.setClipRect(clip_rect_left)
        painter.drawPixmap(rect, self.pixmap_before)
        painter.restore()
        painter.save()
        clip_rect_right = QRect(split_abs_x, rect.top(), rect.right() - split_abs_x, rect.height())
        painter.setClipRect(clip_rect_right)
        painter.drawPixmap(rect, self.pixmap_after)
        painter.restore()
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawLine(split_abs_x, rect.top(), split_abs_x, rect.bottom())
        handle_y = rect.top() + rect.height() // 2
        painter.setBrush(QBrush(QColor("#3b82f6"))) # Indigo/blue handle
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(split_abs_x - 12, handle_y - 12, 24, 24)
        
        # Draw arrows inside handle '< >'
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawLine(split_abs_x - 6, handle_y, split_abs_x - 2, handle_y - 4)
        painter.drawLine(split_abs_x - 6, handle_y, split_abs_x - 2, handle_y + 4)
        painter.drawLine(split_abs_x + 6, handle_y, split_abs_x + 2, handle_y - 4)
        painter.drawLine(split_abs_x + 6, handle_y, split_abs_x + 2, handle_y + 4)
        
        # Overlay labels
        self.draw_text_overlay(painter, rect.left() + 10, rect.top() + 20, "ORIGINAL", "#a1a1aa", transparent=True)
        self.draw_text_overlay(painter, rect.right() - 85, rect.top() + 20, "PROCESSED", "#3b82f6", transparent=True)

    def draw_text_overlay(self, painter, x, y, text, color_str, transparent=False):
        painter.save()
        color = QColor(color_str)
        
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        
        badge_bg = QColor(0, 0, 0, 160) if transparent else QColor("#1f1f23")
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRect(x - 6, y - th + 2, tw + 12, th + 2), 4, 4)
        
        painter.setPen(color)
        painter.drawText(x, y, text)
        painter.restore()

    def get_split_x(self):
        if self.pixmap_after is None: return -1
        rect = self.get_image_draw_rect(self.pixmap_after, self.width(), self.height())
        return int(rect.left() + rect.width() * self.split_ratio)

    def mousePressEvent(self, event):
        if self.view_mode != "split" or self.pixmap_after is None:
            return
            
        split_x = self.get_split_x()
        if abs(event.position().x() - split_x) <= 20: # 20px hit-box
            self.is_dragging = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
            
    def mouseMoveEvent(self, event):
        if self.view_mode != "split" or self.pixmap_after is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
            
        split_x = self.get_split_x()
        
        if self.is_dragging:
            rect = self.get_image_draw_rect(self.pixmap_after, self.width(), self.height())
            new_x = event.position().x()
            ratio = (new_x - rect.left()) / float(rect.width())
            self.split_ratio = max(0.0, min(1.0, ratio))
            self.update()
        else:
            if abs(event.position().x() - split_x) <= 20:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ==============================================================================
# 2. Custom Native Dynamic Histogram Widget
# ==============================================================================
class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.setMinimumHeight(120)
        self.setMaximumHeight(140)

    def setImage(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bg_color = QColor("#141417")
        painter.fillRect(0, 0, w, h, bg_color)

        # Draw subtle grid
        grid_pen = QPen(QColor("#27272a"), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for y in range(h // 4, h, h // 4):
            painter.drawLine(0, y, w, y)
        for x in range(w // 4, w, w // 4):
            painter.drawLine(x, 0, x, h)

        if self.image is None:
            painter.setPen(QColor("#52525b"))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        try:
            hists = []
            colors = []
            if len(self.image.shape) == 3: # BGR
                for i in range(3):
                    hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    hists.append(hist)
                # BGR colors mapping:
                colors = [
                    (QColor(59, 130, 246, 75), QColor(59, 130, 246, 255)),   # Blue
                    (QColor(34, 197, 94, 75), QColor(34, 197, 94, 255)),     # Green
                    (QColor(239, 68, 68, 75), QColor(239, 68, 68, 255))      # Red
                ]
            else: # Grayscale
                hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
                hists.append(hist)
                colors = [
                    (QColor(161, 161, 170, 75), QColor(161, 161, 170, 255))  # Gray
                ]

            for hist_idx, hist in enumerate(hists):
                max_val = np.max(hist)
                if max_val == 0: continue
                
                fill_color, line_color = colors[hist_idx]
                
                # Create fill path
                path = QPainterPath()
                path.moveTo(0, h)
                
                step_x = w / 256.0
                for i in range(256):
                    val = hist[i][0]
                    nh = (val / max_val) * (h - 8)
                    x = i * step_x
                    y = h - nh
                    path.lineTo(x, y)
                path.lineTo(w, h)
                path.closeSubpath()
                
                painter.fillPath(path, QBrush(fill_color))
                
                # Draw outline
                painter.setPen(QPen(line_color, 1.5))
                line_path = QPainterPath()
                for i in range(256):
                    val = hist[i][0]
                    nh = (val / max_val) * (h - 8)
                    x = i * step_x
                    y = h - nh
                    if i == 0:
                        line_path.moveTo(x, y)
                    else:
                        line_path.lineTo(x, y)
                painter.drawPath(line_path)

        except Exception as e:
            print("Error drawing histogram:", e)


# ==============================================================================
# 3. Main Application Class
# ==============================================================================
class MiniPhotoshopApp(QMainWindow):
    model_loaded_signal = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini Photoshop Creative Pro")
        self.setGeometry(100, 100, 1366, 768)
        
        # State variables
        self.original_image = None
        self.current_image = None
        self.preview_image = None
        self.history_images = []  # Stack of image states
        self.history_names = []   # Stack of action names
        
        self.human_detector = HumanDetector()
        self.model_loaded_signal.connect(self.on_model_loaded)
        
        self.initUI()
        self.applyGlobalStyle()
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """Mendaftarkan semua shortcut keyboard aplikasi."""
        from PyQt6.QtGui import QKeySequence, QShortcut
        # Ctrl+Z → Undo
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self.undo_action)
        # Ctrl+O → Buka gambar
        open_sc = QShortcut(QKeySequence("Ctrl+O"), self)
        open_sc.activated.connect(self.openImage)
        # Ctrl+S → Simpan gambar
        save_sc = QShortcut(QKeySequence("Ctrl+S"), self)
        save_sc.activated.connect(self.saveImage)

    def applyGlobalStyle(self):
        # ── Premium Dark Studio Theme ─────────────────────────────────────────────
        qss = """
        /* ── Base ─────────────────────────────────────────── */
        QMainWindow        { background-color: #0a0a0c; }
        QWidget            { font-family: 'Segoe UI', 'Inter', Arial, sans-serif; font-size: 13px; color: #e2e2e6; }

        /* ── Structural Frames ─────────────────────────────── */
        QFrame#top_header  {
            background-color: #111114;
            border-bottom: 1px solid #232328;
        }
        QFrame#right_sidebar {
            background-color: #111114;
            border-left: 1px solid #232328;
        }
        QFrame#nav_bar_frame {
            background-color: #111114;
            border-bottom: 1px solid #232328;
        }
        QWidget#sidebar_inner {
            background-color: transparent;
        }

        /* ── Section Headers (replaces old QGroupBox) ──────── */
        QLabel#section_header {
            color: #52525b;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.5px;
            padding-top: 14px;
            padding-bottom: 4px;
            min-height: 0px;
        }
        QFrame#section_divider {
            background-color: #232328;
            max-height: 1px;
            min-height: 1px;
        }

        /* ── QGroupBox (minimal, borderless) ───────────────── */
        QGroupBox {
            border: none;
            margin-top: 8px;
            padding-top: 6px;
            background-color: transparent;
        }
        QGroupBox::title {
            color: #52525b;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.5px;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 0px;
            top: -4px;
        }

        /* ── Buttons ───────────────────────────────────────── */
        QPushButton {
            background-color: #1e1e24;
            border: 1px solid #2e2e38;
            border-radius: 7px;
            padding: 9px 14px;
            font-weight: 600;
            font-size: 12px;
            color: #c4c4cf;
        }
        QPushButton:hover  { background-color: #28282f; border-color: #3d3d4a; color: #ffffff; }
        QPushButton:pressed { background-color: #141418; }

        QPushButton#primary_btn {
            background-color: #2563eb;
            border: none;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#primary_btn:hover  { background-color: #3b82f6; }
        QPushButton#primary_btn:pressed { background-color: #1d4ed8; }

        QPushButton#hazard_btn {
            background-color: transparent;
            border: 1px solid #3f1515;
            color: #f87171;
        }
        QPushButton#hazard_btn:hover  { background-color: #7f1d1d; border-color: #ef4444; color: #ffffff; }

        /* View Mode Toggle Buttons */
        QPushButton#view_btn {
            background-color: transparent;
            border: 1px solid #2a2a33;
            border-radius: 6px;
            padding: 7px 14px;
            font-size: 12px;
            color: #71717a;
        }
        QPushButton#view_btn:hover { background-color: #1a1a21; color: #a1a1aa; }
        QPushButton#view_btn:checked {
            background-color: #1e293b;
            border-color: #3b82f6;
            color: #60a5fa;
            font-weight: 700;
        }

        /* Sidebar Nav Tabs */
        QPushButton#nav_tab_btn {
            background-color: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0px;
            padding: 10px 6px;
            font-size: 12px;
            font-weight: 600;
            color: #52525b;
        }
        QPushButton#nav_tab_btn:hover  { color: #a1a1aa; border-bottom: 2px solid #3f3f46; }
        QPushButton#nav_tab_btn:checked {
            color: #e2e2e6;
            border-bottom: 2px solid #3b82f6;
        }

        /* ── Sliders ───────────────────────────────────────── */
        QSlider                     { min-height: 28px; }
        QSlider::groove:horizontal  {
            height: 3px;
            background: #27272a;
            border-radius: 2px;
            margin: 0px;
        }
        QSlider::sub-page:horizontal {
            background: #3b82f6;
            border-radius: 2px;
        }
        QSlider::handle:horizontal  {
            background: #ffffff;
            border: 2px solid #3b82f6;
            width: 14px;
            height: 14px;
            margin: -6px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover { background: #bfdbfe; }

        /* ── Labels ────────────────────────────────────────── */
        QLabel { font-size: 12px; color: #a1a1aa; min-height: 18px; }
        QLabel#slider_label {
            color: #d4d4d8;
            font-size: 12px;
            font-weight: 500;
            min-height: 18px;
        }
        QLabel#slider_value {
            color: #52525b;
            font-size: 11px;
            font-weight: 600;
            min-height: 18px;
        }

        /* ── Text Inputs ───────────────────────────────────── */
        QLineEdit {
            background-color: #141418;
            border: 1px solid #2e2e38;
            border-radius: 6px;
            padding: 7px 10px;
            color: #e2e2e6;
            font-size: 12px;
        }
        QLineEdit:focus { border: 1px solid #3b82f6; background-color: #16161d; }

        /* ── Checkboxes ────────────────────────────────────── */
        QCheckBox               { font-size: 12px; color: #a1a1aa; spacing: 9px; }
        QCheckBox:hover         { color: #e2e2e6; }
        QCheckBox::indicator    {
            width: 16px; height: 16px;
            background-color: #141418;
            border: 1.5px solid #3f3f46;
            border-radius: 4px;
        }
        QCheckBox::indicator:hover   { border-color: #3b82f6; }
        QCheckBox::indicator:checked {
            background-color: #3b82f6;
            border-color: #3b82f6;
        }
        QCheckBox::indicator:checked:hover {
            background-color: #60a5fa;
        }

        /* ── List Widget (History) ─────────────────────────── */
        QListWidget {
            background-color: #0f0f13 !important;
            border: 1px solid #1e1e24;
            border-radius: 7px;
            padding: 3px;
            outline: 0;
        }
        QListWidget::item {
            background-color: transparent !important;
            color: #71717a !important;
            padding: 7px 8px;
            border-radius: 5px;
            margin: 1px 0;
            font-size: 11px;
        }
        QListWidget::item:hover    { background-color: #1a1a20 !important; color: #a1a1aa !important; }
        QListWidget::item:selected { background-color: #172554 !important; color: #93c5fd !important; border-left: 2px solid #3b82f6; font-weight: 600; }

        /* ── Scroll Bars ───────────────────────────────────── */
        QScrollArea            { border: none; background-color: transparent; }
        QScrollBar:vertical    { border: none; background: transparent; width: 5px; }
        QScrollBar::handle:vertical { background: #2e2e38; border-radius: 2px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background: #3f3f46; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

        /* ── Status Bar ────────────────────────────────────── */
        QStatusBar {
            background-color: #0a0a0c;
            border-top: 1px solid #1a1a1f;
            color: #52525b;
            font-size: 11px;
            min-height: 26px;
        }
        QStatusBar QLabel { color: #52525b; min-height: 0px; }

        /* ── Menu Bar ──────────────────────────────────────── */
        QMenuBar            { background-color: #111114; border-bottom: 1px solid #232328; padding: 2px 0; }
        QMenuBar::item      { background-color: transparent; padding: 6px 12px; color: #a1a1aa; font-size: 12px; border-radius: 4px; }
        QMenuBar::item:selected { background-color: #1e1e24; color: #e2e2e6; }
        QMenu               { background-color: #17171c; border: 1px solid #2a2a35; padding: 5px; border-radius: 8px; }
        QMenu::item         { padding: 8px 22px 8px 14px; color: #c4c4cf; border-radius: 5px; font-size: 12px; }
        QMenu::item:selected { background-color: #2563eb; color: #ffffff; }
        QMenu::separator    { height: 1px; background-color: #232328; margin: 4px 0; }
        """
        self.setStyleSheet(qss)

    def initUI(self):


        self.createMenuBar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Top Header ──────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("top_header")
        header.setFixedHeight(60)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(10)

        # App name / logo area
        logo_lbl = QLabel("CITRA")
        logo_lbl.setStyleSheet(
            "color: #3b82f6; font-size: 16px; font-weight: 800; letter-spacing: 3px;"
            "margin-right: 8px; min-height: 0px;"
        )
        hl.addWidget(logo_lbl)

        sep_lbl = QLabel("|")
        sep_lbl.setStyleSheet("color: #27272a; font-size: 18px; min-height: 0px;")
        hl.addWidget(sep_lbl)

        self.open_new_btn = QPushButton("  Open Photo")
        self.open_new_btn.setObjectName("primary_btn")
        self.open_new_btn.setFixedHeight(36)
        self.open_new_btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; border: none; border-radius: 7px;"
            "padding: 0 18px; font-size: 12px; font-weight: 700; color: white; }"
            "QPushButton:hover { background-color: #3b82f6; }"
            "QPushButton:pressed { background-color: #1d4ed8; }"
        )
        self.open_new_btn.clicked.connect(self.openImage)
        hl.addWidget(self.open_new_btn)

        save_btn = QPushButton("  Save")
        save_btn.setFixedHeight(36)
        save_btn.setStyleSheet(
            "QPushButton { background-color: #1e1e24; border: 1px solid #2e2e38; border-radius: 7px;"
            "padding: 0 16px; font-size: 12px; font-weight: 600; color: #c4c4cf; }"
            "QPushButton:hover { background-color: #28282f; color: white; }"
        )
        save_btn.clicked.connect(self.saveImage)
        hl.addWidget(save_btn)

        hl.addStretch(1)

        # View mode toggle group (pill style)
        view_group_frame = QFrame()
        view_group_frame.setStyleSheet(
            "QFrame { background-color: #0e0e12; border: 1px solid #232328;"
            "border-radius: 8px; }"
        )
        view_group_layout = QHBoxLayout(view_group_frame)
        view_group_layout.setContentsMargins(4, 4, 4, 4)
        view_group_layout.setSpacing(2)

        self.mode_split_btn = QPushButton("Split")
        self.mode_split_btn.setObjectName("view_btn")
        self.mode_split_btn.setCheckable(True)
        self.mode_split_btn.setChecked(True)
        self.mode_split_btn.setFixedHeight(30)
        self.mode_split_btn.clicked.connect(lambda: self.set_view_mode("split"))
        view_group_layout.addWidget(self.mode_split_btn)

        self.mode_side_btn = QPushButton("Side-by-Side")
        self.mode_side_btn.setObjectName("view_btn")
        self.mode_side_btn.setCheckable(True)
        self.mode_side_btn.setFixedHeight(30)
        self.mode_side_btn.clicked.connect(lambda: self.set_view_mode("side_by_side"))
        view_group_layout.addWidget(self.mode_side_btn)

        self.mode_single_btn = QPushButton("Single")
        self.mode_single_btn.setObjectName("view_btn")
        self.mode_single_btn.setCheckable(True)
        self.mode_single_btn.setFixedHeight(30)
        self.mode_single_btn.clicked.connect(lambda: self.set_view_mode("single"))
        view_group_layout.addWidget(self.mode_single_btn)

        hl.addWidget(view_group_frame)
        hl.addStretch(1)

        self.hold_compare_btn = QPushButton("Hold: View Original")
        self.hold_compare_btn.setFixedHeight(36)
        self.hold_compare_btn.pressed.connect(self.on_hold_pressed)
        self.hold_compare_btn.released.connect(self.on_hold_released)
        hl.addWidget(self.hold_compare_btn)

        root_layout.addWidget(header)

        # ── Workspace ───────────────────────────────────────────────────────────
        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        # Canvas
        self.canvas = InteractiveCanvas()
        workspace_layout.addWidget(self.canvas, 1)

        # ── Right Sidebar ────────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("right_sidebar")
        sidebar.setFixedWidth(300)
        sidebar_root = QVBoxLayout(sidebar)
        sidebar_root.setContentsMargins(0, 0, 0, 0)
        sidebar_root.setSpacing(0)

        # ── Nav Tab Bar ──────────────────────────────────────────────────────────
        nav_bar = QFrame()
        nav_bar.setObjectName("nav_bar_frame")
        nav_bar.setFixedHeight(48)
        nav_bar_layout = QHBoxLayout(nav_bar)
        nav_bar_layout.setContentsMargins(12, 0, 12, 0)
        nav_bar_layout.setSpacing(0)

        self.nav_grp = QButtonGroup(self)
        self.tool_stack = QStackedWidget()

        for idx, label in enumerate(["Adjust", "Filters", "Transform", "AI"]):
            btn = QPushButton(label)
            btn.setObjectName("nav_tab_btn")
            btn.setCheckable(True)
            btn.setFixedHeight(48)
            if idx == 0:
                btn.setChecked(True)
            self.nav_grp.addButton(btn, idx)
            nav_bar_layout.addWidget(btn)

        self.nav_grp.idClicked.connect(self.tool_stack.setCurrentIndex)
        sidebar_root.addWidget(nav_bar)

        # ── Scrollable Tool Pages ────────────────────────────────────────────────
        def make_scroll_page():
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.Shape.NoFrame)
            sa.setStyleSheet("QScrollArea { background: transparent; }")
            inner = QWidget()
            inner.setObjectName("sidebar_inner")
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(16, 8, 16, 16)
            lay.setSpacing(4)
            sa.setWidget(inner)
            return sa, lay

        def add_section(layout, title):
            lbl = QLabel(title.upper())
            lbl.setObjectName("section_header")
            layout.addWidget(lbl)

        def make_slider_row(layout, label_text, slider_obj, value_lbl_attr=None):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setObjectName("slider_label")
            row.addWidget(lbl)
            row.addStretch()
            val_lbl = QLabel("0")
            val_lbl.setObjectName("slider_value")
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val_lbl)
            layout.addLayout(row)
            layout.addWidget(slider_obj)
            # Connect to update value label
            slider_obj.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            val_lbl.setText(str(slider_obj.value()))
            return val_lbl

        # ── PAGE 0: Adjust ───────────────────────────────────────────────────────
        page_adj_sa, adj_lay = make_scroll_page()

        add_section(adj_lay, "Light & Color")

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.preview_adjustments)
        make_slider_row(adj_lay, "Brightness", self.brightness_slider)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.preview_adjustments)
        make_slider_row(adj_lay, "Contrast", self.contrast_slider)

        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self.preview_adjustments)
        make_slider_row(adj_lay, "Saturation", self.saturation_slider)

        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(0)
        self.hue_slider.valueChanged.connect(self.preview_adjustments)
        make_slider_row(adj_lay, "Hue", self.hue_slider)

        add_section(adj_lay, "Binarize")

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.preview_threshold)
        make_slider_row(adj_lay, "Threshold", self.threshold_slider)

        adj_lay.addSpacing(12)

        actions_row = QHBoxLayout()
        self.reset_sliders_btn = QPushButton("Reset")
        self.reset_sliders_btn.setFixedHeight(36)
        self.reset_sliders_btn.clicked.connect(self.reset_sliders)
        actions_row.addWidget(self.reset_sliders_btn)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.setFixedHeight(36)
        self.apply_btn.clicked.connect(self.commit_adjustments)
        actions_row.addWidget(self.apply_btn)
        adj_lay.addLayout(actions_row)
        adj_lay.addStretch()
        self.tool_stack.addWidget(page_adj_sa)

        # ── PAGE 1: Filters ──────────────────────────────────────────────────────
        page_fil_sa, fil_lay = make_scroll_page()
        add_section(fil_lay, "Creative Filters")

        for fname in ["Gaussian Blur", "Median Blur", "Canny Edge", "Grayscale", "Green Seg"]:
            btn = QPushButton(fname)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, n=fname: self.apply_filter(n))
            fil_lay.addWidget(btn)
            fil_lay.addSpacing(3)

        add_section(fil_lay, "Color Channels")
        ch_row = QHBoxLayout()
        for name, col in [("Red", "#ef4444"), ("Green", "#22c55e"), ("Blue", "#3b82f6")]:
            btn = QPushButton(name)
            btn.setFixedHeight(36)
            btn.setStyleSheet(
                f"QPushButton {{ color: {col}; background-color: #141418;"
                f"border: 1px solid {col}33; border-radius: 7px; font-weight: 700; }}"
                f"QPushButton:hover {{ background-color: {col}22; border-color: {col}; }}"
            )
            btn.clicked.connect(lambda checked, n=f"Split {name}": self.apply_filter(n))
            ch_row.addWidget(btn)
        fil_lay.addLayout(ch_row)

        add_section(fil_lay, "Pipeline")

        self.pipe_gblur_chk = QCheckBox("Gaussian Blur")
        self.pipe_mblur_chk = QCheckBox("Median Blur")
        self.pipe_gray_chk  = QCheckBox("Grayscale")
        self.pipe_canny_chk = QCheckBox("Canny Edge")
        self.pipe_gseg_chk  = QCheckBox("Green Seg")
        for chk in [self.pipe_gblur_chk, self.pipe_mblur_chk,
                    self.pipe_gray_chk, self.pipe_canny_chk, self.pipe_gseg_chk]:
            fil_lay.addWidget(chk)

        fil_lay.addSpacing(8)
        self.apply_pipeline_btn = QPushButton("Run Pipeline")
        self.apply_pipeline_btn.setObjectName("primary_btn")
        self.apply_pipeline_btn.setFixedHeight(36)
        self.apply_pipeline_btn.clicked.connect(self.apply_combined_pipeline)
        fil_lay.addWidget(self.apply_pipeline_btn)
        fil_lay.addStretch()
        self.tool_stack.addWidget(page_fil_sa)

        # ── PAGE 2: Transform ────────────────────────────────────────────────────
        page_trn_sa, trn_lay = make_scroll_page()
        add_section(trn_lay, "Geometri")

        for label, cmd in [
            ("Putar 90° Searah Jarum Jam",  "Rotate 90"),
            ("Balik Horizontal",             "Flip Horizontal"),
            ("Balik Vertikal",               "Flip Vertical"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, c=cmd: self.apply_filter(c))
            trn_lay.addWidget(btn)
            trn_lay.addSpacing(3)

        add_section(trn_lay, "Ubah Ukuran")
        dim_row = QHBoxLayout()
        self.resize_w_input = QLineEdit()
        self.resize_w_input.setPlaceholderText("Lebar (px)")
        self.resize_w_input.textEdited.connect(self.on_width_edited)
        dim_row.addWidget(self.resize_w_input)
        x_lbl = QLabel("×")
        x_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        x_lbl.setStyleSheet("color: #52525b; font-size: 14px; min-height: 0;")
        dim_row.addWidget(x_lbl)
        self.resize_h_input = QLineEdit()
        self.resize_h_input.setPlaceholderText("Tinggi (px)")
        self.resize_h_input.textEdited.connect(self.on_height_edited)
        dim_row.addWidget(self.resize_h_input)
        trn_lay.addLayout(dim_row)

        self.aspect_ratio_chk = QCheckBox("Kunci Rasio Aspek")
        self.aspect_ratio_chk.setChecked(True)
        trn_lay.addWidget(self.aspect_ratio_chk)
        trn_lay.addSpacing(4)
        self.apply_resize_btn = QPushButton("Terapkan Ukuran")
        self.apply_resize_btn.setObjectName("primary_btn")
        self.apply_resize_btn.setFixedHeight(36)
        self.apply_resize_btn.clicked.connect(self.apply_resize)
        trn_lay.addWidget(self.apply_resize_btn)

        # ── Crop (Potong Gambar) ─────────────────────────────────────────────────
        add_section(trn_lay, "Potong Gambar (Crop)")

        # Baris X dan Y
        crop_xy_row = QHBoxLayout()
        lbl_x = QLabel("X:")
        lbl_x.setObjectName("slider_label")
        lbl_x.setFixedWidth(18)
        crop_xy_row.addWidget(lbl_x)
        self.crop_x_input = QLineEdit()
        self.crop_x_input.setPlaceholderText("0")
        crop_xy_row.addWidget(self.crop_x_input)
        lbl_y = QLabel("Y:")
        lbl_y.setObjectName("slider_label")
        lbl_y.setFixedWidth(18)
        crop_xy_row.addWidget(lbl_y)
        self.crop_y_input = QLineEdit()
        self.crop_y_input.setPlaceholderText("0")
        crop_xy_row.addWidget(self.crop_y_input)
        trn_lay.addLayout(crop_xy_row)

        # Baris Lebar dan Tinggi
        crop_wh_row = QHBoxLayout()
        lbl_cw = QLabel("L:")
        lbl_cw.setObjectName("slider_label")
        lbl_cw.setFixedWidth(18)
        crop_wh_row.addWidget(lbl_cw)
        self.crop_w_input = QLineEdit()
        self.crop_w_input.setPlaceholderText("Lebar")
        crop_wh_row.addWidget(self.crop_w_input)
        lbl_ch = QLabel("T:")
        lbl_ch.setObjectName("slider_label")
        lbl_ch.setFixedWidth(18)
        crop_wh_row.addWidget(lbl_ch)
        self.crop_h_input = QLineEdit()
        self.crop_h_input.setPlaceholderText("Tinggi")
        crop_wh_row.addWidget(self.crop_h_input)
        trn_lay.addLayout(crop_wh_row)

        trn_lay.addSpacing(4)
        self.apply_crop_btn = QPushButton("Terapkan Crop")
        self.apply_crop_btn.setObjectName("primary_btn")
        self.apply_crop_btn.setFixedHeight(36)
        self.apply_crop_btn.clicked.connect(self.apply_crop)
        trn_lay.addWidget(self.apply_crop_btn)

        trn_lay.addStretch()
        self.tool_stack.addWidget(page_trn_sa)

        # ── PAGE 3: AI ───────────────────────────────────────────────────────────
        page_ai_sa, ai_lay = make_scroll_page()
        add_section(ai_lay, "Machine Learning")

        self.ml_btn = QPushButton("Deteksi Keberadaan Manusia")
        self.ml_btn.setFixedHeight(36)
        self.ml_btn.clicked.connect(self.detect_human)
        ai_lay.addWidget(self.ml_btn)

        # Keterangan singkat tentang model AI
        ml_info = QLabel(
            "Menggunakan MobileNet SSD (OpenCV DNN).\n"
            "Model diunduh otomatis (~15MB) saat pertama kali digunakan."
        )
        ml_info.setWordWrap(True)
        ml_info.setStyleSheet("color: #52525b; font-size: 11px; line-height: 1.4;")
        ai_lay.addWidget(ml_info)

        add_section(ai_lay, "Analitik")
        comp_btn = QPushButton("Simulasi Kompresi Huffman")
        comp_btn.setFixedHeight(36)
        comp_btn.clicked.connect(self.simulate_compression)
        ai_lay.addWidget(comp_btn)

        # Keterangan singkat tentang Huffman
        comp_info = QLabel(
            "Mensimulasikan Huffman Coding untuk menghitung\n"
            "estimasi rasio kompresi dan penghematan ruang."
        )
        comp_info.setWordWrap(True)
        comp_info.setStyleSheet("color: #52525b; font-size: 11px; line-height: 1.4;")
        ai_lay.addWidget(comp_info)

        ai_lay.addStretch()
        self.tool_stack.addWidget(page_ai_sa)

        sidebar_root.addWidget(self.tool_stack, 1)

        # ── Histogram (always visible) ───────────────────────────────────────────
        hist_container = QWidget()
        hist_container.setStyleSheet("background-color: #0d0d10; border-top: 1px solid #1e1e24;")
        hist_vlay = QVBoxLayout(hist_container)
        hist_vlay.setContentsMargins(12, 8, 12, 8)
        hist_vlay.setSpacing(6)
        hist_lbl = QLabel("HISTOGRAM")
        hist_lbl.setObjectName("section_header")
        hist_lbl.setStyleSheet("color: #3f3f46; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; padding: 0; min-height: 0;")
        hist_vlay.addWidget(hist_lbl)
        self.hist_widget = HistogramWidget()
        self.hist_widget.setMinimumHeight(90)
        self.hist_widget.setMaximumHeight(110)
        hist_vlay.addWidget(self.hist_widget)
        sidebar_root.addWidget(hist_container)

        # ── History Panel (always visible) ───────────────────────────────────────
        history_container = QWidget()
        history_container.setStyleSheet("background-color: #0d0d10; border-top: 1px solid #1e1e24;")
        history_vlay = QVBoxLayout(history_container)
        history_vlay.setContentsMargins(12, 8, 12, 10)
        history_vlay.setSpacing(6)

        hist_row = QHBoxLayout()
        history_lbl = QLabel("RIWAYAT PENGEDITAN")
        history_lbl.setStyleSheet("color: #3f3f46; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; min-height: 0;")
        hist_row.addWidget(history_lbl)
        hist_row.addStretch()
        # Tooltip petunjuk penggunaan history
        tip_lbl = QLabel("Klik item untuk kembali ke tahap tersebut")
        tip_lbl.setStyleSheet("color: #3f3f46; font-size: 10px; min-height: 0;")
        hist_row.addWidget(tip_lbl)
        history_vlay.addLayout(hist_row)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(110)
        # Sambungkan klik item history ke fungsi jump-to-state
        self.history_list.itemClicked.connect(self.jump_to_history)
        history_vlay.addWidget(self.history_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setFixedHeight(32)
        self.undo_btn.setToolTip("Batalkan satu langkah terakhir (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_action)
        btn_row.addWidget(self.undo_btn)
        self.reset_all_btn = QPushButton("Reset Semua")
        self.reset_all_btn.setObjectName("hazard_btn")
        self.reset_all_btn.setFixedHeight(32)
        self.reset_all_btn.setToolTip("Kembalikan kanvas ke gambar asli (semua riwayat dihapus)")
        self.reset_all_btn.clicked.connect(self.reset_to_original)
        btn_row.addWidget(self.reset_all_btn)
        history_vlay.addLayout(btn_row)
        sidebar_root.addWidget(history_container)

        workspace_layout.addWidget(sidebar)
        root_layout.addWidget(workspace, 1)

        # ── Status Bar ───────────────────────────────────────────────────────────
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Open a photo to begin.")
        self.res_lbl = QLabel("–")
        self.res_lbl.setStyleSheet("font-size: 11px; color: #3f3f46; margin-right: 16px; min-height: 0;")
        self.statusBar().addPermanentWidget(self.res_lbl)
        self.ai_lbl = QLabel("AI: Not Loaded")
        self.ai_lbl.setStyleSheet("font-size: 11px; color: #3f3f46; margin-right: 8px; min-height: 0;")
        self.statusBar().addPermanentWidget(self.ai_lbl)

    def createMenuBar(self):


        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.openImage)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Image', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.saveImage)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        edit_menu = menubar.addMenu('Edit')
        undo_action = QAction('Undo Step', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        
        reset_action = QAction('Reset Canvas to Original', self)
        reset_action.triggered.connect(self.reset_to_original)
        edit_menu.addAction(reset_action)

    def set_view_mode(self, mode):
        self.mode_split_btn.setChecked(mode == "split")
        self.mode_side_btn.setChecked(mode == "side_by_side")
        self.mode_single_btn.setChecked(mode == "single")
        
        self.canvas.setViewMode(mode)

    def on_hold_pressed(self):
        # Temporarily display original on canvas
        if self.original_image is not None:
            self.canvas.setImages(self.original_image, self.original_image)
            self.statusBar().showMessage("Comparing with Original Image... (Showing Original)")

    def on_hold_released(self):
        # Restore actual comparative view
        if self.original_image is not None:
            active_img = self.preview_image if self.preview_image is not None else self.current_image
            self.canvas.setImages(self.original_image, active_img)
            self.statusBar().showMessage("Preview updated.")

    def on_width_edited(self, text):
        if self.aspect_ratio_chk.isChecked() and self.current_image is not None:
            try:
                new_w = int(text)
                h, w = self.current_image.shape[:2]
                new_h = int(new_w * (h / w))
                self.resize_h_input.blockSignals(True)
                self.resize_h_input.setText(str(new_h))
                self.resize_h_input.blockSignals(False)
            except ValueError:
                pass

    def on_height_edited(self, text):
        if self.aspect_ratio_chk.isChecked() and self.current_image is not None:
            try:
                new_h = int(text)
                h, w = self.current_image.shape[:2]
                new_w = int(new_h * (w / h))
                self.resize_w_input.blockSignals(True)
                self.resize_w_input.setText(str(new_w))
                self.resize_w_input.blockSignals(False)
            except ValueError:
                pass

    def update_image_metadata(self):
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            ch = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
            self.res_lbl.setText(f"Resolution: {w} x {h} px | {ch} Channels")
            self.resize_w_input.setText(str(w))
            self.resize_h_input.setText(str(h))
        else:
            self.res_lbl.setText("Resolution: - px")

    def push_history(self, image, action_name):
        self.history_images.append(image.copy())
        self.history_names.append(action_name)
        self.history_list.addItem(f"#{len(self.history_names)}: {action_name}")
        self.history_list.scrollToBottom()
        
        self.current_image = image.copy()
        self.preview_image = image.copy()
        
        # update canvas & histogram
        self.canvas.setImages(self.original_image, self.current_image)
        self.hist_widget.setImage(self.current_image)
        self.update_image_metadata()

    def undo_action(self):
        """Membatalkan satu langkah pengeditan terakhir (Ctrl+Z)."""
        if len(self.history_images) > 1:
            self.history_images.pop()
            self.history_names.pop()

            # Hapus item terakhir dari daftar riwayat
            item = self.history_list.takeItem(self.history_list.count() - 1)
            del item

            self.current_image = self.history_images[-1].copy()
            self.preview_image = self.current_image.copy()

            self.canvas.setImages(self.original_image, self.current_image)
            self.hist_widget.setImage(self.current_image)
            self.update_image_metadata()

            self.block_slider_signals(True)
            self.reset_slider_values()
            self.block_slider_signals(False)

            self.statusBar().showMessage(
                f"Undo berhasil. Kembali ke: {self.history_names[-1]}"
            )
        else:
            self.statusBar().showMessage("Sudah berada di gambar paling awal. Tidak dapat Undo lebih jauh.")

    def jump_to_history(self, item):
        """Melompat ke state gambar tertentu saat item riwayat diklik."""
        # Cari indeks item yang diklik
        clicked_index = self.history_list.row(item)
        if clicked_index < 0 or clicked_index >= len(self.history_images):
            return

        # Hapus semua state setelah indeks yang dipilih
        self.history_images = self.history_images[:clicked_index + 1]
        self.history_names  = self.history_names[:clicked_index + 1]

        # Perbarui tampilan list: hapus item-item yang lebih baru
        while self.history_list.count() > clicked_index + 1:
            item_to_remove = self.history_list.takeItem(self.history_list.count() - 1)
            del item_to_remove

        # Terapkan state gambar yang dipilih
        self.current_image = self.history_images[-1].copy()
        self.preview_image = self.current_image.copy()

        self.canvas.setImages(self.original_image, self.current_image)
        self.hist_widget.setImage(self.current_image)
        self.update_image_metadata()

        self.block_slider_signals(True)
        self.reset_slider_values()
        self.block_slider_signals(False)

        self.statusBar().showMessage(
            f"Melompat ke riwayat #{clicked_index + 1}: {self.history_names[-1]}"
        )

    def reset_to_original(self):
        """Mengembalikan kanvas ke gambar asli, menghapus seluruh riwayat."""
        if self.original_image is not None:
            reply = QMessageBox.question(
                self,
                "Konfirmasi Reset",
                "Apakah Anda yakin ingin mereset seluruh kanvas ke gambar asli?\n"
                "Semua riwayat pengeditan yang telah dilakukan akan terhapus.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.history_images.clear()
                self.history_names.clear()
                self.history_list.clear()

                self.push_history(self.original_image, "Reset ke Asli")

                self.block_slider_signals(True)
                self.reset_slider_values()
                self.block_slider_signals(False)
                self.statusBar().showMessage("Kanvas berhasil dikembalikan ke gambar asli.")

    def block_slider_signals(self, block):
        self.brightness_slider.blockSignals(block)
        self.contrast_slider.blockSignals(block)
        self.saturation_slider.blockSignals(block)
        self.hue_slider.blockSignals(block)
        self.threshold_slider.blockSignals(block)

    def reset_slider_values(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.hue_slider.setValue(0)
        self.threshold_slider.setValue(127)

    def reset_sliders(self):
        self.block_slider_signals(True)
        self.reset_slider_values()
        self.block_slider_signals(False)
        
        # Reset preview image
        self.preview_image = self.current_image.copy()
        self.canvas.setImages(self.original_image, self.current_image)
        self.hist_widget.setImage(self.current_image)
        self.statusBar().showMessage("Sliders reset.")

    def openImage(self):
        """Membuka file gambar dari dialog pemilihan file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Buka Gambar", "",
            "File Gambar (*.png *.jpg *.jpeg *.bmp);;Semua File (*)"
        )
        if file_name:
            t_start = time.perf_counter()
            img = cv2.imread(file_name)
            t_end = time.perf_counter()

            if img is not None:
                self.original_image = img.copy()

                # Reset riwayat pengeditan
                self.history_images.clear()
                self.history_names.clear()
                self.history_list.clear()

                # Reset slider secara diam-diam (tanpa memicu preview)
                self.block_slider_signals(True)
                self.reset_slider_values()
                self.block_slider_signals(False)

                # Reset pilihan checkbox pipeline
                self.pipe_gblur_chk.setChecked(False)
                self.pipe_mblur_chk.setChecked(False)
                self.pipe_gray_chk.setChecked(False)
                self.pipe_canny_chk.setChecked(False)
                self.pipe_gseg_chk.setChecked(False)

                # Reset input dimensi crop ke ukuran penuh
                h, w = img.shape[:2]
                self.crop_x_input.setText("0")
                self.crop_y_input.setText("0")
                self.crop_w_input.setText(str(w))
                self.crop_h_input.setText(str(h))

                self.push_history(img, f"Dibuka: {os.path.basename(file_name)}")

                size_kb = os.path.getsize(file_name) / 1024.0
                self.statusBar().showMessage(
                    f"Gambar dimuat dalam {(t_end - t_start)*1000:.1f} ms "
                    f"| Ukuran File: {size_kb:.1f} KB"
                )
            else:
                QMessageBox.warning(
                    self, "Gagal Membuka Gambar",
                    f"Tidak dapat membaca file gambar:\n{file_name}\n\n"
                    "Pastikan file tidak rusak dan formatnya didukung (PNG, JPG, BMP)."
                )

    def saveImage(self):
        """Menyimpan gambar aktif ke file dengan dialog pemilihan format."""
        if self.current_image is None:
            QMessageBox.warning(
                self, "Peringatan",
                "Tidak ada gambar aktif untuk disimpan.\n"
                "Silakan buka gambar terlebih dahulu."
            )
            return

        file_filter = "Gambar PNG (*.png);;Gambar JPEG (*.jpg *.jpeg);;Gambar BMP (*.bmp)"
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Simpan Gambar Sebagai", "", file_filter
        )

        if file_name:
            # Tentukan ekstensi berdasarkan format yang dipilih
            ext = ""
            if "PNG" in selected_filter:
                ext = ".png"
            elif "JPEG" in selected_filter:
                ext = ".jpg"
            elif "BMP" in selected_filter:
                ext = ".bmp"

            # Tambahkan ekstensi secara otomatis jika belum ada
            if not any(file_name.lower().endswith(x) for x in [".png", ".jpg", ".jpeg", ".bmp"]):
                file_name += ext

            t_start = time.perf_counter()
            success = cv2.imwrite(file_name, self.current_image)
            t_end = time.perf_counter()

            if success:
                self.statusBar().showMessage(
                    f"Gambar disimpan sebagai '{os.path.basename(file_name)}' "
                    f"dalam {(t_end - t_start)*1000:.1f} ms"
                )
                QMessageBox.information(
                    self, "Berhasil Disimpan",
                    f"Gambar berhasil disimpan ke:\n{file_name}"
                )
            else:
                QMessageBox.critical(
                    self, "Gagal Menyimpan",
                    "Gagal menyimpan gambar.\n"
                    "Periksa izin akses folder tujuan dan coba lagi."
                )

    def preview_adjustments(self):
        if self.current_image is None: return

        b_val = self.brightness_slider.value()
        c_val = self.contrast_slider.value()
        s_val = self.saturation_slider.value()
        h_val = self.hue_slider.value()

        t_start = time.perf_counter()
        # Apply brightness/contrast then hue/saturation
        img_temp = ImageProcessor.apply_brightness_contrast(self.current_image, b_val, c_val)
        self.preview_image = ImageProcessor.apply_hue_saturation_exposure(img_temp, h_val, s_val, 0)
        t_end = time.perf_counter()

        self.canvas.setImages(self.original_image, self.preview_image)
        self.hist_widget.setImage(self.preview_image)
        self.statusBar().showMessage(f"Preview updated in {(t_end - t_start)*1000:.1f} ms")

    def preview_threshold(self):
        if self.current_image is None: return

        t_val = self.threshold_slider.value()

        t_start = time.perf_counter()
        self.preview_image = ImageProcessor.apply_threshold(self.current_image, t_val)
        t_end = time.perf_counter()

        self.canvas.setImages(self.original_image, self.preview_image)
        self.hist_widget.setImage(self.preview_image)
        self.statusBar().showMessage(f"Threshold preview updated in {(t_end - t_start)*1000:.1f} ms")

    def commit_adjustments(self):
        if self.preview_image is not None and self.current_image is not None:
            b_val = self.brightness_slider.value()
            c_val = self.contrast_slider.value()
            s_val = self.saturation_slider.value()
            h_val = self.hue_slider.value()
            t_val = self.threshold_slider.value()
            
            action_name = "Adjustment"
            if t_val != 127:
                action_name = f"Threshold Binarize ({t_val})"
            else:
                details = []
                if b_val != 0: details.append(f"Bri:{b_val}")
                if c_val != 0: details.append(f"Con:{c_val}")
                if s_val != 0: details.append(f"Sat:{s_val}")
                if h_val != 0: details.append(f"Hue:{h_val}")
                if details:
                    action_name = f"Adj ({', '.join(details)})"
                else:
                    action_name = "Sliders Commited"
            
            self.push_history(self.preview_image, action_name)
            
            # Reset sliders silently
            self.block_slider_signals(True)
            self.reset_slider_values()
            self.block_slider_signals(False)
            
            self.statusBar().showMessage("Adjustments committed to edit timeline.")

    def apply_filter(self, filter_name):
        if self.current_image is None: return
        
        t_start = time.perf_counter()
        result_img = None
        
        if filter_name == "Gaussian Blur":
            result_img = ImageProcessor.apply_gaussian_blur(self.current_image)
        elif filter_name == "Median Blur":
            result_img = ImageProcessor.apply_median_blur(self.current_image)
        elif filter_name == "Canny Edge":
            result_img = ImageProcessor.apply_edge_detection_canny(self.current_image)
        elif filter_name == "Grayscale":
            result_img = ImageProcessor.to_grayscale(self.current_image)
        elif filter_name == "Rotate 90":
            result_img = ImageProcessor.rotate_image(self.current_image, -90)
        elif filter_name == "Flip Horizontal":
            result_img = ImageProcessor.flip_image(self.current_image, 1)
        elif filter_name == "Flip Vertical":
            result_img = ImageProcessor.flip_image(self.current_image, 0)
        elif filter_name == "Split Red":
            r, g, b = ImageProcessor.split_rgb(self.current_image)
            result_img = r
        elif filter_name == "Split Green":
            r, g, b = ImageProcessor.split_rgb(self.current_image)
            result_img = g
        elif filter_name == "Split Blue":
            r, g, b = ImageProcessor.split_rgb(self.current_image)
            result_img = b
        elif filter_name == "Green Seg":
            lower = np.array([35, 50, 50])
            upper = np.array([85, 255, 255])
            result_img = ImageProcessor.color_segmentation(self.current_image, lower, upper)
            
        t_end = time.perf_counter()
        
        if result_img is not None:
            self.push_history(result_img, filter_name)
            self.statusBar().showMessage(
                f"Applied filter: {filter_name} in {(t_end - t_start)*1000:.1f} ms"
            )

    # Apply combined filters pipeline checkbox selections
    def apply_combined_pipeline(self):
        if self.current_image is None: return
        
        filters_to_apply = []
        if self.pipe_gblur_chk.isChecked(): filters_to_apply.append("Gaussian Blur")
        if self.pipe_mblur_chk.isChecked(): filters_to_apply.append("Median Blur")
        if self.pipe_gray_chk.isChecked(): filters_to_apply.append("Grayscale")
        if self.pipe_canny_chk.isChecked(): filters_to_apply.append("Canny Edge")
        if self.pipe_gseg_chk.isChecked(): filters_to_apply.append("Green Seg")
        
        if not filters_to_apply:
            QMessageBox.information(self, "Pipeline Kosong", 
                                    "Silakan pilih minimal satu filter pada checkbox pipeline sebelum menerapkan.")
            return
            
        self.statusBar().showMessage("Applying combined multi-filter pipeline...")
        QApplication.processEvents()
        
        t_start = time.perf_counter()
        result_img = self.current_image.copy()
        
        for filter_name in filters_to_apply:
            if filter_name == "Gaussian Blur":
                result_img = ImageProcessor.apply_gaussian_blur(result_img)
            elif filter_name == "Median Blur":
                result_img = ImageProcessor.apply_median_blur(result_img)
            elif filter_name == "Grayscale":
                result_img = ImageProcessor.to_grayscale(result_img)
            elif filter_name == "Canny Edge":
                result_img = ImageProcessor.apply_edge_detection_canny(result_img)
            elif filter_name == "Green Seg":
                lower = np.array([35, 50, 50])
                upper = np.array([85, 255, 255])
                result_img = ImageProcessor.color_segmentation(result_img, lower, upper)
                
        t_end = time.perf_counter()
        
        # Build composite action name
        short_names = []
        for name in filters_to_apply:
            if name == "Gaussian Blur": short_names.append("G-Blur")
            elif name == "Median Blur": short_names.append("M-Blur")
            elif name == "Grayscale": short_names.append("Gray")
            elif name == "Canny Edge": short_names.append("Canny")
            elif name == "Green Seg": short_names.append("Green-Seg")
            
        action_name = f"Pipeline ({'+'.join(short_names)})"
        
        self.push_history(result_img, action_name)
        self.statusBar().showMessage(
            f"Applied pipeline in {(t_end - t_start)*1000:.1f} ms"
        )
        
        # Reset check boxes
        self.pipe_gblur_chk.setChecked(False)
        self.pipe_mblur_chk.setChecked(False)
        self.pipe_gray_chk.setChecked(False)
        self.pipe_canny_chk.setChecked(False)
        self.pipe_gseg_chk.setChecked(False)

    def apply_resize(self):
        """Mengubah ukuran gambar sesuai dimensi yang dimasukkan."""
        if self.current_image is None:
            return
        try:
            w = int(self.resize_w_input.text())
            h = int(self.resize_h_input.text())

            if w <= 0 or h <= 0:
                raise ValueError("Dimensi harus lebih dari nol.")

            t_start = time.perf_counter()
            resized = cv2.resize(self.current_image, (w, h), interpolation=cv2.INTER_LINEAR)
            t_end = time.perf_counter()

            self.push_history(resized, f"Ubah Ukuran ({w}×{h}px)")
            self.statusBar().showMessage(
                f"Gambar diubah ukuran menjadi {w}×{h}px dalam {(t_end - t_start)*1000:.1f} ms"
            )
        except ValueError as e:
            QMessageBox.warning(
                self, "Input Tidak Valid",
                f"Lebar dan tinggi harus berupa bilangan bulat positif.\nDetail: {e}"
            )

    def apply_crop(self):
        """Memotong (crop) gambar sesuai koordinat dan dimensi yang dimasukkan."""
        if self.current_image is None:
            return
        try:
            x = int(self.crop_x_input.text())
            y = int(self.crop_y_input.text())
            cw = int(self.crop_w_input.text())
            ch = int(self.crop_h_input.text())

            img_h, img_w = self.current_image.shape[:2]

            # Validasi: koordinat dan dimensi harus masuk akal
            if x < 0 or y < 0 or cw <= 0 or ch <= 0:
                raise ValueError("Nilai X, Y harus ≥ 0 dan Lebar, Tinggi harus > 0.")
            if x + cw > img_w or y + ch > img_h:
                raise ValueError(
                    f"Area crop ({x}+{cw}={x+cw}, {y}+{ch}={y+ch}) "
                    f"melebihi ukuran gambar ({img_w}×{img_h}px)."
                )

            t_start = time.perf_counter()
            cropped = ImageProcessor.crop_image(
                self.current_image,
                start_y=y, end_y=y + ch,
                start_x=x, end_x=x + cw
            )
            t_end = time.perf_counter()

            if cropped is not None and cropped.size > 0:
                self.push_history(cropped, f"Crop ({x},{y}) {cw}×{ch}px")
                self.statusBar().showMessage(
                    f"Gambar berhasil dipotong menjadi {cw}×{ch}px "
                    f"dalam {(t_end - t_start)*1000:.1f} ms"
                )
                # Perbarui input dimensi crop ke ukuran baru
                self.crop_x_input.setText("0")
                self.crop_y_input.setText("0")
                self.crop_w_input.setText(str(cw))
                self.crop_h_input.setText(str(ch))
            else:
                QMessageBox.warning(
                    self, "Crop Gagal",
                    "Hasil crop kosong. Periksa kembali nilai koordinat yang dimasukkan."
                )
        except ValueError as e:
            QMessageBox.warning(
                self, "Input Tidak Valid",
                f"Pastikan semua nilai berupa bilangan bulat yang valid.\nDetail: {e}"
            )

    def simulate_compression(self):
        """Menjalankan simulasi Huffman Coding dan menampilkan hasil statistik."""
        if self.current_image is None:
            return

        self.statusBar().showMessage("Menjalankan simulasi kompresi Huffman, harap tunggu...")
        QApplication.processEvents()

        t_start = time.perf_counter()
        quantized = CompressionSimulator.quantize(self.current_image, levels=32)
        stats = CompressionSimulator.simulate_huffman(quantized)
        t_end = time.perf_counter()

        if stats:
            duration_ms = (t_end - t_start) * 1000
            msg = (
                "═══ Hasil Simulasi Kompresi Huffman ═══\n\n"
                f"  Ukuran Asli (8-bit tanpa kompresi) : {stats['original_bytes'] / 1024:.2f} KB\n"
                f"  Ukuran Terkompresi (Huffman)        : {stats['compressed_bytes'] / 1024:.2f} KB\n"
                f"  Rasio Kompresi                       : {stats['ratio']:.2f}×\n"
                f"  Penghematan Ruang                    : {stats['space_saving']:.2f}%\n\n"
                f"  Durasi Simulasi                      : {duration_ms:.1f} ms\n\n"
                "Catatan: Gambar yang ditampilkan telah dikuantisasi\n"
                "ke 32 level warna untuk keperluan simulasi ini."
            )
            QMessageBox.information(self, "Hasil Simulasi Kompresi Huffman", msg)
            self.push_history(quantized, "Simulasi Huffman (Kuantisasi 32 Level)")
            self.statusBar().showMessage(
                f"Simulasi Huffman selesai: Rasio {stats['ratio']:.2f}×, "
                f"Hemat {stats['space_saving']:.1f}%"
            )
        else:
            QMessageBox.warning(
                self, "Simulasi Gagal",
                "Tidak dapat menjalankan simulasi kompresi.\n"
                "Pastikan gambar memiliki data piksel yang valid."
            )
            self.statusBar().showMessage("Simulasi kompresi gagal.")

    def detect_human(self):
        """Menjalankan deteksi manusia menggunakan model MobileNet V2 SSD."""
        if self.current_image is None:
            return

        # Model belum dimuat — mulai proses loading asinkron
        if not self.human_detector.is_ready:
            if self.human_detector.is_loading:
                QMessageBox.information(
                    self, "Model Sedang Dimuat",
                    "Model AI masih dalam proses pengunduhan/pemuatan.\n"
                    "Silakan tunggu hingga muncul notifikasi 'Model Siap'."
                )
                return

            QMessageBox.information(
                self, "Memuat Model AI",
                "Model MobileNet V2 SSD akan diunduh dan dimuat (~60MB).\n"
                "Proses ini berjalan di latar belakang.\n"
                "Anda akan mendapat notifikasi saat model siap digunakan."
            )
            self.ai_lbl.setText("Model AI: Memuat... 🟡")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #eab308;")

            def on_loaded(success, err_msg=""):
                self.model_loaded_signal.emit(success, err_msg)

            self.statusBar().showMessage("Mengunduh dan memuat model TensorFlow MobileNetV2...")
            self.human_detector.load_model(callback=on_loaded)
            return

        # Model sudah siap — jalankan inferensi
        self.statusBar().showMessage("Model AI sedang mendeteksi keberadaan manusia...")
        QApplication.processEvents()

        t_start = time.perf_counter()
        result_img, detected = self.human_detector.detect(self.current_image)
        t_end = time.perf_counter()
        duration_ms = (t_end - t_start) * 1000

        if detected:
            self.push_history(result_img, "Deteksi Manusia (AI)")
            self.statusBar().showMessage(
                f"Deteksi AI selesai dalam {duration_ms:.1f} ms — Manusia terdeteksi!"
            )
        else:
            QMessageBox.information(
                self, "Hasil Deteksi AI",
                "Tidak ada manusia yang terdeteksi pada gambar ini.\n\n"
                f"Durasi analisis: {duration_ms:.1f} ms"
            )
            self.statusBar().showMessage(
                f"Deteksi AI selesai dalam {duration_ms:.1f} ms — Tidak ada manusia terdeteksi."
            )

    def on_model_loaded(self, success, err_msg):
        """Callback yang dipanggil saat model AI selesai dimuat (thread-safe via signal)."""
        if success:
            self.ai_lbl.setText("Model AI: Siap 🟢")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #22c55e;")
            QMessageBox.information(
                self, "Model AI Siap",
                "Model AI berhasil dimuat dan siap digunakan!\n"
                "Silakan klik tombol 'Deteksi Keberadaan Manusia' kembali."
            )
            self.statusBar().showMessage("Model AI aktif dan siap digunakan.")
        else:
            self.ai_lbl.setText("Model AI: Error 🔴")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #ef4444;")
            self.statusBar().showMessage("Gagal memuat model AI.")
            QMessageBox.critical(
                self, "Gagal Memuat Model AI",
                f"Model tidak dapat dimuat.\nPenyebab: {err_msg}\n\n"
                "Pastikan koneksi internet aktif untuk mengunduh file model."
            )


# ==============================================================================
# 4. Entry Point
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("CITRA — Aplikasi Pengolahan Citra Digital")
    app.setApplicationVersion("1.0.0")

    # Aktifkan rendering High DPI agar tampilan tajam di layar resolusi tinggi
    app.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True)

    window = MiniPhotoshopApp()
    window.show()
    sys.exit(app.exec())
