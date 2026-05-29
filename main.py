import sys
import cv2
import numpy as np
import time
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QMenuBar, QMenu, QFileDialog,
                             QSlider, QGroupBox, QListWidget, QPushButton, QMessageBox,
                             QSizePolicy, QTabWidget, QStatusBar, QFrame, QLineEdit,
                             QCheckBox, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import (QImage, QPixmap, QAction, QPainter, QColor, QPen, 
                         QBrush, QPainterPath, QPalette)

from image_processor import ImageProcessor
from compression import CompressionSimulator
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
        
    def applyGlobalStyle(self):
        qss = """
        QMainWindow {
            background-color: #0c0c0e;
        }
        
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #f4f4f5;
        }
        
        QFrame#sidebar_frame {
            background-color: #161619;
            border-left: 1px solid #27272a;
        }
        
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            border: 1px solid #27272a;
            border-radius: 8px;
            margin-top: 18px;
            padding-top: 20px;
            background-color: #121214;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            top: 2px;
            padding: 2px 6px;
            color: #3b82f6;
            background-color: #121214;
        }
        
        QTabWidget::pane {
            border: 1px solid #27272a;
            border-radius: 6px;
            background-color: #121214;
            top: -1px;
        }
        
        QTabBar::tab {
            background-color: #1e1e24;
            border: 1px solid #27272a;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: bold;
            color: #a1a1aa;
        }
        
        QTabBar::tab:selected {
            background-color: #121214;
            color: #ffffff;
            border-bottom: 2px solid #3b82f6;
        }
        
        QTabBar::tab:hover {
            background-color: #27272a;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #27272a;
            border: 1px solid #3f3f46;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: bold;
            font-size: 11px;
            color: #f4f4f5;
        }
        
        QPushButton:hover {
            background-color: #3f3f46;
            border-color: #52525b;
        }
        
        QPushButton:pressed {
            background-color: #18181b;
        }
        
        QPushButton#primary_btn {
            background-color: #2563eb;
            border: 1px solid #1d4ed8;
            color: #ffffff;
        }
        
        QPushButton#primary_btn:hover {
            background-color: #3b82f6;
        }
        
        QPushButton#primary_btn:pressed {
            background-color: #1e40af;
        }
        
        QPushButton#hazard_btn {
            background-color: #ef4444;
            border: 1px solid #dc2626;
            color: #ffffff;
        }
        
        QPushButton#hazard_btn:hover {
            background-color: #f87171;
        }
        
        QPushButton#hazard_btn:pressed {
            background-color: #b91c1c;
        }
        
        QSlider::groove:horizontal {
            height: 4px;
            background: #27272a;
            border-radius: 2px;
        }
        
        QSlider::sub-page:horizontal {
            background: #2563eb;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background: #ffffff;
            border: 1.5px solid #2563eb;
            width: 12px;
            height: 12px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 6px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #2563eb;
            border-color: #ffffff;
        }
        
        QLabel {
            font-size: 11px;
            color: #d1d5db;
        }
        
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        
        QLineEdit {
            background-color: #1e1e24;
            border: 1px solid #27272a;
            border-radius: 4px;
            padding: 4px;
            color: #ffffff;
            font-size: 11px;
        }
        
        QLineEdit:focus {
            border: 1px solid #3b82f6;
        }
        
        QCheckBox {
            font-size: 11px;
            color: #d1d5db;
        }
        
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            background-color: #1e1e24;
            border: 1px solid #27272a;
            border-radius: 3px;
        }
        
        QCheckBox::indicator:checked {
            background-color: #2563eb;
            image: url(data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>);
        }
        
        /* Watertight dark list widget timeline styles */
        QListWidget {
            background-color: #121214 !important;
            border: 1px solid #27272a;
            border-radius: 6px;
            padding: 2px;
            color: #f4f4f5 !important;
        }
        
        QListWidget::item {
            background-color: #1a1a1f !important;
            color: #d1d5db !important;
            padding: 6px;
            border-radius: 4px;
            margin-bottom: 2px;
        }
        
        QListWidget::item:hover {
            background-color: #27272a !important;
            color: #ffffff !important;
        }
        
        QListWidget::item:selected {
            background-color: #2563eb !important;
            color: #ffffff !important;
            font-weight: bold;
        }
        
        QMenuBar {
            background-color: #161619;
            border-bottom: 1px solid #27272a;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 12px;
            color: #d1d5db;
            font-size: 11px;
        }
        
        QMenuBar::item:selected {
            background-color: #27272a;
            color: #ffffff;
            border-radius: 4px;
        }
        
        QMenu {
            background-color: #161619;
            border: 1px solid #27272a;
            padding: 4px;
        }
        
        QMenu::item {
            padding: 6px 20px 6px 12px;
            color: #d1d5db;
            border-radius: 4px;
            font-size: 11px;
        }
        
        QMenu::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #27272a;
            margin: 4px 0px;
        }
        
        QStatusBar {
            background-color: #161619;
            border-top: 1px solid #27272a;
            color: #a1a1aa;
            font-size: 11px;
        }
        """
        self.setStyleSheet(qss)
        
    def initUI(self):
        # 1. Menu Bar
        self.createMenuBar()
        
        # 2. Central Widget layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Left Area: Canvas & View Modes Controls ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(12, 12, 12, 12)
        canvas_layout.setSpacing(8)
        
        # View Modes Control Bar (Segmented Look)
        view_bar = QHBoxLayout()
        view_bar.setSpacing(2)
        
        # NEW: Permanent prominent button to load a new photo quickly (replaces having to close/restart)
        self.open_new_btn = QPushButton("Open New Photo 📂")
        self.open_new_btn.setObjectName("primary_btn") # Styled as premium active blue!
        self.open_new_btn.setStyleSheet("padding: 6px 14px; font-size: 11px; font-weight: bold; margin-right: 15px;")
        self.open_new_btn.clicked.connect(self.openImage)
        view_bar.addWidget(self.open_new_btn)
        
        view_lbl = QLabel("Compare Mode:")
        view_lbl.setStyleSheet("font-weight: bold; color: #a1a1aa; margin-right: 5px;")
        view_bar.addWidget(view_lbl)
        
        self.mode_split_btn = QPushButton("Split Compare ↔️")
        self.mode_split_btn.setCheckable(True)
        self.mode_split_btn.setChecked(True)
        self.mode_split_btn.clicked.connect(lambda: self.set_view_mode("split"))
        view_bar.addWidget(self.mode_split_btn)
        
        self.mode_side_btn = QPushButton("Side-by-Side 🗒️")
        self.mode_side_btn.setCheckable(True)
        self.mode_side_btn.clicked.connect(lambda: self.set_view_mode("side_by_side"))
        view_bar.addWidget(self.mode_side_btn)
        
        self.mode_single_btn = QPushButton("Single Result 🖼️")
        self.mode_single_btn.setCheckable(True)
        self.mode_single_btn.clicked.connect(lambda: self.set_view_mode("single"))
        view_bar.addWidget(self.mode_single_btn)
        
        view_bar.addStretch()
        
        # Hold to Compare Button
        self.hold_compare_btn = QPushButton("Press & Hold to view Original 🔍")
        self.hold_compare_btn.setStyleSheet("padding: 6px 12px; background-color: #1e1e24;")
        self.hold_compare_btn.pressed.connect(self.on_hold_pressed)
        self.hold_compare_btn.released.connect(self.on_hold_released)
        view_bar.addWidget(self.hold_compare_btn)
        
        canvas_layout.addLayout(view_bar)
        
        # Custom Canvas Split Comparer
        self.canvas = InteractiveCanvas()
        canvas_layout.addWidget(self.canvas, 1)
        
        main_layout.addWidget(canvas_container, 1)
        
        # --- Right Panel Frame (Sidebar Dashboard) ---
        sidebar = QFrame()
        sidebar.setObjectName("sidebar_frame")
        sidebar.setFixedWidth(330)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)
        
        # Sidebar Heading
        sidebar_hdr = QLabel("TOOL DASHBOARD")
        sidebar_hdr.setStyleSheet("font-weight: 800; font-size: 12px; color: #3b82f6; letter-spacing: 1px;")
        sidebar_layout.addWidget(sidebar_hdr)
        
        # Tabbed Control Workspace
        self.tabs = QTabWidget()
        
        # Tab 1: Adjust Panel
        adjust_widget = QWidget()
        adjust_layout = QVBoxLayout(adjust_widget)
        adjust_layout.setContentsMargins(6, 6, 6, 6)
        adjust_layout.setSpacing(8)
        
        # Group: Colors & Light
        light_grp = QGroupBox("Light & Color Adjustments")
        light_lay = QVBoxLayout(light_grp)
        light_lay.setSpacing(6)
        
        # Brightness Slider
        light_lay.addWidget(QLabel("Brightness"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.preview_adjustments)
        light_lay.addWidget(self.brightness_slider)
        
        # Contrast Slider
        light_lay.addWidget(QLabel("Contrast"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.preview_adjustments)
        light_lay.addWidget(self.contrast_slider)
        
        # Saturation Slider
        light_lay.addWidget(QLabel("Saturation"))
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self.preview_adjustments)
        light_lay.addWidget(self.saturation_slider)
        
        # Hue Slider
        light_lay.addWidget(QLabel("Hue Shift"))
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(0)
        self.hue_slider.valueChanged.connect(self.preview_adjustments)
        light_lay.addWidget(self.hue_slider)
        
        adjust_layout.addWidget(light_grp)
        
        # Group: Thresholding (Binarize)
        thresh_grp = QGroupBox("Thresholding (Binarize)")
        thresh_lay = QVBoxLayout(thresh_grp)
        thresh_lay.setSpacing(6)
        
        thresh_lay.addWidget(QLabel("Threshold Level"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.preview_threshold)
        thresh_lay.addWidget(self.threshold_slider)
        
        adjust_layout.addWidget(thresh_grp)
        
        # Actions Layout
        adj_actions_lay = QHBoxLayout()
        self.reset_sliders_btn = QPushButton("Reset Sliders")
        self.reset_sliders_btn.clicked.connect(self.reset_sliders)
        adj_actions_lay.addWidget(self.reset_sliders_btn)
        
        self.apply_btn = QPushButton("Apply Adjustments")
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.clicked.connect(self.commit_adjustments)
        adj_actions_lay.addWidget(self.apply_btn)
        
        adjust_layout.addLayout(adj_actions_lay)
        adjust_layout.addStretch()
        
        self.tabs.addTab(adjust_widget, "Adjust")
        
        # Tab 2: Filters Panel
        filters_widget = QScrollArea()
        filters_content = QWidget()
        filters_layout = QVBoxLayout(filters_content)
        filters_layout.setContentsMargins(6, 6, 6, 6)
        filters_layout.setSpacing(8)
        
        # Blurs
        blur_grp = QGroupBox("Smoothing & Blurs")
        blur_lay = QVBoxLayout(blur_grp)
        
        gblur_btn = QPushButton("Gaussian Blur (5x5)")
        gblur_btn.clicked.connect(lambda: self.apply_filter("Gaussian Blur"))
        blur_lay.addWidget(gblur_btn)
        
        mblur_btn = QPushButton("Median Blur (5x5)")
        mblur_btn.clicked.connect(lambda: self.apply_filter("Median Blur"))
        blur_lay.addWidget(mblur_btn)
        
        filters_layout.addWidget(blur_grp)
        
        # Feature Extractors & Operations
        feats_grp = QGroupBox("Enhance & Features")
        feats_lay = QVBoxLayout(feats_grp)
        
        canny_btn = QPushButton("Edge Detection (Canny)")
        canny_btn.clicked.connect(lambda: self.apply_filter("Canny Edge"))
        feats_lay.addWidget(canny_btn)
        
        gray_btn = QPushButton("Grayscale Converter")
        gray_btn.clicked.connect(lambda: self.apply_filter("Grayscale"))
        feats_lay.addWidget(gray_btn)
        
        gseg_btn = QPushButton("Green Color Segmentation")
        gseg_btn.clicked.connect(lambda: self.apply_filter("Green Seg"))
        feats_lay.addWidget(gseg_btn)
        
        filters_layout.addWidget(feats_grp)
        
        # RGB Color Extraction
        rgb_grp = QGroupBox("Color Channel Extractor")
        rgb_lay = QHBoxLayout(rgb_grp)
        
        split_r = QPushButton("Red")
        split_r.setStyleSheet("color: #ef4444; font-weight: bold;")
        split_r.clicked.connect(lambda: self.apply_filter("Split Red"))
        rgb_lay.addWidget(split_r)
        
        split_g = QPushButton("Green")
        split_g.setStyleSheet("color: #22c55e; font-weight: bold;")
        split_g.clicked.connect(lambda: self.apply_filter("Split Green"))
        rgb_lay.addWidget(split_g)
        
        split_b = QPushButton("Blue")
        split_b.setStyleSheet("color: #3b82f6; font-weight: bold;")
        split_b.clicked.connect(lambda: self.apply_filter("Split Blue"))
        rgb_lay.addWidget(split_b)
        
        filters_layout.addWidget(rgb_grp)
        
        # Pipeline Combiner Widget Box
        pipe_grp = QGroupBox("Combined Operations Pipeline")
        pipe_lay = QVBoxLayout(pipe_grp)
        pipe_lay.setSpacing(6)
        
        pipe_desc = QLabel("Select multiple filters to apply sequentially:")
        pipe_desc.setStyleSheet("color: #a1a1aa; font-style: italic; margin-bottom: 2px;")
        pipe_lay.addWidget(pipe_desc)
        
        self.pipe_gblur_chk = QCheckBox("Gaussian Blur (5x5)")
        self.pipe_mblur_chk = QCheckBox("Median Blur (5x5)")
        self.pipe_gray_chk = QCheckBox("Grayscale Converter")
        self.pipe_canny_chk = QCheckBox("Canny Edge Detection")
        self.pipe_gseg_chk = QCheckBox("Green Color Segmentation")
        
        pipe_lay.addWidget(self.pipe_gblur_chk)
        pipe_lay.addWidget(self.pipe_mblur_chk)
        pipe_lay.addWidget(self.pipe_gray_chk)
        pipe_lay.addWidget(self.pipe_canny_chk)
        pipe_lay.addWidget(self.pipe_gseg_chk)
        
        self.apply_pipeline_btn = QPushButton("Apply Combined Pipeline ⚡")
        self.apply_pipeline_btn.setObjectName("primary_btn")
        self.apply_pipeline_btn.clicked.connect(self.apply_combined_pipeline)
        pipe_lay.addWidget(self.apply_pipeline_btn)
        
        filters_layout.addWidget(pipe_grp)
        filters_layout.addStretch()
        
        filters_widget.setWidget(filters_content)
        filters_widget.setWidgetResizable(True)
        self.tabs.addTab(filters_widget, "Filters")
        
        # Tab 3: Transform Panel
        transform_widget = QWidget()
        transform_layout = QVBoxLayout(transform_widget)
        transform_layout.setContentsMargins(6, 6, 6, 6)
        transform_layout.setSpacing(10)
        
        # Rotate & Flip Group
        geom_grp = QGroupBox("Geometric Rotation & Flips")
        geom_lay = QVBoxLayout(geom_grp)
        geom_lay.setSpacing(6)
        
        rot_btn = QPushButton("Rotate 90° Clockwise 🔄")
        rot_btn.clicked.connect(lambda: self.apply_filter("Rotate 90"))
        geom_lay.addWidget(rot_btn)
        
        fliph_btn = QPushButton("Flip Horizontal ↔️")
        fliph_btn.clicked.connect(lambda: self.apply_filter("Flip Horizontal"))
        geom_lay.addWidget(fliph_btn)
        
        flipv_btn = QPushButton("Flip Vertical ↕️")
        flipv_btn.clicked.connect(lambda: self.apply_filter("Flip Vertical"))
        geom_lay.addWidget(flipv_btn)
        
        transform_layout.addWidget(geom_grp)
        
        # Resize Custom Panel (Creative Suite Standard)
        resize_grp = QGroupBox("Dynamic Resizing")
        resize_lay = QVBoxLayout(resize_grp)
        resize_lay.setSpacing(8)
        
        dimensions_lay = QHBoxLayout()
        dimensions_lay.addWidget(QLabel("Width (px):"))
        self.resize_w_input = QLineEdit()
        self.resize_w_input.setPlaceholderText("Width")
        self.resize_w_input.textEdited.connect(self.on_width_edited)
        dimensions_lay.addWidget(self.resize_w_input)
        
        dimensions_lay.addWidget(QLabel("Height (px):"))
        self.resize_h_input = QLineEdit()
        self.resize_h_input.setPlaceholderText("Height")
        self.resize_h_input.textEdited.connect(self.on_height_edited)
        dimensions_lay.addWidget(self.resize_h_input)
        
        resize_lay.addLayout(dimensions_lay)
        
        self.aspect_ratio_chk = QCheckBox("Lock Aspect Ratio 🔗")
        self.aspect_ratio_chk.setChecked(True)
        resize_lay.addWidget(self.aspect_ratio_chk)
        
        self.apply_resize_btn = QPushButton("Apply Resize Scale")
        self.apply_resize_btn.setObjectName("primary_btn")
        self.apply_resize_btn.clicked.connect(self.apply_resize)
        resize_lay.addWidget(self.apply_resize_btn)
        
        transform_layout.addWidget(resize_grp)
        transform_layout.addStretch()
        
        self.tabs.addTab(transform_widget, "Transform")
        
        # Tab 4: AI & Compression
        tools_widget = QWidget()
        tools_layout = QVBoxLayout(tools_widget)
        tools_layout.setContentsMargins(6, 6, 6, 6)
        tools_layout.setSpacing(10)
        
        ai_grp = QGroupBox("AI Object Detection")
        ai_lay = QVBoxLayout(ai_grp)
        ai_lay.setSpacing(6)
        
        self.ml_btn = QPushButton("Detect Human Presence (MobileNetV2) 🤖")
        self.ml_btn.clicked.connect(self.detect_human)
        ai_lay.addWidget(self.ml_btn)
        
        tools_layout.addWidget(ai_grp)
        
        comp_grp = QGroupBox("Huffman Compression Analytics")
        comp_lay = QVBoxLayout(comp_grp)
        comp_lay.setSpacing(6)
        
        comp_btn = QPushButton("Huffman Simulation Stats 📊")
        comp_btn.clicked.connect(self.simulate_compression)
        comp_lay.addWidget(comp_btn)
        
        tools_layout.addWidget(comp_grp)
        tools_layout.addStretch()
        
        self.tabs.addTab(tools_widget, "AI / Tools")
        
        sidebar_layout.addWidget(self.tabs, 1)
        
        # Dynamic Histogram Area (Photoshop style - sits in the sidebar!)
        hist_box = QGroupBox("Real-time Color Channels Histogram")
        hist_lay = QVBoxLayout(hist_box)
        hist_lay.setContentsMargins(4, 8, 4, 4)
        
        self.hist_widget = HistogramWidget()
        hist_lay.addWidget(self.hist_widget)
        
        sidebar_layout.addWidget(hist_box)
        
        # History Timeline area
        history_box = QGroupBox("Edit History Timeline")
        history_lay = QVBoxLayout(history_box)
        history_lay.setContentsMargins(6, 8, 6, 6)
        history_lay.setSpacing(6)
        
        self.history_list = QListWidget()
        history_lay.addWidget(self.history_list)
        
        history_btn_lay = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Step ↩️")
        self.undo_btn.clicked.connect(self.undo_action)
        history_btn_lay.addWidget(self.undo_btn)
        
        self.reset_all_btn = QPushButton("Reset Canvas 🗑️")
        self.reset_all_btn.setObjectName("hazard_btn")
        self.reset_all_btn.clicked.connect(self.reset_to_original)
        history_btn_lay.addWidget(self.reset_all_btn)
        
        history_lay.addLayout(history_btn_lay)
        
        sidebar_layout.addWidget(history_box)
        
        main_layout.addWidget(sidebar)
        
        # 3. bottom dynamic status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Welcome to Mini Photoshop Creative Pro. Open an image to start.")
        
        # Status Bar Perm widgets (Resolution & AI state)
        self.res_lbl = QLabel("Resolution: - px")
        self.res_lbl.setStyleSheet("margin-right: 15px; font-weight: bold; color: #a1a1aa;")
        self.statusBar().addPermanentWidget(self.res_lbl)
        
        self.ai_lbl = QLabel("AI Model: Not Loaded ⚪")
        self.ai_lbl.setStyleSheet("font-weight: bold; color: #a1a1aa;")
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
        if len(self.history_images) > 1:
            self.history_images.pop()
            self.history_names.pop()
            
            # Remove last item
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
            
            self.statusBar().showMessage("Undo successful.")
        else:
            self.statusBar().showMessage("Already at the original image state.")

    def reset_to_original(self):
        if self.original_image is not None:
            # Dynamic dialog confirmation pop-up
            reply = QMessageBox.question(
                self, 
                "Konfirmasi Reset",
                "Apakah Anda yakin ingin mereset seluruh kanvas ke gambar asli?\n"
                "Semua riwayat pengeditan yang telah Anda lakukan akan terhapus.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.history_images.clear()
                self.history_names.clear()
                self.history_list.clear()
                
                self.push_history(self.original_image, "Reset Canvas")
                
                self.block_slider_signals(True)
                self.reset_slider_values()
                self.block_slider_signals(False)
                self.statusBar().showMessage("Canvas restored to original state.")

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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            t_start = time.perf_counter()
            img = cv2.imread(file_name)
            t_end = time.perf_counter()
            
            if img is not None:
                self.original_image = img.copy()
                
                # Reset history
                self.history_images.clear()
                self.history_names.clear()
                self.history_list.clear()
                
                # Reset sliders silently
                self.block_slider_signals(True)
                self.reset_slider_values()
                self.block_slider_signals(False)
                
                # Reset pipeline checkboxes
                self.pipe_gblur_chk.setChecked(False)
                self.pipe_mblur_chk.setChecked(False)
                self.pipe_gray_chk.setChecked(False)
                self.pipe_canny_chk.setChecked(False)
                self.pipe_gseg_chk.setChecked(False)
                
                self.push_history(img, f"Opened: {os.path.basename(file_name)}")
                
                # Show status message
                size_kb = os.path.getsize(file_name) / 1024.0
                self.statusBar().showMessage(
                    f"Image loaded in {(t_end - t_start)*1000:.1f} ms | File Size: {size_kb:.1f} KB"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to open image.")

    # REVISED: Enhanced Save Image with explicit format selector (PNG, JPEG, BMP) and automatic extension appender
    def saveImage(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Tidak ada gambar aktif untuk disimpan.")
            return
            
        file_filter = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)"
        file_name, selected_filter = QFileDialog.getSaveFileName(self, "Save Image As", "", file_filter)
        
        if file_name:
            ext = ""
            if "PNG" in selected_filter:
                ext = ".png"
            elif "JPEG" in selected_filter:
                ext = ".jpg"
            elif "BMP" in selected_filter:
                ext = ".bmp"
                
            # If the user did not type the correct extension, append it automatically
            if not file_name.lower().endswith(ext):
                if not any(file_name.lower().endswith(x) for x in [".png", ".jpg", ".jpeg", ".bmp"]):
                    file_name += ext
            
            t_start = time.perf_counter()
            success = cv2.imwrite(file_name, self.current_image)
            t_end = time.perf_counter()
            
            if success:
                self.statusBar().showMessage(
                    f"Image saved successfully as {os.path.basename(file_name)} in {(t_end - t_start)*1000:.1f} ms"
                )
                QMessageBox.information(self, "Success", f"Gambar berhasil disimpan ke:\n{file_name}")
            else:
                QMessageBox.critical(self, "Error", "Gagal menyimpan gambar. Silakan periksa izin file Anda.")

    def preview_adjustments(self):
        if self.current_image is None: return
        
        # Unfocus binarize slider to prevent layout confusion
        self.threshold_slider.setValue(127)
        
        b_val = self.brightness_slider.value()
        c_val = self.contrast_slider.value()
        s_val = self.saturation_slider.value()
        h_val = self.hue_slider.value()
        
        t_start = time.perf_counter()
        
        # Apply standard brightness contrast
        img_temp = ImageProcessor.apply_brightness_contrast(self.current_image, b_val, c_val)
        # Apply HSV Hue Saturation
        self.preview_image = ImageProcessor.apply_hue_saturation_exposure(img_temp, h_val, s_val, 0)
        
        t_end = time.perf_counter()
        
        # Update canvas comparative image & histogram in real-time
        self.canvas.setImages(self.original_image, self.preview_image)
        self.hist_widget.setImage(self.preview_image)
        
        self.statusBar().showMessage(f"Preview updated in {(t_end - t_start)*1000:.1f} ms")

    def preview_threshold(self):
        if self.current_image is None: return
        
        # Reset color adjustment sliders silently
        self.block_slider_signals(True)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.hue_slider.setValue(0)
        self.block_slider_signals(False)
        
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
        if self.current_image is None: return
        try:
            w = int(self.resize_w_input.text())
            h = int(self.resize_h_input.text())
            
            t_start = time.perf_counter()
            resized = cv2.resize(self.current_image, (w, h), interpolation=cv2.INTER_LINEAR)
            t_end = time.perf_counter()
            
            self.push_history(resized, f"Resize ({w}x{h})")
            self.statusBar().showMessage(f"Image resized to {w}x{h} in {(t_end - t_start)*1000:.1f} ms")
        except ValueError:
            QMessageBox.warning(self, "Error", "Lebar dan tinggi harus berupa angka bulat.")

    def simulate_compression(self):
        if self.current_image is None: return
        
        self.statusBar().showMessage("Simulating Huffman compression, please wait...")
        QApplication.processEvents()
        
        t_start = time.perf_counter()
        quantized = CompressionSimulator.quantize(self.current_image, levels=32)
        stats = CompressionSimulator.simulate_huffman(quantized)
        t_end = time.perf_counter()
        
        if stats:
            msg = f"--- Hasil Simulasi Kompresi Huffman ---\n\n"
            msg += f"Ukuran Asli (Uncompressed 8-bit): {stats['original_bytes'] / 1024:.2f} KB\n"
            msg += f"Ukuran Terkompresi (Huffman): {stats['compressed_bytes'] / 1024:.2f} KB\n"
            msg += f"Rasio Kompresi: {stats['ratio']:.2f}x\n"
            msg += f"Penghematan Ruang: {stats['space_saving']:.2f}%\n\n"
            msg += f"Durasi Simulasi: {(t_end - t_start)*1000:.1f} ms"
            
            QMessageBox.information(self, "Hasil Simulasi Kompresi", msg)
            self.push_history(quantized, "Simulasi Huffman (Quantized 32)")
            self.statusBar().showMessage("Huffman compression completed.")
        else:
            QMessageBox.warning(self, "Error", "Gagal mensimulasikan kompresi.")
            self.statusBar().showMessage("Compression simulator failed.")

    def detect_human(self):
        if self.current_image is None: return
        
        if not self.human_detector.is_ready:
            QMessageBox.information(self, "Loading Model", 
                                    "Sedang mengunduh/memuat model MobileNetV2 (~60MB).\n"
                                    "Proses berjalan di latar belakang. Silakan perhatikan notifikasi.")
            self.ai_lbl.setText("AI Model: Loading... 🟡")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #eab308;")
            
            def on_loaded(success, err_msg=""):
                self.model_loaded_signal.emit(success, err_msg)
                    
            self.statusBar().showMessage("Loading TensorFlow MobileNetV2 Model...")
            self.human_detector.load_model(callback=on_loaded)
            return
            
        self.statusBar().showMessage("AI detecting human shapes...")
        QApplication.processEvents()
        
        t_start = time.perf_counter()
        result_img, detected = self.human_detector.detect(self.current_image)
        t_end = time.perf_counter()
        
        if detected:
            self.push_history(result_img, "Human Detection AI")
            self.statusBar().showMessage(f"AI Detection finished in {(t_end - t_start)*1000:.1f} ms (Human Detected!)")
        else:
            QMessageBox.information(self, "Hasil AI", "Tidak ada manusia yang terdeteksi.")
            self.statusBar().showMessage(f"AI Detection finished in {(t_end - t_start)*1000:.1f} ms (None detected)")

    def on_model_loaded(self, success, err_msg):
        if success:
            self.ai_lbl.setText("AI Model: Ready 🟢")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #22c55e;")
            QMessageBox.information(self, "Success", "Model AI berhasil dimuat! Silakan klik tombol Human Detection kembali.")
            self.statusBar().showMessage("AI Model is active and ready.")
        else:
            self.ai_lbl.setText("AI Model: Error 🔴")
            self.ai_lbl.setStyleSheet("font-weight: bold; color: #ef4444;")
            self.statusBar().showMessage("Failed to load AI model.")
            QMessageBox.critical(self, "Error", f"Gagal memuat model.\nAlasan: {err_msg}")


# ==============================================================================
# 4. Entry Point
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Enable High DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True)
    
    window = MiniPhotoshopApp()
    window.show()
    sys.exit(app.exec())
