from cusift import CuSift, ExtractOptions

import colorsys
import math

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout,
    QHBoxLayout, QWidget, QPushButton, QSplitter, QSizePolicy,
    QToolBar, QAction, QGroupBox, QFormLayout, QDoubleSpinBox,
    QSpinBox, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter
from PyQt5.QtCore import Qt, QPointF

import sys


class ImagePanel(QGraphicsView):
    """A panel that displays an image with mouse-wheel zoom and click-drag pan."""

    _ZOOM_IN_FACTOR = 1.15
    _ZOOM_OUT_FACTOR = 1.0 / 1.15

    def __init__(self, placeholder_text: str = "No image loaded", parent=None):
        super().__init__(parent)
        self._placeholder_text = placeholder_text
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom_level = 0

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.darkGray)
        self.setStyleSheet("border: none;")

    def set_image(self, pixmap: QPixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().x(), pixmap.rect().y(),
                                 pixmap.width(), pixmap.height())
        self._zoom_level = 0
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def clear(self):
        self._scene.clear()
        self._pixmap_item = None
        self._zoom_level = 0

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        angle = event.angleDelta().y()
        if angle > 0:
            factor = self._ZOOM_IN_FACTOR
            self._zoom_level += 1
        elif angle < 0:
            factor = self._ZOOM_OUT_FACTOR
            self._zoom_level -= 1
        else:
            return
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._zoom_level == 0:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        """Reset zoom to fit the image in the view."""
        if self._pixmap_item is not None:
            self._zoom_level = 0
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)


class ExtractOptionsPanel(QWidget):
    """Editable panel for all ExtractOptions fields."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        group = QGroupBox("Extract Options")
        form = QFormLayout()

        # -- thresh --
        self.thresh = QDoubleSpinBox()
        self.thresh.setRange(0.0, 100.0)
        self.thresh.setDecimals(2)
        self.thresh.setSingleStep(0.5)
        self.thresh.setValue(3.0)
        self.thresh.setToolTip("DoG contrast threshold (higher = fewer keypoints)")
        form.addRow("Threshold:", self.thresh)

        # -- lowest_scale --
        self.lowest_scale = QDoubleSpinBox()
        self.lowest_scale.setRange(0.0, 1000.0)
        self.lowest_scale.setDecimals(2)
        self.lowest_scale.setSingleStep(0.5)
        self.lowest_scale.setValue(0.0)
        self.lowest_scale.setToolTip("Minimum feature scale in pixels (0 = keep all)")
        form.addRow("Lowest Scale:", self.lowest_scale)

        # -- highest_scale --
        self.highest_scale = QDoubleSpinBox()
        self.highest_scale.setRange(0.0, 32.0)
        self.highest_scale.setDecimals(2)
        self.highest_scale.setSingleStep(2.0)
        self.highest_scale.setValue(32.0)
        self.highest_scale.setToolTip("Maximum feature scale in pixels (large value = no limit)")
        form.addRow("Highest Scale:", self.highest_scale)

        # -- edge_thresh --
        self.edge_thresh = QDoubleSpinBox()
        self.edge_thresh.setRange(1.0, 100.0)
        self.edge_thresh.setDecimals(2)
        self.edge_thresh.setSingleStep(1.0)
        self.edge_thresh.setValue(10.0)
        self.edge_thresh.setToolTip("Edge rejection threshold (ratio of principal curvatures)")
        form.addRow("Edge Threshold:", self.edge_thresh)

        # -- init_blur --
        self.init_blur = QDoubleSpinBox()
        self.init_blur.setRange(0.0, 10.0)
        self.init_blur.setDecimals(2)
        self.init_blur.setSingleStep(0.1)
        self.init_blur.setValue(1.0)
        self.init_blur.setToolTip("Assumed blur sigma of the input image")
        form.addRow("Init Blur:", self.init_blur)

        # -- max_keypoints --
        self.max_keypoints = QSpinBox()
        self.max_keypoints.setRange(1, 131072)
        self.max_keypoints.setSingleStep(1024)
        self.max_keypoints.setValue(32768)
        self.max_keypoints.setToolTip("Maximum number of keypoints returned")
        form.addRow("Max Keypoints:", self.max_keypoints)

        # -- num_octaves --
        self.num_octaves = QSpinBox()
        self.num_octaves.setRange(3, 7)
        self.num_octaves.setValue(5)
        self.num_octaves.setToolTip("Number of scale-space octaves")
        form.addRow("Num Octaves:", self.num_octaves)

        # -- scale_suppression_radius --
        self.scale_suppression_radius = QDoubleSpinBox()
        self.scale_suppression_radius.setRange(0.0, 50.0)
        self.scale_suppression_radius.setDecimals(2)
        self.scale_suppression_radius.setSingleStep(1.0)
        self.scale_suppression_radius.setValue(0.0)
        self.scale_suppression_radius.setToolTip(
            "Scale-NMS radius multiplier (0 = disabled, 6.0 is a good starting value)"
        )
        form.addRow("Scale Suppress R:", self.scale_suppression_radius)

        group.setLayout(form)

        # -- Results display -----------------------------------------------
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        self.keypoints_count_label = QLabel("—")
        self.keypoints_count_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        results_layout.addRow("Keypoints:", self.keypoints_count_label)
        results_group.setLayout(results_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)
        layout.addWidget(results_group)
        layout.addStretch()

    def to_extract_options(self) -> ExtractOptions:
        """Read current spinner values and return an ExtractOptions."""
        return ExtractOptions(
            thresh=self.thresh.value(),
            lowest_scale=self.lowest_scale.value(),
            highest_scale=self.highest_scale.value(),
            edge_thresh=self.edge_thresh.value(),
            init_blur=self.init_blur.value(),
            max_keypoints=self.max_keypoints.value(),
            num_octaves=self.num_octaves.value(),
            scale_suppression_radius=self.scale_suppression_radius.value(),
        )

    def from_extract_options(self, opts: ExtractOptions):
        """Populate spinners from an existing ExtractOptions."""
        self.thresh.setValue(opts.thresh)
        self.lowest_scale.setValue(opts.lowest_scale)
        hs = opts.highest_scale if opts.highest_scale != float('inf') else 99999.0
        self.highest_scale.setValue(hs)
        self.edge_thresh.setValue(opts.edge_thresh)
        self.init_blur.setValue(opts.init_blur)
        self.max_keypoints.setValue(opts.max_keypoints)
        self.num_octaves.setValue(opts.num_octaves)
        self.scale_suppression_radius.setValue(opts.scale_suppression_radius)


class SiftVisualizer(QMainWindow):
    def __init__(self, sift):
        super().__init__()
        self.setWindowTitle("CUDA SIFT Visualizer")
        self.setGeometry(100, 100, 1400, 700)

        self._sift = sift
        self._current_pixmap = None
        self._as_numpy_array = None
        self._extract_opts = ExtractOptions()
        self._keypoints = None
        self._overlay_pixmap = None
        self._show_overlay = True

        # -- Toolbar with Load button, Re-extract, and overlay toggle ------
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_action = QAction("Load Image...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_image)
        toolbar.addAction(load_action)

        reextract_action = QAction("Re-extract", self)
        reextract_action.setShortcut("Ctrl+R")
        reextract_action.triggered.connect(self._on_reextract)
        toolbar.addAction(reextract_action)

        toolbar.addSeparator()

        self._overlay_action = QAction("Toggle Overlay", self)
        self._overlay_action.setShortcut("Ctrl+T")
        self._overlay_action.setCheckable(True)
        self._overlay_action.setChecked(True)
        self._overlay_action.triggered.connect(self._toggle_overlay)
        toolbar.addAction(self._overlay_action)

        # -- Options panel (left sidebar) ----------------------------------
        self._options_panel = ExtractOptionsPanel()
        self._options_panel.from_extract_options(self._extract_opts)
        self._options_panel.setMinimumWidth(220)
        self._options_panel.setMaximumWidth(300)

        scroll = QScrollArea()
        scroll.setWidget(self._options_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(220)
        scroll.setMaximumWidth(300)

        # -- Single image panel --------------------------------------------
        self.image_panel = ImagePanel("No image loaded")

        # -- Main layout: options sidebar | image panel --------------------
        central = QWidget()
        hlayout = QHBoxLayout(central)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(scroll, 0)
        hlayout.addWidget(self.image_panel, 1)
        self.setCentralWidget(central)

    # -- Slots -------------------------------------------------------------

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.load_image(path)
        
        # After loading the image, convert to a numpy array for CuSift processing
        # The numpy array needs to be grayscale 32 bit floating point in range [0, 255]
        if self._current_pixmap is not None:
            qimg = self._current_pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(w * h)
            self._as_numpy_array = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w)).astype(np.float32)

        # Immediately run SIFT extraction after loading the image
        self._on_extract_sift()

    def _on_reextract(self):
        """Read options from the panel and re-run SIFT extraction."""
        self._extract_opts = self._options_panel.to_extract_options()
        self._on_extract_sift()

    def _on_extract_sift(self):
        if self._current_pixmap is None or self._as_numpy_array is None:
            return

        self._keypoints = self._sift.extract(self._as_numpy_array, options=self._extract_opts)
        print(f"Extracted {len(self._keypoints)} keypoints from the image.")
        self._options_panel.keypoints_count_label.setText(str(len(self._keypoints)))

        # Draw keypoints and display in the panel
        self._overlay_pixmap = self._draw_keypoints_to_pixmap(
            self._current_pixmap, self._keypoints
        )
        self._refresh_display()

    @staticmethod
    def _draw_keypoints_to_pixmap(
        source_pixmap: QPixmap,
        keypoints,
        radius_scale: float = 3.0,
        orientation_color: str = "red",
        line_width: int = 1,
    ) -> QPixmap:
        """Draw keypoints onto the image and return a QPixmap (no disk I/O).

        Uses the original QPixmap as the base so RGB images stay RGB."""
        from PIL import Image, ImageDraw

        # Convert QPixmap -> QImage -> PIL Image (preserves colour)
        qimg = source_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(w * h * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
        img = Image.fromarray(arr, mode="RGBA").convert("RGB")

        # Subsampling -> colour mapping (HSV hue ramp)
        subsample_vals = sorted({kp.subsampling for kp in keypoints})
        if len(subsample_vals) <= 1:
            scale_color = {s: (0, 255, 0) for s in subsample_vals}
        else:
            scale_color = {}
            for idx, s in enumerate(subsample_vals):
                hue = idx / len(subsample_vals)
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                scale_color[s] = (int(r * 255), int(g * 255), int(b * 255))

        draw = ImageDraw.Draw(img)
        for kp in keypoints:
            x, y = kp.x, kp.y
            s = kp.scale * radius_scale
            orient = kp.orientation
            rgb = scale_color[kp.subsampling]

            draw.ellipse(
                [x - s, y - s, x + s, y + s],
                outline=rgb,
                width=line_width,
            )
            dx = s * math.cos(orient)
            dy = s * math.sin(orient)
            draw.line(
                [(x, y), (x + dx, y + dy)],
                fill=orientation_color,
                width=line_width,
            )

        # Convert PIL Image -> QPixmap
        img_rgba = img.convert("RGBA")
        data = img_rgba.tobytes("raw", "RGBA")
        w, h = img_rgba.size
        qimg = QImage(data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    def _toggle_overlay(self, checked):
        self._show_overlay = checked
        self._refresh_display()

    def _refresh_display(self):
        """Show the overlay or the plain image depending on the toggle."""
        if self._show_overlay and self._overlay_pixmap is not None:
            self.image_panel.set_image(self._overlay_pixmap)
        elif self._current_pixmap is not None:
            self.image_panel.set_image(self._current_pixmap)

    # -- Public API --------------------------------------------------------

    def load_image(self, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self._current_pixmap = pixmap
        self._overlay_pixmap = None
        self.image_panel.set_image(pixmap)


def main():

    # Attempt to init CuSift to check if CUDA is available
    try:
        # The user can optionally pass in the path to the shared library, check if it exists first
        if len(sys.argv) > 1:
            lib_path = sys.argv[1]
            print(f"Attempting to load CuSift from: {lib_path}")
            sift = CuSift(lib_path)
        else:
            sift = CuSift()
    except Exception as e:
        print("Error initializing CuSift:", e)
        print("Make sure you have a compatible NVIDIA GPU and the CUDA toolkit installed.")
        return


    app = QApplication(sys.argv)
    visualizer = SiftVisualizer(sift=sift)
    visualizer.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


