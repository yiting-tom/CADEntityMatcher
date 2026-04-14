import json
import sys
from pathlib import Path

from smdr_core import (
    DEFAULT_DROP_ENTITY_TYPES,
    bounds_to_scene_rect,
    build_session_from_path,
    extract_template_from_scene_polygon,
    geometry_to_scene,
    load_template,
    scan_session,
)

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QPainter,
        QPainterPath,
        QPen,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGraphicsEllipseItem,
        QGraphicsPathItem,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None


if QT_IMPORT_ERROR is None:
    class DxfGraphicsView(QGraphicsView):
        selectionClosed = Signal(list)
        selectionCleared = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setScene(QGraphicsScene(self))
            self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
            self.setDragMode(QGraphicsView.NoDrag)
            self.setBackgroundBrush(QBrush(QColor("#f7f7f2")))
            self.setMouseTracking(True)
            self.viewport().setCursor(Qt.CrossCursor)

            self._drawing_item = None
            self._template_item = None
            self._match_item = None
            self._selection_items = []
            self._selection_points = []
            self._selection_closed = False
            self._hover_point = None
            self._is_panning = False
            self._pan_anchor = None

            self._preview_pen = QPen(QColor("#253238"), 0)
            self._preview_pen.setCosmetic(True)
            self._template_pen = QPen(QColor("#16803c"), 0)
            self._template_pen.setCosmetic(True)
            self._template_pen.setWidthF(1.8)
            self._match_pen = QPen(QColor("#b42318"), 0)
            self._match_pen.setCosmetic(True)
            self._match_pen.setWidthF(1.6)
            self._selection_pen = QPen(QColor("#0b63ce"), 0)
            self._selection_pen.setCosmetic(True)
            self._selection_pen.setWidthF(1.8)
            self._selection_fill = QBrush(QColor(11, 99, 206, 45))
            self._template_brush = QBrush(QColor(22, 128, 60, 18))

        def set_preview_geometries(self, geometries, bounds):
            if self._drawing_item is not None:
                self.scene().removeItem(self._drawing_item)
                self._drawing_item = None

            path = build_path_from_geometries(geometries)
            item = QGraphicsPathItem(path)
            item.setPen(self._preview_pen)
            item.setBrush(Qt.NoBrush)
            item.setZValue(0)
            self.scene().addItem(item)
            self._drawing_item = item

            x, y, w, h = bounds_to_scene_rect(bounds)
            self.scene().setSceneRect(QRectF(x, y, w, h))
            self.reset_view()

        def set_template_highlights(self, geometries):
            self._replace_highlight_item("_template_item", geometries, self._template_pen, self._template_brush, 20)

        def clear_template_highlights(self):
            if self._template_item is not None:
                self.scene().removeItem(self._template_item)
                self._template_item = None

        def set_match_highlights(self, matches):
            geometries = []
            for match in matches:
                geometries.extend(match.get("highlights", []))
            self._replace_highlight_item("_match_item", geometries, self._match_pen, Qt.NoBrush, 30)

        def clear_match_highlights(self):
            if self._match_item is not None:
                self.scene().removeItem(self._match_item)
                self._match_item = None

        def clear_selection(self):
            self._selection_points = []
            self._selection_closed = False
            self._hover_point = None
            self._clear_selection_items()
            self.selectionCleared.emit()

        def reset_view(self):
            rect = self.scene().sceneRect()
            if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
                return
            self.fitInView(rect, Qt.KeepAspectRatio)

        def mousePressEvent(self, event):
            if event.button() in (Qt.RightButton, Qt.MiddleButton):
                self._is_panning = True
                self._pan_anchor = event.position()
                self.viewport().setCursor(Qt.ClosedHandCursor)
                event.accept()
                return

            if event.button() == Qt.LeftButton and not self._selection_closed:
                scene_pos = self.mapToScene(event.position().toPoint())
                if len(self._selection_points) >= 3 and self._is_near_first_vertex(scene_pos, event.position()):
                    self.close_selection()
                else:
                    self._selection_points.append(scene_pos)
                    self._hover_point = scene_pos
                    self._redraw_selection()
                event.accept()
                return

            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._is_panning and self._pan_anchor is not None:
                delta = event.position() - self._pan_anchor
                self._pan_anchor = event.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                event.accept()
                return

            if self._selection_points and not self._selection_closed:
                self._hover_point = self.mapToScene(event.position().toPoint())
                self._redraw_selection()

            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() in (Qt.RightButton, Qt.MiddleButton) and self._is_panning:
                self._is_panning = False
                self._pan_anchor = None
                self.viewport().setCursor(Qt.CrossCursor)
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.LeftButton and len(self._selection_points) >= 3 and not self._selection_closed:
                self.close_selection()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)

        def wheelEvent(self, event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.clear_selection()
                event.accept()
                return
            super().keyPressEvent(event)

        def close_selection(self):
            if len(self._selection_points) < 3:
                return
            self._selection_closed = True
            self._hover_point = None
            self._redraw_selection()
            points = [(point.x(), point.y()) for point in self._selection_points]
            self.selectionClosed.emit(points)

        def _replace_highlight_item(self, attr_name, geometries, pen, brush, z_value):
            existing = getattr(self, attr_name)
            if existing is not None:
                self.scene().removeItem(existing)
            path = build_path_from_geometries(geometries)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(z_value)
            self.scene().addItem(item)
            setattr(self, attr_name, item)

        def _clear_selection_items(self):
            for item in self._selection_items:
                self.scene().removeItem(item)
            self._selection_items = []

        def _redraw_selection(self):
            self._clear_selection_items()
            if not self._selection_points:
                return

            path = QPainterPath()
            first = self._selection_points[0]
            path.moveTo(first)
            for point in self._selection_points[1:]:
                path.lineTo(point)
            if self._selection_closed:
                path.closeSubpath()
            elif self._hover_point is not None:
                path.lineTo(self._hover_point)

            path_item = QGraphicsPathItem(path)
            path_item.setPen(self._selection_pen)
            path_item.setBrush(self._selection_fill if self._selection_closed else Qt.NoBrush)
            path_item.setZValue(40)
            self.scene().addItem(path_item)
            self._selection_items.append(path_item)

            radius = self._scene_radius_for_pixels(5)
            for idx, point in enumerate(self._selection_points):
                vertex_pen = QPen(QColor("white"), 0)
                vertex_pen.setCosmetic(True)
                ellipse = QGraphicsEllipseItem(
                    point.x() - radius,
                    point.y() - radius,
                    radius * 2,
                    radius * 2,
                )
                ellipse.setPen(vertex_pen)
                ellipse.setBrush(QBrush(QColor("#16803c") if idx == 0 and len(self._selection_points) >= 3 else QColor("#0b63ce")))
                ellipse.setZValue(41)
                self.scene().addItem(ellipse)
                self._selection_items.append(ellipse)

        def _scene_radius_for_pixels(self, pixels):
            p0 = self.mapToScene(0, 0)
            p1 = self.mapToScene(pixels, 0)
            return abs(p1.x() - p0.x())

        def _is_near_first_vertex(self, scene_pos, view_pos):
            if not self._selection_points:
                return False
            first = self._selection_points[0]
            first_view = self.mapFromScene(first)
            dx = first_view.x() - view_pos.x()
            dy = first_view.y() - view_pos.y()
            return (dx * dx + dy * dy) ** 0.5 <= 12


    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.session = None
            self.session_stats = None
            self.current_template_id = None
            self.current_matches = None
            self.current_file = None

            self.setWindowTitle("SMDR Qt DXF Pattern Scanner")
            self.resize(1600, 980)

            self._build_ui()
            self._connect_signals()
            self._set_idle_text()

        def _build_ui(self):
            root = QWidget()
            layout = QHBoxLayout(root)
            splitter = QSplitter(Qt.Horizontal)
            layout.addWidget(splitter)
            self.setCentralWidget(root)

            controls = QWidget()
            controls_layout = QVBoxLayout(controls)
            controls_layout.setContentsMargins(8, 8, 8, 8)
            controls_layout.setSpacing(10)

            file_box = QGroupBox("1. Load DXF")
            file_layout = QVBoxLayout(file_box)
            file_row = QHBoxLayout()
            self.file_path_edit = QLineEdit()
            self.file_path_edit.setPlaceholderText("Choose a DXF file")
            browse_button = QPushButton("Browse…")
            browse_button.clicked.connect(self.choose_file)
            file_row.addWidget(self.file_path_edit)
            file_row.addWidget(browse_button)
            file_layout.addLayout(file_row)

            self.fast_build_cb = QCheckBox("Fast Cache Build (skip ELLIPSE/SPLINE)")
            self.drop_noisy_cb = QCheckBox("Enable Entity Type Filtering")
            self.drop_noisy_cb.setChecked(True)
            self.drop_common_circle_cb = QCheckBox("Drop Most Common Circle Size")
            self.drop_types_edit = QLineEdit(",".join(sorted(DEFAULT_DROP_ENTITY_TYPES)))
            self.drop_types_edit.setPlaceholderText("Comma-separated DXF types to drop")
            self.load_button = QPushButton("Load Drawing")
            file_layout.addWidget(self.fast_build_cb)
            file_layout.addWidget(self.drop_noisy_cb)
            file_layout.addWidget(self.drop_common_circle_cb)
            file_layout.addWidget(QLabel("Drop entity types"))
            file_layout.addWidget(self.drop_types_edit)
            file_layout.addWidget(self.load_button)

            selection_box = QGroupBox("2. Select Feature")
            selection_layout = QVBoxLayout(selection_box)
            selection_layout.addWidget(QLabel("Left click: add vertex | Double-click or click first point: close | Right drag: pan | Wheel: zoom | ESC: clear"))
            selection_row = QHBoxLayout()
            self.clear_selection_button = QPushButton("Clear Selection")
            self.reset_view_button = QPushButton("Reset View")
            self.download_template_button = QPushButton("Download Fingerprint")
            self.download_template_button.setEnabled(False)
            selection_row.addWidget(self.clear_selection_button)
            selection_row.addWidget(self.reset_view_button)
            selection_row.addWidget(self.download_template_button)
            selection_layout.addLayout(selection_row)
            self.extract_result = QPlainTextEdit()
            self.extract_result.setReadOnly(True)
            self.extract_result.setMinimumHeight(180)
            selection_layout.addWidget(self.extract_result)

            scan_box = QGroupBox("3. Scan")
            scan_layout = QVBoxLayout(scan_box)
            form = QFormLayout()
            self.scan_mode_combo = QComboBox()
            self.scan_mode_combo.addItems(["Standard Scan", "Fast Scan"])
            self.score_max_spin = QDoubleSpinBox()
            self.score_max_spin.setDecimals(3)
            self.score_max_spin.setSingleStep(0.01)
            self.score_max_spin.setRange(0.0, 1.0)
            self.score_max_spin.setValue(0.40)
            form.addRow("Mode", self.scan_mode_combo)
            form.addRow("Score max", self.score_max_spin)
            scan_layout.addLayout(form)
            scan_row = QHBoxLayout()
            self.scan_button = QPushButton("Start Pattern Matching")
            self.scan_button.setEnabled(False)
            self.download_matches_button = QPushButton("Download Match Results")
            self.download_matches_button.setEnabled(False)
            scan_row.addWidget(self.scan_button)
            scan_row.addWidget(self.download_matches_button)
            scan_layout.addLayout(scan_row)
            self.scan_status = QLabel()
            scan_layout.addWidget(self.scan_status)
            self.scan_result = QPlainTextEdit()
            self.scan_result.setReadOnly(True)
            self.scan_result.setMinimumHeight(220)
            scan_layout.addWidget(self.scan_result)

            stats_box = QGroupBox("Load Stats")
            stats_layout = QVBoxLayout(stats_box)
            self.upload_result = QPlainTextEdit()
            self.upload_result.setReadOnly(True)
            self.upload_result.setMinimumHeight(220)
            stats_layout.addWidget(self.upload_result)

            controls_layout.addWidget(file_box)
            controls_layout.addWidget(selection_box)
            controls_layout.addWidget(scan_box)
            controls_layout.addWidget(stats_box)
            controls_layout.addStretch(1)

            view_container = QWidget()
            view_layout = QVBoxLayout(view_container)
            view_layout.setContentsMargins(8, 8, 8, 8)
            self.view = DxfGraphicsView()
            self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view_layout.addWidget(self.view)

            splitter.addWidget(controls)
            splitter.addWidget(view_container)
            splitter.setSizes([430, 1170])

            file_menu = self.menuBar().addMenu("File")
            open_action = QAction("Open DXF…", self)
            open_action.triggered.connect(self.choose_file)
            file_menu.addAction(open_action)

        def _connect_signals(self):
            self.load_button.clicked.connect(self.load_dxf)
            self.clear_selection_button.clicked.connect(self.clear_selection)
            self.reset_view_button.clicked.connect(self.view.reset_view)
            self.download_template_button.clicked.connect(self.download_template)
            self.scan_button.clicked.connect(self.scan_template)
            self.download_matches_button.clicked.connect(self.download_matches)
            self.view.selectionClosed.connect(self.extract_from_selection)
            self.view.selectionCleared.connect(self.on_selection_cleared)

        def _set_idle_text(self):
            self.upload_result.setPlainText("No DXF loaded.")
            self.extract_result.setPlainText("No template extracted.")
            self.scan_result.setPlainText("Waiting for scan.")
            self.scan_status.setText("")

        def choose_file(self):
            path, _ = QFileDialog.getOpenFileName(self, "Open DXF", str(Path.cwd()), "DXF Files (*.dxf)")
            if path:
                self.file_path_edit.setText(path)

        def load_dxf(self):
            path = self.file_path_edit.text().strip()
            if not path:
                self._show_error("Please choose a DXF file first.")
                return

            self.statusBar().showMessage("Loading DXF…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                session, stats = build_session_from_path(
                    path,
                    fast_build=self.fast_build_cb.isChecked(),
                    drop_noisy_types=self.drop_noisy_cb.isChecked(),
                    drop_entity_types=self.drop_types_edit.text(),
                    drop_most_common_circle=self.drop_common_circle_cb.isChecked(),
                )
            except Exception as exc:
                self._show_error(str(exc))
                self.statusBar().clearMessage()
                return
            finally:
                QApplication.restoreOverrideCursor()

            self.session = session
            self.session_stats = stats
            self.current_file = path
            self.current_template_id = None
            self.current_matches = None

            self.view.set_preview_geometries(session["preview_primitives"], session["bounds"])
            self.view.clear_selection()
            self.view.clear_template_highlights()
            self.view.clear_match_highlights()
            self.download_template_button.setEnabled(False)
            self.download_matches_button.setEnabled(False)
            self.scan_button.setEnabled(False)
            self.upload_result.setPlainText(json.dumps(stats, indent=2, ensure_ascii=False))
            self.extract_result.setPlainText("Drawing loaded. Add vertices in the canvas to select a feature.")
            self.scan_result.setPlainText("Waiting for scan.")
            self.scan_status.setText("")
            self.statusBar().showMessage(f"Loaded {Path(path).name}", 4000)

        def clear_selection(self):
            self.view.clear_selection()

        def on_selection_cleared(self):
            self.current_template_id = None
            self.current_matches = None
            self.view.clear_template_highlights()
            self.view.clear_match_highlights()
            self.download_template_button.setEnabled(False)
            self.download_matches_button.setEnabled(False)
            self.scan_button.setEnabled(False)
            self.extract_result.setPlainText("Selection cleared.")
            self.scan_result.setPlainText("Waiting for scan.")
            self.scan_status.setText("")

        def extract_from_selection(self, scene_polygon):
            if self.session is None:
                self._show_error("Load a DXF before extracting a template.")
                return

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                result = extract_template_from_scene_polygon(self.session, scene_polygon)
            except Exception as exc:
                self._show_error(str(exc))
                return
            finally:
                QApplication.restoreOverrideCursor()

            if result.get("entity_count", 0) <= 0 or not result.get("template_id"):
                self.current_template_id = None
                self.current_matches = None
                self.scan_button.setEnabled(False)
                self.download_template_button.setEnabled(False)
                self.download_matches_button.setEnabled(False)
                self.view.clear_template_highlights()
                self.view.clear_match_highlights()
                self.extract_result.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
                return

            self.current_template_id = result["template_id"]
            self.current_matches = None
            self.view.clear_match_highlights()
            self.view.set_template_highlights(result.get("highlights", []))
            self.download_template_button.setEnabled(True)
            self.scan_button.setEnabled(True)
            self.download_matches_button.setEnabled(False)

            preview = {
                "template_id": result["template_id"],
                "group_center": result.get("group_center"),
                "entity_count": result.get("entity_count", 0),
                "highlight_count_total": result.get("highlight_count_total", 0),
                "highlight_count_returned": result.get("highlight_count_returned", 0),
                "entities_preview": result.get("entities_preview", []),
            }
            self.extract_result.setPlainText(json.dumps(preview, indent=2, ensure_ascii=False))
            self.scan_result.setPlainText("Waiting for scan.")
            self.scan_status.setText("")
            self.statusBar().showMessage("Template extracted.", 3000)

        def scan_template(self):
            if self.session is None or not self.current_template_id:
                self._show_error("Extract a template before scanning.")
                return

            fast = self.scan_mode_combo.currentIndex() == 1
            self.scan_status.setText("Scanning…")
            self.scan_result.setPlainText("")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                data = scan_session(
                    self.session,
                    template_id=self.current_template_id,
                    score_max=self.score_max_spin.value(),
                    fast=fast,
                )
            except Exception as exc:
                self._show_error(str(exc))
                return
            finally:
                QApplication.restoreOverrideCursor()

            self.current_matches = data
            self.download_matches_button.setEnabled(True)
            self.view.set_match_highlights(data.get("matches", []))
            stats = data.get("scan_stats", {})
            self.scan_status.setText(
                f"Scan complete. {data.get('match_count', 0)} matches, "
                f"{stats.get('elapsed_ms', 0)} ms, engine={stats.get('engine', 'n/a')}"
            )

            preview_matches = []
            for match in (data.get("matches", []) or [])[:80]:
                preview_matches.append(
                    {
                        "dxf_x": match.get("dxf_x"),
                        "dxf_y": match.get("dxf_y"),
                        "match_score": match.get("match_score"),
                        "highlight_count_total": match.get("highlight_count_total", 0),
                        "highlight_count_returned": match.get("highlight_count_returned", 0),
                    }
                )
            preview = {
                "match_count": data.get("match_count", 0),
                "scan_stats": stats,
                "matches_preview": preview_matches,
            }
            self.scan_result.setPlainText(json.dumps(preview, indent=2, ensure_ascii=False))

        def download_template(self):
            if self.session is None or not self.current_template_id:
                return
            template = load_template(self.session, self.current_template_id)
            if template is None:
                self._show_error("Template not found.")
                return

            payload = {
                "template_id": self.current_template_id,
                "group_center": template.get("group_center"),
                "entities": template.get("entities", []),
            }
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Template Fingerprint",
                str(Path(self.current_file or "template_fingerprint.json").with_suffix(".json")),
                "JSON Files (*.json)",
            )
            if path:
                Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        def download_matches(self):
            if not self.current_matches:
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Match Results",
                str(Path(self.current_file or "match_results.json").with_suffix(".matches.json")),
                "JSON Files (*.json)",
            )
            if path:
                Path(path).write_text(json.dumps(self.current_matches, indent=2, ensure_ascii=False), encoding="utf-8")

        def _show_error(self, message):
            QMessageBox.critical(self, "SMDR", message)


def build_path_from_geometries(geometries):
    path = QPainterPath()
    for geometry in geometries or []:
        scene_geometry = geometry_to_scene(geometry)
        if scene_geometry["kind"] == "circle":
            radius = float(scene_geometry["r"])
            path.addEllipse(
                QRectF(
                    float(scene_geometry["cx"]) - radius,
                    float(scene_geometry["cy"]) - radius,
                    radius * 2,
                    radius * 2,
                )
            )
            continue

        points = scene_geometry.get("points", [])
        if not points:
            continue
        first = points[0]
        path.moveTo(QPointF(float(first[0]), float(first[1])))
        for point in points[1:]:
            path.lineTo(QPointF(float(point[0]), float(point[1])))
    return path


def main():
    if QT_IMPORT_ERROR is not None:
        raise SystemExit(
            "PySide6 is required to run the Qt desktop app. "
            "Install it in your environment first, then rerun `python app.py`."
        )

    app = QApplication(sys.argv)
    app.setApplicationName("SMDR")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
