# Файл: Meshropractor.py
import sys
import time
import numpy as np
import trimesh
import zipfile
import json
import io
import os

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QColorDialog, QTreeWidgetItem, QToolButton, \
    QLabel, QSplashScreen, QTableWidgetItem, QMenu, QPushButton, QWidget, QHBoxLayout, QCheckBox
from PySide6.QtCore import Qt, QSettings, QSize, QEvent
from PySide6.QtGui import QColor, QFont, QTextCursor, QPixmap, QIcon, QCursor, QAction
import pyvista as pv

from UI_Meshropractor import Ui_MainWindow
from Workers_Meshropractor import AlignmentThread, CompensationThread, HAS_O3D

if HAS_O3D:
    import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MeshropractorTeam", "Meshropractor")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.lbl_app_title = QLabel("Meshropractor - Без названия")
        self.lbl_app_title.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold; background: transparent;")
        self.setWindowTitle("Meshropractor - Без названия")
        self.ui.title_layout.insertWidget(3, self.lbl_app_title)
        self.ui.title_layout.insertStretch(4)

        self.cad_mesh = None
        self.scan_mesh = None
        self.result_mesh = None

        # --- Переменные Слайсера ---
        self.slicer_parts = []  # Список для хранения всех загруженных деталей

        self.actors = {"CAD": None, "Scan": None, "Result": None, "Heatmap": None}
        self.pick_mode = None
        self.cad_pts = []
        self.scan_pts = []
        self.pt_actors = []

        self.ui.action_save.triggered.connect(self.save_project)
        self.ui.action_undo.triggered.connect(self.undo_action)
        self.ui.action_redo.triggered.connect(self.redo_action)

        self.ui.btn_new_project.clicked.connect(self.action_new_project)
        self.ui.btn_open_project.clicked.connect(self.action_open_project)
        self.ui.btn_recent_projects.clicked.connect(self.open_recent_gallery)
        self.ui.btn_donate.clicked.connect(self.action_show_donate)
        self.ui.btn_back_to_start.clicked.connect(lambda: self.ui.stack.setCurrentWidget(self.ui.page_start))

        self.ui.tree.itemChanged.connect(self.on_tree_visibility_changed)

        self.ui.btn_load_cad.clicked.connect(self.load_cad)
        self.ui.btn_load_scan.clicked.connect(self.load_scan)
        self.ui.btn_pick_cad.clicked.connect(self.start_pick_cad)
        self.ui.btn_pick_scan.clicked.connect(self.start_pick_scan)
        self.ui.btn_clear_pts.clicked.connect(self.clear_picks)
        self.ui.btn_run_icp.clicked.connect(self.run_icp)

        self.ui.btn_run_comp.clicked.connect(self.run_comp)
        self.ui.btn_save.clicked.connect(self.save_result)
        self.ui.btn_cancel_comp.clicked.connect(self.cancel_comp)

        # --- Подключение кнопок ленты Слайсера ---
        self.ui.ribbon_btns["Создание срезов Concept Laser"].clicked.connect(self.open_export_dialog)
        self.ui.ribbon_btns["Новый проект"].clicked.connect(self.action_new_project)
        self.ui.ribbon_btns["Загрузить проект"].clicked.connect(self.action_open_project)
        self.ui.ribbon_btns["Сохранить проект"].clicked.connect(self.save_project)
        self.ui.ribbon_btns["Сохранить проект как"].clicked.connect(self.save_project)
        self.ui.ribbon_btns["Импорт детали"].clicked.connect(self.import_slicer_part)
        self.ui.ribbon_btns["Выгрузить деталь"].clicked.connect(self.unload_slicer_part)

        # Кнопки-пустышки, чтобы не было крашей, если на них нажмут
        self.ui.ribbon_btns["Сохранить выбранные детали как"].clicked.connect(self.save_selected_slicer_parts)
        self.ui.ribbon_btns["Сохранить все в папку"].clicked.connect(lambda: self.log("Функция в разработке"))

        # Обработка выделения детали кликом мыши (для фокусировки камеры)
        self.ui.tbl_parts.itemSelectionChanged.connect(self.on_slicer_part_selection_changed)

        # Клик по ячейке столбца "Затенение" -> всплывающее мини-меню выбора режима отображения детали
        self.ui.tbl_parts.cellClicked.connect(self.on_slicer_part_cell_clicked)


        self.ui.chk_view_cad.stateChanged.connect(self.update_visibility)
        self.ui.chk_view_scan.stateChanged.connect(self.update_visibility)
        self.ui.chk_view_res.stateChanged.connect(self.update_visibility)

        self.ui.sld_op_cad.valueChanged.connect(lambda v: self.update_opacity("CAD", v))
        self.ui.sld_op_scan.valueChanged.connect(lambda v: self.update_opacity("Scan", v))
        self.ui.sld_op_res.valueChanged.connect(lambda v: self.update_opacity("Result", v))

        self.ui.btn_col_cad.clicked.connect(lambda: self.pick_color("CAD", self.ui.btn_col_cad))
        self.ui.btn_col_scan.clicked.connect(lambda: self.pick_color("Scan", self.ui.btn_col_scan))
        self.ui.btn_col_res.clicked.connect(lambda: self.pick_color("Result", self.ui.btn_col_res))

        self.ui.btn_heatmap.clicked.connect(self.generate_heatmap)
        self.ui.btn_clear_heat.clicked.connect(self.clear_heatmap)
        self.ui.sliders["heat_limit"][0].valueChanged.connect(self.update_heatmap_limit)

        # Глобальный перехватчик движений мыши для изменения размера окна
        QApplication.instance().installEventFilter(self)

    def _check_resize_zone(self, pos):
        x, y = pos.x(), pos.y()
        margin = 6
        dir = ""
        if y < margin:
            dir += "T"
        elif y > self.height() - margin:
            dir += "B"
        if x < margin:
            dir += "L"
        elif x > self.width() - margin:
            dir += "R"
        return dir

    def _update_cursor(self, dir):
        if dir in ["T", "B"]:
            self.setCursor(Qt.SizeVerCursor)
        elif dir in ["L", "R"]:
            self.setCursor(Qt.SizeHorCursor)
        elif dir in ["TL", "BR"]:
            self.setCursor(Qt.SizeFDiagCursor)
        elif dir in ["TR", "BL"]:
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.unsetCursor()

    def eventFilter(self, obj, event):
        # ЗАЩИТА ОТ КРАША: не ловим мышь, пока окно не загрузилось полностью
        if not self.isVisible():
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseMove and not getattr(self, '_resizing', False):
            if event.buttons() == Qt.NoButton:
                pos = self.mapFromGlobal(event.globalPosition().toPoint())
                dir = self._check_resize_zone(pos)
                if dir:
                    self._update_cursor(dir)
                else:
                    self.unsetCursor()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._resize_dir = self._check_resize_zone(event.position().toPoint())
            if self._resize_dir:
                self._resizing = True
                self._start_geometry = self.geometry()
                self._start_mouse_pos = event.globalPosition().toPoint()
            elif event.position().y() < 45:
                self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        global_pos = event.globalPosition().toPoint()
        if getattr(self, '_resizing', False):
            dx = global_pos.x() - self._start_mouse_pos.x()
            dy = global_pos.y() - self._start_mouse_pos.y()
            x, y, w, h = self._start_geometry.getRect()

            if 'L' in self._resize_dir:
                w -= dx; x += dx
            elif 'R' in self._resize_dir:
                w += dx
            if 'T' in self._resize_dir:
                h -= dy; y += dy
            elif 'B' in self._resize_dir:
                h += dy

            if w < 800:
                if 'L' in self._resize_dir: x += (w - 800)
                w = 800
            if h < 600:
                if 'T' in self._resize_dir: y += (h - 600)
                h = 600

            self.setGeometry(x, y, w, h)
            event.accept()
        elif hasattr(self, 'drag_pos'):
            self.move(global_pos - self.drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._resizing = False
        if hasattr(self, 'drag_pos'): del self.drag_pos
        self.setCursor(Qt.ArrowCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def closeEvent(self, event):
        if hasattr(self, 'comp_thread') and self.comp_thread.isRunning():
            self.log("Принудительная остановка расчетов...")
            self.comp_thread.terminate()
            self.comp_thread.wait()

        if hasattr(self, 'align_thread') and self.align_thread.isRunning():
            self.align_thread.terminate()
            self.align_thread.wait()

        try:
            if hasattr(self, 'ui') and getattr(self.ui, 'plotter', None):
                self.ui.plotter.close()
            if hasattr(self, 'ui') and getattr(self.ui, 'slicer_plotter', None):
                self.ui.slicer_plotter.close()
        except Exception:
            pass

        event.accept()

    def log(self, text, replace=False):
        clean_text = text.replace("REPLACE_FLAG", "")
        print(clean_text)

    def pick_color(self, key, btn):
        initial_color = QColor(self.ui.mesh_colors[key])
        color = QColorDialog.getColor(initial_color, self, f"Выберите цвет для {key}")
        if color.isValid() and self.ui.plotter:
            hex_color = color.name()
            self.ui.mesh_colors[key] = hex_color
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 3px;")
            if self.actors[key]:
                self.actors[key].GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())
                self.ui.plotter.render()

    def trimesh_to_pyvista(self, tmesh):
        faces = np.pad(tmesh.faces, ((0, 0), (1, 0)), constant_values=3)
        return pv.PolyData(tmesh.vertices, faces)

    def show_mesh(self, key, mesh):
        if not self.ui.plotter: return
        pv_mesh = self.trimesh_to_pyvista(mesh)
        if self.actors[key]: self.ui.plotter.remove_actor(self.actors[key])

        if key == "CAD":
            op = self.ui.sld_op_cad.value() / 100.0
        elif key == "Scan":
            op = self.ui.sld_op_scan.value() / 100.0
        elif key == "Result":
            op = self.ui.sld_op_res.value() / 100.0
        else:
            op = 0.8

        self.actors[key] = self.ui.plotter.add_mesh(pv_mesh, color=self.ui.mesh_colors[key], opacity=op,
                                                    show_edges=(key == "Result"))
        self.actors[key].pickable = True
        self.ui.plotter.reset_camera()
        self.update_visibility()

    def update_visibility(self):
        if not self.ui.plotter: return
        if self.actors["CAD"]: self.actors["CAD"].SetVisibility(self.ui.chk_view_cad.isChecked())
        if self.actors["Scan"]: self.actors["Scan"].SetVisibility(self.ui.chk_view_scan.isChecked())
        if self.actors["Result"]: self.actors["Result"].SetVisibility(self.ui.chk_view_res.isChecked())
        self.ui.plotter.render()

    def update_opacity(self, key, value):
        if self.actors[key] and self.ui.plotter:
            self.actors[key].GetProperty().SetOpacity(value / 100.0)
            self.ui.plotter.render()

    def add_tree_item(self, parent_category, name, actor_key):
        parent_category.takeChildren()
        item = QTreeWidgetItem(parent_category, [name])
        item.setCheckState(0, Qt.Checked)
        item.setData(0, Qt.UserRole, actor_key)
        self.ui.tree.setCurrentItem(item)

    def on_tree_visibility_changed(self, item, column):
        actor_key = item.data(0, Qt.UserRole)
        if actor_key and self.actors.get(actor_key) and self.ui.plotter:
            is_visible = (item.checkState(0) == Qt.Checked)
            self.actors[actor_key].SetVisibility(is_visible)
            self.ui.plotter.render()

    def undo_action(self):
        self.log("> Возврат к предыдущему состоянию")

    def redo_action(self):
        self.log("> Повтор отмененного действия")

    def load_cad(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить CAD", "", "STL Files (*.stl)")
        if path:
            filename = path.split('/')[-1]
            self.log(f"> Загружен CAD: {filename}")
            self.cad_mesh = trimesh.load(path)
            self.show_mesh("CAD", self.cad_mesh)
            self.add_tree_item(self.ui.cat_cad, filename, "CAD")

    def load_scan(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить Скан", "", "STL Files (*.stl)")
        if path:
            filename = path.split('/')[-1]
            self.log(f"> Загружен Скан: {filename}")
            self.scan_mesh = trimesh.load(path)
            self.show_mesh("Scan", self.scan_mesh)
            self.add_tree_item(self.ui.cat_scan, filename, "Scan")

    # === ФУНКЦИИ СЛАЙСЕРА ===
    def import_slicer_part(self):
        """Загружает STL деталь в 3D-сцену слайсера"""
        path, _ = QFileDialog.getOpenFileName(self, "Импорт детали для печати", "", "STL Files (*.stl)")
        if path:
            filename = os.path.basename(path)
            self.log(f"\n>>> Загрузка детали в слайсер: {filename}...")
            try:
                mesh = trimesh.load(path)
                pv_mesh = self.trimesh_to_pyvista(mesh)

                # Убеждаемся, что сцена создана
                self.ui._ensure_slicer_plotter()

                # Уникальное имя для движка, чтобы он не перезаписывал старую модель
                part_id = len(self.slicer_parts)
                actor_name = f"slicer_part_{part_id}"

                if self.ui.slicer_plotter:
                    self.ui.slicer_plotter.add_mesh(pv_mesh, color="#d3d3d3", show_edges=True, name=actor_name)
                    if part_id == 0:  # Центрируем камеру только для первой детали
                        self.ui.slicer_plotter.reset_camera()
                    self.ui.slicer_plotter.render()

                # Сохраняем в список
                self.slicer_parts.append({
                    "mesh": mesh,
                    "mesh_pv": pv_mesh,  # pyvista-версия меша - нужна, например, для рамки-bbox в меню "Затенение"
                    "filename": filename,
                    "actor_name": actor_name,
                    "last_visible_mode": "shaded_wire",  # соответствует show_edges=True в add_mesh выше
                })

                # --- ДОБАВЛЯЕМ ДЕТАЛЬ В ТАБЛИЦУ ---
                row = self.ui.tbl_parts.rowCount()
                self.ui.tbl_parts.insertRow(row)

                item_id = QTableWidgetItem(str(row + 1))
                item_id.setTextAlignment(Qt.AlignCenter)
                self.ui.tbl_parts.setItem(row, 0, item_id)

                # --- Столбец 1: Выбранные (Чекбокс по центру) ---
                sel_container = QWidget()
                sel_layout = QHBoxLayout(sel_container)
                sel_layout.setContentsMargins(0, 0, 0, 0)
                sel_layout.setAlignment(Qt.AlignCenter)
                chk_sel = QCheckBox()
                chk_sel.setFixedSize(20, 20)  # <--- Обрубаем невидимый текст, делая ровный квадрат
                chk_sel.setChecked(True)
                chk_sel.setCursor(Qt.PointingHandCursor)
                sel_layout.addWidget(chk_sel)
                self.ui.tbl_parts.setCellWidget(row, 1, sel_container)

                # --- Столбец 2: Видимые (Чекбокс по центру) ---
                vis_container = QWidget()
                vis_layout = QHBoxLayout(vis_container)
                vis_layout.setContentsMargins(0, 0, 0, 0)
                vis_layout.setAlignment(Qt.AlignCenter)
                chk_vis = QCheckBox()
                chk_vis.setFixedSize(20, 20)  # <--- То же самое здесь
                chk_vis.setChecked(True)
                chk_vis.setCursor(Qt.PointingHandCursor)
                chk_vis.toggled.connect(lambda checked, r=row: self._on_vis_checkbox_changed(r, checked))
                vis_layout.addWidget(chk_vis)
                self.ui.tbl_parts.setCellWidget(row, 2, vis_container)

                self.ui.tbl_parts.setItem(row, 3, QTableWidgetItem("Зат.+каркас"))  # соответствует show_edges=True при add_mesh выше
                self.ui.tbl_parts.setItem(row, 4, QTableWidgetItem("0%"))
                # --- Создаем кнопку цвета (столбец 5) ---
                color_container = QWidget()
                color_layout = QHBoxLayout(color_container)
                color_layout.setContentsMargins(0, 0, 0, 0)
                color_layout.setAlignment(Qt.AlignCenter)

                btn_color = QPushButton()
                btn_color.setFixedSize(24, 24)
                btn_color.setCursor(Qt.PointingHandCursor)
                # Стартовый цвет берем из настроек сцены слайсера (#d3d3d3)
                btn_color.setStyleSheet("background-color: #d3d3d3; border: 1px solid #555; border-radius: 3px;")

                # Привязываем клик к новой функции, передавая ей номер строки и саму кнопку
                btn_color.clicked.connect(lambda checked=False, r=row, b=btn_color: self.pick_slicer_part_color(r, b))

                color_layout.addWidget(btn_color)
                # Используем setCellWidget вместо setItem!
                self.ui.tbl_parts.setCellWidget(row, 5, color_container)
                self.ui.tbl_parts.setItem(row, 6, QTableWidgetItem("STL"))
                self.ui.tbl_parts.setItem(row, 7, QTableWidgetItem(filename))

                self.ui.lbl_part_count.setText(f"Кол-во деталей: {self.ui.tbl_parts.rowCount()}")
                self.log("✅ Деталь успешно импортирована в сцену слайсера.")
            except Exception as e:
                self.log(f"[!] Ошибка при импорте детали: {str(e)}")

    def unload_slicer_part(self):
        """Очищает сцену слайсера и удаляет все детали"""
        self.slicer_parts = []
        if getattr(self.ui, 'slicer_plotter', None):
            self.ui.slicer_plotter.clear()
            self.ui.slicer_plotter.add_axes()
            self.ui.slicer_plotter.render()

        self.ui.tbl_parts.setRowCount(0)
        self.ui.lbl_part_count.setText("Кол-во деталей: 0")
        self.log("\n[i] Детали выгружены, сцена слайсера очищена.")

    def save_selected_slicer_parts(self):
        """Сохраняет выбранные галочкой детали из слайсера в STL"""
        if self.ui.tbl_parts.rowCount() == 0 or not self.slicer_parts:
            self.log("[!] ОШИБКА: Нет загруженных деталей для сохранения.")
            return

        meshes_to_save = []
        # Собираем меши всех деталей, у которых стоит галочка в колонке 1
        for row in range(self.ui.tbl_parts.rowCount()):
            container = self.ui.tbl_parts.cellWidget(row, 1)
            if container:
                chk = container.findChild(QCheckBox)
                if chk and chk.isChecked():
                    meshes_to_save.append(self.slicer_parts[row]["mesh"])

        if not meshes_to_save:
            self.log("[!] ВНИМАНИЕ: Нет выбранных деталей. Отметьте деталь галочкой в таблице.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить выбранные детали", "Exported_Parts.stl",
                                              "STL Files (*.stl)")
        if path:
            self.log(f"\n⏳ Экспорт деталей ({len(meshes_to_save)} шт.) в {path}...")
            try:
                # Если выбрано несколько деталей, склеиваем их в один STL файл
                if len(meshes_to_save) == 1:
                    final_mesh = meshes_to_save[0]
                else:
                    final_mesh = trimesh.util.concatenate(meshes_to_save)

                final_mesh.export(path)
                self.log("✅ Детали успешно сохранены в STL!")
            except Exception as e:
                self.log(f"[!] Ошибка при сохранении: {str(e)}")

    def _on_vis_checkbox_changed(self, row, is_visible):
        """Реагирует на переключение виджета галочки 'Видимые'."""
        if row >= len(self.slicer_parts):
            return

        # is_visible теперь — это строгий True или False
        if is_visible:
            # Если галочка стоит, возвращаем деталь в последнем выбранном режиме отображения
            last_mode = self.slicer_parts[row].get("last_visible_mode", "shaded_wire")
            self._apply_part_display_mode(row, last_mode, sync_visible_checkbox=False)
        else:
            # Если снята, полностью прячем со сцены
            self._apply_part_display_mode(row, "hide", sync_visible_checkbox=False)

    def on_slicer_part_selection_changed(self):
        """Центрирует камеру и ось вращения на выбранной в таблице детали"""
        if not getattr(self.ui, 'slicer_plotter', None) or not self.ui.tbl_parts.selectedItems():
            return

        row = self.ui.tbl_parts.currentRow()

        if 0 <= row < len(self.slicer_parts):
            actor_name = self.slicer_parts[row]["actor_name"]

            if actor_name in self.ui.slicer_plotter.actors:
                actor = self.ui.slicer_plotter.actors[actor_name]

                # --- ИСПРАВЛЕНИЕ БАГА: Не центрируем камеру на скрытых деталях ---
                if not actor.GetVisibility():
                    return
                # -----------------------------------------------------------------

                new_focal_point = np.array(actor.center)

                camera = self.ui.slicer_plotter.camera
                old_focal_point = np.array(camera.GetFocalPoint())

                shift = new_focal_point - old_focal_point

                old_pos = np.array(camera.GetPosition())
                camera.SetFocalPoint(*new_focal_point)
                camera.SetPosition(*(old_pos + shift))

                self.ui.slicer_plotter.reset_camera_clipping_range()
                self.ui.slicer_plotter.render()

    # ==========================================================
    # МИНИ-МЕНЮ СТОЛБЦА "ЗАТЕНЕНИЕ" (режим отображения детали)
    # ==========================================================
    # Индексы столбцов tbl_parts для справки (см. setHorizontalHeaderLabels в UI_Meshropractor.py):
    #   0 - "#"            1 - "Выбранные"      2 - "Видимые"     3 - "Затенение"
    #   4 - "Прозр."        5 - "Цвет"           6 - "Способ"      7 - "Название"
    COL_VISIBLE = 2
    COL_SHADING = 3
    COL_TRANSPARENCY = 4

    def on_slicer_part_cell_clicked(self, row, column):
        """Клик по ячейке таблицы деталей слайсера."""
        if row < 0 or row >= len(self.slicer_parts):
            return

        # === ЕСЛИ КЛИКНУЛИ ПО СТОЛБЦУ "ЗАТЕНЕНИЕ" ===
        if column == self.COL_SHADING:
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu { background-color: #333333; color: white; border: 1px solid #555; font-size: 13px; }
                QMenu::item { padding: 6px 24px 6px 12px; }
                QMenu::item:selected { background-color: #b31b1b; }
                QMenu::separator { height: 1px; background: #555; margin: 4px 6px; }
            """)

            modes = [
                ("Скрыть", "hide", True),
                ("Затенение", "shaded", False),
                ("Треугольники", "triangles", False),
                ("Затенение и каркас", "shaded_wire", False),
                ("Каркас", "wireframe", False),
                ("Ограничивающий параллелепипед", "bbox", False),
                ("Прозрачность", "transparent", False),
                ("Без затенения", "flat", False),
            ]
            for label, mode_key, add_separator_after in modes:
                action = QAction(label, self)
                action.triggered.connect(
                    lambda checked=False, r=row, mk=mode_key: self._apply_part_display_mode(r, mk))
                menu.addAction(action)
                if add_separator_after:
                    menu.addSeparator()

            menu.exec(QCursor.pos())

        # === ЕСЛИ КЛИКНУЛИ ПО СТОЛБЦУ "ПРОЗРАЧНОСТЬ" ===
        elif column == self.COL_TRANSPARENCY:
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu { background-color: #333333; color: white; border: 1px solid #555; font-size: 13px; }
                QMenu::item { padding: 6px 24px 6px 12px; }
                QMenu::item:selected { background-color: #b31b1b; }
            """)

            # Уровни прозрачности в процентах (0 = сплошная деталь, 100 = невидимая)
            levels = [0, 25, 50, 75, 90]
            for val in levels:
                action = QAction(f"{val}%", self)
                action.triggered.connect(lambda checked=False, r=row, v=val: self._apply_part_transparency(r, v))
                menu.addAction(action)

            menu.exec(QCursor.pos())

    def _apply_part_display_mode(self, row, mode_key, sync_visible_checkbox=True):
        """Применяет выбранный в мини-меню режим отображения к 3D-актеру детали.

        mode_key - один из:
            'hide'         - Скрыть (деталь полностью пропадает со сцены)
            'shaded'       - Затенение (обычная сплошная закрашенная поверхность)
            'triangles'    - Треугольники (та же поверхность, но без сглаживания
                              нормалей - видна огранка/триангуляция меша)
            'shaded_wire'  - Затенение и каркас (закрашенная поверхность + ребра сетки поверх)
            'wireframe'    - Каркас (только ребра сетки, без закрашенных граней)
            'bbox'         - Ограничивающий параллелепипед (вместо детали - габаритный "ящик")
            'transparent'  - Прозрачность (полупрозрачная поверхность)
            'flat'         - Без затенения (ровная заливка цветом, без учета освещения сцены)

        sync_visible_checkbox - обновлять ли галочку "Видимые" (COL_VISIBLE) под новый
            режим. По умолчанию True (вызов из меню "Затенение" - тогда галочка должна
            подстроиться под режим). При вызове ИЗ обработчика самой галочки
            (on_slicer_part_item_changed) передают False, чтобы не дергать ее
            повторно и не плодить лишние сигналы - она там уже в нужном состоянии.
        """
        if row >= len(self.slicer_parts):
            return
        plotter = getattr(self.ui, 'slicer_plotter', None)
        if not plotter:
            return

        part = self.slicer_parts[row]
        actor_name = part["actor_name"]
        actor = plotter.actors.get(actor_name)
        if actor is None:
            return

        prop = actor.GetProperty()
        bbox_name = f"{actor_name}__bbox"  # имя вспомогательного актера с рамкой-bbox
        bbox_actor = plotter.actors.get(bbox_name)

        # Перед применением конкретного режима сбрасываем к "нейтральному" состоянию:
        # сама деталь видима, рамка (если создавалась раньше) скрыта, освещение/прозрачность
        # по умолчанию. Дальше каждый режим включает только то, что ему нужно.
        actor.SetVisibility(True)
        if bbox_actor is not None:
            bbox_actor.SetVisibility(False)
        # Считываем текущую прозрачность из таблицы, чтобы не сбросить ее случайно
        current_trans_pct = 0
        trans_item = self.ui.tbl_parts.item(row, self.COL_TRANSPARENCY)
        if trans_item and trans_item.text().endswith('%'):
            current_trans_pct = int(trans_item.text()[:-1])
        prop.SetOpacity(1.0 - (current_trans_pct / 100.0))
        prop.SetLighting(True)
        prop.SetEdgeVisibility(False)
        prop.SetRepresentationToSurface()
        prop.SetInterpolationToGouraud()  # гладкое (сглаженное по нормалям) освещение

        if mode_key == "hide":
            # --- Скрыть: полностью прячем деталь со сцены ---
            actor.SetVisibility(False)

        elif mode_key == "shaded":
            # --- Затенение: сплошная закрашенная поверхность без видимых ребер ---
            pass  # это и есть "нейтральное" состояние, заданное выше

        elif mode_key == "triangles":
            # --- Треугольники: та же поверхность, но БЕЗ сглаживания нормалей -
            # каждая треугольная грань меша видна отдельной плоской фасеткой ---
            prop.SetInterpolationToFlat()

        elif mode_key == "shaded_wire":
            # --- Затенение и каркас: закрашенная поверхность + поверх видны ребра треугольников ---
            prop.SetEdgeVisibility(True)
            prop.SetEdgeColor(0.0, 0.0, 0.0)

        elif mode_key == "wireframe":
            # --- Каркас: только ребра сетки, без закрашенных граней ---
            prop.SetRepresentationToWireframe()

        elif mode_key == "bbox":
            # --- Ограничивающий параллелепипед: прячем саму деталь и показываем вместо
            # нее габаритный "ящик". Актер рамки создается лениво один раз на деталь
            # и переиспользуется при повторных выборах этого режима ---
            actor.SetVisibility(False)
            if bbox_actor is None and part.get("mesh_pv") is not None:
                outline_mesh = part["mesh_pv"].outline()
                bbox_actor = plotter.add_mesh(outline_mesh, color="yellow", line_width=2, name=bbox_name)
            if bbox_actor is not None:
                bbox_actor.SetVisibility(True)

        elif mode_key == "transparent":
            # --- Прозрачность: полупрозрачная поверхность, чтобы видеть, что за деталью ---
            prop.SetOpacity(0.35)

        elif mode_key == "flat":
            # --- Без затенения: ровная заливка цветом без учета освещения сцены
            # (деталь не темнеет/не светлеет в зависимости от угла к источнику света) ---
            prop.SetLighting(False)

        # Запоминаем последний НЕ-скрывающий режим - он нужен, чтобы при повторном
        # включении галочки "Видимые" деталь вернулась именно в него (а не всегда
        # в "Затенение и каркас" по умолчанию), и чтобы для 'bbox' при показе
        # обратно появлялась рамка, а не сам меш.
        if mode_key != "hide":
            part["last_visible_mode"] = mode_key

        # Ячейка столбца "Затенение" - подпись текущего режима
        mode_labels = {
            "hide": "Скрыто", "shaded": "Затенение", "triangles": "Треугольники",
            "shaded_wire": "Зат.+каркас", "wireframe": "Каркас",
            "bbox": "Огр. паралл.", "transparent": "Прозрачность", "flat": "Без затенения",
        }
        cell = self.ui.tbl_parts.item(row, self.COL_SHADING)
        if cell:
            cell.setText(mode_labels.get(mode_key, ""))

            # Галочка "Видимые" (COL_VISIBLE) - держим в согласии с фактической видимостью:
            if sync_visible_checkbox:
                container = self.ui.tbl_parts.cellWidget(row, self.COL_VISIBLE)
                if container:
                    chk = container.findChild(QCheckBox)
                    if chk:
                        chk.blockSignals(True)
                        chk.setChecked(mode_key != "hide")
                        chk.blockSignals(False)

            # --- ИСПРАВЛЕНИЕ БАГА: Принудительно пересчитываем глубину видимости камеры ---
            plotter.reset_camera_clipping_range()

            plotter.render()

    def _apply_part_transparency(self, row, trans_pct):
        """Меняет уровень прозрачности актера в PyVista и обновляет текст в таблице"""
        if row >= len(self.slicer_parts):
            return

        plotter = getattr(self.ui, 'slicer_plotter', None)
        if not plotter:
            return

        part = self.slicer_parts[row]
        actor_name = part["actor_name"]
        actor = plotter.actors.get(actor_name)

        if actor is not None:
            # PyVista принимает Opacity от 1.0 (сплошной) до 0.0 (полностью прозрачный).
            # Поэтому инвертируем наши проценты: 75% прозрачности = 0.25 Opacity
            opacity_value = 1.0 - (trans_pct / 100.0)
            actor.GetProperty().SetOpacity(opacity_value)

            # Если мы сделали деталь прозрачной вручную, обновляем ее статус
            # в столбце "Затенение", чтобы не было конфликтов логики
            if trans_pct > 0:
                cell_shading = self.ui.tbl_parts.item(row, self.COL_SHADING)
                if cell_shading and cell_shading.text() not in ["Каркас", "Огр. паралл.", "Скрыто"]:
                    cell_shading.setText("Прозрачность")
                    part["last_visible_mode"] = "transparent"
            elif trans_pct == 0:
                # Если вернули 0% прозрачности, логично вернуть надпись "Затенение"
                cell_shading = self.ui.tbl_parts.item(row, self.COL_SHADING)
                if cell_shading and cell_shading.text() == "Прозрачность":
                    cell_shading.setText("Затенение")
                    part["last_visible_mode"] = "shaded"

        # Обновляем текст в ячейке "Прозр."
        cell_trans = self.ui.tbl_parts.item(row, self.COL_TRANSPARENCY)
        if cell_trans:
            cell_trans.setText(f"{trans_pct}%")

        plotter.render()

    def pick_slicer_part_color(self, row, btn):
        """Вызывает окно выбора цвета и перекрашивает 3D-деталь в слайсере"""
        if row >= len(self.slicer_parts):
            return

        plotter = getattr(self.ui, 'slicer_plotter', None)
        if not plotter:
            return

        part = self.slicer_parts[row]
        actor_name = part["actor_name"]
        actor = plotter.actors.get(actor_name)

        if not actor:
            return

        # 1. Считываем текущий цвет детали, чтобы палитра открывалась не с белого цвета
        current_rgb = actor.GetProperty().GetColor()
        initial_color = QColor(int(current_rgb[0] * 255), int(current_rgb[1] * 255), int(current_rgb[2] * 255))

        # 2. Вызываем стандартное окно палитры
        color = QColorDialog.getColor(initial_color, self, f"Выберите цвет для детали {part['filename']}")

        # 3. Если пользователь выбрал цвет и нажал "ОК"
        if color.isValid():
            hex_color = color.name()
            # Перекрашиваем квадратик в таблице
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 3px;")
            # Перекрашиваем саму деталь в 3D-движке (PyVista ждет доли от 0 до 1, поэтому redF, greenF)
            actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())

            plotter.render()

    def start_pick_cad(self):
        if not self.cad_mesh: return self.log("[!] Сначала загрузите CAD!")

        # Ленивая привязка горячей клавиши для VTK-сцены
        if not getattr(self, '_space_bound', False) and self.ui.plotter:
            self.ui.plotter.add_key_event('space', self.on_space_pressed)
            self._space_bound = True

        self.pick_mode = 'CAD'
        self.ui.chk_view_cad.setChecked(True)
        self.ui.chk_view_scan.setChecked(False)
        self.ui.sld_op_cad.setValue(100)
        self.log("\n[РЕЖИМ CAD] Наведите курсор на деталь и нажмите ПРОБЕЛ.")

    def start_pick_scan(self):
        if not self.scan_mesh: return self.log("[!] Сначала загрузите Скан!")

        if not getattr(self, '_space_bound', False) and self.ui.plotter:
            self.ui.plotter.add_key_event('space', self.on_space_pressed)
            self._space_bound = True

        self.pick_mode = 'Scan'
        self.ui.chk_view_cad.setChecked(False)
        self.ui.chk_view_scan.setChecked(True)
        self.ui.sld_op_scan.setValue(100)
        self.log("\n[РЕЖИМ СКАНА] Наведите курсор на деталь и нажмите ПРОБЕЛ.")

    def on_space_pressed(self):
        if not self.pick_mode or not self.ui.plotter: return
        try:
            import vtk
            pos = self.ui.plotter.interactor.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.Pick(pos[0], pos[1], 0, self.ui.plotter.renderer)
            if picker.GetActor():
                self.place_marker(picker.GetPickPosition())
        except Exception as e:
            self.log(f"[!] Ошибка лучемета: {str(e)}")

    def place_marker(self, point):
        if not self.ui.plotter: return
        radius = self.cad_mesh.scale * 0.015 if self.cad_mesh else 1.0
        if self.pick_mode == 'CAD':
            self.cad_pts.append(point)
            actor = self.ui.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color='red')
            actor.pickable = False
            self.pt_actors.append(actor)
            self.log(f"📍 CAD-точка {len(self.cad_pts)} установлена.")
        elif self.pick_mode == 'Scan':
            self.scan_pts.append(point)
            actor = self.ui.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color='yellow')
            actor.pickable = False
            self.pt_actors.append(actor)
            self.log(f"📍 Скан-точка {len(self.scan_pts)} установлена.")
        self.ui.lbl_pts.setText(f"Точек на CAD: {len(self.cad_pts)} | Точек на Скане: {len(self.scan_pts)}")

    def clear_picks(self):
        self.cad_pts.clear()
        self.scan_pts.clear()
        self.pick_mode = None
        if getattr(self.ui, 'plotter', None):
            for actor in self.pt_actors: self.ui.plotter.remove_actor(actor)
        self.pt_actors.clear()
        self.ui.lbl_pts.setText("Точек на CAD: 0 | Точек на Скане: 0")

    def run_icp(self):
        if not self.cad_mesh or not self.scan_mesh: return self.log("[!] ОШИБКА: Загрузите обе модели!")
        if len(self.cad_pts) > 0 and len(self.cad_pts) != len(self.scan_pts): return self.log(
            "[!] ОШИБКА: Точки не совпадают!")
        self.pick_mode = None
        self.ui.btn_run_icp.setEnabled(False)
        self.ui.btn_run_icp.setText("⏳ ИДЕТ СОВМЕЩЕНИЕ...")

        self.align_thread = AlignmentThread(self.cad_mesh, self.scan_mesh, self.cad_pts, self.scan_pts)
        self.align_thread.log_signal.connect(self.log)
        self.align_thread.finished_signal.connect(self.on_icp_done)
        self.align_thread.start()

    def on_icp_done(self, aligned_scan):
        self.scan_mesh = aligned_scan
        self.show_mesh("Scan", self.scan_mesh)
        self.ui.chk_view_cad.setChecked(True)
        self.ui.chk_view_scan.setChecked(True)
        self.ui.sld_op_cad.setValue(40)
        self.ui.sld_op_scan.setValue(100)
        self.ui.btn_run_icp.setEnabled(True)
        self.ui.btn_run_icp.setText("▶ СОВМЕСТИТЬ МОДЕЛИ (ICP)")
        self.clear_picks()
        self.ui.tabs.setCurrentIndex(1)
        self.log("\n>>> Модели совмещены. Перейдите к предеформации (Шаг 2).")

    def generate_heatmap(self):
        if not self.cad_mesh or not self.scan_mesh: return self.log("[!] ОШИБКА: Загрузите модели.")
        if not getattr(self.ui, 'plotter', None): return

        self.log("\n>>> Расчет цветовой карты...")
        self.ui.btn_heatmap.setEnabled(False)
        try:
            if not HAS_O3D: return self.log("[!] Для Heatmap требуется Open3D.")
            cad_tmesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.array(self.cad_mesh.vertices, dtype=np.float32)),
                                                    o3d.core.Tensor(np.array(self.cad_mesh.faces, dtype=np.int32)))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(cad_tmesh)
            query_points = o3d.core.Tensor(np.array(self.scan_mesh.vertices, dtype=np.float32))
            signed_dists = scene.compute_signed_distance(query_points).numpy()
            pv_heatmap = self.trimesh_to_pyvista(self.scan_mesh)
            pv_heatmap['Deviation'] = signed_dists
            if self.actors["Heatmap"]: self.ui.plotter.remove_actor(self.actors["Heatmap"])
            self.ui.chk_view_cad.setChecked(False)
            self.ui.chk_view_scan.setChecked(False)
            limit = self.ui.sliders["heat_limit"][0].value() / self.ui.sliders["heat_limit"][1]
            self.actors["Heatmap"] = self.ui.plotter.add_mesh(
                pv_heatmap, scalars='Deviation', cmap='turbo', clim=[-limit, limit],
                show_scalar_bar=True, scalar_bar_args={
                    'title': 'Отклонение (мм)', 'color': 'black', 'vertical': True,
                    'position_x': 0.88, 'position_y': 0.05, 'height': 0.9, 'width': 0.08,
                    'title_font_size': 18, 'label_font_size': 14, 'fmt': '%1.3f'
                }
            )
            self.ui.plotter.reset_camera()
            self.log(f"✅ Готово! Красный = Наплыв, Синий = Усадка.")
        except Exception as e:
            self.log(f"[!] Ошибка: {str(e)}")
        finally:
            self.ui.btn_heatmap.setEnabled(True)

    def update_heatmap_limit(self):
        if self.actors.get("Heatmap") and hasattr(self.actors["Heatmap"].mapper, 'dataset') and self.ui.plotter:
            limit = self.ui.sliders["heat_limit"][0].value() / self.ui.sliders["heat_limit"][1]
            self.actors["Heatmap"].mapper.scalar_range = [-limit, limit]
            self.ui.plotter.render()

    def clear_heatmap(self):
        if self.actors.get("Heatmap") and self.ui.plotter:
            self.ui.plotter.remove_actor(self.actors["Heatmap"])
            self.actors["Heatmap"] = None
        self.ui.chk_view_scan.setChecked(True)
        self.ui.chk_view_cad.setChecked(True)
        self.log("Отображение сброшено в базовый режим.")

    def update_progress_safe(self, value):
        if hasattr(self, 'ui') and hasattr(self.ui, 'comp_progress_bar'):
            self.ui.comp_progress_bar.setValue(value)

    def run_comp(self):
        if not self.cad_mesh or not self.scan_mesh: return

        self.ui.comp_stack.setCurrentIndex(1)
        self.ui.comp_progress_bar.setValue(0)
        self.ui.btn_run_comp.setEnabled(False)
        self.ui.btn_run_comp.setText("⏳ ИДЕТ РАСЧЕТ...")

        settings = {
            "points": int(self.ui.sliders["points"][0].value() / self.ui.sliders["points"][1]),
            "smooth": float(self.ui.sliders["smooth"][0].value() / self.ui.sliders["smooth"][1]),
            "use_remesh": self.ui.chk_remesh.isChecked(),
            "edge_len": float(self.ui.sliders["edge_len"][0].value() / self.ui.sliders["edge_len"][1]),
            "limit": float(self.ui.sliders["limit"][0].value() / self.ui.sliders["limit"][1]),
            "norm": float(self.ui.sliders["norm"][0].value() / self.ui.sliders["norm"][1]),
            "anchor": self.ui.chk_anchor.isChecked(),
            "neighbors": int(self.ui.sliders["neighbors"][0].value() / self.ui.sliders["neighbors"][1])
        }

        self.comp_thread = CompensationThread(self.cad_mesh, self.scan_mesh, settings)
        self.comp_thread.log_signal.connect(self.log)
        self.comp_thread.progress_signal.connect(self.update_progress_safe)
        self.comp_thread.finished_signal.connect(self.on_comp_done)
        self.comp_thread.start()

    def cancel_comp(self):
        if hasattr(self, 'comp_thread') and self.comp_thread.isRunning():
            self.log("[!] Расчет принудительно остановлен пользователем.")
            self.comp_thread.terminate()
            self.comp_thread.wait()

        self.ui.comp_stack.setCurrentIndex(0)
        self.ui.btn_run_comp.setEnabled(True)
        self.ui.btn_run_comp.setText("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")

    def on_comp_done(self, result_mesh):
        self.result_mesh = result_mesh
        self.show_mesh("Result", self.result_mesh)
        self.add_tree_item(self.ui.cat_res, "Compensated_Part.stl", "Result")

        if self.ui.cat_scan.childCount() > 0:
            self.ui.cat_scan.child(0).setCheckState(0, Qt.Unchecked)

        self.ui.comp_stack.setCurrentIndex(0)
        self.ui.btn_run_comp.setEnabled(True)
        self.ui.btn_run_comp.setText("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.ui.btn_save.setEnabled(True)

    def save_result(self):
        if self.result_mesh:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "Compensated_Part.stl", "STL Files (*.stl)")
            if path:
                self.result_mesh.export(path)
                self.log(f"✅ Успешно сохранено: {path}")

    def open_export_dialog(self):
        from UI_Meshropractor import DialogExportCLS
        dialog = DialogExportCLS(self)
        if dialog.exec():
            self.log("\n>>> Окно экспорта подтверждено. Скоро здесь будет запуск нарезки .CLS с новыми параметрами!")

    def update_title(self, project_name=None):
        title = f"Meshropractor - {project_name}" if project_name else "Meshropractor - Без названия"
        self.setWindowTitle(title)
        self.lbl_app_title.setText(title)

    def clear_project_data(self):
        """Очищает память и ОБЕ 3D-сцены для старта нового проекта"""
        self.clear_picks()
        self.cad_mesh = None
        self.scan_mesh = None
        self.result_mesh = None

        # Полностью очищаем переменные слайсера
        self.slicer_parts = []

        # Безопасная очистка сцены предеформации
        if getattr(self.ui, 'plotter', None):
            self.ui.plotter.clear()
            self.ui.plotter.add_axes()

        # Безопасная очистка сцены слайсера
        if getattr(self.ui, 'slicer_plotter', None):
            self.ui.slicer_plotter.clear()
            self.ui.slicer_plotter.add_axes()
            self.ui.slicer_plotter.render()

        self.ui.cat_cad.takeChildren()
        self.ui.cat_scan.takeChildren()
        self.ui.cat_res.takeChildren()

        self.ui.tbl_parts.setRowCount(0)
        self.ui.lbl_part_count.setText("Кол-во деталей: 0")

        self.log("\n[i] Память и 3D-сцены полностью очищены. Начат новый проект.")

    def add_to_recent(self, path):
        recent = self.settings.value("recent_files", [])
        if path in recent: recent.remove(path)
        recent.insert(0, path)
        recent = recent[:12]
        self.settings.setValue("recent_files", recent)

    def open_recent_gallery(self):
        self.ui.stack.setCurrentWidget(self.ui.page_recent)
        for i in reversed(range(self.ui.recent_layout.count())):
            widget = self.ui.recent_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        recent_files = self.settings.value("recent_files", [])
        row, col = 0, 0
        for path in recent_files:
            if not os.path.exists(path): continue

            card = QToolButton()
            card.setFixedSize(220, 240)
            card.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            card.setCursor(Qt.PointingHandCursor)

            pixmap = QPixmap()
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    if 'preview.png' in zf.namelist():
                        pixmap.loadFromData(zf.read('preview.png'))
            except:
                pass

            if not pixmap.isNull():
                card.setIcon(QIcon(pixmap))
                card.setIconSize(QSize(200, 180))

            card.setText(os.path.basename(path))
            card.clicked.connect(lambda ch=False, p=path: self.load_mrp_file(p))
            card.setStyleSheet(
                "QToolButton { background-color: white; border: 1px solid #ccc; border-radius: 5px; color: black; font-weight: bold;} QToolButton:hover { border: 2px solid #b31b1b; }")

            self.ui.recent_layout.addWidget(card, row, col)
            col += 1
            if col > 3:
                col = 0;
                row += 1

    def save_project(self):
        has_data = self.cad_mesh is not None or self.scan_mesh is not None or len(self.slicer_parts) > 0

        if not has_data:
            self.log("[!] ОШИБКА: Проект пуст, нечего сохранять.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "New_Project.mrp", "MeshRopractor Project (*.mrp)")
        if not path: return

        self.log(f"\n⏳ Сохранение проекта в {path}...")
        try:
            if self.ui.plotter:
                img_array = self.ui.plotter.screenshot(transparent_background=False)
                from PIL import Image
                img = Image.fromarray(img_array)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
            else:
                img_bytes = b""

            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if self.cad_mesh: zf.writestr('meshes/cad.stl', self.cad_mesh.export(file_type='stl'))
                if self.scan_mesh: zf.writestr('meshes/scan.stl', self.scan_mesh.export(file_type='stl'))
                if self.result_mesh: zf.writestr('meshes/result.stl', self.result_mesh.export(file_type='stl'))

                meta = {
                    "version": "1.2",
                    "cad_pts": [list(p) for p in self.cad_pts],
                    "scan_pts": [list(p) for p in self.scan_pts]
                }

                # Сохраняем ВСЕ детали слайсера и их оригинальные имена
                for i, part_data in enumerate(self.slicer_parts):
                    zf.writestr(f'meshes/slicer_part_{i}.stl', part_data['mesh'].export(file_type='stl'))
                    meta[f"slicer_name_{i}"] = part_data['filename']

                zf.writestr('project.json', json.dumps(meta, indent=4))
                if img_bytes: zf.writestr('preview.png', img_bytes)

            self.add_to_recent(path)
            self.update_title(os.path.basename(path))
            self.log("✅ Проект успешно сохранен!")
        except Exception as e:
            self.log(f"[!] Ошибка при сохранении: {str(e)}")

    def action_new_project(self):
        # Нажатие на "Новый проект" теперь полностью очищает систему
        self.clear_project_data()
        self.update_title(None)

    def action_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "MeshRopractor (*.mrp)")
        if path: self.load_mrp_file(path)

    def load_mrp_file(self, path):
        self.log(f"\n⏳ Открытие проекта {path}...")
        self.clear_project_data()

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                file_list = zf.namelist()
                meta = json.loads(zf.read('project.json').decode('utf-8')) if 'project.json' in file_list else {}
                has_predef = False

                if 'meshes/cad.stl' in file_list:
                    self.cad_mesh = trimesh.load(io.BytesIO(zf.read('meshes/cad.stl')), file_type='stl')
                    self.show_mesh("CAD", self.cad_mesh)
                    self.add_tree_item(self.ui.cat_cad, "CAD (из проекта)", "CAD")
                    has_predef = True

                if 'meshes/scan.stl' in file_list:
                    self.scan_mesh = trimesh.load(io.BytesIO(zf.read('meshes/scan.stl')), file_type='stl')
                    self.show_mesh("Scan", self.scan_mesh)
                    self.add_tree_item(self.ui.cat_scan, "Scan (из проекта)", "Scan")
                    has_predef = True

                if 'meshes/result.stl' in file_list:
                    self.result_mesh = trimesh.load(io.BytesIO(zf.read('meshes/result.stl')), file_type='stl')
                    self.show_mesh("Result", self.result_mesh)
                    self.add_tree_item(self.ui.cat_res, "Result (из проекта)", "Result")
                    has_predef = True

                self.pick_mode = 'CAD'
                for pt in meta.get("cad_pts", []): self.place_marker(pt)
                self.pick_mode = 'Scan'
                for pt in meta.get("scan_pts", []): self.place_marker(pt)
                self.pick_mode = None

                # Восстанавливаем детали слайсера
                slicer_files = [f for f in file_list if f.startswith('meshes/slicer_part_')]
                if slicer_files:
                    self.ui._ensure_slicer_plotter()
                    # Блокируем сигналы таблицы, чтобы галочки не вызывали ошибки при загрузке
                    self.ui.tbl_parts.blockSignals(True)

                    for i, s_file in enumerate(slicer_files):
                        mesh = trimesh.load(io.BytesIO(zf.read(s_file)), file_type='stl')
                        pv_mesh = self.trimesh_to_pyvista(mesh)

                        part_id = len(self.slicer_parts)
                        actor_name = f"slicer_part_{part_id}"

                        if self.ui.slicer_plotter:
                            self.ui.slicer_plotter.add_mesh(pv_mesh, color="#d3d3d3", show_edges=True, name=actor_name)

                        original_name = meta.get(f"slicer_name_{i}", f"Project_Part_{part_id + 1}.stl")

                        self.slicer_parts.append({
                            "mesh": mesh,
                            "mesh_pv": pv_mesh,
                            "filename": original_name,
                            "actor_name": actor_name,
                            "last_visible_mode": "shaded_wire",
                        })

                        row = self.ui.tbl_parts.rowCount()
                        self.ui.tbl_parts.insertRow(row)

                        item_id = QTableWidgetItem(str(row + 1))
                        item_id.setTextAlignment(Qt.AlignCenter)
                        self.ui.tbl_parts.setItem(row, 0, item_id)

                        item_sel = QTableWidgetItem()
                        item_sel.setTextAlignment(Qt.AlignCenter)
                        item_sel.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                        item_sel.setCheckState(Qt.Checked)
                        self.ui.tbl_parts.setItem(row, 1, item_sel)

                        item_vis = QTableWidgetItem()
                        item_vis.setTextAlignment(Qt.AlignCenter)
                        item_vis.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                        item_vis.setCheckState(Qt.Checked)
                        self.ui.tbl_parts.setItem(row, 2, item_vis)

                        self.ui.tbl_parts.setItem(row, 3, QTableWidgetItem("Зат.+каркас"))  # соответствует show_edges=True при add_mesh выше
                        self.ui.tbl_parts.setItem(row, 4, QTableWidgetItem("0%"))
                        self.ui.tbl_parts.setItem(row, 5, QTableWidgetItem("Серый"))
                        self.ui.tbl_parts.setItem(row, 6, QTableWidgetItem("STL"))
                        self.ui.tbl_parts.setItem(row, 7, QTableWidgetItem(original_name))

                    self.ui.lbl_part_count.setText(f"Кол-во деталей: {self.ui.tbl_parts.rowCount()}")
                    if self.ui.slicer_plotter:
                        self.ui.slicer_plotter.reset_camera()
                        self.ui.slicer_plotter.render()

                    self.ui.tbl_parts.blockSignals(False)  # Включаем сигналы обратно

            self.add_to_recent(path)
            self.update_title(os.path.basename(path))

            if self.ui.plotter:
                self.ui.plotter.reset_camera()

            # Умное переключение: если загрузили только Слайсер - открываем его!
            if slicer_files and not has_predef:
                self.ui.stack.setCurrentWidget(self.ui.page_slicer)
            else:
                self.ui.stack.setCurrentWidget(self.ui.page_predef)

            self.log("✅ Проект успешно восстановлен!")

        except zipfile.BadZipFile:
            self.log("[!] ОШИБКА: Файл поврежден или не является архивом .mrp")
        except Exception as e:
            self.log(f"[!] Ошибка: {str(e)}")

    def action_show_donate(self):
        # Импортируем наше новое окно
        from UI_Meshropractor import DialogDonate
        dialog = DialogDonate(self)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash_path = os.path.join(ASSETS_DIR, "splash_screen.jpg")
    if not os.path.exists(splash_path):
        splash_path = os.path.join(ASSETS_DIR, "splash_screen.png")

    original_pixmap = QPixmap(splash_path)

    if original_pixmap.isNull():
        pixmap = QPixmap(600, 350)
        pixmap.fill(QColor("#2b2b2b"))
    else:
        pixmap = original_pixmap.scaled(600, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    font = QFont("Segoe UI", 10)
    splash.setFont(font)
    splash.showMessage("Инициализация ядра 3D-графики...", Qt.AlignBottom | Qt.AlignCenter, QColor("#FFFFFF"))
    app.processEvents()

    window = MainWindow()

    splash.showMessage("Загрузка компонентов интерфейса...", Qt.AlignBottom | Qt.AlignCenter, QColor("#FFFFFF"))
    app.processEvents()
    time.sleep(1.0)

    window.show()
    splash.finish(window)

    sys.exit(app.exec())