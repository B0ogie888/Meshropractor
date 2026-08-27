# Файл: Meshropractor.py
import sys
import time
import numpy as np
import trimesh
import zipfile
import json
import io
import os

# Получаем абсолютный путь к папке, где лежит этот скрипт (Meshropractor.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Указываем путь к папке с картинками
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Импорты интерфейса (Добавили QLabel)
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QColorDialog, QTreeWidgetItem, QToolButton, QLabel, QSplashScreen
from PySide6.QtCore import Qt, QSettings, QSize, QEvent
from PySide6.QtGui import QColor, QFont, QTextCursor, QPixmap, QIcon

import pyvista as pv

# ИМПОРТИРУЕМ НАШИ СОБСТВЕННЫЕ МОДУЛИ
from UI_Meshropractor import Ui_MainWindow
from Workers_Meshropractor import AlignmentThread, CompensationThread, HAS_O3D

if HAS_O3D:
    import open3d as o3d


# ==========================================
# ГЛАВНАЯ ЛОГИКА ОКНА (Контроллер)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Инициализируем хранилище настроек
        self.settings = QSettings("MeshropractorTeam", "Meshropractor")

        # 1. ЗАГРУЖАЕМ ИНТЕРФЕЙС
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --- ВСТРАИВАЕМ ЗАГОЛОВОК В КАСТОМНУЮ ПАНЕЛЬ ---
        self.lbl_app_title = QLabel("Meshropractor - Без названия")
        self.lbl_app_title.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: bold; background: transparent;")
        self.setWindowTitle("Meshropractor - Без названия")

        # Вставляем заголовок ПОСЛЕ иконок (индекс 3) и добавляем вторую пружину (индекс 4)
        self.ui.title_layout.insertWidget(3, self.lbl_app_title)
        self.ui.title_layout.insertStretch(4)

        # 2. ПЕРЕМЕННЫЕ ЛОГИКИ
        self.cad_mesh = None
        self.scan_mesh = None
        self.result_mesh = None
        self.actors = {"CAD": None, "Scan": None, "Result": None, "Heatmap": None}
        self.pick_mode = None
        self.cad_pts = []
        self.scan_pts = []
        self.pt_actors = []

        # 3. ПОДКЛЮЧАЕМ ЛОГИКУ К КНОПКАМ ИНТЕРФЕЙСА
        self.ui.action_save.triggered.connect(self.save_project)
        self.ui.action_undo.triggered.connect(self.undo_action)
        self.ui.action_redo.triggered.connect(self.redo_action)

        # --- Стартовая страница ---
        self.ui.btn_new_project.clicked.connect(self.action_new_project)
        self.ui.btn_open_project.clicked.connect(self.action_open_project)
        self.ui.btn_recent_projects.clicked.connect(self.open_recent_gallery)
        self.ui.btn_back_to_start.clicked.connect(lambda: self.ui.stack.setCurrentWidget(self.ui.page_start))

        self.ui.tree.itemChanged.connect(self.on_tree_visibility_changed)
        self.ui.plotter.add_key_event('space', self.on_space_pressed)

        # Вкладка 1
        self.ui.btn_load_cad.clicked.connect(self.load_cad)
        self.ui.btn_load_scan.clicked.connect(self.load_scan)
        self.ui.btn_pick_cad.clicked.connect(self.start_pick_cad)
        self.ui.btn_pick_scan.clicked.connect(self.start_pick_scan)
        self.ui.btn_clear_pts.clicked.connect(self.clear_picks)
        self.ui.btn_run_icp.clicked.connect(self.run_icp)

        # Вкладка 2
        self.ui.btn_run_comp.clicked.connect(self.run_comp)
        self.ui.btn_save.clicked.connect(self.save_result)

        # --- Вкладка Слайсера ---
        # Подключаем единственную кнопку "Экспорт .CLS" к открытию модального окна
        self.ui.ribbon_btns["Создание срезов Concept Laser"].clicked.connect(self.open_export_dialog)

        # Панель слоев
        self.ui.chk_view_cad.stateChanged.connect(self.update_visibility)
        self.ui.chk_view_scan.stateChanged.connect(self.update_visibility)
        self.ui.chk_view_res.stateChanged.connect(self.update_visibility)

        self.ui.sld_op_cad.valueChanged.connect(lambda v: self.update_opacity("CAD", v))
        self.ui.sld_op_scan.valueChanged.connect(lambda v: self.update_opacity("Scan", v))
        self.ui.sld_op_res.valueChanged.connect(lambda v: self.update_opacity("Result", v))

        self.ui.btn_col_cad.clicked.connect(lambda: self.pick_color("CAD", self.ui.btn_col_cad))
        self.ui.btn_col_scan.clicked.connect(lambda: self.pick_color("Scan", self.ui.btn_col_scan))
        self.ui.btn_col_res.clicked.connect(lambda: self.pick_color("Result", self.ui.btn_col_res))

        # Панель Анализа
        self.ui.btn_heatmap.clicked.connect(self.generate_heatmap)
        self.ui.btn_clear_heat.clicked.connect(self.clear_heatmap)
        self.ui.sliders["heat_limit"][0].valueChanged.connect(self.update_heatmap_limit)

        # Глобальный перехватчик движений мыши, чтобы курсор не залипал
        QApplication.instance().installEventFilter(self)

    # === ЛОГИКА ПЕРЕТАСКИВАНИЯ БЕЗРАМОЧНОГО ОКНА ===
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
        if dir in ["T", "B"]: self.setCursor(Qt.SizeVerCursor)
        elif dir in ["L", "R"]: self.setCursor(Qt.SizeHorCursor)
        elif dir in ["TL", "BR"]: self.setCursor(Qt.SizeFDiagCursor)
        elif dir in ["TR", "BL"]: self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.unsetCursor() # <--- Сбрасываем курсор окна на стандартный

    def eventFilter(self, obj, event):
        # Глобально ловим движение мыши, где бы она ни находилась
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

        # Оставили здесь только логику физического изменения размера и перетаскивания окна
        if getattr(self, '_resizing', False):
            dx = global_pos.x() - self._start_mouse_pos.x()
            dy = global_pos.y() - self._start_mouse_pos.y()
            x, y, w, h = self._start_geometry.getRect()

            if 'L' in self._resize_dir: w -= dx; x += dx
            elif 'R' in self._resize_dir: w += dx
            if 'T' in self._resize_dir: h -= dy; y += dy
            elif 'B' in self._resize_dir: h += dy

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
        """Корректно закрываем обе 3D-сцены перед выходом"""
        try:
            if hasattr(self, 'ui') and hasattr(self.ui, 'plotter'):
                self.ui.plotter.close()
            if hasattr(self, 'slicer_plotter'):
                self.slicer_plotter.close()
        except Exception:
            pass
        event.accept()

    # === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ЛОГИКИ ===
    def log(self, text, replace=False):
        if text.startswith("REPLACE_FLAG"):
            cursor = self.ui.console.textCursor()
            cursor.movePosition(cursor.End)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text.replace("REPLACE_FLAG", ""))
        else:
            self.ui.console.append(text)
        self.ui.console.verticalScrollBar().setValue(self.ui.console.verticalScrollBar().maximum())

    def pick_color(self, key, btn):
        initial_color = QColor(self.ui.mesh_colors[key])
        color = QColorDialog.getColor(initial_color, self, f"Выберите цвет для {key}")
        if color.isValid():
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
        if self.actors["CAD"]: self.actors["CAD"].SetVisibility(self.ui.chk_view_cad.isChecked())
        if self.actors["Scan"]: self.actors["Scan"].SetVisibility(self.ui.chk_view_scan.isChecked())
        if self.actors["Result"]: self.actors["Result"].SetVisibility(self.ui.chk_view_res.isChecked())
        self.ui.plotter.render()

    def update_opacity(self, key, value):
        if self.actors[key]:
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
        if actor_key and self.actors.get(actor_key):
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

    def start_pick_cad(self):
        if not self.cad_mesh: return self.log("[!] Сначала загрузите CAD!")
        self.pick_mode = 'CAD'
        self.ui.chk_view_cad.setChecked(True)
        self.ui.chk_view_scan.setChecked(False)
        self.ui.sld_op_cad.setValue(100)
        self.log("\n[РЕЖИМ CAD] Наведите курсор на деталь и нажмите ПРОБЕЛ.")

    def start_pick_scan(self):
        if not self.scan_mesh: return self.log("[!] Сначала загрузите Скан!")
        self.pick_mode = 'Scan'
        self.ui.chk_view_cad.setChecked(False)
        self.ui.chk_view_scan.setChecked(True)
        self.ui.sld_op_scan.setValue(100)
        self.log("\n[РЕЖИМ СКАНА] Наведите курсор на деталь и нажмите ПРОБЕЛ.")

    def on_space_pressed(self):
        if not self.pick_mode: return
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
        self.log("\n>>> Расчет цветовой карты...")
        self.ui.btn_heatmap.setEnabled(False)
        try:
            if not HAS_O3D: return self.log("[!] Для Heatmap требуется Open3D.")
            cad_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(self.cad_mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.array(self.cad_mesh.faces, dtype=np.int32))
            )
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
        if self.actors.get("Heatmap") and hasattr(self.actors["Heatmap"].mapper, 'dataset'):
            limit = self.ui.sliders["heat_limit"][0].value() / self.ui.sliders["heat_limit"][1]
            self.actors["Heatmap"].mapper.scalar_range = [-limit, limit]
            self.ui.plotter.render()

    def clear_heatmap(self):
        if self.actors.get("Heatmap"):
            self.ui.plotter.remove_actor(self.actors["Heatmap"])
            self.actors["Heatmap"] = None
        self.ui.chk_view_scan.setChecked(True)
        self.ui.chk_view_cad.setChecked(True)
        self.log("Отображение сброшено в базовый режим.")

    def run_comp(self):
        if not self.cad_mesh or not self.scan_mesh: return
        self.ui.btn_run_comp.setEnabled(False)
        self.ui.btn_run_comp.setText("⏳ ИДЕТ РАСЧЕТ МАТРИЦ...")

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
        self.comp_thread.finished_signal.connect(self.on_comp_done)
        self.comp_thread.start()

    def on_comp_done(self, result_mesh):
        self.result_mesh = result_mesh
        self.show_mesh("Result", self.result_mesh)
        self.add_tree_item(self.ui.cat_res, "Compensated_Part.stl", "Result")
        if self.ui.cat_scan.childCount() > 0:
            self.ui.cat_scan.child(0).setCheckState(0, Qt.Unchecked)
        self.ui.btn_run_comp.setEnabled(True)
        self.ui.btn_run_comp.setText("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.ui.btn_save.setEnabled(True)

    def save_result(self):
        if self.result_mesh:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить", "Compensated_Part.stl", "STL Files (*.stl)")
            if path:
                self.result_mesh.export(path)
                self.log(f"✅ Успешно сохранено: {path}")

    # ==========================================
    # ЛОГИКА СЛАЙСЕРА
    # ==========================================
    def open_export_dialog(self):
        """Открывает окно параметров Concept Laser"""
        from UI_Meshropractor import DialogExportCLS
        dialog = DialogExportCLS(self)

        # Если пользователь нажал "Да"
        if dialog.exec():
            self.log("\n>>> Окно экспорта подтверждено. Скоро здесь будет запуск нарезки .CLS с новыми параметрами!")

    # ==========================================
    # ПАМЯТЬ, ОЧИСТКА И НЕДАВНИЕ ПРОЕКТЫ
    # ==========================================
    def update_title(self, project_name=None):
        """Обновляет заголовок и в панели задач, и на кастомной рамке"""
        title = f"Meshropractor - {project_name}" if project_name else "Meshropractor - Без названия"
        self.setWindowTitle(title)
        self.lbl_app_title.setText(title)

    def clear_project_data(self):
        """Полностью очищает память и 3D-сцену для нового проекта"""
        self.clear_picks()
        self.cad_mesh = None
        self.scan_mesh = None
        self.result_mesh = None
        self.ui.plotter.clear()
        self.ui.plotter.add_axes()
        self.ui.cat_cad.takeChildren()
        self.ui.cat_scan.takeChildren()
        self.ui.cat_res.takeChildren()

        # Сбрасываем слайсер
        if hasattr(self, 'slicer_part_path'):
            self.slicer_part_path = ""
            self.ui.lbl_slicer_part.setText("Деталь: Не выбрана")
        if hasattr(self, 'slicer_supp_path'):
            self.slicer_supp_path = ""
            self.ui.lbl_slicer_supp.setText("Поддержки: Не выбраны (Опционально)")

        self.log("\n[i] Память и 3D-сцена очищены.")

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
                col = 0
                row += 1

    def save_project(self):
        if not self.cad_mesh and not self.scan_mesh:
            self.log("[!] ОШИБКА: Проект пуст, нечего сохранять.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "New_Project.mrp",
                                              "MeshRopractor Project (*.mrp)")
        if not path: return

        self.log(f"\n⏳ Сохранение проекта в {path}...")
        try:
            img_array = self.ui.plotter.screenshot(transparent_background=False)
            from PIL import Image
            img = Image.fromarray(img_array)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if self.cad_mesh: zf.writestr('meshes/cad.stl', self.cad_mesh.export(file_type='stl'))
                if self.scan_mesh: zf.writestr('meshes/scan.stl', self.scan_mesh.export(file_type='stl'))
                if self.result_mesh: zf.writestr('meshes/result.stl', self.result_mesh.export(file_type='stl'))

                zf.writestr('preview.png', img_bytes)

                meta = {
                    "version": "1.1",
                    "cad_pts": [list(p) for p in self.cad_pts],
                    "scan_pts": [list(p) for p in self.scan_pts]
                }
                zf.writestr('project.json', json.dumps(meta, indent=4))

            self.add_to_recent(path)
            self.update_title(os.path.basename(path))  # <-- ОБНОВЛЯЕМ НАЗВАНИЕ ПОСЛЕ СОХРАНЕНИЯ
            self.log("✅ Проект успешно сохранен!")
        except Exception as e:
            self.log(f"[!] Ошибка при сохранении: {str(e)}")

    def action_new_project(self):
        # ОЧИЩАЕМ СЦЕНУ ПЕРЕД НОВЫМ ПРОЕКТОМ
        self.clear_project_data()
        self.update_title(None)
        self.ui.stack.setCurrentWidget(self.ui.page_predef)
        self.log("\n>>> Создан новый проект. Загрузите CAD и Скан для начала работы.")

    def action_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "MeshRopractor (*.mrp)")
        if path: self.load_mrp_file(path)

    def load_mrp_file(self, path):
        self.log(f"\n⏳ Открытие проекта {path}...")
        self.ui.stack.setCurrentWidget(self.ui.page_predef)

        # ОЧИЩАЕМ СЦЕНУ ПЕРЕД ЗАГРУЗКОЙ НОВОГО
        self.clear_project_data()

        try:
            with zipfile.ZipFile(path, 'r') as zf:
                file_list = zf.namelist()
                meta = json.loads(zf.read('project.json').decode('utf-8')) if 'project.json' in file_list else {}

                if 'meshes/cad.stl' in file_list:
                    self.cad_mesh = trimesh.load(io.BytesIO(zf.read('meshes/cad.stl')), file_type='stl')
                    self.show_mesh("CAD", self.cad_mesh)
                    self.add_tree_item(self.ui.cat_cad, "CAD (из проекта)", "CAD")

                if 'meshes/scan.stl' in file_list:
                    self.scan_mesh = trimesh.load(io.BytesIO(zf.read('meshes/scan.stl')), file_type='stl')
                    self.show_mesh("Scan", self.scan_mesh)
                    self.add_tree_item(self.ui.cat_scan, "Scan (из проекта)", "Scan")

                if 'meshes/result.stl' in file_list:
                    self.result_mesh = trimesh.load(io.BytesIO(zf.read('meshes/result.stl')), file_type='stl')
                    self.show_mesh("Result", self.result_mesh)
                    self.add_tree_item(self.ui.cat_res, "Result (из проекта)", "Result")

                self.pick_mode = 'CAD'
                for pt in meta.get("cad_pts", []): self.place_marker(pt)
                self.pick_mode = 'Scan'
                for pt in meta.get("scan_pts", []): self.place_marker(pt)
                self.pick_mode = None

            self.add_to_recent(path)
            self.update_title(os.path.basename(path))  # <-- ОБНОВЛЯЕМ НАЗВАНИЕ ПОСЛЕ ЗАГРУЗКИ
            self.ui.plotter.reset_camera()
            self.log("✅ Проект успешно восстановлен!")

        except zipfile.BadZipFile:
            self.log("[!] ОШИБКА: Файл поврежден или не является архивом .mrp")
        except Exception as e:
            self.log(f"[!] Ошибка: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Загружаем картинку для сплеш-скрина
    splash_path = os.path.join(ASSETS_DIR, "splash_screen.jpg")
    original_pixmap = QPixmap(splash_path)

    # Сжимаем картинку до классического "инженерного" размера (как в SolidWorks)
    # Qt.SmoothTransformation гарантирует, что текст и сетка не станут пиксельными
    pixmap = original_pixmap.scaled(600, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # Если картинка не найдена, создаем темный фон-заглушку
    if pixmap.isNull():
        pixmap = QPixmap(600, 350)
        pixmap.fill(QColor("#2b2b2b"))

    # Создаем и показываем сплеш-скрин
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    # Настраиваем шрифт для текста загрузки
    font = QFont("Segoe UI", 10)
    splash.setFont(font)

    # Имитируем процесс загрузки (чтобы текст успел отрисоваться)
    splash.showMessage("Инициализация ядра 3D-графики...", Qt.AlignBottom | Qt.AlignCenter, QColor("#FFFFFF"))
    app.processEvents()

    # --- Создание главного окна ---
    # (В этот момент программа "задумывается", загружая UI и PyVista)
    window = MainWindow()

    splash.showMessage("Загрузка компонентов интерфейса...", Qt.AlignBottom | Qt.AlignCenter, QColor("#FFFFFF"))
    app.processEvents()

    # Небольшая пауза, чтобы пользователь успел увидеть логотип (можно убрать)
    time.sleep(1.0)

    # Показываем основное окно и закрываем загрузочный экран
    window.show()
    splash.finish(window)

    sys.exit(app.exec())