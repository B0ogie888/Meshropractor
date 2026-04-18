# Файл: App.py
import sys
import numpy as np
import trimesh

# Импорты интерфейса
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QColorDialog, QTreeWidgetItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor

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

        # 1. ЗАГРУЖАЕМ ИНТЕРФЕЙС
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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
        if dir in ["T", "B"]:
            self.setCursor(Qt.SizeVerCursor)
        elif dir in ["L", "R"]:
            self.setCursor(Qt.SizeHorCursor)
        elif dir in ["TL", "BR"]:
            self.setCursor(Qt.SizeFDiagCursor)
        elif dir in ["TR", "BL"]:
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

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
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()

        if event.buttons() == Qt.NoButton:
            self._update_cursor(self._check_resize_zone(pos))
        elif getattr(self, '_resizing', False):
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

    # === ЛОГИКА КНОПОК ===
    def save_project(self):
        self.log("> Функция сохранения проекта в разработке...")

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
        self.log("Все выбранные точки сброшены.")

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
            if not HAS_O3D:
                self.log("[!] Для Heatmap требуется Open3D.")
                return

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())