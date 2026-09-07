"""Project lifecycle and background operation coordination for the desktop UI."""
from copy import deepcopy
import os
import io
from uuid import uuid4

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QCheckBox, QTableWidgetItem, QWidget, QHBoxLayout, QPushButton, QInputDialog

from background_tasks import FunctionWorker
from project_store import ProjectState, load_project, save_project, load_mesh


from project_history import HistoryMixin
from slicer_tools import SlicerToolsMixin


class ProjectController(HistoryMixin, SlicerToolsMixin, QMainWindow):
    def init_project_controller(self):
        self.scene_models = {}
        self.active_result_key = None
        self.active_heatmap_key = None
        self.callout_records = []
        self.project_path = None
        self.dirty = False
        self._restoring = False
        self._job = None
        self._job_generation = 0
        self._generation = 0
        self._closing = False
        self._after_save = None
        self._job_callback = None
        self._job_completed = False
        self.pick_mode = None
        self._job_previous_enabled = []
        self._pickability = {}
        self.ui.section_panel.bind(self)
        self.ui.btn_cancel_align.clicked.connect(self.cancel_current_job)
        for control in (self.ui.cb_def_type, self.ui.cb_samples, self.ui.sb_points,
                        self.ui.sb_factor, self.ui.sb_factor_z, self.ui.chk_link_factor,
                        self.ui.cb_search_time, self.ui.chk_icp, self.ui.sb_align_tolerance, self.ui.sb_align_coverage, self.ui.sb_max_deviation,
                        self.ui.sb_min_coverage):
            signal = next(getattr(control, name) for name in ("valueChanged", "currentIndexChanged", "toggled") if hasattr(control, name))
            signal.connect(self.mark_dirty)
        self.ui.sliders["heat_limit"][0].valueChanged.connect(self.mark_dirty)
        self.ui.tbl_res.cellClicked.connect(self.select_result)
        self.ui.tbl_heat.cellClicked.connect(self.select_heatmap)
        self.ui.action_save.setShortcut("Ctrl+S")

    def mark_dirty(self, *args):
        if not getattr(self, "_restoring", True):
            self.dirty = True
            self.update_title()
            self.queue_history()

    def update_title(self, project_name=None):
        name = project_name or (os.path.basename(self.project_path) if self.project_path else "Без названия")
        title = f"Meshropractor — {name}{' *' if self.dirty else ''}"
        self.setWindowTitle(title)
        self.lbl_app_title.setText(title)

    def _busy(self):
        if getattr(self, '_transform_session', None) is not None and not getattr(self, '_applying_transform', False):
            self.log("Завершите преобразование в открытом окне инструмента.")
            return True
        if self._job is not None:
            self.log("[i] Дождитесь завершения текущей операции или отмените расчёт.")
            return True
        return False

    def start_job(self, worker, callback):
        if self._busy():
            return False
        self.flush_history()
        self._job = worker
        if hasattr(self, 'workspace_tools'):
            self.workspace_tools.manual = None
            self.workspace_tools.measurements.stop()
            self.workspace_tools.cancel_button.show()
        self.update_history_actions()
        self.pick_mode = None
        self._job_callback = callback
        self._job_generation = self._generation
        self._job_completed = False
        # Keep navigation and cancellation available; freeze all project mutations.
        controls = [self.ui.action_save, self.ui.btn_new_project, self.ui.btn_open_project,
                    self.ui.btn_load_cad, self.ui.btn_load_scan, self.ui.btn_run_icp,
                    self.ui.btn_run_def, self.ui.btn_run_comp, self.ui.btn_heatmap,
                    self.ui.btn_clear_heat, self.ui.btn_save, self.ui.btn_pick_cad,
                    self.ui.btn_pick_scan, self.ui.btn_clear_pts, self.ui.tbl_parts,
                    self.ui.tbl_cad, self.ui.tbl_scan, self.ui.tbl_res, self.ui.tbl_heat,
                    self.ui.chk_preview_pts, self.ui.chk_show_vectors, self.ui.chk_callouts,
                    self.ui.btn_clear_callouts, self.ui.section_panel]
        controls += list(self.ui.ribbon_btns.values()) + list(self.ui.position_buttons.values())
        if hasattr(self.ui, 'surface_toolbar'): controls.append(self.ui.surface_toolbar)
        controls.append(self.ui.measurement_panel)
        if hasattr(self, 'workspace_tools') and self.workspace_tools.supports.panel:
            controls.append(self.workspace_tools.supports.panel)
        controls += [self.ui.cb_def_type, self.ui.cb_samples, self.ui.sb_points_wrapper,
                     self.ui.sb_factor, self.ui.sb_factor_z, self.ui.chk_link_factor,
                     self.ui.cb_search_time, self.ui.chk_icp, self.ui.sb_align_tolerance, self.ui.sb_align_coverage, self.ui.sb_max_deviation,
                     self.ui.sb_min_coverage, self.ui.sliders["heat_limit"][0]]
        self._job_previous_enabled = [(control, control.isEnabled()) for control in dict.fromkeys(controls)]
        for control, _ in self._job_previous_enabled:
            control.setEnabled(False)
        self.ui.section_panel._make_widget()
        result_signal = worker.result if isinstance(worker, FunctionWorker) else worker.finished_signal
        result_signal.connect(self._receive_job_result)
        worker.error.connect(self._receive_job_error)
        worker.finished.connect(self._finish_job)
        if hasattr(worker, "log_signal"):
            worker.log_signal.connect(self.log)
        self.ui.btn_cancel_align.setEnabled(worker.__class__.__name__ == "AlignmentThread")
        self.ui.status_label.setText("Выполняется операция…")
        worker.start()
        return True

    @Slot(object)
    def _receive_job_result(self, value):
        if self.sender() is self._job and self._generation == self._job_generation and not self._job.isInterruptionRequested():
            try:
                self._job_callback(value)
                self._job_completed = True
            except Exception as exc:
                self.log(f"[!] Не удалось применить результат: {exc}")

    @Slot(str)
    def _receive_job_error(self, message):
        self.log(f"[!] {message}")
        self.ui.status_label.setText(f"Ошибка: {message}")

    @Slot()
    def _finish_job(self):
        if self.sender() is not self._job:
            return
        worker = self._job
        completed = self._job_completed
        self._job = None
        if hasattr(self, 'workspace_tools'): self.workspace_tools.cancel_button.hide()
        self._job_callback = None
        for control, enabled in self._job_previous_enabled:
            control.setEnabled(enabled)
        self.ui.section_panel._make_widget()
        self._job_previous_enabled = []
        self.ui.def_stack.setCurrentIndex(0)
        self.ui.comp_stack.setCurrentIndex(0)
        self.ui.btn_cancel_def.setEnabled(True)
        self.ui.btn_cancel_comp.setEnabled(True)
        self.ui.btn_run_icp.setText("▶ ВЫПОЛНИТЬ ВЫРАВНИВАНИЕ")
        self.ui.btn_cancel_align.setEnabled(False)
        self.ui.btn_save.setEnabled(self.result_mesh is not None)
        self.ui.sb_points_wrapper.setEnabled(self.ui.cb_samples.currentIndex() == 0)
        self.ui.sb_factor_z.setEnabled(not self.ui.chk_link_factor.isChecked())
        if not completed:
            self.ui.lbl_rmse.setText("—")
        self.ui.status_label.setText("Готово" if completed else "Операция остановлена или завершилась ошибкой; подробности в журнале")
        self.flush_history()
        worker.deleteLater()
        continuation, self._after_save = self._after_save, None
        if self._closing:
            self._closing = False
            QTimer.singleShot(0, self.close)
        elif continuation and completed:
            QTimer.singleShot(0, continuation)

    def cancel_def(self):
        self.cancel_current_job()

    def cancel_comp(self):
        self.cancel_current_job()

    def cancel_current_job(self):
        if self._job:
            self._job.requestInterruption()
            self.ui.btn_cancel_align.setEnabled(False)
            self.ui.btn_cancel_def.setEnabled(False)
            self.ui.btn_cancel_comp.setEnabled(False)
            self.ui.status_label.setText("Отмена: ожидается завершение текущего шага…")
            self.log("[i] Запрошена безопасная отмена. Ожидается завершение текущего шага.")

    def closeEvent(self, event):
        session = getattr(self, '_transform_session', None)
        if session is not None:
            session.dialog.reject()
        if self._job:
            event.ignore()
            self._closing = True
            # A save is allowed to finish; calculations/imports may be cancelled.
            if not (isinstance(self._job, FunctionWorker) and self._job.function is save_project):
                self.cancel_current_job()
            return
        if not self._confirm_discard(self.close):
            event.ignore()
            return
        self._history_timer.stop()
        if hasattr(self, 'workspace_tools') and self.workspace_tools.cube:
            self.workspace_tools.cube.dispose()
        for plotter in (self.ui.plotter, self.ui.slicer_plotter):
            if plotter is not None:
                plotter.close()
        event.accept()

    def _confirm_discard(self, continuation):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "Несохранённые изменения", "Сохранить изменения текущего проекта?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Save:
            self.save_project(after_save=continuation)
            return False
        return answer == QMessageBox.Discard

    def calculation_settings(self, is_comp):
        xy = self.ui.sb_factor.value()
        return dict(def_type=self.ui.cb_def_type.currentIndex(),
                    points=0 if self.ui.cb_samples.currentIndex() == 1 else self.ui.sb_points.value(),
                    factor=(xy if self.ui.chk_link_factor.isChecked() else [xy, xy, self.ui.sb_factor_z.value()]) if is_comp else 1.0,
                    is_comp=is_comp, limit=self.ui.sb_max_deviation.value(),
                    min_coverage=self.ui.sb_min_coverage.value() / 100.0, seed=42)

    def _style_for(self, table, row, key=None):
        def checked(col):
            container = table.cellWidget(row, col)
            return container.findChild(QCheckBox).isChecked() if container else True
        if key:
            style = dict(self.def_actors_meta.get(key, {}))
            style["color"] = self.ui.mesh_colors.get(key, "#d3d3d3")
        else:
            part = self.slicer_parts[row]
            actor = self.ui.slicer_plotter.actors.get(part["actor_name"])
            style = dict(last_visible_mode=part.get("last_visible_mode", "shaded_wire"),
                         transparency=int(table.item(row, 4).text().rstrip("%")),
                         color=list(actor.GetProperty().GetColor()) if actor else "#d3d3d3")
        style.update(is_selected=checked(1), is_visible=checked(2))
        return style

    def capture_project(self):
        state = ProjectState(platforms=deepcopy(self.platforms), cad_pts=np.asarray(self.cad_pts).tolist(), scan_pts=np.asarray(self.scan_pts).tolist(),
                             callouts=deepcopy(self.callout_records), active_result=self.active_result_key,
                             active_heatmap=self.active_heatmap_key,
                             page="slicer" if self.ui.stack.currentWidget() is self.ui.page_slicer else "predef")
        for table in (self.ui.tbl_cad, self.ui.tbl_scan, self.ui.tbl_res, self.ui.tbl_heat):
            for row in range(table.rowCount()):
                key = table.item(row, 0).data(Qt.UserRole)
                record = deepcopy(self.scene_models[key])
                record["name"] = table.item(row, 6).text()
                record["style"] = self._style_for(table, row, key)
                state.models.append(record)
        for row, part in enumerate(self.slicer_parts):
            state.parts.append(dict(mesh=part["mesh"].copy(), filename=part["filename"], platform=part.get("platform"),
                                    style=self._style_for(self.ui.tbl_parts, row), supports=deepcopy(part.get('supports', []))))
        state.sections = self.ui.section_panel.snapshot()
        state.settings = dict(def_type=self.ui.cb_def_type.currentIndex(), samples=self.ui.cb_samples.currentIndex(),
                              points=self.ui.sb_points.value(), factor_xy=self.ui.sb_factor.value(), factor_z=self.ui.sb_factor_z.value(),
                              linked=self.ui.chk_link_factor.isChecked(), search_time=self.ui.cb_search_time.currentIndex(),
                              icp=self.ui.chk_icp.isChecked(), align_tolerance=self.ui.sb_align_tolerance.value(), align_coverage=self.ui.sb_align_coverage.value(), limit=self.ui.sb_max_deviation.value(),
                              min_coverage=self.ui.sb_min_coverage.value(), heat_limit=self.ui.sliders["heat_limit"][0].value())
        return state

    def save_project(self, checked=False, *, save_as=False, after_save=None):
        if self._busy():
            return False
        path = self.project_path if not save_as else None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", self.project_path or "New_Project.mrp", "Meshropractor (*.mrp)")
        if not path:
            return False
        if not path.lower().endswith(".mrp"):
            path += ".mrp"
        try:
            state = self.capture_project()
            preview = b""
            plotter = self.ui.slicer_plotter if self.ui.stack.currentWidget() is self.ui.page_slicer else self.ui.plotter
            if plotter is not None and hasattr(plotter, "screenshot"):
                try:
                    from PIL import Image
                    image = Image.fromarray(plotter.screenshot(return_img=True))
                    image.thumbnail((400, 400))
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    preview = stream.getvalue()
                except Exception as exc:
                    self.log(f"[i] Сохранение без миниатюры: {exc}")
        except Exception as exc:
            self.log(f"[!] Не удалось подготовить проект: {exc}")
            return False
        def saved(_):
            self.history.saved = self.history.key
            self.project_path = path
            self.dirty = False
            self.update_title()
            self.add_to_recent(path)
            self.log("✅ Проект сохранён со всеми результатами и настройками.")
        self._after_save = after_save
        return self.start_job(FunctionWorker(save_project, path, state, preview), saved)

    def action_new_project(self):
        if self._busy():
            return
        if not self._confirm_discard(self.action_new_project):
            return
        from UI_Meshropractor import DialogNewProject
        dialog = DialogNewProject(self)
        if not dialog.exec():
            return
        self.clear_project_data()
        self.project_path = None
        self.dirty = False
        self.update_title()
        self.ui.stack.setCurrentWidget(self.ui.page_slicer if dialog.selected_mode == "slicer" else self.ui.page_predef)
        self.reset_history()

    def action_open_project(self):
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Meshropractor (*.mrp)")
        if path:
            self.load_mrp_file(path)

    def load_mrp_file(self, path):
        if self._busy() or not self._confirm_discard(lambda: self.load_mrp_file(path)):
            return
        def loaded(state):
            previous = self.capture_project()
            old_dirty, old_path = self.dirty, self.project_path
            try:
                self.restore_project(state)
            except Exception:
                self.restore_project(previous)
                self.dirty, self.project_path = old_dirty, old_path
                self.update_title()
                raise
            self.project_path = path
            self.dirty = False
            self.update_title()
            self.add_to_recent(path)
            self.reset_history()
            self.log("✅ Проект восстановлен.")
        self.start_job(FunctionWorker(load_project, path), loaded)

    def restore_project(self, state):
        self._restoring = True
        try:
            self.clear_project_data()
            self._restoring = True
            self.platforms = deepcopy(state.platforms)
            for platform in self.platforms:
                platform.setdefault("id", str(uuid4()))
            self.update_platform_ui(draw=False)
            if state.models:
                self.ui._ensure_def_plotter()
            for record in state.models:
                kind = record["kind"]
                table = {"CAD": self.ui.tbl_cad, "Scan": self.ui.tbl_scan, "Heatmap": self.ui.tbl_heat}.get(kind, self.ui.tbl_res)
                key = self.add_def_table_item(table, record["name"], kind, actor_key=record["key"])
                if kind == "Heatmap":
                    self._render_heatmap(key, record["mesh"], record["deviations"])
                else:
                    self.show_mesh(key, record["mesh"])
                self.scene_models[key] = deepcopy(record)
                if kind == "CAD": self.cad_mesh = record["mesh"]
                if kind == "Scan": self.scan_mesh = record["mesh"]
                self._restore_style(table, table.rowCount() - 1, record.get("style", {}), key)
            self.active_result_key = state.active_result
            self.result_mesh = self.scene_models.get(state.active_result, {}).get("mesh")
            self.update_result_quality()
            self.update_alignment_quality()
            self.active_heatmap_key = state.active_heatmap
            self._activate_heatmap(state.active_heatmap)
            self.def_count = sum(r["kind"] == "Def" for r in state.models)
            self.comp_count = sum(r["kind"] == "Result" for r in state.models)
            self.heat_count = sum(r["kind"] == "Heatmap" for r in state.models)
            for part in state.parts:
                self._append_slicer_part(part["mesh"], part["filename"], part.get("platform"), part.get("style", part), part.get('supports', []))
            for kind, points in (("CAD", state.cad_pts), ("Scan", state.scan_pts)):
                self.pick_mode = kind
                for point in points:
                    self.place_marker(point)
            self.pick_mode = None
            self.ui.section_panel.restore(state.sections)
            settings = state.settings
            for key, control, default in (("def_type", self.ui.cb_def_type, 1), ("samples", self.ui.cb_samples, 3), ("search_time", self.ui.cb_search_time, 1)):
                control.setCurrentIndex(settings.get(key, default))
            self.ui.chk_link_factor.setChecked(settings.get("linked", True))
            self.ui.chk_icp.setChecked(settings.get("icp", True))
            for key, control, default in (("align_tolerance", self.ui.sb_align_tolerance, 0), ("align_coverage", self.ui.sb_align_coverage, 30), ("points", self.ui.sb_points, 20000), ("factor_xy", self.ui.sb_factor, 1.0),
                                          ("factor_z", self.ui.sb_factor_z, 1.0), ("limit", self.ui.sb_max_deviation, 5.0),
                                          ("min_coverage", self.ui.sb_min_coverage, 30.0),
                                          ("heat_limit", self.ui.sliders["heat_limit"][0], 100)):
                control.setValue(settings.get(key, default))
            self.callout_records = []
            for callout in state.callouts:
                if callout["key"] in self.scene_models:
                    self._activate_heatmap(callout["key"])
                    self.add_heatmap_callout(callout["point"])
            self._activate_heatmap(state.active_heatmap)
            self.ui.stack.setCurrentWidget(self.ui.page_slicer if state.page == "slicer" else self.ui.page_predef)
            self.update_info_combobox()
            self.refresh_scene_visibility()
            self.update_parts_table_filter(self.ui.scene_tabs.currentIndex())
            self._sync_heatmap_legend()
        finally:
            self._restoring = False

    def _restore_style(self, table, row, style, key=None):
        if key:
            self.def_actors_meta[key].update(style)
            color = style.get("color", "#d3d3d3")
            self.ui.mesh_colors[key] = color
            if not key.startswith("Heatmap"):
                self.actors[key].prop.color = color
            button = table.cellWidget(row, 5).findChild(QPushButton)
            button.setStyleSheet(f"background-color: {color}; border: 1px solid #555;")
            self._apply_def_display_mode(table, row, key, style.get("last_visible_mode", "triangles"))
            self._apply_def_transparency(table, row, key, int(style.get("transparency", 0)))
        table.cellWidget(row, 1).findChild(QCheckBox).setChecked(style.get("is_selected", True))
        table.cellWidget(row, 2).findChild(QCheckBox).setChecked(style.get("is_visible", True))

    def _append_slicer_part(self, mesh, filename, platform=None, style=None, supports=None):
        self.ui._ensure_slicer_plotter()
        style = style or {}
        row = len(self.slicer_parts)
        key = f"slicer_part_{row}"
        pv_mesh = self.trimesh_to_pyvista(mesh)
        color = style.get("color", "#d3d3d3")
        self.ui.slicer_plotter.add_mesh(pv_mesh, color=color, name=key)
        transparency = int(str(style.get("transparency", 0)).rstrip("%"))
        self.slicer_parts.append(dict(mesh=mesh, mesh_pv=pv_mesh, filename=filename, actor_name=key,
                                     platform=platform, last_visible_mode=style.get("last_visible_mode", "shaded_wire"),
                                     supports=deepcopy(supports if supports is not None else style.get('_supports', []))))
        table = self.ui.tbl_parts
        table.insertRow(row)
        for col, value in ((0, str(row + 1)), (3, "Зат.+каркас"), (4, f"{transparency}%"), (6, filename)):
            item = QTableWidgetItem(value)
            if col != 6: item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, col, item)
        for col, checked in ((1, style.get("is_selected", True)), (2, style.get("is_visible", True))):
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            checkbox = QCheckBox()
            checkbox.setChecked(checked)
            layout.addWidget(checkbox)
            table.setCellWidget(row, col, container)
            if col == 2: checkbox.toggled.connect(lambda value, r=row: self._on_vis_checkbox_changed(r, value))
            checkbox.toggled.connect(self.mark_dirty)
        button = QPushButton()
        button.setFixedSize(24, 24)
        button.setStyleSheet(f"background-color: {pv.Color(color).hex_rgb}; border: 1px solid #555;")
        button.clicked.connect(lambda checked=False, r=row, b=button: self.pick_slicer_part_color(r, b))
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(button)
        table.setCellWidget(row, 5, container)
        if getattr(self, '_slicer_batch', False): return
        self.update_info_combobox()
        self.refresh_scene_visibility()
        self.update_parts_table_filter(self.ui.scene_tabs.currentIndex())
        self.mark_dirty()

    def import_slicer_part(self):
        if self._busy(): return
        path, _ = QFileDialog.getOpenFileName(self, "Импорт детали", "", "Модели (*.stl *.step *.stp);;STL (*.stl);;STEP (*.step *.stp)")
        if not path: return
        precision = self._step_import_precision(path)
        if precision is None: return
        index = self.ui.scene_tabs.currentIndex()
        platforms = [p for p in self.platforms if p["is_default"]]
        platform = platforms[index - 1]["name"] if 0 < index <= len(platforms) else None
        self.start_job(FunctionWorker(load_mesh, path, *precision), lambda mesh: self._append_slicer_part(mesh, os.path.basename(path), platform))

    def _step_import_precision(self, path):
        if not path.lower().endswith((".step", ".stp")):
            return 0.05, 0.25
        import math
        from import_dialog import StepImportDialog
        dialog = StepImportDialog(self)
        if not dialog.exec(): return None
        linear, angle = dialog.values()
        self.log(f"[i] Импорт STEP: отклонение {linear:g} мм, угол {math.degrees(angle):g}°, единицы — мм.")
        return linear, angle

    def load_cad(self):
        self._import_model("CAD")

    def load_scan(self):
        self._import_model("Scan")

    def _import_model(self, kind):
        if self._busy(): return
        file_filter = "Модели (*.stl *.step *.stp);;STL (*.stl);;STEP (*.step *.stp)" if kind == "CAD" else "STL (*.stl)"
        path, _ = QFileDialog.getOpenFileName(self, f"Загрузить {kind}", "", file_filter)
        if not path: return
        precision = self._step_import_precision(path)
        if precision is None: return
        def loaded(mesh):
            self.ui._ensure_def_plotter()
            self.clear_picks()
            self.ui.chk_show_vectors.setChecked(False)
            self.ui.chk_preview_pts.setChecked(False)
            table = self.ui.tbl_cad if kind == "CAD" else self.ui.tbl_scan
            if kind == "CAD": self.cad_mesh = mesh
            else: self.scan_mesh = mesh
            key = self.add_def_table_item(table, os.path.basename(path), kind, clear_table=True)
            self.show_mesh(key, mesh)
            self.mark_dirty()
        self.start_job(FunctionWorker(load_mesh, path, *precision), loaded)

    def update_alignment_quality(self):
        quality = self.scan_mesh.metadata.get("alignment", {}) if self.scan_mesh is not None else {}
        if all(key in quality for key in ("rmse", "coverage", "tolerance", "p95")):
            self.ui.lbl_rmse.setText(f"{quality['rmse']:.4f} mm")
            warning = " · Проверьте ориентацию по маркерам" if quality.get("ambiguous") or quality.get("weak_geometry") else ""
            self.ui.lbl_align_quality.setText(f"Площадь скана в допуске: {quality['coverage']:.1%} · Допуск: {quality['tolerance']:.4g} мм · P95: {quality['p95']:.4g} мм{warning}")
        else:
            self.ui.lbl_rmse.setText("—")
            self.ui.lbl_align_quality.setText("Площадь в допуске и P95 появятся после совмещения.")

    def select_result(self, row, column):
        item = self.ui.tbl_res.item(row, 0)
        if item:
            key = item.data(Qt.UserRole)
            if key in self.scene_models:
                self.active_result_key = key
                self.result_mesh = self.scene_models[key]["mesh"]
                self.update_result_quality()
                if self.ui.chk_show_vectors.isChecked(): self.toggle_vector_field()
                self.mark_dirty()

    def update_result_quality(self):
        if self.result_mesh is None:
            self.ui.lbl_result_quality.setText("Выберите результат для экспорта и просмотра показателей")
            return
        quality = self.result_mesh.metadata.get("quality", {})
        if not quality:
            self.ui.lbl_result_quality.setText("Для этого результата показатели качества не сохранены")
            return
        self.ui.lbl_result_quality.setText(
            f"Покрытие: {quality.get('coverage_percent', 0):.1f}% · "
            f"Контрольная RMSE: {quality.get('validation_rmse_mm', 0):.4f} мм\n"
            f"Макс. расстояние до измерений: {quality.get('max_support_distance_mm', 0):.3f} мм")
        self.ui.lbl_result_quality.setToolTip("Большое расстояние до ближайшего подтверждённого измерения означает, что поле в этой области хуже подкреплено данными. RMSE оценивает контрольную выборку, а не точность производства.")

    def select_heatmap(self, row, column):
        item = self.ui.tbl_heat.item(row, 0)
        if item:
            self._activate_heatmap(item.data(Qt.UserRole))
            self.mark_dirty()

    def _activate_heatmap(self, key):
        self.active_heatmap_key = key
        actor = self.actors.get(key)
        self.pv_heatmap = actor.mapper.dataset if actor is not None else None
        self._configure_heatmap_picking()
        if key in self.scene_models:
            self.ui.lbl_active_heatmap.setText(f"Активная карта: {self.scene_models[key].get('name', key)}")
        else:
            self.ui.lbl_active_heatmap.setText("Выберите карту в таблице слева")
        self._sync_heatmap_legend()

    def _configure_heatmap_picking(self):
        if self.ui.plotter is None: return
        if self.ui.chk_callouts.isChecked():
            for key, actor in self.actors.items():
                if actor is None: continue
                self._pickability.setdefault(key, actor.pickable)
                actor.pickable = key == self.active_heatmap_key
        else:
            for key, pickable in self._pickability.items():
                actor = self.actors.get(key)
                if actor is not None: actor.pickable = pickable
            self._pickability.clear()

    def _sync_heatmap_legend(self):
        if self.ui.plotter is None: return
        visible = any(key.startswith("Heatmap_") and actor is not None and actor.GetVisibility() for key, actor in self.actors.items())
        for bar in self.ui.plotter.scalar_bars.values():
            bar.SetVisibility(visible)
        for record, actor in zip(self.callout_records, self.callout_actors):
            parent = self.actors.get(record["key"])
            actor.SetVisibility(parent is not None and bool(parent.GetVisibility()))
        self.ui.plotter.render()

    def _render_heatmap(self, key, mesh, deviations):
        data = self.trimesh_to_pyvista(mesh)
        data["Deviation"] = deviations
        limit = self.ui.sliders["heat_limit"][0].value() / self.ui.sliders["heat_limit"][1]
        self.actors[key] = self.ui.plotter.add_mesh(data, scalars="Deviation", cmap="coolwarm", clim=[-limit, limit],
                                                  name=key, show_scalar_bar=True,
                                                  scalar_bar_args=dict(title="Отклонение (мм)", color="black", fmt="%.3f"))
        self.scene_models[key] = dict(key=key, kind="Heatmap", name=key, mesh=mesh, deviations=np.asarray(deviations).copy())
        self._activate_heatmap(key)
