"""Interactive surface regions, reversible plans and asynchronous relief work."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import numpy as np
from PySide6.QtCore import QObject, QEvent, QPoint, QRect, QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import QFileDialog
from background_tasks import FunctionWorker
from marking_dialog import MarkingDialog
from marking_geometry import plans, store_plans, merge_relief
from marking_previews import area_lines
from scene_navigation import SceneOutline


class MarkingSession(QObject):
    is_marking = True
    def __init__(self, window, row):
        super().__init__(window)
        self.window, self.row = window, row
        self.plotter = window.ui.slicer_plotter
        self.source = window.slicer_parts[row]['mesh']
        try:
            self.items = plans(self.source)
            stale = False
        except ValueError:
            # Do not project saved areas onto an edited, unrelated surface. Keep
            # the original metadata untouched until a new plan is saved.
            self.items = []; stale = True
        self.current = -1; self.drag = None; self.new_pending = not self.items
        self.closed = False; self.relief = None; self.preview_actor = None; self.area_actor = None
        self.revision = 0; self.dirty = False; self.loading = False; self.params = None
        self.dialog = MarkingDialog(window.slicer_parts[row]['filename'],window.settings,window)
        self.rubber = SceneOutline(self.plotter)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(500)
        self.timer.timeout.connect(self.prepare)
        controls = [window.ui.magics_ribbon, window.ui.toolbar, window.ui.tbl_parts,
                    window.ui.scene_tabs,window.ui.section_panel,window.ui.surface_toolbar,
                    window.ui.measurement_panel,window.ui.action_undo,window.ui.action_redo]
        self.controls = [(control,control.isEnabled()) for control in controls]
        for control,_ in self.controls: control.setEnabled(False)
        window.workspace_tools.measurements.stop(); window.workspace_tools.manual = None
        window._repair_session = self
        window.flush_history(); window.update_history_actions()
        self.dialog.changed.connect(self.changed)
        self.dialog.new_area.connect(self.new_area)
        self.dialog.delete_area.connect(self.delete_area)
        self.dialog.areas.currentIndexChanged.connect(self.select_area)
        self.dialog.preview_requested.connect(self.prepare)
        self.dialog.save_requested.connect(self.save)
        self.dialog.apply_requested.connect(self.apply)
        self.dialog.export_requested.connect(self.export)
        self.dialog.cancel_requested.connect(self.cancel)
        self.dialog.finished.connect(self.close)
        self.plotter.installEventFilter(self)
        window.ui.stack.currentChanged.connect(self.workspace_changed)
        self.refresh_areas()
        if self.items: self.select_area(0)
        window.marking_previews.sync()
        self.dialog.show()
        self.dialog.move(window.mapToGlobal(window.rect().topRight())-QPoint(self.dialog.width()+24,-90))
        if stale: self.dialog.status.setText('Геометрия изменилась: задайте области заново. Старый план заменится при сохранении.')

    def workspace_changed(self, *_):
        if self.window.ui.stack.currentWidget() is not self.window.ui.page_slicer: self.dialog.reject()

    def source_current(self):
        if self.row>=len(self.window.slicer_parts) or self.window.slicer_parts[self.row]['mesh'] is not self.source:
            raise ValueError('Деталь изменилась. Откройте маркировку заново.')

    def commit_draft(self):
        if 0<=self.current<len(self.items): self.items[self.current]['params'] = self.dialog.values()

    def refresh_areas(self):
        with QSignalBlocker(self.dialog.areas):
            self.dialog.areas.clear()
            self.dialog.areas.addItems([item['name'] for item in self.items])
            self.dialog.areas.setCurrentIndex(self.current)

    def select_area(self,index):
        if self.loading or index<0 or index>=len(self.items): return
        self.commit_draft(); self.current = index; self.new_pending = False
        with QSignalBlocker(self.dialog.areas): self.dialog.areas.setCurrentIndex(index)
        self.loading = True
        try: self.dialog.load_values(self.items[index]['params'])
        finally: self.loading = False
        dirty = self.dirty
        self.changed()
        self.dirty = dirty

    def new_area(self):
        self.commit_draft(); self.new_pending = True
        self.dialog.status.setText('Протяните новую рамку ЛКМ на выбранной детали.')

    def delete_area(self):
        if self.current<0: return
        self.items.pop(self.current); self.current = -1; self.clear_preview()
        self.dirty = True; self.refresh_areas()
        if self.items: self.select_area(0)
        else: self.new_area(); self.draw_area()

    def changed(self):
        if self.loading or self.closed: return
        self.dirty = True; self.revision += 1; self.relief = None
        self.dialog.apply.setEnabled(False); self.dialog.export.setEnabled(False)
        self.window.marking_previews.cancel_editor(self)
        self.commit_draft(); self.draw_area()
        if self.current>=0 and self.dialog.flags['auto_preview'].isChecked(): self.timer.start()

    def draw_area(self):
        if self.area_actor is not None: self.plotter.remove_actor(self.area_actor,render=False); self.area_actor = None
        if self.current<0: self.plotter.render(); return
        self.area_actor = self.plotter.add_mesh(area_lines(self.items[self.current]),color='#c6aa18',line_width=2,
            lighting=False,pickable=False,reset_camera=False,render=False,name='marking_area')
        self.plotter.render()

    def clear_preview(self):
        if self.preview_actor is not None:
            self.plotter.remove_actor(self.preview_actor,render=False); self.preview_actor = None
            self.plotter.render()

    def start_work(self,function,args,success):
        window = self.window
        worker = FunctionWorker(function,*args,with_progress=True)
        worker.progress.connect(self.dialog.status.setText)
        worker.error.connect(self.dialog.status.setText)
        worker.finished.connect(lambda: QTimer.singleShot(0,self.finished_work))
        window._applying_repair = True
        try:
            accepted = window.start_job(worker,lambda result:setattr(window,'_job_next',lambda:success(result)))
        finally: window._applying_repair = False
        if accepted: self.dialog.set_running(True)

    def finished_work(self):
        if self.closed: return
        self.dialog.set_running(False)
        if self.relief is not None: self.dialog.apply.setEnabled(True); self.dialog.export.setEnabled(True)

    def prepare(self):
        if self.closed or self.dialog.running or self.current<0: return
        self.timer.stop()
        try:
            self.source_current(); self.params = self.dialog.parameters()
            self.params['frame'] = self.items[self.current]['frame']
            self.commit_draft()
            revision = self.revision
            def ready(mesh):
                if self.closed or self.revision!=revision: return
                self.relief = mesh; self.clear_preview()
                self.preview_actor = self.plotter.add_mesh(self.window.trimesh_to_pyvista(mesh),color='#e5d943',
                    opacity=.9,pickable=False,reset_camera=False,name='marking_preview')
                self.dialog.apply.setEnabled(True); self.dialog.export.setEnabled(True)
                self.dialog.status.setText('Предпросмотр готов. Объединение изменит сетку; Ctrl+Z отменит результат.')
            def failed(message):
                if self.closed or self.revision!=revision: return
                self.clear_preview(); self.dialog.status.setText(message)
            self.window.marking_previews.request_editor(self,self.source,self.items[self.current],self.params,ready,failed)
        except Exception as exc:
            self.clear_preview(); self.dialog.status.setText(str(exc))

    def cancel(self):
        if self.dialog.running: self.window.cancel_current_job()
        else:
            self.timer.stop(); self.window.marking_previews.cancel_editor(self)
            self.dialog.status.setText('Предпросмотр отменён. Можно продолжить ввод или нажать «Обновить».')

    def save(self):
        try:
            self.source_current(); self.commit_draft()
            self.window.flush_history()
            store_plans(self.source,self.items)
            self.window.mark_dirty(); self.window.flush_history('Области маркировки')
            self.dirty = False
            self.window.marking_previews.sync()
            self.dialog.status.setText('Запланированные области сохранены в детали. Сохраните проект .mrp для записи на диск.')
            if self.dialog.flags['remember'].isChecked(): self.window.settings.setValue('marking/text',self.dialog.text.toPlainText())
        except Exception as exc: self.dialog.status.setText(str(exc))

    def apply(self):
        if self.relief is None or self.dialog.running: return
        if self.window.slicer_parts[self.row].get('supports'):
            self.dialog.status.setText('Перед изменением геометрии удалите поддержки детали. Области можно сохранить запланированными.'); return
        try: self.source_current()
        except ValueError as exc: self.dialog.status.setText(str(exc)); return
        self.commit_draft()
        def applied(mesh):
            if self.closed: return
            self.source_current(); self.clear_preview()
            remaining = [item for index,item in enumerate(self.items) if index!=self.current]
            store_plans(mesh,remaining)
            self.window.flush_history()
            before = self.window.capture_project()
            try:
                self.window.replace_slicer_mesh(self.row,mesh)
                self.window.refresh_scene_visibility()
                self.window.update_info_combobox()
                self.plotter.reset_camera_clipping_range()
                self.window.mark_dirty(); self.window.flush_history('Маркировка детали')
            except Exception as exc:
                self.window.restore_project(before)
                self.source = self.window.slicer_parts[self.row]['mesh']
                self.dialog.status.setText(str(exc)); return
            self.dirty = False; self.dialog.set_running(False); self.dialog.accept()
        self.start_work(merge_relief,(self.source,self.relief,deepcopy(self.params)),applied)

    def export(self):
        if self.relief is None or self.dialog.running: return
        path,_ = QFileDialog.getSaveFileName(self.dialog,'Сохранить маркировку','marking.stl','STL (*.stl)')
        if not path: return
        try:
            self.relief.export(path if Path(path).suffix.lower()=='.stl' else path+'.stl')
            self.dialog.status.setText('Отдельная сетка маркировки сохранена.')
        except Exception as exc: self.dialog.status.setText(str(exc))

    def point_on_plane(self, position, origin, normal):
        renderer = self.plotter.renderer; ratio = self.plotter.devicePixelRatioF()
        height = self.plotter.render_window.GetSize()[1]; points = []
        for depth in (0.,1.):
            renderer.SetDisplayPoint(position.x()*ratio,height-1-position.y()*ratio,depth)
            renderer.DisplayToWorld(); value = np.asarray(renderer.GetWorldPoint()); points.append(value[:3]/value[3])
        direction = points[1]-points[0]; divisor = direction@normal
        if abs(divisor)<1e-8: raise ValueError('Поверхность видна с ребра. Поверните сцену к ней.')
        return points[0]+direction*((origin-points[0])@normal/divisor)

    def region_from_rect(self,rect):
        hit = self.window.workspace_tools.picker(rect.center(),rows=[self.row])
        if hit is None: raise ValueError('Центр рамки должен находиться на выбранной детали.')
        _,face,origin = hit; normal = np.array(self.source.face_normals[face],copy=True)
        right = self.point_on_plane(rect.center()+QPoint(20,0),origin,normal)-origin
        right /= np.linalg.norm(right); up = np.cross(normal,right); up /= np.linalg.norm(up)
        top = self.point_on_plane(rect.topLeft(),origin,normal)
        bottom = self.point_on_plane(rect.bottomRight(),origin,normal)
        width = abs((bottom-top)@right); height = abs((bottom-top)@up)
        if min(width,height)<.1: raise ValueError('Нарисуйте область крупнее.')
        frame = np.eye(4); frame[:3,:3] = np.column_stack((right,up,normal)); frame[:3,3] = origin
        return frame,width,height

    def eventFilter(self,obj,event):
        if obj is not self.plotter or self.closed: return False
        kind = event.type()
        if kind in (QEvent.Hide,QEvent.FocusOut,QEvent.WindowDeactivate):
            self.drag = None; self.rubber.hide(); return False
        if kind==QEvent.KeyPress and event.key()==Qt.Key_Escape:
            self.drag = None; self.rubber.hide(); self.dialog.status.setText('Разметка рамки отменена.'); return True
        if kind==QEvent.MouseButtonPress and event.button()==Qt.LeftButton:
            cube = self.window.workspace_tools.cube
            if cube and cube.geometry().contains(event.position().toPoint()): return False
            if self.dialog.running: return True
            self.drag = event.position().toPoint(); return True
        if kind==QEvent.MouseMove and self.drag is not None:
            if not event.buttons() & Qt.LeftButton: self.drag = None; self.rubber.hide(); return True
            self.rubber.setGeometry(QRect(self.drag,event.position().toPoint()).normalized()); self.rubber.show(); return True
        if kind==QEvent.MouseButtonRelease and event.button()==Qt.LeftButton and self.drag is not None:
            start,self.drag = self.drag,None; self.rubber.hide()
            try:
                self.source_current()
                rect = QRect(start,event.position().toPoint()).normalized()
                if min(rect.width(),rect.height())<6:
                    hit = self.window.workspace_tools.picker(event.position().toPoint(),rows=[self.row])
                    if hit:
                        for index,item in reversed(list(enumerate(self.items))):
                            p = np.linalg.solve(np.asarray(item['frame']),np.r_[hit[2],1.])
                            values = item['params']
                            if abs(p[0])<=values['width']/2 and abs(p[1])<=values['area_height']/2:
                                self.dialog.areas.setCurrentIndex(index); return True
                    return True
                frame,width,height = self.region_from_rect(rect)
                self.commit_draft()
                if self.new_pending or self.current<0:
                    if len(self.items)>=50: raise ValueError('Допускается до 50 областей на детали.')
                    self.items.append(dict(id=str(uuid4()),name=f'Область {len(self.items)+1}',frame=frame.tolist(),params=self.dialog.values()))
                    self.current = len(self.items)-1
                self.items[self.current]['frame'] = frame.tolist(); self.new_pending = False
                self.loading = True
                try:
                    self.dialog.fields['width'].setValue(width); self.dialog.fields['area_height'].setValue(height)
                finally: self.loading = False
                self.refresh_areas(); self.changed()
                self.dialog.status.setText('Область создана. Введите текст или выберите рисунок / Data Matrix.')
            except Exception as exc: self.dialog.status.setText(str(exc))
            return True
        return False

    def close(self,*_):
        if self.closed: return
        if self.dirty and self.dialog.flags['auto_save'].isChecked(): self.save()
        self.window.marking_previews.cancel_editor(self)
        self.closed = True; self.timer.stop(); self.clear_preview()
        self.rubber.dispose()
        if self.area_actor is not None: self.plotter.remove_actor(self.area_actor,render=False)
        self.plotter.removeEventFilter(self); self.plotter.render()
        self.window.ui.stack.currentChanged.disconnect(self.workspace_changed)
        self.window._repair_session = None
        self.window.marking_previews.sync()
        for control,enabled in self.controls: control.setEnabled(enabled)
        self.window.update_history_actions()
        self.dialog.deleteLater(); self.deleteLater()
