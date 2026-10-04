"""Non-destructive, shared-geometry virtual copies in the active slicer scene."""
import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from duplicate_dialog import DuplicateDialog
from duplicate_layout import duplicate_plan
from part_supports import combined_mesh


class DuplicateSession:
    def __init__(self,window,operation,rows):
        self.window,self.operation,self.rows=window,operation,list(rows)
        self.plotter=window.ui.slicer_plotter; self.closed=False; self.actors={}; self.sources=[]
        self.meshes=[combined_mesh(window.slicer_parts[row]) for row in self.rows]
        self.dialog=DuplicateDialog(operation,len(rows),window)
        self.timer=QTimer(self.dialog); self.timer.setSingleShot(True); self.timer.setInterval(60)
        self.timer.timeout.connect(self.update_preview); self.dialog.changed.connect(self.timer.start)
        self.dialog.apply_requested.connect(self.apply); self.dialog.finished.connect(self.close)
        for row in self.rows:
            part=window.slicer_parts[row]
            datasets=[('part',part['mesh_pv'])]
            for group in part.get('supports',[]):
                actor=self.plotter.actors.get('part_support_'+group['id'])
                if actor is not None and group.get('visible',True): datasets.append((group['id'],actor.mapper.dataset))
            for tag,data in datasets:
                mapper=pv.DataSetMapper(dataset=data); mapper.scalar_visibility=False
                for plane in window.ui.section_panel._planes: mapper.AddClippingPlane(plane)
                self.sources.append((row,tag,mapper))
        controls=[window.ui.magics_ribbon,window.ui.toolbar,window.ui.tbl_parts,window.ui.scene_tabs,
                  window.ui.section_panel,window.ui.action_save,window.ui.action_undo,window.ui.action_redo,
                  window.ui.btn_back_to_start,window.ui.measurement_panel]
        if hasattr(window.ui,'surface_toolbar'): controls.append(window.ui.surface_toolbar)
        if window.workspace_tools.supports.panel: controls.append(window.workspace_tools.supports.panel)
        self.controls=[(control,control.isEnabled()) for control in controls]
        for control,_ in self.controls: control.setEnabled(False)
        window._duplicate_session=self
        if hasattr(window.ui,'part_inspector'): window.ui.part_inspector.schedule_refresh()
        self.dialog.show()
        pos=window.mapToGlobal(window.rect().topLeft()); self.dialog.move(pos.x()+25,pos.y()+110)
        self.update_preview()

    def clear_preview(self):
        for actor in self.actors.values(): self.plotter.remove_actor(actor,render=False)
        self.actors.clear()

    def update_preview(self):
        if self.closed: return
        params=self.dialog.parameters()
        self.dialog.diagram.counts=params['counts']; self.dialog.diagram.update()
        try:
            plan=duplicate_plan(self.meshes,params,self.operation)
            self.dialog.total.setValue(plan['total_parts'])
            self.dialog.selection.setText(f'Исходных: {len(self.rows)} · Новых: {plan["new_parts"]}')
            self.dialog.status.setText('Шаг ячейки: '+ ' / '.join(f'{n:g}' for n in plan['steps'])+' мм')
            self.dialog.apply_button.setEnabled(plan['new_parts']>0)
            expected=set()
            if self.dialog.preview.isChecked():
                for row,tag,mapper in self.sources:
                    for index,offset in enumerate(plan['offsets']):
                        key=(row,tag,index); expected.add(key)
                        actor=self.actors.get(key)
                        if actor is None:
                            actor=pv.Actor(mapper=mapper); actor.prop.color='#d6c746'; actor.prop.opacity=.38
                            actor.prop.show_edges=False; actor.prop.ambient=.3
                            self.plotter.add_actor(actor,name=f'duplicate_preview_{row}_{tag}_{index}',
                                                   reset_camera=False,pickable=False,render=False)
                            self.actors[key]=actor
                        matrix=np.eye(4); matrix[:3,3]=offset; actor.user_matrix=matrix
            for key in set(self.actors)-expected: self.plotter.remove_actor(self.actors.pop(key),render=False)
            self.plotter.reset_camera_clipping_range(); self.plotter.render()
        except (ValueError,FloatingPointError) as exc:
            self.clear_preview(); self.dialog.status.setText(str(exc)); self.dialog.apply_button.setEnabled(False)
            self.plotter.render()

    def apply(self):
        self.timer.stop()
        try:
            params=self.dialog.parameters(); plan=duplicate_plan(self.meshes,params,self.operation)
            if not plan['new_parts']: raise ValueError('Увеличьте количество хотя бы по одной оси.')
            self.clear_preview(); self.window._applying_duplicate=True
            try: self.window.apply_slicer_tool(self.operation,params,self.rows)
            finally: self.window._applying_duplicate=False
        except Exception as exc:
            self.update_preview(); self.dialog.status.setText(str(exc)); return
        try: self.dialog.save_values()
        except Exception as exc: self.window.log(f'Не удалось запомнить матрицу: {exc}')
        self.dialog.accept()

    def close(self,*_):
        if self.closed: return
        self.closed=True; self.timer.stop(); self.clear_preview(); self.sources.clear()
        self.window._duplicate_session=None
        for control,enabled in self.controls: control.setEnabled(enabled)
        if hasattr(self.window.ui,'part_inspector'): self.window.ui.part_inspector.schedule_refresh()
        self.window.update_history_actions(); self.plotter.reset_camera_clipping_range(); self.plotter.render()
        self.dialog.deleteLater()
