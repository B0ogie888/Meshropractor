"""Reversible slicer editing commands sharing repair preview and cancellation."""
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMenu,QToolButton
from model_tool_ribbon import COMMANDS
from model_tool_dialog import ModelToolDialog
from model_tool_geometry import calculate
from repair_tools import RepairSession


class ModelTools(QObject):
    def __init__(self,window):
        super().__init__(window);self.window=window;self.buttons=window.ui.model_tool_buttons
        for operation,button in self.buttons.items():button.clicked.connect(lambda checked=False,op=operation:self.open('union' if op=='boolean' else op))
        menu=QMenu(self.buttons['boolean'])
        for op in ('union','difference','intersection'):
            menu.addAction(COMMANDS[op]).triggered.connect(lambda checked=False,o=op:self.open(o))
        self.buttons['boolean'].setMenu(menu);self.buttons['boolean'].setPopupMode(QToolButton.InstantPopup)

    def open(self,operation):
        window=self.window
        if window._busy():return
        if operation=='fragments':window.repair_tools.open('split');return
        rows=window.selected_slicer_rows()
        try:
            if not rows:raise ValueError('Выберите детали в текущей сцене.')
            if operation in ('merge','union','difference','intersection','remove_volume'):
                if len(rows)<2:raise ValueError('Выберите минимум две детали. Первая в списке — основная.')
                if len({window.slicer_parts[r].get('platform') for r in rows})>1:raise ValueError('Детали должны относиться к одной платформе.')
            if operation in ('label','struts','rapidfit','formfit') and len(rows)!=1:raise ValueError('Для этой команды выберите одну деталь.')
            if operation not in ('struts','rapidfit','formfit','surface_array','label') and any(window.slicer_parts[r].get('supports') for r in rows):
                raise ValueError('Изменение поверхности нарушит привязку поддержек. Сначала удалите поддержки выбранных деталей и после операции создайте их заново.')
            if operation in ('extrude','surface_array') and not all(window.workspace_tools.selection.get(r) for r in rows):
                raise ValueError('Сначала выделите поверхности каждой выбранной детали нижней панелью выбора.')
            if operation=='label':
                from marking_tools import MarkingSession
                MarkingSession(window,rows[0]);return
            ModelSession(window,operation,rows,dialog_factory=ModelToolDialog,calculator=calculate)
        except Exception as exc:window.log('[!] '+str(exc));window.ui.status_label.setText(str(exc))


class ModelSession(RepairSession):
    def preview(self,*args):
        super().preview(*args)
        if self.result and self.result['mode']=='add':
            for actor,visible in self.hidden_actors:actor.SetVisibility(visible)
            self.plotter.render()

    def apply(self):
        if not self.result or self.dialog.running:return
        window=self.window;before=None;camera=deepcopy(self.plotter.camera_position)
        try:
            self.sources_current();self.clear_preview();window.flush_history();before=window.capture_project()
            state=deepcopy(before,{id(record['mesh']):record['mesh'] for record in before.models+before.parts})
            by_row={item['row']:item for item in self.result['items']};parts=[];mode=self.result['mode']
            for row,record in enumerate(state.parts):
                if mode=='add' or row not in self.rows:parts.append(record)
                if row not in by_row:continue
                for i,mesh in enumerate(by_row[row]['meshes'],1):
                    replacement=dict(record,mesh=mesh,supports=[],filename=f"{Path(record['filename']).stem} — {COMMANDS[self.operation]} {i}.stl")
                    parts.append(replacement)
            state.parts=parts
            window.restore_project(state);self.plotter.camera_position=camera
            window.mark_dirty();window.flush_history(COMMANDS[self.operation]);window.log(COMMANDS[self.operation]+': применено. Ctrl+Z — отменить.')
            self.dialog.accept()
        except Exception as exc:
            if before is not None:window.restore_project(before);self.plotter.camera_position=camera
            self.dialog.status.setText(str(exc))
