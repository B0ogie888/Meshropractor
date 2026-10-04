"""Non-blocking previews of editable marking plans, inside and outside the editor."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QTimer, Slot
from background_tasks import FunctionWorker
from marking_content import content_parameters
from marking_geometry import KEY, build_relief, digest, plans

PREFIX = 'marking_plan_'


def area_lines(item):
    values = item['params']; w, h = values['width']/2, values['area_height']/2
    if values.get('circular'):
        t = np.linspace(0,2*np.pi,65)
        points = np.c_[w*np.cos(t),h*np.sin(t),np.full(len(t),.03)]
    else: points = np.array([[-w,-h,.03],[w,-h,.03],[w,h,.03],[-w,h,.03],[-w,-h,.03]])
    frame = np.asarray(item['frame'],float)
    return pv.lines_from_points(points@frame[:3,:3].T+frame[:3,3])


def preview_key(geometry, item):
    raw = json.dumps(dict(frame=item['frame'],params=item['params']),sort_keys=True,ensure_ascii=False)
    return geometry, hashlib.sha256(raw.encode('utf-8')).hexdigest()


class MarkingPreviews(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window; self.plotter = None
        self.records = {}; self.cache = OrderedDict()
        self.worker = None; self.current = None; self.editor = None
        self.stopping = False
        self.timer = QTimer(self); self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_next)

    def request_editor(self, owner, source, item, params, ready, failed):
        request = dict(owner=owner,source=source,key=preview_key(digest(source),item),
                       params=deepcopy(params),ready=ready,failed=failed)
        self.cancel_editor(owner)
        self.editor = request
        owner.dialog.set_preview_running(True)
        self.timer.start(0)

    def cancel_editor(self, owner):
        if self.editor and self.editor.get('owner') is owner: self.editor = None
        if self.current and self.current.get('owner') is owner and self.worker:
            self.worker.requestInterruption()
        if not owner.closed: owner.dialog.set_preview_running(False)

    def remember(self, key, mesh):
        self.cache[key] = mesh; self.cache.move_to_end(key)
        while len(self.cache)>12 or sum(len(m.faces) for m in self.cache.values())>1_000_000:
            self.cache.popitem(last=False)

    def sync(self):
        if self.stopping: return
        plotter = getattr(self.window.ui,'slicer_plotter',None)
        if plotter is None: return
        if plotter is not self.plotter:
            self.clear_actors(); self.plotter = plotter
        visible = set(self.window.display_tools.visible_rows())
        if self.window.ui.stack.currentWidget() is not self.window.ui.page_slicer: visible.clear()
        editor = getattr(self.window,'_repair_session',None)
        editing_row = editor.row if getattr(editor,'is_marking',False) else None
        desired = {}
        for row,part in enumerate(self.window.slicer_parts):
            source = part['mesh']; data = source.metadata.get(KEY)
            if not data or row==editing_row: continue
            try: items = plans(source)
            except ValueError: continue
            actor = plotter.actors.get(part['actor_name'])
            shown = row in visible and actor is not None and bool(actor.GetVisibility())
            for item in items:
                key = preview_key(data['digest'],item); identifier = (row,key)
                record = self.records.get(identifier)
                if record is None:
                    record = dict(key=key,source=source,item=item,shown=shown,failed=False,
                                  name=f'{PREFIX}{row}_{key[1][:20]}')
                record['shown'] = shown; desired[identifier] = record
        for identifier,record in self.records.items():
            if identifier not in desired: self.remove_record(record)
        self.records = desired
        for record in self.records.values():
            name = record['name']
            if record['shown'] and name+'_area' not in plotter.actors:
                plotter.add_mesh(area_lines(record['item']),name=name+'_area',color='#c6aa18',
                    line_width=1.5,lighting=False,pickable=False,reset_camera=False,render=False)
            if record['shown'] and record['key'] in self.cache and name not in plotter.actors:
                self.add_relief(record,self.cache[record['key']])
            for suffix in ('','_area'):
                actor = plotter.actors.get(name+suffix)
                if actor is not None: actor.SetVisibility(record['shown'])
        self.timer.start(0)

    def add_relief(self, record, mesh):
        if self.plotter is None: return
        self.plotter.add_mesh(self.window.trimesh_to_pyvista(mesh),name=record['name'],
            color='#e5d943',opacity=.9,pickable=False,reset_camera=False,render=False)

    def remove_record(self,record):
        if self.plotter is not None:
            for suffix in ('','_area'): self.plotter.remove_actor(record['name']+suffix,render=False)

    def clear_actors(self):
        for record in self.records.values(): self.remove_record(record)
        self.records.clear()

    def deliver(self, request, mesh):
        if self.editor is request and not request['owner'].closed:
            self.editor = None
            request['owner'].dialog.set_preview_running(False)
            request['ready'](mesh)

    @Slot()
    def start_next(self):
        if self.stopping or self.worker is not None: return
        request = self.editor
        if request:
            cached = self.cache.get(request['key'])
            if cached is not None:
                self.deliver(request,cached); self.timer.start(0); return
        else:
            record = next((r for r in self.records.values() if r['shown'] and not r['failed']
                           and r['key'] not in self.cache and r['name'] not in self.plotter.actors),None)
            if record is None: return
            try:
                params = content_parameters(record['item']['params'])
                params['frame'] = record['item']['frame']
            except Exception:
                record['failed'] = True; self.timer.start(0); return
            request = dict(source=record['source'],key=record['key'],params=params)
        self.current = request
        self.worker = FunctionWorker(build_relief,request['source'],deepcopy(request['params']),with_progress=True)
        self.worker.setParent(self)
        self.worker.result.connect(self.received)
        self.worker.error.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    @Slot(object)
    def received(self, mesh):
        if self.sender() is not self.worker or self.stopping: return
        request = self.current; self.remember(request['key'],mesh)
        if request.get('owner'): self.deliver(request,mesh)
        for record in self.records.values():
            if record['key']==request['key'] and record['shown']: self.add_relief(record,mesh)
        if self.plotter is not None: self.plotter.render()

    @Slot(str)
    def failed(self, message):
        if self.sender() is not self.worker or self.stopping: return
        request = self.current
        if self.editor is request and not request['owner'].closed:
            self.editor = None; request['owner'].dialog.set_preview_running(False)
            request['failed'](message)
        for record in self.records.values():
            if record['key']==request['key']: record['failed'] = True

    @Slot()
    def finished(self):
        if self.sender() is not self.worker: return
        request,worker = self.current,self.worker
        self.current = self.worker = None; worker.deleteLater()
        if self.editor is request:
            self.editor = None
            if not request['owner'].closed: request['owner'].dialog.set_preview_running(False)
        if self.stopping: QTimer.singleShot(0,self.window.close)
        else: self.timer.start(0)

    def stop_for_close(self):
        self.timer.stop()
        if self.worker is None: return True
        self.stopping = True; self.editor = None; self.worker.requestInterruption()
        return False

    def resume(self):
        self.stopping = False; self.sync()

    def dispose(self):
        self.stopping = True; self.timer.stop(); self.clear_actors(); self.cache.clear()
