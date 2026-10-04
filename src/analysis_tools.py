"""Background inspections, measurements and portable reports for selected parts."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv
import html
import json
import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject
from PySide6.QtGui import QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QInputDialog, QPlainTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QMenu, QToolButton)
from background_tasks import FunctionWorker
from display_settings import DIALOG_STYLE
from analysis_ribbon import COMMANDS
from analysis_geometry import wall_samples, collisions, cavities, slice_distribution, build_estimate

DEFAULTS=dict(layer_height=.05,volume_rate=5.,layer_seconds=8.,setup_minutes=15.,hour_price=0.,density=1.,material_price=0.,fixed_cost=0.)


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def inspect(operation, records, params, *, progress=None, cancelled=None):
    from analysis_geometry import check_cancel
    if operation in ('intersections','trapping'):
        return collisions(records, progress=progress,cancelled=cancelled, **params)
    if operation=='slices': return slice_distribution(records,progress=progress,cancelled=cancelled,**params)
    result=[]
    for record in records:
        check_cancel(cancelled)
        if progress: progress('Анализ: '+record['filename'])
        value=wall_samples(record['mesh'],cancelled=cancelled,**params) if operation=='walls' else cavities(record['mesh'],cancelled=cancelled)
        result.append(dict(name=record['filename'],**value))
    return result


class AnalysisTools(QObject):
    def __init__(self,window):
        super().__init__(window); self.window=window; self.dialog=None; self.results={}; self.annotations=[]
        self.params=dict(DEFAULTS); self.title='Отчёт Meshropractor'; self.author=''; self.wall_owners={}
        self.buttons=window.ui.analysis_buttons
        for op,button in self.buttons.items(): button.clicked.connect(lambda checked=False,operation=op:self.trigger(operation))
        mapping={'volume':'volume','density':'packing_density','dimensions':'dimensions','mass_center':'center_mass','bounds':'bbox'}
        for op,target in mapping.items():
            window.ui.display_buttons[target].toggled.connect(lambda checked,b=self.buttons[op]:b.setChecked(checked))
        menu=QMenu(self.buttons['view'])
        for title,axis,sign in [('Изометрия',None,1),('Сверху',2,1),('Снизу',2,-1),('Спереди',1,-1),('Сзади',1,1),('Справа',0,1),('Слева',0,-1)]:
            menu.addAction(title).triggered.connect(lambda checked=False,a=axis,s=sign:window.display_tools.orient(a,s))
        self.buttons['view'].setMenu(menu);self.buttons['view'].setPopupMode(QToolButton.InstantPopup)

    def rows(self):
        rows=self.window.selected_slicer_rows()
        if not rows: raise ValueError('Выберите детали в текущей сцене.')
        return rows

    def configuration(self):
        dialog=QDialog(self.window); dialog.setWindowTitle('Параметры оценки времени и стоимости'); dialog.setStyleSheet(DIALOG_STYLE)
        form=QFormLayout(dialog); fields={}
        names={'layer_height':'Высота слоя, мм','volume_rate':'Объёмная производительность, мм³/с','layer_seconds':'Время на слой, с','setup_minutes':'Подготовка, мин','hour_price':'Стоимость часа','density':'Плотность, г/см³','material_price':'Цена материала за кг','fixed_cost':'Дополнительные расходы'}
        for key,title in names.items():
            spin=QDoubleSpinBox(); spin.setDecimals(4); spin.setRange(.0001 if key in ('layer_height','volume_rate','density') else 0,1e9)
            spin.setValue(self.params[key]); form.addRow(title,spin); fields[key]=spin
        label=QLabel('Оценка по выбранной группе, включая поддержки. Не является прогнозом конкретной машины. Валюта всех цен должна совпадать.');label.setWordWrap(True);form.addRow(label)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);form.addRow(buttons)
        accepted=dialog.exec()
        if accepted:
            self.params={key:spin.value() for key,spin in fields.items()}
            self.window.display_tools._density,self.window.display_tools._price=self.params['density'],self.params['material_price']
            self.window.mark_dirty()
        dialog.deleteLater();return bool(accepted)

    def trigger(self,op):
        if self.window._busy():return
        try:
            display=self.window.display_tools
            mapping={'volume':'volume','density':'packing_density','dimensions':'dimensions','mass_center':'center_mass','bounds':'bbox'}
            if op in mapping:
                target=mapping[op];display.buttons[target].setChecked(self.buttons[op].isChecked());display.trigger(target);return
            if op=='view':display.orient();return
            if op=='risks':display.show_report('build_risk');return
            if op=='material':display.show_report('material_cost');return
            if op=='distance':
                panel=self.window.ui.measurement_panel;panel.tabs.setCurrentIndex(0);panel.modes[0].setCurrentIndex(0);panel.start();return
            if op=='thickness':
                panel=self.window.ui.measurement_panel;panel.tabs.setCurrentIndex(0);panel.modes[0].setCurrentIndex(4);panel.start();return
            if op=='precision':
                panel=self.window.ui.measurement_panel
                digits,ok=QInputDialog.getInt(self.window,'Точность отображения','Знаков после запятой (не точность сетки):',panel.decimals,0,6)
                if ok:panel.decimals=digits;self.window.mark_dirty()
                return
            if op=='actual':self.actual();return
            if op=='template':
                title,ok=QInputDialog.getText(self.window,'Настройки отчёта','Заголовок:',text=self.title)
                if not ok:return
                author,ok=QInputDialog.getText(self.window,'Настройки отчёта','Автор:',text=self.author)
                if ok:self.title,self.author=title.strip() or 'Отчёт Meshropractor',author;self.window.mark_dirty()
                return
            rows=self.rows();records=[dict(mesh=self.window.slicer_parts[r]['mesh'].copy(),filename=self.window.slicer_parts[r]['filename'],supports=deepcopy(self.window.slicer_parts[r].get('supports',[]))) for r in rows]
            if op in ('time','cost','report'):
                if op!='report' and not self.configuration():return
                estimate=build_estimate(records,self.params)
                data=dict(title=self.title,author=self.author,date=datetime.now().isoformat(timespec='seconds'),version=__import__('app_version').APP_VERSION,
                    estimate=estimate,actual_measurements=self.annotations,
                    scene_measurements=[self.window.ui.measurement_panel.results.item(i).text() for i in range(self.window.ui.measurement_panel.results.count())],
                    analyses=self.current_results(),parts=[r['filename'] for r in records],
                    notice='Оценки приблизительные; объёмы пересекающихся деталей и поддержек суммируются. Толщина и срезы рассчитываются по выборке.')
                self.show_result(op,data);return
            params={}
            if op=='walls':
                threshold,ok=QInputDialog.getDouble(self.window,'Толщина стенок','Порог тонкой стенки, мм:',1.,.0001,1e6,4)
                if not ok:return
                params=dict(threshold=threshold,count=2000)
            elif op=='slices':
                height,ok=QInputDialog.getDouble(self.window,'Распределение срезов','Высота слоя, мм:',self.params['layer_height'],.001,100,4)
                if not ok:return
                params=dict(layer_height=height,samples=80)
            elif op=='trapping':
                direction,ok=QInputDialog.getItem(self.window,'Извлечение первой детали','Направление:', ['+Z','−Z','+X','−X','+Y','−Y'],0,False)
                if not ok:return
                travel,ok=QInputDialog.getDouble(self.window,'Извлечение первой детали','Перемещение, мм:',20,.001,1e6,3)
                if not ok:return
                params=dict(direction={'+'+axis:np.eye(3)[i] for i,axis in enumerate('XYZ')} | {'−'+axis:-np.eye(3)[i] for i,axis in enumerate('XYZ')})
                params=dict(direction=params['direction'][direction],travel=travel,steps=24)
            if op in ('intersections','trapping') and len(records)<2:raise ValueError('Для проверки выберите минимум две детали.')
            identity=tuple((self.window.slicer_parts[r]['filename'],id(self.window.slicer_parts[r]['mesh'])) for r in rows)
            def ready(value):
                data=dict(parts=[r['filename'] for r in records],result=plain(value),params=plain(params))
                self.results[op]=(identity,data);self.show_result(op,data)
                if op=='walls':self.highlight_walls(rows,value)
            self.window.start_job(FunctionWorker(inspect,op,records,params,with_progress=True),ready)
        except Exception as exc:self.window.display_tools.notify(COMMANDS[op]+': '+str(exc))

    def current_results(self):
        current={(p['filename'],id(p['mesh'])) for p in self.window.slicer_parts}
        selected={(self.window.slicer_parts[r]['filename'],id(self.window.slicer_parts[r]['mesh'])) for r in self.window.selected_slicer_rows()}
        results = {op:deepcopy(data) for op,(identity,data) in self.results.items() if set(identity)==selected and set(identity)<=current}
        for record in results.get('walls', {}).get('result', []):
            for key in ('points','values','source_faces'): record.pop(key,None)
        return results

    def actual(self):
        name,ok=QInputDialog.getText(self.window,'Фактические измерения','Название измерения / детали:')
        if not ok or not name.strip():return
        unit,ok=QInputDialog.getItem(self.window,'Фактические измерения','Единицы:',['мм','°'],0,False)
        if not ok:return
        values=[]
        for title,initial,low in [('Номинал',0.,-1e9),('Фактическое значение',0.,-1e9),('Допуск ±',.1,0.)]:
            value,ok=QInputDialog.getDouble(self.window,'Фактические измерения',title+':',initial,low,1e9,6)
            if not ok:return
            values.append(value)
        self.window.flush_history();self.annotations.append(dict(name=name.strip(),unit=unit,nominal=values[0],actual=values[1],tolerance=values[2],passed=abs(values[1]-values[0])<=values[2]))
        self.window.mark_dirty();self.window.flush_history('Фактическое измерение');self.show_result('actual',dict(measurements=self.annotations))

    def highlight_walls(self,rows,results):
        self.clear_highlights()
        plotter=self.window.ui.slicer_plotter
        for row,result in zip(rows,results):
            values=np.asarray(result['values']);mask=np.isfinite(values)&(values<result['threshold'])
            name='analysis_walls_'+str(row);plotter.remove_actor(name,render=False)
            if mask.any():
                plotter.add_mesh(pv.PolyData(np.asarray(result['points'])[mask]),name=name,color='#e74435',point_size=7,render_points_as_spheres=True,pickable=False,reset_camera=False,render=False)
                self.wall_owners[row]=id(self.window.slicer_parts[row]['mesh'])
        self.sync_highlights()
        plotter.render()

    def sync_highlights(self):
        plotter=self.window.ui.slicer_plotter
        for row,identity in list(self.wall_owners.items()):
            name='analysis_walls_'+str(row)
            if row>=len(self.window.slicer_parts) or id(self.window.slicer_parts[row]['mesh'])!=identity:
                plotter.remove_actor(name,render=False);self.wall_owners.pop(row);continue
            actor=plotter.actors.get(name);source=plotter.actors.get(self.window.slicer_parts[row]['actor_name'])
            if actor is None: self.wall_owners.pop(row);continue
            actor.SetVisibility(source is not None and source.GetVisibility())
            actor.SetUserMatrix(source.GetUserMatrix() if source is not None else None)
            mapper=actor.GetMapper();mapper.RemoveAllClippingPlanes()
            for plane in self.window.ui.section_panel._planes: mapper.AddClippingPlane(plane)

    def show_result(self,operation,data):
        if self.dialog:self.dialog.close();self.dialog.deleteLater()
        dialog=QDialog(self.window);dialog.setWindowTitle(COMMANDS[operation]);dialog.setStyleSheet(DIALOG_STYLE);dialog.resize(850,620);dialog.setModal(False)
        layout=QVBoxLayout(dialog);report=QPlainTextEdit();report.setReadOnly(True)
        text=self.report_text(operation,data);report.setPlainText(text);layout.addWidget(report)
        if operation=='slices':
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            fig=Figure(figsize=(6,2),tight_layout=True);axis=fig.add_subplot(111);axis.plot(data['result']['z'],data['result']['area_mm2']);axis.set_xlabel('Z, mm');axis.set_ylabel('Area, mm²');axis.grid(True)
            canvas=FigureCanvasQTAgg(fig);canvas.setMinimumHeight(180);layout.addWidget(canvas)
        row=QHBoxLayout()
        for label,kind in [('HTML','html'),('PDF','pdf'),('JSON','json'),('CSV','csv')]:
            button=QPushButton('Сохранить '+label);button.clicked.connect(lambda checked=False,k=kind:self.export(data,text,k));row.addWidget(button)
        clear=QPushButton('Очистить подсветку');clear.clicked.connect(self.clear_highlights);row.addWidget(clear)
        close=QPushButton('Закрыть');close.clicked.connect(dialog.close);row.addWidget(close);layout.addLayout(row)
        dialog.report=report;self.dialog=dialog;dialog.show()

    def clear_highlights(self):
        self.wall_owners.clear()
        plotter=self.window.ui.slicer_plotter
        if plotter:
            for name in list(plotter.actors):
                if name.startswith('analysis_walls_'):plotter.remove_actor(name,render=False)
            plotter.render()

    def report_text(self,op,data):
        if 'estimate' in data:
            e=data['estimate'];lines=[data['title'],f"Автор: {data['author']}",f"Дата: {data['date']}",f"Объём: {e['total_mm3']:.3f} мм³; неопределённых объёмов: {e['unknown']}",f"Масса: {e['mass_g']:.3f} г; слоёв: {e['layers']}",f"Оценка времени: {e['seconds']/3600:.3f} ч",f"Материал: {e['material_cost']:.2f}; работа машины: {e['machine_cost']:.2f}; всего: {e['total_cost']:.2f}",data['notice']]
            p=e['params'];lines += [f"Параметры: слой {p['layer_height']:g} мм; производительность {p['volume_rate']:g} мм³/с; {p['layer_seconds']:g} с/слой; подготовка {p['setup_minutes']:g} мин.",f"Плотность {p['density']:g} г/см³; цена материала {p['material_price']:g}/кг; час машины {p['hour_price']:g}; дополнительные расходы {p['fixed_cost']:g}."]
            lines+=['\nДетали: '+', '.join(data['parts']), '\nИзмерения сцены:']+data['scene_measurements']
            lines+=['\nФактические измерения:']+[f"{m['name']}: {m['actual']} {m['unit']}, номинал {m['nominal']}, допуск ±{m['tolerance']}; {'в допуске' if m['passed'] else 'вне допуска'}" for m in data['actual_measurements']]
            lines+=['\nАнализы:']+[COMMANDS[key]+'\n'+self.report_text(key,value) for key,value in data['analyses'].items()];return '\n'.join(lines)
        if op=='walls':return '\n\n'.join(f"{r['name']}: минимум {r['minimum']} мм; медиана {r['median']} мм; максимум {r['maximum']} мм\nТонких проб: {r['thin']}; попаданий: {r['hits']}/{r['count']}; замкнутое согласованное тело: {r['reliable']}\nВыборка по нормали, не строгий глобальный минимум. Красные точки — пробы ниже порога." for r in data['result'])
        if op in ('intersections','trapping'):
            note='Проверены детали без поддержек. Соприкосновение без объёма пересечения не считается ошибкой.'
            if op=='trapping':note+=' Извлечение — выборка 25 положений первой детали каждой пары в заданном направлении; между пробами препятствия могут быть пропущены.'
            return '\n'.join(f"{data['parts'][r['first']]} / {data['parts'][r['second']]}: {r['status']}; объём {r['volume']} мм³; смещение {r.get('offset',0)} мм" for r in data['result'])+'\n\n'+note
        if op=='cavities':
            lines=[]
            for r in data['result']:
                lines += [r['name'],f"Оболочек: {len(r['shells'])}; замкнутых с отрицательным объёмом: {r['inward_shells']}",f"Замкнутая согласованная сетка: {'да' if r['reliable'] else 'нет'}"]
                for shell in r['shells']:
                    volume='не определён' if shell['signed_volume'] is None else f"{shell['signed_volume']:.3f} мм³"
                    lines.append(f"  Оболочка {shell['index']}: {'замкнута' if shell['closed'] else 'открыта'}; знаковый объём {volume}; площадь {shell['area']:.3f} мм²; треугольников {shell['triangles']}")
            return '\n'.join(lines)+'\n\nОтрицательная внутренняя оболочка может обозначать полость при корректных нормалях. Дренаж и проходимость каналов не проверяются.'
        if op=='slices':
            r=data['result'];lines=[f"Слой {r['layer_height']:g} мм; слоёв до высоты сборки: {r['layers']}; проб с незамкнутыми контурами: {r['open_paths']}",r['note'],'Z, мм | Площадь, мм² | Длина контуров, мм']
            lines += [f'{z:.4f} | {area:.4f} | {perimeter:.4f}' for z,area,perimeter in zip(r['z'],r['area_mm2'],r['perimeter_mm'])]
            return '\n'.join(lines)
        if op=='actual':
            return '\n'.join(f"{m['name']}: {m['actual']} {m['unit']}; номинал {m['nominal']}; допуск ±{m['tolerance']}; {'в допуске' if m['passed'] else 'вне допуска'}" for m in data['measurements'])
        return json.dumps(plain(data),ensure_ascii=False,indent=2)

    def export(self,data,text,kind):
        path,_=QFileDialog.getSaveFileName(self.window,'Сохранить отчёт','report.'+kind,kind.upper()+' (*.'+kind+')')
        if not path:return
        if not path.lower().endswith('.'+kind):path+='.'+kind
        try:
            if kind=='json':Path(path).write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            elif kind=='csv':
                with open(path,'w',encoding='utf-8-sig',newline='') as file:
                    writer=csv.writer(file);writer.writerow(['Поле','Значение'])
                    for key,value in plain(data).items():writer.writerow([key,json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value])
            else:
                document='<html><meta charset="utf-8"><body><h1>'+html.escape(data.get('title',self.title))+'</h1><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre></body></html>'
                if kind=='html':Path(path).write_text(document,encoding='utf-8')
                else:
                    printer=QPrinter(QPrinter.HighResolution);printer.setOutputFormat(QPrinter.PdfFormat);printer.setOutputFileName(path)
                    doc=QTextDocument();doc.setHtml(document);doc.print_(printer)
            self.window.log('Отчёт сохранён: '+path)
        except Exception as exc:self.window.log('[!] Не удалось сохранить отчёт: '+str(exc))
