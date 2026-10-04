"""Model preparation parameters with the existing preview/apply workflow."""
import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox,QSpinBox,QCheckBox,QLineEdit,QComboBox
from PySide6.QtGui import QPainterPath,QFont
from repair_dialog import RepairDialog
from model_tool_ribbon import COMMANDS,HINTS,IMPLICIT


class ModelToolDialog(RepairDialog):
    def __init__(self,operation,count,center,parent):
        super().__init__(operation,count,center,parent,caption=COMMANDS[operation])
        self.note.setText(HINTS.get(operation,'Первая выбранная деталь — основная, остальные — инструменты.'))
        self.support_note.setText('Изменённая геометрия сохраняется как сетка. Ctrl+Z восстановит исходную деталь. Для изменения поверхности сначала удалите её поддержки и после операции создайте заново.' if operation not in ('struts','rapidfit','formfit','surface_array') else 'Создаётся новая геометрия; исходная деталь и её поддержки сохраняются.')
        source=parent.slicer_parts[parent.selected_slicer_rows()[0]]['mesh'];bounds=source.bounds
        if operation in IMPLICIT:self.field('step','Шаг расчёта, мм',.4,.001,100)
        if operation in ('hollow','shell_core','honeycomb','lattice','slice_lattice','tetra','tetra_slices','formfit','rapidfit'):
            is_structure=operation in ('honeycomb','lattice','slice_lattice','tetra','tetra_slices')
            self.field('wall','Наружная стенка, мм (0 — без стенки)' if is_structure else 'Толщина стенки, мм',1.,0 if is_structure else .001,1000)
        if operation in ('round','round_offset','perforate','struts'):self.field('radius','Радиус, мм',1.,.001,1000)
        if operation in ('offset','round_offset'):self.field('offset','Смещение, мм',1.,-1000,1000)
        if operation in ('rapidfit','formfit'):self.field('gap','Зазор, мм',.2,0,1000)
        if operation in ('honeycomb','lattice','slice_lattice','tetra','tetra_slices','perforate'):self.field('cell','Размер ячейки / шаг, мм',5.,.001,10000)
        if operation in ('honeycomb','lattice','slice_lattice','tetra','tetra_slices'):self.field('thickness','Толщина стенок / рёбер, мм',1.,.001,1000)
        if operation in ('extrude','surface_array'):
            self.field('distance','Выдвижение / шаг, мм',2.,-1000,1000)
            self.only_faces.setChecked(True);self.only_faces.setVisible(False)
        if operation=='surface_array':self.field('count','Количество копий',3,1,100)
        if operation=='cut':self.add_vector('point','Точка плоскости, мм',center);self.add_vector('normal','Нормаль плоскости',[0,0,1])
        if operation=='perforate':
            self.axis=QComboBox();self.axis.addItems(['X','Y','Z']);self.axis.setCurrentIndex(2)
            self.axis.currentIndexChanged.connect(lambda *_:self.changed.emit());self.form.addRow('Направление отверстий:',self.axis)
        if operation=='struts':
            self.add_vector('start','Начало, мм',bounds[0]);self.add_vector('end','Конец, мм',bounds[1])
        if operation=='label':
            self.text=QLineEdit('Meshropractor');self.text.textChanged.connect(lambda *_:self.changed.emit());self.form.addRow('Текст:',self.text)
            self.text.setMaxLength(128)
            self.field('text_size','Высота текста, мм',5.,.01,1000)
            self.field('height','Высота / глубина рельефа, мм',1.,.001,1000)
            self.add_vector('position','Центр XY / низ текста Z, мм',[center[0],center[1],bounds[1,2]-.5])
            for key,title in [('engrave','Выгравировать вместо выдвижения'),('standalone','Создать отдельной деталью')]:
                check=QCheckBox(title);check.toggled.connect(lambda *_:self.changed.emit());self.flags[key]=check;self.form.addRow(check)

    def field(self,key,title,value,low,high):
        widget=QSpinBox() if isinstance(value,int) else QDoubleSpinBox()
        if isinstance(widget,QDoubleSpinBox):widget.setDecimals(4)
        widget.setRange(low,high);widget.setValue(value);widget.setKeyboardTracking(False)
        widget.valueChanged.connect(lambda *_:self.changed.emit());self.fields[key]=widget;self.form.addRow(title+':',widget)

    def parameters(self):
        result=super().parameters()
        if self.operation=='perforate':result['axis']=self.axis.currentIndex()
        if self.operation=='label':
            if not self.text.text().strip():raise ValueError('Введите текст маркировки.')
            path=QPainterPath();path.addText(0,0,QFont('Arial',40),self.text.text())
            rect=path.boundingRect()
            if rect.height()<=0:raise ValueError('Текст не образует контуров.')
            scale=result['text_size']/rect.height()
            result['text_contours']=[[(scale*(point.x()-rect.center().x()),-scale*(point.y()-rect.center().y())) for point in polygon] for polygon in path.toSubpathPolygons()]
        return result
