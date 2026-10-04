"""Theme-aware matrix duplication controls and an illustrative placement diagram."""
from itertools import product
import json
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QGridLayout,QWidget,
    QLabel,QSpinBox,QDoubleSpinBox,QCheckBox,QToolButton,QDialogButtonBox)
from app_branding import app_icon


class DuplicateDiagram(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.counts=[2,1,1]; self.setMinimumSize(175,135)

    def paintEvent(self,event):
        owner = self
        while owner is not None and not hasattr(owner,'engineering_theme'): owner=owner.parentWidget()
        dark = owner is not None and owner.engineering_theme.mode=='dark'
        painter=QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        ink=QColor('#e6ede7' if dark else '#252b2b'); painter.setPen(QPen(ink,1))
        colors = [QColor(c) for c in (('#a7a46e','#555b49','#737951') if dark else ('#faf4aa','#e0e4bd','#c6cca1'))]
        corners=np.array([[x,y,z] for x,y,z in product((0,.7),repeat=3)])
        projection=np.array([[.866,-.866,0],[.5,.5,-1]])
        cells=list(product(*(range(min(n,3)) for n in self.counts)))
        positions=[corners+cell for cell in cells]
        flat=np.vstack(positions)@projection.T
        scale=min((self.width()-20)/max(np.ptp(flat[:,0]),1),(self.height()-20)/max(np.ptp(flat[:,1]),1))
        middle=(flat.min(0)+flat.max(0))/2
        for vertices in sorted(positions,key=lambda v:np.mean(v@np.ones(3))):
            projected=(vertices@projection.T-middle)*scale+np.array([self.width()/2,self.height()/2])
            points=[QPointF(*p) for p in projected]
            for color,indices in zip(colors,((1,5,7,3),(4,6,7,5),(2,3,7,6))):
                painter.setBrush(color); painter.drawPolygon(QPolygonF([points[i] for i in indices]))


class DuplicateDialog(QDialog):
    changed=Signal()
    apply_requested=Signal()

    def __init__(self,operation,selected,parent=None):
        super().__init__(parent); self.operation=operation
        self.setWindowTitle('Дублировать детали · Виртуальные копии'); self.setWindowIcon(app_icon())
        self.setMinimumWidth(495); self.setWindowModality(Qt.NonModal)
        root=QVBoxLayout(self); root.setContentsMargins(18,18,18,16); root.setSpacing(12)
        heading=QLabel('ДУБЛИРОВАНИЕ / МАТРИЦА'); heading.setStyleSheet('font-size:16px; font-weight:600;'); root.addWidget(heading)
        top=QHBoxLayout(); summary=QVBoxLayout()
        summary.addWidget(QLabel('Общее количество деталей'))
        self.total=QSpinBox(); self.total.setRange(1,1_000_000); self.total.setReadOnly(True)
        self.total.setButtonSymbols(QSpinBox.NoButtons); summary.addWidget(self.total)
        self.selection=QLabel(f'Выбрано: {selected}'); summary.addWidget(self.selection)
        self.preview=QCheckBox('Предпросмотр в сцене'); self.preview.setChecked(True); summary.addWidget(self.preview)
        top.addLayout(summary,1); self.diagram=DuplicateDiagram(); top.addWidget(self.diagram,1); root.addLayout(top)
        self.toggle=QToolButton(); self.toggle.setText('▾ МАТРИЦА РАЗМЕЩЕНИЯ'); self.toggle.setCheckable(True)
        self.toggle.setChecked(True); root.addWidget(self.toggle)
        self.matrix=QWidget(); grid=QGridLayout(self.matrix); grid.setContentsMargins(0,0,0,0)
        for column,text in enumerate(('Ось','Количество','Промежуток, мм')): grid.addWidget(QLabel(text),0,column)
        self.counts=[]; self.gaps=[]
        defaults=[2,2,1] if operation=='Пакетное дублирование' else [2,1,1]
        remembered={}
        self.settings_key='duplicate_matrix_v1_'+('batch' if operation=='Пакетное дублирование' else 'single')
        try: remembered=json.loads(parent.settings.value(self.settings_key,'{}'))
        except (ValueError,TypeError,AttributeError): pass
        if not isinstance(remembered,dict): remembered={}
        for index,axis in enumerate('XYZ'):
            count=QSpinBox(); count.setRange(1,1001)
            gap=QDoubleSpinBox(); gap.setRange(0,1e6); gap.setDecimals(3); gap.setSingleStep(1)
            try:
                count.setValue(int(remembered.get('counts',defaults)[index]))
                gap.setValue(float(remembered.get('gaps',[5,5,1])[index]))
            except (ValueError,TypeError,IndexError,OverflowError): count.setValue(defaults[index]); gap.setValue([5,5,1][index])
            gap.setToolTip('Расстояние между габаритами соседних комплектов выбранных деталей, включая поддержки.')
            self.counts.append(count); self.gaps.append(gap)
            grid.addWidget(QLabel(axis),index+1,0); grid.addWidget(count,index+1,1); grid.addWidget(gap,index+1,2)
            count.valueChanged.connect(self.changed); gap.valueChanged.connect(self.changed)
        root.addWidget(self.matrix)
        self.remember=QCheckBox('Запомнить значения'); self.remember.setChecked(True); root.addWidget(self.remember)
        hint=QLabel('Количество по осям включает исходную ячейку. Выбранные детали копируются вместе, сохраняя расположение между собой.')
        hint.setWordWrap(True); root.addWidget(hint)
        self.status=QLabel(); self.status.setWordWrap(True); root.addWidget(self.status)
        self.buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        self.apply_button=self.buttons.button(QDialogButtonBox.Ok); self.apply_button.setText('Создать копии')
        self.buttons.button(QDialogButtonBox.Cancel).setText('Закрыть')
        self.buttons.accepted.connect(self.apply_requested); self.buttons.rejected.connect(self.reject); root.addWidget(self.buttons)
        self.preview.toggled.connect(self.changed); self.toggle.toggled.connect(self.toggle_matrix)

    def toggle_matrix(self,shown):
        self.matrix.setVisible(shown); self.toggle.setText(('▾' if shown else '▸')+' МАТРИЦА РАЗМЕЩЕНИЯ')

    def parameters(self):
        return dict(matrix_layout=True,counts=[w.value() for w in self.counts],gaps=[w.value() for w in self.gaps])

    def save_values(self):
        if self.remember.isChecked(): self.parent().settings.setValue(self.settings_key,json.dumps(self.parameters()))
        else: self.parent().settings.remove(self.settings_key)
