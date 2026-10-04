"""Modeless marking editor: content, typography and surface projection."""
import base64
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSignalBlocker, QSize
from PySide6.QtGui import QFont, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QToolButton, QComboBox, QCheckBox, QPlainTextEdit,
    QFontComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QScrollArea, QFrame)
from app_branding import app_icon
from ribbon_layout import asset_icon


class MarkingDialog(QDialog):
    changed = Signal()
    new_area = Signal()
    delete_area = Signal()
    save_requested = Signal()
    preview_requested = Signal()
    apply_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, name, settings, parent):
        super().__init__(parent)
        self.name, self.settings = name, settings
        self.running = False; self.preview_running = False; self.content = 'text'; self.image_data = ''; self.fields = {}; self.flags = {}
        self.setWindowTitle('Маркировка'); self.setWindowIcon(app_icon()); self.setMinimumWidth(500)
        self.setStyleSheet('''QDialog {background:#fafbf8; color:#252b2b;}
            QLabel#MarkingTitle {font-size:18px; font-weight:600;}
            QLabel#MarkingHint {color:#5b6463;}
            QToolButton:checked {background:#e5d943; color:#252b2b;}
            QPushButton#MarkingApply {background:#e5d943; color:#252b2b; padding:8px;}
        ''')
        layout = QVBoxLayout(self); layout.setContentsMargins(18,18,18,16); layout.setSpacing(10)
        title = QLabel('МАРКИРОВКА / ПОВЕРХНОСТЬ'); title.setObjectName('MarkingTitle'); layout.addWidget(title)
        hint = QLabel('На выбранной детали протяните рамку ЛКМ.\nПКМ — вращение; параметры области можно менять ниже.')
        hint.setObjectName('MarkingHint'); hint.setWordWrap(True); layout.addWidget(hint)
        self.editor = QWidget(); outer = QVBoxLayout(self.editor); outer.setContentsMargins(0,0,0,0)
        row = QHBoxLayout(); self.areas = QComboBox(); row.addWidget(self.areas,1)
        add = QPushButton('Новая область'); add.setIcon(asset_icon('main','Новый проект'))
        add.clicked.connect(self.new_area); row.addWidget(add)
        remove = QToolButton(); remove.setIcon(asset_icon('texture','delete')); remove.setToolTip('Удалить выбранную область')
        remove.clicked.connect(self.delete_area); row.addWidget(remove); outer.addLayout(row)
        self.tabs = QTabWidget(); outer.addWidget(self.tabs)
        pages = []
        for text in ('Текст','Рисунки','Проекция','Матричный штрихкод'):
            page = QWidget(); form = QFormLayout(page); form.setContentsMargins(10,10,10,10)
            self.tabs.addTab(page,text); pages.append(form)
        self.tabs.currentChanged.connect(self.tab_changed)
        form = pages[0]
        self.text = QPlainTextEdit(str(settings.value('marking/text', 'Meshropractor')))
        self.text.setMaximumHeight(90); self.text.textChanged.connect(self.changed); form.addRow(self.text)
        name_row = QHBoxLayout()
        self.flags['auto_name'] = QCheckBox('Название детали'); name_row.addWidget(self.flags['auto_name'])
        self.flags['auto_name'].toggled.connect(self.use_name)
        self.flags['remember'] = QCheckBox('Запомнить текст'); self.flags['remember'].setChecked(True); name_row.addWidget(self.flags['remember'])
        form.addRow(name_row)
        fonts = QHBoxLayout(); self.font = QFontComboBox(); fonts.addWidget(self.font,1)
        self.font.currentFontChanged.connect(self.changed)
        for key, label in (('bold','B'),('italic','I'),('underline','U'),('strike','S')):
            button = QToolButton(); button.setText(label); button.setCheckable(True)
            button.setToolTip(dict(bold='Полужирный',italic='Курсив',underline='Подчёркнутый',strike='Зачёркнутый')[key])
            button.toggled.connect(self.changed); self.flags[key] = button; fonts.addWidget(button)
        form.addRow(fonts)
        self.spin(form,'text_size','Высота текста',5.,.1,1000,' мм')
        self.points = QDoubleSpinBox(); self.points.setRange(.28,2835); self.points.setSuffix(' пт'); self.points.setValue(5*72/25.4)
        self.points.valueChanged.connect(self.from_points); self.fields['text_size'].valueChanged.connect(self.to_points)
        form.addRow('Размер в пунктах',self.points)
        self.align = QComboBox(); self.align.addItems(['По левому краю','По центру','По правому краю']); self.align.setCurrentIndex(1)
        self.align.currentIndexChanged.connect(self.changed); form.addRow('Выравнивание',self.align)
        self.spin(form,'spacing','Межстрочный интервал',1.15,.6,3.,' ×')
        form = pages[1]
        choose = QPushButton('Загрузить рисунок…'); choose.setIcon(asset_icon('display','export_png'))
        choose.clicked.connect(self.load_image); form.addRow(choose)
        self.image_label = QLabel('PNG, JPEG, BMP, SVG → монохромный рельеф'); self.image_label.setWordWrap(True); form.addRow(self.image_label)
        self.spin(form,'threshold','Порог яркости',128.,1.,254.,'')
        self.spin(form,'raster_size','Разрешение рисунка',96.,16.,256.,' px')
        self.check(form,'invert_image','Инвертировать рисунок',False)
        form = pages[2]
        self.check(form,'project','Проецировать на поверхность детали',True)
        self.check(form,'fit','Уместить содержимое в области',True)
        self.spin(form,'width','Ширина области',20.,.1,10000,' мм')
        self.spin(form,'area_height','Высота области',10.,.1,10000,' мм')
        self.spin(form,'angle','Поворот',0.,-360,360,'°')
        self.spin(form,'shift_x','Сдвиг по горизонтали',0.,-10000,10000,' мм')
        self.spin(form,'shift_y','Сдвиг по вертикали',0.,-10000,10000,' мм')
        self.spin(form,'resolution','Шаг проекции',1.,.05,20.,' мм')
        tip = QLabel('Рельеф следует видимой стороне поверхности.\nПри выходе за край уменьшите область или высоту текста.')
        tip.setWordWrap(True); form.addRow(tip)
        form = pages[3]
        self.code = QPlainTextEdit('101-1138.2'); self.code.setMaximumHeight(90); self.code.textChanged.connect(self.changed)
        form.addRow('Data Matrix ECC 200',self.code)
        tip = QLabel('Код с коррекцией ошибок и свободной зоной.\nРазмер задаётся областью. Для читаемости используйте плоский участок.')
        tip.setWordWrap(True); form.addRow(tip)
        self.code_preview = QLabel(); self.code_preview.setAlignment(Qt.AlignCenter); form.addRow(self.code_preview)
        common = QFormLayout(); outer.addLayout(common)
        self.shape = QComboBox(); self.shape.addItems(['Прямоугольная маркировка','Круговая маркировка'])
        self.shape.currentIndexChanged.connect(self.changed); common.addRow('Форма',self.shape)
        self.spin(common,'depth','Высота / глубина рельефа',.4,.005,1000,' мм')
        self.method = QComboBox(); self.method.addItems(['Выступающий рельеф','Гравировка']); self.method.currentIndexChanged.connect(self.changed)
        common.addRow('Способ нанесения',self.method)
        advanced = QGroupBox('Расширенные'); advanced.setCheckable(True); advanced.setChecked(False)
        advanced_layout = QVBoxLayout(advanced)
        options = QWidget(); opts = QFormLayout(options); opts.setContentsMargins(0,0,0,0)
        for key,title,value in [('through','Сквозная маркировка',False),('auto_preview','Автообновление предпросмотра',True),
                                ('auto_save','Автосохранение областей при закрытии',True)]: self.check(opts,key,title,value)
        advanced_layout.addWidget(options); options.hide(); advanced.toggled.connect(options.setVisible)
        outer.addWidget(advanced); outer.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.editor); layout.addWidget(scroll,1)
        self.status = QLabel('Задайте область на детали.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        row = QHBoxLayout()
        self.preview = QPushButton('Обновить'); self.preview.clicked.connect(self.preview_requested); row.addWidget(self.preview)
        self.save = QPushButton('Сохранить запланированное'); self.save.clicked.connect(self.save_requested); row.addWidget(self.save)
        self.export = QPushButton('Отдельная STL…'); self.export.clicked.connect(self.export_requested); row.addWidget(self.export)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.apply = QPushButton('Срастить с деталью'); self.apply.setObjectName('MarkingApply'); self.apply.clicked.connect(self.apply_requested); row.addWidget(self.apply,1)
        self.cancel = QPushButton('Отменить расчёт'); self.cancel.clicked.connect(self.cancel_requested); self.cancel.setEnabled(False); row.addWidget(self.cancel)
        self.close_button = QPushButton('Закрыть'); self.close_button.clicked.connect(self.reject); row.addWidget(self.close_button)
        layout.addLayout(row)
        self.apply.setEnabled(False); self.export.setEnabled(False)
        available = self.screen().availableGeometry()
        self.resize(560, min(820, available.height()-80))

    def spin(self, form, key, title, value, low, high, suffix):
        spin = QDoubleSpinBox(); spin.setRange(low,high); spin.setDecimals(3); spin.setValue(value)
        spin.setSuffix(suffix); spin.setKeyboardTracking(False); spin.valueChanged.connect(self.changed)
        self.fields[key] = spin; form.addRow(title,spin)

    def check(self, form, key, title, value):
        check = QCheckBox(title); check.setChecked(value); check.toggled.connect(self.changed)
        self.flags[key] = check; form.addRow(check)

    def tab_changed(self,index):
        if index in (0,1,3): self.content = {0:'text',1:'image',3:'datamatrix'}[index]
        self.changed.emit()

    def from_points(self, value):
        with QSignalBlocker(self.fields['text_size']): self.fields['text_size'].setValue(value*25.4/72)
        self.changed.emit()

    def to_points(self, value):
        with QSignalBlocker(self.points): self.points.setValue(value*72/25.4)

    def use_name(self, checked):
        if checked: self.text.setPlainText(Path(self.name).stem)
        self.text.setReadOnly(checked); self.changed.emit()

    def load_image(self):
        path,_ = QFileDialog.getOpenFileName(self,'Рисунок маркировки','','Рисунки (*.png *.jpg *.jpeg *.bmp *.svg)')
        if not path: return
        if Path(path).stat().st_size > 8*1024**2:
            self.status.setText('Рисунок должен быть меньше 8 МБ.'); return
        if Path(path).suffix.lower()=='.svg':
            from PySide6.QtSvg import QSvgRenderer
            renderer = QSvgRenderer(path)
            if not renderer.isValid(): self.status.setText('Не удалось прочитать SVG.'); return
            size = renderer.defaultSize().scaled(QSize(384,384),Qt.KeepAspectRatio)
            image = QImage(size,QImage.Format_ARGB32_Premultiplied); image.fill(Qt.transparent)
            painter = QPainter(image); renderer.render(painter); painter.end()
        else: image = QImage(path)
        if image.isNull(): self.status.setText('Не удалось прочитать рисунок.'); return
        from PySide6.QtCore import QBuffer, QIODevice
        buffer = QBuffer(); buffer.open(QIODevice.WriteOnly)
        image.scaled(384,384,Qt.KeepAspectRatio,Qt.SmoothTransformation).save(buffer,'PNG')
        self.image_data = base64.b64encode(bytes(buffer.data())).decode('ascii')
        self.refresh_image()
        self.changed.emit()

    def refresh_image(self):
        try: image = QImage.fromData(base64.b64decode(self.image_data)) if self.image_data else QImage()
        except ValueError: image = QImage()
        if image.isNull(): self.image_label.setText('PNG, JPEG, BMP, SVG → монохромный рельеф')
        else: self.image_label.setPixmap(QPixmap.fromImage(image).scaled(160,90,Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def values(self):
        return dict({key:widget.value() for key,widget in self.fields.items()},
            **{key:widget.isChecked() for key,widget in self.flags.items()}, text=self.text.toPlainText(),
            font=self.font.currentFont().family(), align=('left','center','right')[self.align.currentIndex()],
            circular=bool(self.shape.currentIndex()), engrave=bool(self.method.currentIndex()),
            content=self.content, code=self.code.toPlainText(), image=self.image_data)

    def load_values(self, values):
        widgets = list(self.fields.values())+list(self.flags.values())+[self.text,self.font,self.align,self.shape,self.method,self.tabs,self.code]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        for key,widget in self.fields.items():
            if key in values: widget.setValue(values[key])
        for key,widget in self.flags.items():
            if key in values: widget.setChecked(bool(values[key]))
        self.text.setPlainText(values.get('text','')); self.text.setReadOnly(values.get('auto_name',False))
        self.font.setCurrentFont(QFont(values.get('font','Arial')))
        self.align.setCurrentIndex(('left','center','right').index(values.get('align','center')))
        self.shape.setCurrentIndex(int(values.get('circular',False))); self.method.setCurrentIndex(int(values.get('engrave',False)))
        self.content = values.get('content','text'); self.tabs.setCurrentIndex({'text':0,'image':1,'datamatrix':3}[self.content])
        self.code.setPlainText(values.get('code','')); self.image_data = values.get('image','')
        self.refresh_image()
        self.to_points(self.fields['text_size'].value())
        del blockers

    def parameters(self):
        from marking_content import content_parameters
        def preview(png):
            image = QImage.fromData(png)
            self.code_preview.setPixmap(QPixmap.fromImage(image).scaled(112,112,Qt.KeepAspectRatio,Qt.FastTransformation))
        return content_parameters(self.values(),preview)

    def set_preview_running(self, running):
        # Keep keyboard focus and text selection during background calculation.
        self.preview_running = running
        self.preview.setEnabled(not running)
        self.cancel.setEnabled(running)
        if running: self.apply.setEnabled(False); self.export.setEnabled(False)

    def set_running(self, running):
        self.running = running; self.editor.setEnabled(not running)
        for widget in (self.preview,self.save,self.close_button): widget.setEnabled(not running)
        self.cancel.setEnabled(running)
        if running: self.apply.setEnabled(False); self.export.setEnabled(False)

    def reject(self):
        if self.running: self.cancel_requested.emit(); return
        super().reject()
