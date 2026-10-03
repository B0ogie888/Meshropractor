"""Image preview and editable texture projection parameters."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QLabel, QLineEdit, QPushButton)
from display_settings import DIALOG_STYLE
from texture_geometry import read_image

IMAGE_FILTER = 'Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)'


class TextureDialog(QDialog):
    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Параметры текстуры'); self.setStyleSheet(DIALOG_STYLE); self.setMinimumWidth(470)
        self.image = np.asarray(layer['image']).copy(); self.path = layer.get('path', '')
        form = QFormLayout(self)
        self.name = QLineEdit(layer.get('name', 'Текстура')); form.addRow('Название:', self.name)
        self.preview = QLabel(); self.preview.setAlignment(Qt.AlignCenter); self.preview.setFixedHeight(170); form.addRow(self.preview)
        self.file = QPushButton('Заменить изображение…'); self.file.clicked.connect(self.choose_image); form.addRow(self.file)
        self.projection = QComboBox(); self.projection.addItems(['По граням', 'XY', 'XZ', 'YZ', 'Цилиндр', 'Сфера'])
        params = layer.get('params', {}); projection = params.get('projection', 'По граням')
        if projection not in [self.projection.itemText(i) for i in range(self.projection.count())]:
            self.projection.addItem(projection)
        self.projection.setCurrentText(projection); form.addRow('Проекция:', self.projection)
        self.fields = {}
        for key, title, default, low, high in (
            ('repeat_u', 'Повторения U:', 1., .1, 8.), ('repeat_v', 'Повторения V:', 1., .1, 8.),
            ('offset_u', 'Смещение U:', 0., -1., 1.), ('offset_v', 'Смещение V:', 0., -1., 1.),
            ('angle', 'Угол, °:', 0., -360., 360.)):
            spin = QDoubleSpinBox(); spin.setRange(low, high); spin.setDecimals(3); spin.setKeyboardTracking(False)
            spin.setValue(params.get(key, default)); self.fields[key] = spin; form.addRow(title, spin)
        self.projection.currentTextChanged.connect(self.update_fields); self.update_fields(projection)
        self.status = QLabel('Текстура меняет внешний вид. Рельеф и геометрия для печати не изменяются.'); self.status.setWordWrap(True); form.addRow(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('Применить'); buttons.button(QDialogButtonBox.Cancel).setText('Отмена')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)
        self.show_image()

    def show_image(self):
        image = np.ascontiguousarray(self.image)
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(410, 165, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.file.setToolTip(self.path or 'Изображение хранится в проекте')

    def update_fields(self, mode):
        for spin in self.fields.values(): spin.setEnabled(mode not in ('Атлас цветов', 'Исходная UV'))

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Изображение текстуры', self.path, IMAGE_FILTER)
        if not path: return
        try:
            self.image = read_image(path); self.path = path; self.show_image()
        except Exception as exc: self.status.setText(str(exc))

    def parameters(self):
        return dict(projection=self.projection.currentText(), **{key: spin.value() for key, spin in self.fields.items()})
