"""STEP import options; UI angles are degrees, OCCT angles are radians."""
import math
from PySide6.QtWidgets import QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, QLabel, QCheckBox, QRadioButton

from display_settings import DIALOG_STYLE


class StepImportDialog(QDialog):
    def __init__(self, parent=None, *, allow_split=True):
        super().__init__(parent)
        self._allow_split = bool(allow_split)
        self.setWindowTitle("Импорт STEP — CAD/BREP или сетка STL")
        self.setMinimumWidth(500)
        self.setStyleSheet(DIALOG_STYLE + '''
            QDoubleSpinBox {background: #262626; color: #e0e0e0; border: 1px solid #666; padding: 4px;}
            QCheckBox:disabled {color: #888;}
            QRadioButton {color: #e0e0e0;}
            QRadioButton::indicator {width: 13px; height: 13px; border-radius: 7px;}
            QRadioButton::indicator:unchecked {background: #2b2b2b; border: 1px solid #aaa;}
            QRadioButton::indicator:checked {background: #4da6ff; border: 2px solid #e0e0e0;}
        ''')
        form = QFormLayout(self)
        self.native = QRadioButton('Оставить CAD-геометрию (BREP): тела и поверхности')
        self.native.setChecked(True)
        self.native.setToolTip('Хранит исходные поверхности и тела STEP вместе с сеткой для отображения.')
        form.addRow(self.native)
        self.mesh = QRadioButton('Преобразовать в треугольную сетку STL')
        self.mesh.setToolTip('Загрузить как обычную сетку для работы и экспорта в STL. Исходный STEP-файл сохраняется.')
        form.addRow(self.mesh)
        self.split_bodies = QCheckBox('Загрузить тела как отдельные детали')
        self.split_bodies.setChecked(self._allow_split)
        self.split_bodies.setEnabled(self._allow_split)
        self.split_bodies.setToolTip('Каждое CAD-тело станет отдельной деталью.' if self._allow_split else
                                    'В предеформации тела загружаются одной моделью.')
        form.addRow(self.split_bodies)
        self.native.toggled.connect(lambda enabled: self._native_changed(enabled))
        geometry_hint = QLabel('С CAD-геометрией можно повторно изменить качество сетки, '
                               'получить сведения о поверхностях и сохранить STEP. '
                               'Без неё загружается только треугольная сетка.')
        geometry_hint.setWordWrap(True)
        form.addRow(geometry_hint)
        self.linear = QDoubleSpinBox()
        self.linear.setDecimals(4)
        self.linear.setRange(.0001, 10)
        self.linear.setValue(.05)
        self.linear.setSuffix(" мм")
        self.linear.setKeyboardTracking(False)
        self.angle = QDoubleSpinBox()
        self.angle.setDecimals(3)
        self.angle.setRange(.1, 179)
        self.angle.setValue(math.degrees(.25))
        self.angle.setSuffix(" °")
        self.angle.setKeyboardTracking(False)
        form.addRow("Линейное отклонение:", self.linear)
        form.addRow("Угловое отклонение:", self.angle)
        hint = QLabel("Меньше значения — подробнее сетка и больше размер модели.\nЕдиницы STEP автоматически переводятся в миллиметры.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText('Загрузить')
        self.buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

    def _native_changed(self, enabled):
        self.split_bodies.setEnabled(bool(enabled) and self._allow_split)

    def values(self):
        return self.linear.value(), math.radians(self.angle.value())

    def options(self):
        native = self.native.isChecked()
        return dict(native=native, split_bodies=native and self._allow_split and self.split_bodies.isChecked())
