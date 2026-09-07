"""STEP tessellation settings; UI angles are degrees, OCCT angles are radians."""
import math
from PySide6.QtWidgets import QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, QLabel


class StepImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Качество триангуляции STEP")
        self.setMinimumWidth(380)
        form = QFormLayout(self)
        self.linear = QDoubleSpinBox()
        self.linear.setDecimals(4)
        self.linear.setRange(.0001, 10)
        self.linear.setValue(.05)
        self.linear.setSuffix(" мм")
        self.angle = QDoubleSpinBox()
        self.angle.setDecimals(3)
        self.angle.setRange(.1, 179)
        self.angle.setValue(math.degrees(.25))
        self.angle.setSuffix(" °")
        form.addRow("Линейное отклонение:", self.linear)
        form.addRow("Угловое отклонение:", self.angle)
        hint = QLabel("Меньше значения — подробнее сетка и больше размер модели.\nЕдиницы STEP автоматически переводятся в миллиметры.")
        hint.setWordWrap(True)
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return self.linear.value(), math.radians(self.angle.value())
