"""Modeless CAD tools and a scalable icon; geometry is supplied by the controller."""
from app_branding import app_icon
from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGridLayout, QLabel,
                               QPlainTextEdit, QPushButton, QVBoxLayout)

from display_settings import DIALOG_STYLE


def cad_icon(size=32):
    """A CAD solid with a curved face, rendered at multiple icon resolutions."""
    from ribbon_layout import asset_icon
    icon = asset_icon('tools', 'cad')
    if icon is not None:
        resolutions = QIcon()
        for pixels in sorted({24, 32, 48, 64, 96, max(1, int(size))}):
            resolutions.addPixmap(icon.pixmap(pixels, pixels))
        return resolutions
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">
      <g stroke="#a8c4d7" stroke-width="1.5" stroke-linejoin="round" fill="none">
        <path d="M5 11 20 4 35 12 35 29 20 37 5 28Z"/>
        <path d="M5 11 20 19 35 12 20 4Z"/>
        <path d="M20 19 35 12 35 29 20 37Z"/>
        <path d="M20 19V37 M5 11 20 19"/>
        <path d="M6 25C13 14 27 35 34 20" stroke="#b2da77" stroke-width="2.5"/>
      </g>
      <g fill="#d7efa9" stroke="#416645" stroke-width=".8">
        <circle cx="6" cy="25" r="2"/><circle cx="34" cy="20" r="2"/>
      </g>
    </svg>'''
    renderer = QSvgRenderer(QByteArray(svg))
    icon = QIcon()
    for pixels in sorted({24, 32, 48, 64, 96, max(1, int(size))}):
        pixmap = QPixmap(pixels, pixels)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class CADToolsDialog(QDialog):
    retessellate_requested = Signal()
    split_requested = Signal()
    export_requested = Signal()
    convert_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CAD / STEP')
        self.setWindowIcon(app_icon())
        self.setModal(False)
        self.setMinimumWidth(590)
        self.setStyleSheet(DIALOG_STYLE + '''
            QPlainTextEdit {background: #262626; color: #e0e0e0; border: 1px solid #555; padding: 8px;}
            QPushButton:disabled {color: #777; background: #303030; border-color: #494949;}
        ''')
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        hint = QLabel('Выбирайте детали и CAD-поверхности в рабочей сцене. '
                      'Сведения о телах, гранях и выбранной поверхности отображаются ниже.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Выберите деталь с сохранённой CAD-геометрией.')
        self.details.setMinimumHeight(175)
        layout.addWidget(self.details, 1)
        actions = QGridLayout()
        self.retessellate = QPushButton('Качество сетки…')
        self.retessellate.setToolTip('Повторно построить сетку отображения по сохранённой CAD-геометрии.')
        self.split = QPushButton('Разделить на тела')
        self.split.setToolTip('Создать отдельные детали из тел выбранной CAD-модели.')
        self.export = QPushButton('Сохранить STEP…')
        self.export.setToolTip('Сохранить CAD-геометрию выбранных деталей в файл STEP.')
        self.convert = QPushButton('Преобразовать в сетку')
        self.convert.setToolTip('Оставить треугольную сетку и удалить сохранённую CAD-геометрию.')
        for index, (button, signal) in enumerate((
                (self.retessellate, self.retessellate_requested), (self.split, self.split_requested),
                (self.export, self.export_requested), (self.convert, self.convert_requested))):
            actions.addWidget(button, index // 2, index % 2)
            button.clicked.connect(lambda checked=False, target=signal: target.emit())
        layout.addLayout(actions)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.button(QDialogButtonBox.Close).setText('Закрыть')
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.update_selection('CAD-детали не выбраны.', False, False)

    def update_selection(self, text, native_enabled, split_enabled):
        """The controller decides eligibility, including mixed and multiple selections."""
        self.summary.setText(str(text))
        enabled = bool(native_enabled)
        for button in (self.retessellate, self.export, self.convert):
            button.setEnabled(enabled)
        self.split.setEnabled(enabled and bool(split_enabled))
        if not enabled:
            self.details.clear()

    def set_info(self, text):
        """Display controller-supplied body/face measurements without executing markup."""
        self.details.setPlainText(str(text))
