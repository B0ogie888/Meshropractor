"""Compact creation and positioning groups for the slicer ribbon."""
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QLabel, QFrame
from slicer_tools import TOOL_NAMES


def main_icon(name):
    sheet = '<path d="M12 5H30L39 14V43H12Z M30 5V14H39"/>'
    folder = '<path d="M5 13H21L26 18H43V39H5Z M5 21H43"/>'
    disk = '<path d="M7 5H35L42 12V43H7Z M15 5V19H33V5 M15 43V28H34V43"/>'
    cube = '<path d="M8 14 24 6 40 14V34L24 42 8 34Z M8 14 24 23 40 14 M24 23V42"/>'
    body = sheet if name == 'Новый проект' else folder if name in ('Загрузить проект', 'Сохранить все в папку') else disk if name.startswith('Сохранить проект') else cube
    accent = {
        'Новый проект': '<path d="M26 27H40M33 20V34"/>',
        'Загрузить проект': '<path d="M25 5V24M18 17 25 24 32 17"/>',
        'Сохранить проект': '',
        'Сохранить проект как': '<path d="M25 39 39 25 43 29 29 43Z"/>',
        'Импорт детали': '<path d="M3 24H24M17 17 24 24 17 31"/>',
        'Сохранить выбранные детали как': '<path d="M25 24H45M38 17 45 24 38 31"/>',
        'Сохранить все в папку': '<path d="M31 26V44M24 37 31 44 38 37"/>',
        'Выгрузить деталь': '<path d="M29 8 42 21M42 8 29 21"/>',
    }.get(name, '')
    pixmap = QPixmap()
    pixmap.loadFromData(f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"><g fill="none" stroke="#b9c2ca" stroke-width="2" stroke-linejoin="round">{body}</g><g fill="none" stroke="#67b9ec" stroke-width="2.8">{accent}</g></svg>'.encode(), 'SVG')
    return QIcon(pixmap)


def tool_icon(index):
    cube = '<path d="M8 12 22 5 36 12 36 29 22 37 8 29Z M8 12 22 20 36 12 M22 20V37" fill="none" stroke="#b9c2ca" stroke-width="1.8" stroke-linejoin="round"/>'
    overlays = [
        '<path d="M34 26V40M27 33H41" stroke="#f0b949" stroke-width="3"/>',
        '<g transform="translate(18 14) scale(.65)">' + cube + '</g>',
        '<g transform="translate(22 0) scale(.6)">' + cube + '</g><g transform="translate(22 22) scale(.6)">' + cube + '</g>',
        '<path d="M26 34H44M39 29 44 34 39 39M35 25V43M30 30 35 25 40 30" fill="none" stroke="#67b9ec" stroke-width="2"/>',
        '<path d="M27 37A10 10 0 1 1 43 29M37 29H43V23" fill="none" stroke="#67b9ec" stroke-width="2"/>',
        '<path d="M27 40 43 24M35 24H43V32M27 32V40H35" fill="none" stroke="#67b9ec" stroke-width="2"/>',
        '<path d="M24 3V43" stroke="#67b9ec" stroke-width="2" stroke-dasharray="3 2"/>',
    ]
    pixmap = QPixmap()
    pixmap.loadFromData(f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{cube}{overlays[index]}</svg>'.encode(), 'SVG')
    return QIcon(pixmap)


def create_tools_ribbon():
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(8, 0, 8, 0)
    layout.setSpacing(12)
    buttons = {}
    for title, indices in (("Создать", range(3)), ("Расположение", range(3, 7))):
        if buttons:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            layout.addWidget(line)
        group = QVBoxLayout()
        group.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(4)
        for index in indices:
            name = TOOL_NAMES[index]
            button = QToolButton()
            button.setText(name.replace('Пакетное ', 'Пакетное\n'))
            button.setIcon(tool_icon(index))
            button.setIconSize(QSize(28, 28))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumWidth(75)
            button.setStyleSheet('QToolButton {border: none; padding: 0 6px; color: #ddd; font-size: 11px;} QToolButton:hover {background: #444;} QToolButton:disabled {color: #777;}')
            button.setToolTip(name + (": отметьте детали в таблице" if index else ": параллелепипед, цилиндр или сфера"))
            row.addWidget(button)
            buttons[name] = button
        group.addLayout(row)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color: #aaa; font-size: 10px;')
        group.addWidget(label)
        layout.addLayout(group)
    layout.addStretch()
    return container, buttons
