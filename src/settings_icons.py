"""Settings/help commands rendered consistently with the slicer ribbon."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


SETTINGS_COMMANDS = ('Параметры', 'Горячие клавиши', 'Справка', 'Проверить обновления', 'О программе')


def settings_icon(name):
    shapes = {
        'Параметры': '<path d="M9 10V38M24 10V38M39 10V38"/><path d="M4 18H14V25H4ZM19 28H29V35H19ZM34 13H44V20H34Z" fill="#67b9ec"/>',
        'Горячие клавиши': '<rect x="4" y="11" width="40" height="27" rx="4"/><path d="M10 18H13M19 18H22M28 18H31M37 18H39M10 25H13M19 25H22M28 25H31M37 25H39M13 32H34"/>',
        'Справка': '<path d="M24 12C18 7 11 7 5 9V37C12 35 19 37 24 41C29 37 36 35 43 37V9C37 7 30 7 24 12ZM24 12V41M10 16L19 18M10 23L19 25M29 18L38 16M29 25L38 23"/>',
        'Проверить обновления': '<path d="M10 18A15 15 0 0 1 36 12L40 17M40 6V17H29M38 30A15 15 0 0 1 12 36L8 31M8 42V31H19"/><path d="M24 16V31M18 25L24 31 30 25" stroke="#67b9ec"/>',
        'О программе': '<circle cx="24" cy="24" r="19"/><path d="M24 22V35M21 35H27M21 22H24" stroke="#67b9ec"/><circle cx="24" cy="14" r="1.5" fill="#67b9ec"/>',
    }
    pixmap = QPixmap()
    pixmap.loadFromData(('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48">'
                        '<g fill="none" stroke="#b9c2ca" stroke-width="2" stroke-linejoin="round" '
                        f'stroke-linecap="round">{shapes[name]}</g></svg>').encode(), 'SVG')
    return QIcon(pixmap)


def create_settings_ribbon():
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(8, 0, 8, 0)
    buttons = {}
    for title, names in (('Отображение', SETTINGS_COMMANDS[:1]), ('Помощь', SETTINGS_COMMANDS[1:])):
        if buttons:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            layout.addWidget(line)
        group = QVBoxLayout()
        group.setSpacing(0)
        row = QHBoxLayout()
        for name in names:
            button = QToolButton()
            button.setText(name.replace(' ', '\n', 1))
            button.setToolTip(name)
            button.setIcon(settings_icon(name))
            button.setIconSize(QSize(28, 28))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumWidth(85)
            button.setStyleSheet('QToolButton {border: none; padding: 0 8px; color: #ddd; font-size: 11px;} '
                                'QToolButton:hover {background: #444;} QToolButton:disabled {color: #777;}')
            row.addWidget(button)
            buttons[name] = button
        group.addLayout(row)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color: #aaa; font-size: 10px;')
        group.addWidget(label)
        layout.addLayout(group)
    layout.addStretch()
    return widget, buttons
