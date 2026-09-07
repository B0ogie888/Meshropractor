"""Shared vector icons for workspace tools; crisp on high-DPI displays."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QWidget, QHBoxLayout, QToolButton

SUPPORT_NAMES = ['Генерация поддержек', 'Поддержки для выбранных', 'Ветвящиеся поддержки',
                 'Поддержки вручную', 'Предпросмотр области поддержек']


def workspace_icon(kind):
    paths = {
        'platform': '<path d="M4 29 23 19 44 29 24 40Z M4 29V34L24 45 44 34V29 M10 32V40M38 33V40"/>',
        'part': '<path d="M10 12 25 4 40 12V31L25 40 10 31Z M10 12 25 21 40 12 M25 21V40"/>',
        'triangle': '<path d="M6 38 24 6 42 38Z M24 6V38 M6 38 33 22M42 38 15 22"/>',
        'plane': '<path d="M7 14 39 6V36L7 43Z M7 14 39 36"/>',
        'smooth': '<path d="M6 38C5 12 32 38 38 6L44 15C35 44 20 19 16 44Z"/>',
        'component': '<path d="M7 13 20 6 32 14V29L20 36 7 29Z M22 23 35 17 44 25V39L32 45 22 38Z"/>',
        'brush': '<path d="M12 30 33 5 42 13 20 35Z M12 30C2 31 12 39 5 44 25 43 25 36 20 35"/>',
        'rectangle': '<path d="M5 9H43V39H5Z M13 31 23 17 33 31Z"/>',
        'clear': '<path d="M8 8 40 40M40 8 8 40"/>',
        'support': '<path d="M6 12 24 4 42 12 24 21Z M6 12V21L24 29 42 21V12 M11 25V41M24 30V41M37 25V41M5 43H43"/>',
        'selected': '<path d="M6 12 24 4 42 12 24 21Z M6 12V21L24 29 42 21V12 M11 25V41M24 30V41M37 25V41M5 43H43 M17 12 23 17 33 7"/>',
        'tree': '<path d="M5 12 24 4 43 12 24 22Z M10 19 24 32 38 19 M24 22V44M12 44H36 M15 24V15M33 24V15"/>',
        'manual': '<path d="M6 12 24 4 42 12 24 21Z M24 21V42M12 44H36 M34 24V36M28 30H40"/>',
        'preview': '<path d="M5 12 23 4 40 12 23 21Z M5 12V27L20 36 M7 36Q25 20 43 36Q25 50 7 36Z"/><circle cx="25" cy="36" r="5"/>',
    }
    pixmap = QPixmap()
    pixmap.loadFromData(('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"><g fill="none" stroke="#9bce73" stroke-width="2.2" stroke-linejoin="round">' + paths[kind] + '</g></svg>').encode(), 'SVG')
    return QIcon(pixmap)


def create_workspace_ribbon(names, icons):
    panel = QWidget()
    layout = QHBoxLayout(panel)
    layout.setContentsMargins(8, 0, 8, 0)
    buttons = {}
    for name, icon in zip(names, icons):
        button = QToolButton()
        button.setText(name.replace(' области ', '\nобласти ').replace(' для ', '\nдля '))
        button.setIcon(workspace_icon(icon))
        button.setIconSize(QSize(28, 28))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setFixedHeight(64)
        button.setToolTip(name)
        button.setStyleSheet('QToolButton {color: #ddd; border: none; padding: 0 8px;} QToolButton:hover {background: #444;} QToolButton:checked {background: #485b36;}')
        layout.addWidget(button)
        buttons[name] = button
    layout.addStretch()
    return panel, buttons
