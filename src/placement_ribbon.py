"""Basic and automatic placement tools in a compact, scrollable slicer ribbon."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from tool_ribbon import tool_icon


PLACEMENT_COMMANDS = {
    'move': 'Перемещать',
    'rotate': 'Вращать',
    'free_move': 'Свободное перемещение',
    'scale': 'Масштабировать',
    'mirror': 'Отзеркалить',
    'top_bottom': 'Верхняя/нижняя поверхность',
    'auto_arrange': 'Автоматическое размещение',
    'optimize_orientation': 'Оптимизация положения',
    'compare_orientations': 'Сравнение положений',
    'minimize_bbox': 'Минимизировать ограничивающий параллелепипед',
    'fit_platform': 'Подогнать под платформу',
    'sort_by_shape': 'Сортировать по форме',
    'pack_3d': '3D-размещение',
}

_CAPTIONS = {
    'move': 'Перемещать',
    'rotate': 'Вращать',
    'free_move': 'Свободное\nперемещение',
    'scale': 'Масштабировать',
    'mirror': 'Отзеркалить',
    'top_bottom': 'Верхняя/нижняя\nповерхность',
    'auto_arrange': 'Автоматическое\nразмещение',
    'optimize_orientation': 'Оптимизация\nположения',
    'compare_orientations': 'Сравнение\nположений',
    'minimize_bbox': 'Минимизировать ограничивающий\nпараллелепипед',
    'fit_platform': 'Подогнать\nпод платформу',
    'sort_by_shape': 'Сортировать\nпо форме',
    'pack_3d': '3D-размещение',
}

_HINTS = {
    'move': 'Переместить выбранные детали по координатам или относительно текущего положения.',
    'rotate': 'Повернуть выбранные детали вокруг центра или заданной оси.',
    'free_move': 'Перемещать выбранные детали непосредственно в рабочей сцене.',
    'scale': 'Изменить масштаб или конечные размеры выбранных деталей.',
    'mirror': 'Отразить выбранные детали относительно плоскости.',
    'top_bottom': 'Ориентировать выбранную поверхность детали вверх или к платформе.',
    'auto_arrange': 'Расположить выбранные детали на платформе с заданными промежутками.',
    'optimize_orientation': 'Подобрать ориентацию выбранных деталей по заданным критериям.',
    'compare_orientations': 'Сравнить варианты ориентации детали перед применением.',
    'minimize_bbox': 'Подобрать ориентацию с компактным ограничивающим параллелепипедом.',
    'fit_platform': 'Подогнать выбранные детали под рабочие габариты платформы.',
    'sort_by_shape': 'Перенести ориентацию детали-образца на детали похожей формы, сохраняя их центры и размеры.',
    'pack_3d': 'Разместить выбранные детали в объёме платформы по трём координатам.',
}

_GROUPS = (
    ('Базовый', ('move', 'rotate', 'free_move', 'scale', 'mirror', 'top_bottom')),
    ('Автоматический', ('auto_arrange', 'optimize_orientation', 'compare_orientations',
                         'minimize_bbox', 'fit_platform', 'sort_by_shape', 'pack_3d')),
)


def placement_icon(operation):
    from ribbon_layout import asset_icon
    icon = asset_icon('placement', operation)
    if icon is not None: return icon
    """Reuse familiar transform symbols; draw the other placement actions individually."""
    basic_icons = {'move': 3, 'rotate': 4, 'scale': 5, 'mirror': 6}
    if operation in basic_icons:
        return tool_icon(basic_icons[operation])
    cube = '<path d="M9 13 24 5 39 13V32L24 41 9 32ZM9 13 24 22 39 13M24 22V41"/>'
    shapes = {
        'free_move': (
            '<path d="M16 18 24 14 32 18V28L24 33 16 28ZM16 18 24 23 32 18M24 23V33"/>',
            '<path d="M4 24H14M9 19 4 24 9 29M34 24H44M39 19 44 24 39 29M24 3V12M19 8 24 3 29 8M24 35V45M19 40 24 45 29 40"/>'),
        'top_bottom': (
            cube,
            '<path d="M9 13 24 5 39 13 24 22Z" fill="#345867"/><path d="M5 43H43M44 31V9M40 14 44 9 48 14M44 31 40 26M44 31 48 26"/>'),
        'auto_arrange': (
            '<path d="M4 17 24 6 44 17V37L24 46 4 37ZM4 17 24 28 44 17M24 28V46"/>',
            '<path d="M10 17 17 13 24 17 17 21ZM24 17 31 13 38 17 31 21ZM17 24 24 20 31 24 24 28Z" fill="#3d5960"/>'),
        'optimize_orientation': (
            '<path d="M12 18 27 9 36 21 22 32ZM12 18V31L22 41V32M22 41 36 31V21"/>',
            '<path d="M6 19A18 18 0 0 1 36 7M36 2V9H29M42 29A18 18 0 0 1 12 41M12 46V39H19"/>'),
        'compare_orientations': (
            '<path d="M4 11 13 6 22 11V26L13 31 4 26ZM4 11 13 16 22 11M13 16V31M29 7 44 12V27L35 34 25 22ZM29 7 35 20 44 12M35 20V34"/>',
            '<path d="M8 40H39M13 35 8 40 13 45M34 35 39 40 34 45"/>'),
        'minimize_bbox': (
            '<path d="M5 11 24 2 43 11V36L24 46 5 36ZM5 11 24 22 43 11M24 22V46" stroke-dasharray="3 2"/><path d="M16 17 28 11 34 24 22 30ZM16 17V29L22 37V30M22 37 34 31V24"/>',
            '<path d="M1 24H12M7 19 12 24 7 29M47 24H36M41 19 36 24 41 29"/>'),
        'fit_platform': (
            '<path d="M3 34 24 23 45 34 24 45ZM3 34V38L24 49 45 38V34M14 9 25 3 36 9V23L25 29 14 23ZM14 9 25 15 36 9M25 15V29"/>',
            '<path d="M8 9V30M4 25 8 30 12 25M41 9V30M37 25 41 30 45 25M15 39 24 35 33 39"/>'),
        'sort_by_shape': (
            '<path d="M6 29V15L15 10 24 15V29L15 34ZM6 15 15 20 24 15M15 20V34M30 16C30 11 42 11 42 16V30C42 35 30 35 30 30ZM30 16C30 21 42 21 42 16"/>',
            '<path d="M8 42H40M35 37 40 42 35 47"/><circle cx="12" cy="4" r="2"/><circle cx="36" cy="4" r="2"/>'),
        'pack_3d': (
            '<path d="M4 12 24 2 44 12V36L24 46 4 36ZM4 12 24 23 44 12M24 23V46"/>',
            '<path d="M8 14 16 10 24 14V24L16 28 8 24ZM8 14 16 18 24 14M16 18V28M24 25 32 21 40 25V35L32 39 24 35ZM24 25 32 29 40 25M32 29V39" fill="#3a5663"/>'),
    }
    outline, accent = shapes[operation]
    colour = '#ebbb66' if operation in ('auto_arrange', 'minimize_bbox', 'pack_3d') else '#70bddb'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 50">'
           f'<g fill="none" stroke="#bec7ce" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{outline}</g>'
           f'<g fill="none" stroke="{colour}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{accent}</g></svg>')
    icon = QIcon()
    for size in (24, 48, 96):
        pixmap = QPixmap()
        pixmap.loadFromData(svg.replace('width="48" height="48"', f'width="{size}" height="{size}"').encode(), 'SVG')
        icon.addPixmap(pixmap)
    return icon


class _PlacementScrollArea(QScrollArea):
    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            pixels, angle = event.pixelDelta(), event.angleDelta()
            delta = pixels.x() or pixels.y() or (angle.x() or angle.y()) / 120 * 96
            bar.setValue(bar.value() - round(delta))
            event.accept()
        else:
            super().wheelEvent(event)


def create_placement_ribbon():
    scroll = _PlacementScrollArea()
    scroll.setObjectName('placement_ribbon')
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroll.setStyleSheet('''
        QScrollArea, QScrollArea > QWidget > QWidget {background: #2b2b2b;}
        QToolButton {color: #e0e0e0; border: 1px solid transparent; border-radius: 3px;
                     padding: 0 5px; font-size: 10px;}
        QToolButton:hover {background: #41464a; border-color: #626c73;}
        QToolButton:pressed {background: #244650; border-color: #70bddb;}
        QToolButton:focus {border-color: #70bddb;}
        QToolButton:disabled {color: #777;}
        QScrollBar:horizontal {height: 7px; border: none; background: #242424; margin: 0;}
        QScrollBar::handle:horizontal {background: #60666b; min-width: 28px; border-radius: 3px;}
        QScrollBar::handle:horizontal:hover {background: #8a949d;}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {width: 0;}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {background: none;}
    ''')
    scroll.horizontalScrollBar().setToolTip('Прокрутка инструментов расположения. Можно использовать колесо мыши.')
    panel = QWidget()
    layout = QHBoxLayout(panel)
    layout.setContentsMargins(7, 1, 7, 1)
    layout.setSpacing(7)
    buttons = {}
    for title, operations in _GROUPS:
        if buttons:
            divider = QFrame()
            divider.setFrameShape(QFrame.VLine)
            divider.setFixedWidth(1)
            divider.setStyleSheet('background: #515151;')
            layout.addWidget(divider)
        group = QVBoxLayout()
        group.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(1)
        for operation in operations:
            button = QToolButton()
            button.setObjectName(f'placement_{operation}')
            button.setText(_CAPTIONS[operation])
            button.setAccessibleName(PLACEMENT_COMMANDS[operation])
            button.setToolTip(f'{PLACEMENT_COMMANDS[operation]}\n\n{_HINTS[operation]}')
            button.setIcon(placement_icon(operation))
            button.setIconSize(QSize(24, 24))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setCursor(Qt.PointingHandCursor)
            font = QFont(button.font())
            font.setPixelSize(10)
            button.setFont(font)
            metrics = QFontMetrics(font)
            button.setFixedWidth(max(72, max(metrics.horizontalAdvance(line)
                                            for line in _CAPTIONS[operation].split('\n')) + 16))
            button.setFixedHeight(56)
            row.addWidget(button)
            buttons[operation] = button
        group.addLayout(row)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color: #929ca3; font-size: 9px;')
        label.setFixedHeight(12)
        group.addWidget(label)
        layout.addLayout(group)
    layout.addStretch()
    scroll.setWidget(panel)
    return scroll, buttons
