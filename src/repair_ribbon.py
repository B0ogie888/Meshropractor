"""Repair commands with individual vector icons and a horizontally scrollable ribbon."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)


REPAIR_COMMANDS = {
    'wizard': 'Мастер исправлений',
    'auto': 'Автоисправление',
    'wrap': 'Обернуть деталь',
    'normals': 'Исправление нормалей',
    'stitch': 'Авто-сшивание',
    'holes': 'Исправление дыр',
    'noise': 'Фрагменты шума',
    'unify': 'Унифицировать',
    'split': 'Фрагменты в детали',
    'remove_small': 'Удалить мелкие детали',
    'slivers': 'Фильтровать острые треугольники',
    'duplicates': 'Удалить идентичные треугольники',
    'overlaps': 'Обнаружить треугольники внахлёст',
    'fill_hole': 'Режим заливки дыр',
    'bridge': 'Создать перемычку',
    'add_triangle': 'Создать треугольник',
    'delete_faces': 'Удалить треугольники',
    'clip': 'Обрезать треугольники',
    'move_vertices': 'Перемещать точки детали',
    'drag_vertices': 'Передвинуть точки детали',
    'decimate': 'Редукция треугольников',
    'smooth': 'Сглаживание',
    'clean_smooth': 'Очистка и сглаживание',
    'subdivide': 'Подразделение поверхности',
    'remesh': 'Переразбивка поверхности',
}

_CAPTIONS = {
    'wizard': 'Мастер\nисправлений', 'auto': 'Автоисправление', 'wrap': 'Обернуть\nдеталь',
    'normals': 'Исправление\nнормалей', 'stitch': 'Авто-сшивание', 'holes': 'Исправление\nдыр',
    'noise': 'Фрагменты\nшума', 'unify': 'Унифицировать', 'split': 'Фрагменты\nв детали',
    'remove_small': 'Удалить мелкие\nдетали', 'slivers': 'Фильтровать острые\nтреугольники',
    'duplicates': 'Удалить идентичные\nтреугольники', 'overlaps': 'Обнаружить треугольники\nвнахлёст',
    'fill_hole': 'Режим заливки\nдыр', 'bridge': 'Создать\nперемычку', 'add_triangle': 'Создать\nтреугольник',
    'delete_faces': 'Удалить\nтреугольники', 'clip': 'Обрезать\nтреугольники',
    'move_vertices': 'Перемещать\nточки детали', 'drag_vertices': 'Передвинуть\nточки детали',
    'decimate': 'Редукция\nтреугольников', 'smooth': 'Сглаживание',
    'clean_smooth': 'Очистка\nи сглаживание', 'subdivide': 'Подразделение\nповерхности',
    'remesh': 'Переразбивка\nповерхности',
}

_HINTS = {
    'wizard': 'Открыть диагностику и мастер исправлений для загруженных моделей.',
    'auto': 'Проверить выбранные детали и подобрать автоматическое исправление сетки.',
    'wrap': 'Построить замкнутую оболочку выбранных деталей. Операция может изменить геометрию.',
    'normals': 'Согласовать направление нормалей треугольников выбранных деталей.',
    'stitch': 'Соединить близкие открытые рёбра сетки с заданным допуском.',
    'holes': 'Найти и заполнить открытые контуры выбранных деталей.',
    'noise': 'Найти небольшие изолированные фрагменты сетки.',
    'unify': 'Объединить выбранные детали в одну сетку.',
    'split': 'Разделить несвязанные фрагменты сетки на самостоятельные детали.',
    'remove_small': 'Удалить детали, размер которых меньше заданного порога.',
    'slivers': 'Найти и подсветить треугольники с углом ниже заданного порога.',
    'duplicates': 'Удалить повторяющиеся треугольники выбранных деталей.',
    'overlaps': 'Проверить выбранные детали на пересечения и наложения треугольников.',
    'fill_hole': 'Выбрать открытый контур на детали и заполнить его новыми гранями.',
    'bridge': 'Соединить выбранные участки границы сетки новыми треугольниками.',
    'add_triangle': 'Создать треугольник по трём точкам детали.',
    'delete_faces': 'Удалить треугольники, выделенные инструментами выбора поверхностей.',
    'clip': 'Обрезать сетку выбранных деталей плоскостью. В отличие от сечений, изменяет геометрию.',
    'move_vertices': 'Задать смещение выбранных точек детали по координатным осям.',
    'drag_vertices': 'Указать новое положение точек детали.',
    'decimate': 'Уменьшить количество треугольников с заданной степенью редукции.',
    'smooth': 'Сгладить поверхность выбранных деталей. Изменяет положение вершин сетки.',
    'clean_smooth': 'Сшить совпадающие вершины, убрать дубли/вырождения и сгладить поверхность.',
    'subdivide': 'Разделить треугольники выбранных деталей на более мелкие.',
    'remesh': 'Приблизить плотность сетки к заданному размеру элементов разбиением, сокращением и сглаживанием.',
}

_GROUPS = (
    ('Мастер и восстановление', ('wizard', 'auto', 'wrap')),
    ('Автоматические операции', ('normals', 'stitch', 'holes', 'noise', 'unify', 'split',
                                 'remove_small', 'slivers', 'duplicates', 'overlaps')),
    ('Ручное редактирование', ('fill_hole', 'bridge', 'add_triangle', 'delete_faces', 'clip',
                              'move_vertices', 'drag_vertices')),
    ('Обработка поверхности', ('decimate', 'smooth', 'clean_smooth', 'subdivide', 'remesh')),
)


def repair_icon(operation):
    from ribbon_layout import asset_icon
    icon = asset_icon('repair', operation)
    if icon is not None: return icon
    """Original SVG symbols; colour highlights the part affected by each operation."""
    mesh = '<path d="M7 35 12 12 31 6 42 25 31 42ZM12 12 24 24 31 6M7 35 24 24 31 42M24 24 42 25"/>'
    triangle = '<path d="M6 38 24 7 42 38Z"/>'
    icons = {
        'wizard': ('<path d="M6 6H30L37 13V41H6ZM30 6V13H37M11 15H23M11 23H23M11 31H20"/>',
                   '<path d="M25 41 41 25M36 22V16M41 19H46M41 13 44 10"/>'),
        'auto': (mesh, '<path d="M13 25 21 33 37 16" stroke-width="4"/>'),
        'wrap': (mesh, '<path d="M3 13Q17-1 34 4Q47 8 47 26Q46 44 30 46Q8 49 2 35M2 6V14H10"/>'),
        'normals': (triangle + '<path d="M6 38 24 27 42 38M24 7V27"/>', '<path d="M24 27V12M19 17 24 12 29 17M13 33 6 24M5 30 6 24 12 25M35 33 42 24M36 25 42 24 43 30"/>'),
        'stitch': ('<path d="M7 7 20 10 18 38 5 41M41 7 28 10 30 38 43 41"/>', '<path d="M16 14 31 20M17 25 31 31M17 20 29 14M17 31 30 25M17 36 31 36"/>'),
        'holes': (mesh, '<path d="M17 18 30 17 32 29 19 32Z" fill="#33585b"/><path d="M17 18 32 29M30 17 19 32"/>'),
        'noise': (triangle, '<circle cx="7" cy="8" r="2"/><circle cx="41" cy="10" r="2"/><circle cx="43" cy="42" r="2"/><path d="M5 16 10 21M10 16 5 21"/>'),
        'unify': ('<path d="M4 14 16 6 27 15 16 27ZM22 32 34 24 45 33 34 44Z"/>', '<path d="M4 33H17M12 28 17 33 12 38M43 14H30M35 9 30 14 35 19"/>'),
        'split': ('<path d="M5 10 19 5V25L5 30ZM28 22 43 17V38L28 43Z"/>', '<path d="M23 3V45" stroke-dasharray="3 3"/><path d="M19 35H5M10 30 5 35 10 40M28 10H42M37 5 42 10 37 15"/>'),
        'remove_small': ('<path d="M6 36 20 8 34 36ZM35 13 39 5 43 13Z"/>', '<path d="M35 26 45 36M45 26 35 36"/>'),
        'slivers': ('<path d="M5 39 29 5 33 39ZM8 35H30"/>', '<path d="M5 43 43 43M36 10 43 17M43 10 36 17"/>'),
        'duplicates': ('<path d="M5 31 21 4 37 31Z"/>', '<path d="M11 39 27 12 43 39Z" stroke-dasharray="3 2"/><path d="M31 5 41 15M41 5 31 15"/>'),
        'overlaps': ('<path d="M5 35 15 6 36 35ZM13 40 36 11 43 40Z"/>', '<path d="M16 35 27 21 36 35Z" fill="#645333"/><circle cx="32" cy="31" r="10"/><path d="M39 39 46 46"/>'),
        'fill_hole': ('<path d="M5 13 21 5 42 14 40 38 23 44 5 35ZM5 13 15 22 5 35M21 5 25 17 42 14M40 38 30 29 23 44"/>', '<path d="M15 22 25 17 34 24 30 29 20 32Z" fill="#33585b"/><path d="M15 22 30 29M25 17 20 32"/>'),
        'bridge': ('<path d="M4 9H15V40H4ZM33 9H44V40H33Z"/>', '<path d="M15 18H33V31H15ZM15 18 33 31M15 31 33 18" fill="#33585b"/>'),
        'add_triangle': (triangle, '<circle cx="6" cy="38" r="2" fill="#70bec9"/><circle cx="24" cy="7" r="2" fill="#70bec9"/><circle cx="42" cy="38" r="2" fill="#70bec9"/><path d="M24 23V35M18 29H30"/>'),
        'delete_faces': (mesh, '<path d="M16 18 32 34M32 18 16 34" stroke-width="3"/>'),
        'clip': (mesh, '<path d="M3 27H46" stroke-width="3"/><path d="M9 40 40 9" stroke-dasharray="3 2"/>'),
        'move_vertices': (mesh, '<circle cx="24" cy="24" r="3" fill="#70bec9"/><path d="M24 24H44M39 19 44 24 39 29M24 24V4M19 9 24 4 29 9"/>'),
        'drag_vertices': (mesh, '<path d="M23 18 39 29 31 30 28 39ZM31 30 38 40" fill="#334e58"/><path d="M7 9Q3 2 15 3M11 1 15 3 12 7"/>'),
        'decimate': ('<path d="M3 10 17 3 29 12 24 26 7 28ZM3 10 14 16 17 3M14 16 29 12M14 16 7 28 24 26Z"/>', '<path d="M21 44 35 19 46 44ZM21 34H11M16 29 11 34 16 39"/>'),
        'smooth': ('<path d="M4 28 12 12 22 34 31 10 44 28" stroke-dasharray="2 3"/>', '<path d="M4 30C12 7 18 40 26 23S38 8 44 28" stroke-width="3"/>'),
        'clean_smooth': ('<path d="M4 33C11 10 18 40 26 24S37 12 44 31"/>', '<path d="M12 10 18 3 24 10 18 17ZM31 35 37 28 43 35 37 42Z" fill="#33585b"/>'),
        'subdivide': (triangle, '<path d="M15 23H33L24 38ZM24 7 24 38M6 38 33 23M42 38 15 23"/>'),
        'remesh': ('<path d="M5 13 25 5 43 16 40 37 22 44 6 34Z"/>', '<path d="M5 13 20 17 25 5M20 17 32 22 43 16M6 34 20 17 22 44M20 17 21 31 40 37M21 31 32 22 22 44M32 22 40 37"/>'),
    }
    outline, accent = icons[operation]
    colour = '#e5aa5c' if operation in ('remove_small', 'delete_faces', 'clip', 'overlaps', 'slivers', 'noise') else '#78c3ce'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">'
           f'<g fill="none" stroke="#bec7ce" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round">{outline}</g>'
           f'<g fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round">{accent}</g></svg>')
    # Include multiple raster sizes so Qt does not enlarge a tiny source on a
    # high-DPI monitor. The SVG source remains independent of external assets.
    icon = QIcon()
    for size in (24, 48, 96):
        pixmap = QPixmap()
        pixmap.loadFromData(svg.replace('width="48" height="48"', f'width="{size}" height="{size}"').encode(), 'SVG')
        icon.addPixmap(pixmap)
    return icon


class _RibbonScrollArea(QScrollArea):
    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            pixels = event.pixelDelta()
            angle = event.angleDelta()
            delta = pixels.x() or pixels.y() or (angle.x() or angle.y()) / 120 * 96
            bar.setValue(bar.value() - round(delta))
            event.accept()
        else:
            super().wheelEvent(event)


def create_repair_ribbon():
    from ribbon_layout import compact_button_column
    scroll = _RibbonScrollArea()
    scroll.setObjectName('repair_ribbon')
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
        QToolButton:pressed {background: #244650; border-color: #78c3ce;}
        QToolButton:focus {border-color: #78c3ce;}
        QToolButton:disabled {color: #777;}
        QScrollBar:horizontal {height: 7px; border: none; background: #242424; margin: 0;}
        QScrollBar::handle:horizontal {background: #60666b; min-width: 28px; border-radius: 3px;}
        QScrollBar::handle:horizontal:hover {background: #8a949d;}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {width: 0;}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {background: none;}
    ''')
    scroll.horizontalScrollBar().setToolTip('Прокрутка инструментов исправления. Можно использовать колесо мыши.')
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
            button.setObjectName(f'repair_{operation}')
            button.setText(_CAPTIONS[operation])
            button.setAccessibleName(REPAIR_COMMANDS[operation])
            button.setToolTip(f'{REPAIR_COMMANDS[operation]}\n\n{_HINTS[operation]}')
            button.setIcon(repair_icon(operation))
            button.setIconSize(QSize(24, 24))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setCursor(Qt.PointingHandCursor)
            font = QFont(button.font())
            font.setPixelSize(10)
            button.setFont(font)
            metrics = QFontMetrics(font)
            # QScrollArea otherwise compresses every button to its minimum,
            # eliding long Russian labels even though horizontal scrolling is available.
            button.setFixedWidth(max(72, max(metrics.horizontalAdvance(line)
                                            for line in _CAPTIONS[operation].split('\n')) + 16))
            button.setFixedHeight(56)
            buttons[operation] = button
            if operation in ('unify', 'split', 'slivers', 'duplicates'): continue
            if operation == 'remove_small':
                row.addWidget(compact_button_column([buttons[key] for key in ('unify', 'split', 'remove_small')]))
            elif operation == 'overlaps':
                row.addWidget(compact_button_column([buttons[key] for key in ('slivers', 'duplicates', 'overlaps')]))
            else: row.addWidget(button)
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
