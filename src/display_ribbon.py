"""Scrollable display ribbon: rendering, scene annotations, checks and image export."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QToolButton, QVBoxLayout, QWidget


DISPLAY_COMMANDS = {
    'view': 'Вид сцены', 'smooth': 'Сглаженное затенение', 'simplified': 'Упрощённый вид',
    'grid': 'Сетка платформы', 'ruler': 'Линейки', 'zones': 'Запретные зоны',
    'dimensions': 'Размеры деталей', 'center_mass': 'Центры масс', 'bbox': 'Общий габарит',
    'origin': 'Начало координат', 'part_number': 'Номера деталей', 'part_name': 'Названия деталей',
    'part_path': 'Пути файлов', 'coordinates': 'Координатный куб', 'texture': 'Текстура / цвет модели',
    'triangle_colors': 'Цвет треугольников', 'overhang': 'Зоны нависаний',
    'outside': 'Вне платформы', 'build_risk': 'Геометрические проверки',
    'volume': 'Оценка объёма', 'material_cost': 'Оценка стоимости материала', 'packing_density': 'Плотность размещения',
    'export_png': 'Экспорт изображения', 'clipboard': 'Копировать изображение', 'print': 'Распечатать',
}
STATISTICS = {'volume', 'material_cost', 'packing_density'}
TOGGLES = set(DISPLAY_COMMANDS) - {'view', 'export_png', 'clipboard', 'print'}
DEFAULTS = {operation: operation in {'zones', 'coordinates'} for operation in TOGGLES}
GROUPS = (
    ('Отображение', ('view', 'smooth', 'simplified')),
    ('Элементы', ('grid', 'ruler', 'zones', 'dimensions', 'center_mass', 'bbox', 'origin',
                  'part_number', 'part_name', 'part_path', 'coordinates')),
    ('Дополнительные', ('texture', 'triangle_colors', 'overhang', 'outside', 'build_risk')),
    ('Статистика', ('volume', 'material_cost', 'packing_density')),
    ('Экспорт изображения', ('export_png', 'clipboard', 'print')),
)
_CAPTIONS = {
    'view': 'Вид\nсцены', 'smooth': 'Сглаженное\nзатенение', 'simplified': 'Упрощённый\nвид',
    'grid': 'Сетка\nплатформы', 'ruler': 'Линейки', 'zones': 'Запретные\nзоны',
    'dimensions': 'Размеры\nдеталей', 'center_mass': 'Центры\nмасс', 'bbox': 'Общий\nгабарит',
    'origin': 'Начало\nкоординат', 'part_number': 'Номера\nдеталей', 'part_name': 'Названия\nдеталей',
    'part_path': 'Пути\nфайлов', 'coordinates': 'Координатный\nкуб', 'texture': 'Текстура /\nцвет модели',
    'triangle_colors': 'Цвет\nтреугольников', 'overhang': 'Зоны\nнависаний', 'outside': 'Вне\nплатформы',
    'build_risk': 'Геометрические\nпроверки', 'volume': 'Оценка\nобъёма', 'material_cost': 'Стоимость\nматериала',
    'packing_density': 'Плотность\nразмещения', 'export_png': 'Экспорт\nизображения',
    'clipboard': 'Копировать\nизображение', 'print': 'Распечатать',
}
HINTS = {
    'view': 'Изометрия и шесть ортогональных видов. Камера поворачивается вокруг текущего центра.',
    'smooth': 'Усредняет нормали только для отображения. Может визуально смягчить острые углы; геометрия и индексы граней сохраняются.',
    'simplified': 'Показывает габариты вместо поверхностей. В этом режиме выбирайте детали в таблице.',
    'grid': 'Сетка в плоскости XY с подписанным шагом в миллиметрах.',
    'ruler': 'Мировые линейки X и Y в миллиметрах вдоль краёв платформы.',
    'zones': 'Показать или скрыть настроенные запретные зоны активной платформы.',
    'dimensions': 'Габаритные размеры X/Y/Z видимых деталей в миллиметрах.',
    'center_mass': 'Центр масс однородного замкнутого тела. Для открытой сетки отмечается только центр габаритов.',
    'bbox': 'Ограничивающий параллелепипед всех видимых деталей.',
    'origin': 'Мировое начало координат и оси XYZ в рабочей сцене.',
    'part_number': 'Номер детали соответствует её строке в общем списке.',
    'part_name': 'Подписи видимых деталей по имени файла.',
    'part_path': 'Исходный путь файла, если он сохранён в модели.',
    'coordinates': 'Показать или скрыть куб ориентации в углу сцены.',
    'texture': 'Показывает сохранённые цвета граней/вершин или UV-текстуру. Если данных нет, подскажет, какой импорт нужен.',
    'triangle_colors': 'Контрастные цвета соседних треугольников для просмотра триангуляции; цвета проекта сохраняются.',
    'overhang': 'Нижние грани с наклоном менее 45° к горизонтали выше Z=0.01 мм. Это геометрическая подсветка, не расчёт поддержек.',
    'outside': 'Подсветить детали и поддержки, выходящие за габариты активной платформы.',
    'build_risk': 'Геометрическая проверка: открытые сетки, выход за платформу, возможные пересечения габаритов и запретных зон. Это не симуляция печати.',
    'volume': 'Показать в углу сцены объём выбранных деталей и поддержек. При пустом выборе — ноль; открытые сетки отмечаются как неопределённые.',
    'material_cost': 'Показать массу и стоимость выбранных деталей и поддержек. Плотность и цена за килограмм задаются через стрелку справа.',
    'packing_density': 'Показать использование объёма камеры, плотность до высоты сборки и её высоту для выбранных деталей с поддержками. Перекрытия не вычитаются.',
    'export_png': 'Сохранить изображение текущей 3D-сцены в PNG.',
    'clipboard': 'Скопировать изображение текущей 3D-сцены в буфер обмена.',
    'print': 'Открыть системный диалог печати изображения текущей сцены.',
}


def display_icon(operation):
    cube = '<path d="M8 13 24 5 40 13V33L24 42 8 33ZM8 13 24 22 40 13M24 22V42"/>'
    shapes = {
        'view': cube + '<path d="M24 5V22L8 13" fill="#5fabc8"/>',
        'smooth': '<circle cx="24" cy="24" r="18" fill="url(#sphere)"/>',
        'simplified': cube + '<path d="M4 24H15M10 19 15 24 10 29M44 24H33M38 19 33 24 38 29" stroke="#78c4dd"/>',
        'grid': '<path d="M4 40 20 8H44L28 40ZM8 32H32M12 24H36M16 16H40M12 40 28 8M20 40 36 8"/>',
        'ruler': '<path d="M5 8H43V20H17V42H5ZM13 8V15M21 8V12M29 8V15M37 8V12M5 25H12M5 33H9"/>',
        'zones': cube + '<path d="M12 39 24 18 36 39Z" fill="#c86149"/><path d="M24 25V31M24 34V36"/>',
        'dimensions': cube + '<path d="M5 45H43M5 41V48M43 41V48M1 10V34M0 10H4M0 34H4" stroke="#e7bb70"/>',
        'center_mass': '<circle cx="24" cy="24" r="16"/><path d="M24 8V40M8 24H40"/><path d="M24 8A16 16 0 0 1 40 24H24ZM24 24V40A16 16 0 0 1 8 24Z" fill="#a3ca74"/>',
        'bbox': cube.replace('/>', ' stroke-dasharray="4 3"/>') + '<path d="M17 19 26 14 33 21 23 28Z" fill="#73b6cd"/>',
        'origin': '<path d="M14 35H43M14 35V5M14 35 3 45"/><path d="M38 31 43 35 38 39M10 10 14 5 18 10M4 39 3 45 9 44" stroke="#ebbb66"/>',
        'part_number': cube + '<path d="M23 13 26 10V20M18 15H28" stroke="#a1d375"/>',
        'part_name': '<path d="M5 8H32L44 24 32 40H5Z"/><path d="M12 31 19 16 26 31M15 25H23" stroke="#a1d375"/><circle cx="35" cy="24" r="2"/>',
        'part_path': '<path d="M5 11H20L24 16H43V39H5ZM5 23H43"/><path d="M11 31H35M29 26 35 31 29 36" stroke="#e8b867"/>',
        'coordinates': cube + '<path d="M24 42H47M24 42 4 31M24 42V16" stroke="#73c8dd"/>',
        'texture': '<path d="M6 8H42V40H6Z"/><path d="M6 8H18V20H6ZM30 8H42V20H30ZM18 20H30V32H18ZM6 32H18V40H6ZM30 32H42V40H30Z" fill="#7dbb76"/>',
        'triangle_colors': '<path d="M24 5 43 40H5Z" fill="#598fb8"/><path d="M24 5 24 29 5 40Z" fill="#cd7965"/><path d="M24 29 43 40H5Z" fill="#b2c06c"/>',
        'overhang': '<path d="M8 8H40V20H20V41H8Z"/><path d="M20 20H40L30 30Z" fill="#e79545"/><path d="M28 35V43M24 39 28 43 32 39"/>',
        'outside': '<path d="M3 32 24 22 45 32 24 43Z"/><path d="M25 6 37 0 47 6V21L37 27 25 21ZM25 6 37 12 47 6M37 12V27" fill="#bf6454"/>',
        'build_risk': '<path d="M24 4 46 42H2Z" fill="#97652c"/><path d="M24 15V29M24 34V37" stroke="#f7cf7b" stroke-width="3"/>',
        'volume': cube + '<path d="M8 24 24 32 40 24V33L24 42 8 33Z" fill="#688eac"/>',
        'material_cost': '<ellipse cx="24" cy="12" rx="18" ry="7"/><path d="M6 12V24C6 34 42 34 42 24V12M6 24V35C6 45 42 45 42 35V24"/><path d="M18 36H30" stroke="#e8bf74"/>',
        'packing_density': cube + '<path d="M12 18 19 14 26 18V28L19 32 12 28ZM25 28 32 24 38 28V34L32 38 25 34Z" fill="#96bb70"/>',
        'export_png': '<path d="M5 5H43V33H5ZM8 29 18 18 27 25 33 17 40 29"/><circle cx="31" cy="12" r="3"/><path d="M24 31V46M18 40 24 46 30 40" stroke="#80c2dd"/>',
        'clipboard': '<path d="M11 9H39V43H11ZM18 4H32V14H18ZM5 17V47H31"/><path d="M16 35 23 25 29 31 34 25" stroke="#80c2dd"/>',
        'print': '<path d="M12 16V4H36V16M12 35H4V16H44V35H36M12 28H36V44H12ZM18 34H30M18 39H30"/><circle cx="37" cy="22" r="2" fill="#91ca7d"/>',
    }
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><defs><radialGradient id="sphere" cx="30%" cy="25%"><stop stop-color="#e5f5ff"/><stop offset="1" stop-color="#446478"/></radialGradient></defs><g fill="none" stroke="#c3ccd2" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round">' + shapes[operation] + '</g></svg>'
    icon = QIcon()
    for size in (24, 48, 96):
        pixmap = QPixmap()
        pixmap.loadFromData(svg.replace('width="48" height="48"', f'width="{size}" height="{size}"').encode(), 'SVG')
        icon.addPixmap(pixmap)
    return icon


class _DisplayScroll(QScrollArea):
    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            pixels, angle = event.pixelDelta(), event.angleDelta()
            delta = pixels.x() or pixels.y() or (angle.x() or angle.y()) / 120 * 96
            bar.setValue(bar.value() - round(delta)); event.accept()
        else: super().wheelEvent(event)


def create_display_ribbon():
    scroll = _DisplayScroll()
    scroll.setObjectName('display_ribbon')
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet('''
        QScrollArea, QScrollArea > QWidget > QWidget {background: #2b2b2b;}
        QToolButton {color: #e0e0e0; border: 1px solid transparent; border-radius: 3px; padding: 0 4px;}
        QToolButton:hover {background: #41464a; border-color: #626c73;}
        QToolButton:checked {background: #345365; border-color: #70bddb;}
        QToolButton:pressed {background: #244650;}
        QToolButton:disabled {color: #777;}
        QMenu {background: #303030; color: #e0e0e0; border: 1px solid #666;}
        QMenu::item:selected {background: #426071;}
        QScrollBar:horizontal {height: 7px; border: none; background: #242424; margin: 0;}
        QScrollBar::handle:horizontal {background: #60666b; min-width: 28px; border-radius: 3px;}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {width: 0;}
    ''')
    panel = QWidget(); layout = QHBoxLayout(panel)
    layout.setContentsMargins(7, 1, 7, 1); layout.setSpacing(7)
    buttons = {}
    for title, operations in GROUPS:
        if buttons:
            divider = QFrame(); divider.setFixedWidth(1); divider.setStyleSheet('background: #515151;')
            layout.addWidget(divider)
        group = QVBoxLayout(); group.setSpacing(0)
        row = QVBoxLayout() if title == 'Статистика' else QHBoxLayout()
        row.setSpacing(1)
        for operation in operations:
            button = QToolButton()
            button.setObjectName('display_' + operation)
            button.setText(DISPLAY_COMMANDS[operation] if operation in STATISTICS else _CAPTIONS[operation]); button.setAccessibleName(DISPLAY_COMMANDS[operation])
            button.setToolTip(DISPLAY_COMMANDS[operation] + '\n\n' + HINTS[operation])
            button.setIcon(display_icon(operation)); button.setIconSize(QSize(24, 24))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon); button.setCursor(Qt.PointingHandCursor)
            button.setCheckable(operation in TOGGLES)
            if operation in DEFAULTS: button.setChecked(DEFAULTS[operation])
            font = QFont(button.font()); font.setPixelSize(10); button.setFont(font)
            metrics = QFontMetrics(font)
            button.setFixedWidth(max(65, max(metrics.horizontalAdvance(line) for line in _CAPTIONS[operation].split('\n')) + 16))
            button.setFixedHeight(56)
            if operation in STATISTICS:
                button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                button.setIconSize(QSize(16, 16))
                button.setFixedHeight(18)
                button.setFixedWidth(210)
            row.addWidget(button); buttons[operation] = button
        group.addLayout(row)
        label = QLabel(title); label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color: #929ca3; font-size: 9px;'); label.setFixedHeight(12)
        group.addWidget(label); layout.addLayout(group)
    layout.addStretch(); scroll.setWidget(panel)
    return scroll, buttons
