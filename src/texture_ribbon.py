"""Texture and colour commands in the same order as the reference ribbon."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget
from repair_ribbon import _RibbonScrollArea

COMMANDS = {
    'new': 'Новая текстура', 'bake': 'Деталь-в-текстуру', 'select': 'Выбрать текстуру',
    'edit': 'Редактировать текстуру', 'update': 'Обновить текстуру', 'copy': 'Копировать текстуру',
    'paste': 'Вставить текстуру', 'clear': 'Удалить текстуру с треугольников', 'delete': 'Удалить текстуру',
    'show': 'Переключить отображение текстур', 'invert': 'Обратить отображение текстур',
    'paint_part': 'Покрасить деталь', 'split': 'Разделить деталь по цветам',
    'paint_faces': 'Покрасить поверхности', 'colors': 'Цвет треугольников',
}
HINTS = {
    'new': 'Наложить PNG/JPEG/BMP/TIFF на выбранные поверхности или детали.',
    'bake': 'Преобразовать текущие цвета треугольников детали в растровую текстуру без изменения геометрии.',
    'select': 'Список текстур всех загруженных деталей. Выбор выделяет владельца и поверхности.',
    'edit': 'Название, изображение, проекция, повторения, смещение и угол текстуры.',
    'update': 'Перечитать изображение с диска. Для атласа цветов — пересчитать цвета детали.',
    'copy': 'Скопировать выбранную текстуру во внутренний буфер приложения.',
    'paste': 'Наложить копию текстуры на выбранные детали или поверхности с новой привязкой.',
    'clear': 'Снять выбранную текстуру только с выделенных треугольников.',
    'delete': 'Удалить выбранную текстуру целиком, сохраняя геометрию и цвета.',
    'show': 'Показать или скрыть все сохранённые текстуры и цвета модели.',
    'invert': 'Поменять видимость выбранной текстуры. Остальные текстуры остаются на месте.',
    'paint_part': 'Задать сохранённый цвет всех треугольников выбранных деталей.',
    'split': 'Создать отдельные сетки по цветам. Цвет изображения определяется в центре треугольника; CAD становится сеткой.',
    'paint_faces': 'Окрасить выделенные треугольники или CAD-поверхности. Выделите их нижней панелью.',
    'colors': 'Показывать сохранённые цвета и текстуры поверхности.',
}
GROUPS = (('Главная', ('new', 'bake', 'select', 'edit', 'update', 'copy', 'paste', 'clear', 'delete')),
          ('Отображение', ('show', 'invert')), ('Цвет', ('paint_part', 'split', 'paint_faces', 'colors')))


def texture_icon(operation):
    motifs = {
        'new': '<path d="M34 4V16M28 10H40"/>', 'bake': '<path d="M7 8 17 3 27 9 17 16ZM7 8V20L17 27 27 20V9M17 16V27"/>',
        'select': '<path d="M28 24 40 32 34 34 31 41Z"/>', 'edit': '<path d="M9 38 13 28 34 7 40 13 19 34ZM30 11 36 17"/>',
        'update': '<path d="M38 15A16 16 0 1 0 40 31M31 15H40V6"/>', 'copy': '<path d="M7 8H29V30H7ZM16 16H38V38H16"/>',
        'paste': '<path d="M10 10H36V40H10ZM17 5H29V14H17M17 26H29M23 20V32"/>',
        'clear': '<path d="M6 40 21 6 42 40ZM24 23 38 37M38 23 24 37"/>',
        'delete': '<path d="M9 10H37M17 5H29M13 10 16 40H32L35 10M21 17V33M27 17V33"/>',
        'show': '<path d="M3 24Q24 3 45 24Q24 45 3 24Z"/><circle cx="24" cy="24" r="7"/>',
        'invert': '<path d="M4 24H44M24 4V44"/><circle cx="24" cy="24" r="16"/><path d="M24 8A16 16 0 0 1 24 40Z" fill="#e6ba74"/>',
        'paint_part': '<path d="M10 17 25 5 39 17 25 29ZM10 17V34L25 44 39 34V17M25 29V44"/>',
        'split': '<path d="M5 12 19 5V32L5 39ZM29 16 43 9V36L29 43ZM24 3V45"/>',
        'paint_faces': '<path d="M5 38 23 7 43 38ZM5 38 31 21 23 7"/><path d="M5 38 31 21 43 38Z" fill="#e6ba74"/>',
        'colors': '<path d="M6 14 24 4 42 14 24 24Z" fill="#e6ba74"/><path d="M6 14V35L24 45V24Z" fill="#a2c880"/><path d="M24 24 42 14V35L24 45Z" fill="#74bce0"/>',
    }
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">'
           '<path d="M9 17H35V39H9Z" fill="#395461" stroke="#c5d0d7"/>'
           '<path d="M9 17H17V25H9ZM25 17H35V25H25ZM17 25H25V33H17ZM9 33H17V39H9ZM25 33H35V39H25Z" fill="#81c4d2"/>'
           '<g fill="none" stroke="#e6ba74" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round">' + motifs[operation] + '</g></svg>')
    icon = QIcon()
    for size in (24, 48, 96):
        pixmap = QPixmap(); pixmap.loadFromData(svg.replace('width="48" height="48"', f'width="{size}" height="{size}"').encode(), 'SVG'); icon.addPixmap(pixmap)
    return icon


def create_texture_ribbon():
    scroll = _RibbonScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setObjectName('texture_ribbon')
    scroll.setStyleSheet('QToolButton {color:#e0e0e0;border:1px solid transparent;padding:2px;font-size:10px;} QToolButton:hover {background:#41464a;} QToolButton:checked {background:#244650;border-color:#78c3ce;} QLabel {color:#929ca3;font-size:9px;}')
    panel = QWidget(); layout = QHBoxLayout(panel); layout.setContentsMargins(7, 1, 7, 1); buttons = {}
    for title, operations in GROUPS:
        if buttons:
            divider = QFrame(); divider.setFrameShape(QFrame.VLine); layout.addWidget(divider)
        group = QVBoxLayout(); group.setSpacing(0); row = QHBoxLayout(); row.setSpacing(1)
        for op in operations:
            button = QToolButton(); button.setObjectName('texture_' + op)
            words = COMMANDS[op].split(); text = ' '.join(words)
            if len(text) > 15:
                at = len(words) // 2; text = ' '.join(words[:at]) + '\n' + ' '.join(words[at:])
            button.setText(text); button.setAccessibleName(COMMANDS[op]); button.setToolTip(COMMANDS[op] + '\n\n' + HINTS[op])
            button.setIcon(texture_icon(op)); button.setIconSize(QSize(24, 24)); button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setFixedSize(max(78, max(len(line) for line in text.split('\n')) * 6 + 14), 56)
            button.setCheckable(op in ('show', 'colors')); button.setCursor(Qt.PointingHandCursor)
            row.addWidget(button); buttons[op] = button
        group.addLayout(row); label = QLabel(title); label.setAlignment(Qt.AlignCenter); label.setFixedHeight(12); group.addWidget(label)
        layout.addLayout(group)
    layout.addStretch(); scroll.setWidget(panel)
    return scroll, buttons
