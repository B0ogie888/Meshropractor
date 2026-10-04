"""Editable commands and original vector icons for mesh preparation."""
from PySide6.QtCore import QSize,Qt
from PySide6.QtGui import QIcon,QPixmap
from PySide6.QtWidgets import QFrame,QToolButton,QHBoxLayout,QVBoxLayout,QLabel
from ribbon_layout import asset_icon, compact_button_column

COMMANDS={
 'hollow':'Пустотелая деталь','cut':'Разрезание','perforate':'Перфорации',
 'shell_core':'Корпус и ядро','surface_array':'Поверхность в массив','round':'Скругление',
 'extrude':'Выдвинуть','offset':'Смещение','round_offset':'Заокругленное смещение',
 'merge':'Объединить детали','boolean':'Булевы операции','fragments':'Фрагменты в детали',
 'label':'Маркировка','struts':'Распорки','honeycomb':'Сотовые структуры',
 'lattice':'Структуры','slice_lattice':'Структуры на основе срезов',
 'tetra':'Тетраэдрическая решётка','tetra_slices':'Диагональная решётка по срезам',
 'rapidfit':'Фиксатор (лоток)','formfit':'Формообразующая оболочка','remove_volume':'Удаление объёмов',
 'union':'Булево объединение','difference':'Булево вычитание','intersection':'Булево пересечение'}
GROUPS=(('Редактировать',tuple(COMMANDS)[:9]),('Объединение',tuple(COMMANDS)[9:12]),
 ('Сгенерировать',tuple(COMMANDS)[12:14]),('Структуры',tuple(COMMANDS)[14:19]),('Оснастка',tuple(COMMANDS)[19:22]))
IMPLICIT={'hollow','shell_core','round','round_offset','honeycomb','lattice','slice_lattice','tetra','tetra_slices','formfit'}
HINTS={
 'hollow':'Внутренняя полость с заданной толщиной стенки. Поле расстояний; отверстия отвода задаются отдельно.',
 'cut':'Разрезать плоскостью на две детали, закрыв срезы.',
 'perforate':'Сквозные цилиндрические отверстия по регулярной сетке вдоль X/Y/Z.',
 'shell_core':'Создать отдельные наружный корпус и внутреннее ядро.',
 'surface_array':'Копии выделенной поверхности вдоль её средней нормали; результат — открытые сетки.',
 'round':'Приближённое скругление выпуклых областей сферическим открытием. Это изменение сетки, не CAD-филлет.',
 'extrude':'Выдвижение выбранной поверхности вдоль средней нормали; отрицательное значение вычитает материал.',
 'offset':'Смещение всех вершин по нормалям. Самопересечения не исключаются автоматически.',
 'round_offset':'Смещение поля и сферическое открытие с заданным радиусом.',
 'merge':'Объединить сетки в одну запись без удаления внутренних пересечений.',
 'boolean':'Объединение, вычитание или пересечение замкнутых деталей; первая в списке — основная.',
 'fragments':'Разделить связные оболочки на детали с переносом поддержек.',
 'label':'Область рамкой на детали: текст, рисунок или Data Matrix; проекция, рельеф/гравировка, предпросмотр и сохранение областей.',
 'struts':'Цилиндрическая распорка между двумя координатами.',
 'honeycomb':'Регулярные вертикальные шестигранные стенки внутри выбранного тела.',
 'lattice':'Периодическая кубическая решётка цилиндрических рёбер.',
 'slice_lattice':'Слои рёбер с чередованием направлений X/Y, ограниченные телом.',
 'tetra':'Периодические диагональные рёбра. Собственная решётка, без профилей DSM Somos.',
 'tetra_slices':'Диагональные рёбра по слоям с чередованием направления.',
 'rapidfit':'Открытый прямоугольный лоток по габаритам детали с зазором и стенкой; собственный аналог фиксатора.',
 'formfit':'Оболочка вокруг детали с зазором. Замкнутая форма: извлечение и прочность не проверяются.',
 'remove_volume':'Вычесть остальные выбранные тела из первого. Это геометрическая операция без формата Concept Laser.'}


def model_tool_icon(operation):
    icon=asset_icon('tools',operation)
    if icon is not None:return icon
    cube='<path d="M7 13 23 5 39 13V33L23 42 7 33ZM7 13 23 23 39 13M23 23V42"/>'
    motif={
    'hollow':'M15 17 23 13 31 17V29L23 33 15 29Z','cut':'M19 3 28 44',
    'perforate':'M15 26V30M23 29V33M31 26V30','shell_core':'M23 5V42M31 18H43V35H31Z',
    'surface_array':'M20 8 35 2 44 9V27M26 16 41 10 47 15V33',
    'round':'M25 5Q40 3 40 18','extrude':'M23 15V2M18 7 23 2 28 7',
    'offset':'M2 10 23 0 46 10M2 10V38M46 10V38','round_offset':'M2 14Q2 0 18 0H32Q46 0 46 14V38',
    'merge':'M29 4 45 12V29L29 37','boolean':'M26 16H43V35H26ZM29 22 38 30M38 22 29 30',
    'fragments':'M5 6H17V17H5ZM31 32H44V44H31Z','label':'M12 35 20 18 28 35M15 29H25',
    'struts':'M3 41 43 4M7 44 46 8','honeycomb':'M11 19 17 15 23 19V27L17 31 11 27ZM23 19 29 15 35 19V27L29 31 23 27Z',
    'lattice':'M7 13 39 33M39 13 7 33M23 5V42M7 23H39','slice_lattice':'M7 18 23 27 39 18M7 25 23 34 39 25',
    'tetra':'M7 13 23 42 39 13ZM23 5 7 33 39 33Z','tetra_slices':'M7 18 39 26M39 18 7 26M7 29 39 37M39 29 7 37',
    'rapidfit':'M2 28V44H46V28M9 28V37H39V28','formfit':'M2 10 23 0 46 10V37L23 48 2 37Z',
    'remove_volume':'M28 10H44V26H28ZM30 13 42 23M42 13 30 23'}[operation]
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><g fill="none" stroke="#bbcbd3" stroke-width="1.6" stroke-linejoin="round">{cube}</g><path d="{motif}" fill="none" stroke="#76c6e6" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/></svg>'
    pixmap=QPixmap();pixmap.loadFromData(svg.encode(),'SVG');return QIcon(pixmap)


def append_groups(panel,layout):
    buttons={}
    for title,operations in GROUPS:
        line=QFrame();line.setFrameShape(QFrame.VLine);layout.addWidget(line)
        group=QVBoxLayout();group.setSpacing(0);row=QHBoxLayout();row.setSpacing(1)
        for operation in operations:
            button=QToolButton();button.setObjectName('model_tool_'+operation)
            words=COMMANDS[operation].split();split=max(1,len(words)//2)
            button.setText(' '.join(words[:split])+'\n'+' '.join(words[split:]) if len(words)>1 else words[0])
            button.setAccessibleName(COMMANDS[operation]);button.setToolTip(HINTS[operation])
            button.setIcon(model_tool_icon(operation));button.setIconSize(QSize(28,28))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon);button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet('QToolButton {color:#ddd;font-size:11px;border:1px solid transparent;padding:2px;} QToolButton:hover {background:#41464a;}')
            buttons[operation]=button
            if operation not in ('extrude', 'offset', 'round_offset'): row.addWidget(button)
            elif operation == 'round_offset':
                row.addWidget(compact_button_column([buttons[key] for key in ('extrude', 'offset', 'round_offset')]))
        group.addLayout(row);label=QLabel(title);label.setAlignment(Qt.AlignCenter);label.setStyleSheet('color:#aaa;font-size:10px;');group.addWidget(label);layout.addLayout(group)
    panel.model_tool_buttons=buttons;return buttons
