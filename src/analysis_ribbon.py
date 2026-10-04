"""Working inspection/report commands arranged like the reference ribbon."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget
from repair_ribbon import _RibbonScrollArea
from ribbon_layout import asset_icon
from display_ribbon import display_icon
COMMANDS = {
    'view':'Вид платформы', 'intersections':'Определить пересечения', 'trapping':'Анализ заклинивания',
    'walls':'Анализ толщины стенок', 'cavities':'Замкнутые внутренние оболочки',
    'risks':'Анализ рисков построения', 'slices':'График распределения срезов',
    'time':'Оценка времени построения', 'cost':'Оценка стоимости', 'material':'Оценка стоимости материала',
    'volume':'Оценка объёма', 'density':'Плотность размещения',
    'dimensions':'Размеры детали', 'mass_center':'Центр масс', 'bounds':'Общий габарит',
    'distance':'Измерение расстояния', 'thickness':'Измерить толщину', 'actual':'Фактические измерения',
    'precision':'Точность измерений', 'report':'Сгенерировать отчёт', 'template':'Настройки отчёта',
}
GROUPS = (('Анализ платформы', tuple(COMMANDS)[:7]), ('Оценка', tuple(COMMANDS)[7:12]),
          ('Измерение', tuple(COMMANDS)[12:19]), ('Отчёты', tuple(COMMANDS)[19:]))
HINTS = {'intersections':'Объёмные пересечения выбранных замкнутых деталей. Касания имеют нулевой объём.',
 'trapping':'Проверить препятствия при поступательном извлечении первой детали в выбранном направлении; дискретная проверка.',
 'walls':'Выборка толщины по внутренней нормали. Не гарантирует обнаружение всех тонких областей.',
 'cavities':'Проверить замкнутые отдельные оболочки и оболочки с отрицательным объёмом. Связность дренажных каналов не моделируется.',
 'slices':'Площадь и длина контуров выборки горизонтальных слоёв с поддержками.',
 'time':'Приближённая оценка по производительности, числу слоёв и накладным затратам.',
 'report':'Отчёт выбранных деталей с результатами анализа и измерениями. Экспорт HTML, PDF, JSON и CSV.',
 'thickness':'Один щелчок на поверхности — расстояние до противоположной стенки вдоль нормали.',
 'actual':'Внести фактическое и номинальное значение, допуск и единицы; сохранить в проекте.'}


def create_analysis_ribbon():
    scroll = _RibbonScrollArea(); scroll.setFrameShape(QFrame.NoFrame); scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    panel = QWidget(); layout = QHBoxLayout(panel); layout.setContentsMargins(7, 1, 7, 1); buttons = {}
    panel.setStyleSheet('QToolButton {color:#ddd;font-size:11px;border:1px solid transparent;padding:2px;} QToolButton:hover {background:#41464a;} QToolButton:checked {background:#244650;border-color:#78c3ce;} QLabel {color:#929ca3;font-size:9px;}')
    for title, operations in GROUPS:
        if buttons:
            line=QFrame(); line.setFrameShape(QFrame.VLine); layout.addWidget(line)
        group=QVBoxLayout(); group.setSpacing(0); row=QHBoxLayout(); row.setSpacing(1)
        for op in operations:
            name=COMMANDS[op]; words=name.split(); at=max(1, len(words)//2)
            button=QToolButton(); button.setObjectName('analysis_'+op)
            button.setText(' '.join(words[:at])+'\n'+' '.join(words[at:]) if len(words)>1 else name)
            button.setAccessibleName(name); button.setToolTip(name+'\n\n'+HINTS.get(op, 'Работа с выбранными деталями текущей сцены.'))
            button.setIcon(asset_icon('analysis',op) or display_icon({'time':'ruler','cost':'material_cost','distance':'dimensions','thickness':'overhang','report':'print','template':'part_name'}.get(op,'bbox')))
            button.setIconSize(QSize(28,28)); button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon); button.setCursor(Qt.PointingHandCursor)
            button.setCheckable(op in ('volume','density','dimensions','mass_center','bounds')); row.addWidget(button); buttons[op]=button
        group.addLayout(row); label=QLabel(title); label.setAlignment(Qt.AlignCenter); label.setFixedHeight(14); group.addWidget(label); layout.addLayout(group)
    layout.addStretch(); scroll.setWidget(panel); return scroll,buttons
