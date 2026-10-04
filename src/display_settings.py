"""Persistent rendering preferences and the slicer's functional help commands."""
from dataclasses import asdict, dataclass, replace
import json

from PySide6.QtCore import QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout)

from app_version import APP_VERSION


DIALOG_STYLE = '''
    QDialog {background: #2b2b2b; color: #e0e0e0;}
    QLabel, QCheckBox {color: #e0e0e0;}
    QPushButton {background: #3b3b3b; color: #e0e0e0; border: 1px solid #666;
                 padding: 6px 12px; border-radius: 3px;}
    QPushButton:hover {background: #484848;}
    QComboBox {background: #383838; color: #e0e0e0; padding: 5px;}
    QTextBrowser {background: #262626; color: #e0e0e0; border: 1px solid #555; padding: 10px;}
'''


@dataclass(frozen=True)
class DisplayPreferences:
    background: str = '#ffffff'
    anti_aliasing: str = 'msaa'
    samples: int = 4
    interactive_edges: bool = True


def load_preferences(settings):
    """Ignore invalid/obsolete user values instead of preventing application startup."""
    default = DisplayPreferences()
    try:
        data = json.loads(settings.value('display/preferences', '{}'))
        if not isinstance(data, dict):
            return default
        color = QColor(data.get('background', default.background))
        return DisplayPreferences(
            background=color.name() if color.isValid() else default.background,
            anti_aliasing=data.get('anti_aliasing') if data.get('anti_aliasing') in ('none', 'fxaa', 'msaa') else default.anti_aliasing,
            samples=data.get('samples') if type(data.get('samples')) is int and data['samples'] in (4, 8) else default.samples,
            interactive_edges=data.get('interactive_edges') if type(data.get('interactive_edges')) is bool else default.interactive_edges,
        )
    except (ValueError, TypeError, OverflowError):
        return default


def save_preferences(settings, preferences):
    settings.setValue('display/preferences', json.dumps(asdict(preferences)))
    settings.sync()


def scalar_bar_contrast(plotter, background):
    """Change legend lettering, never the deviation lookup table or mesh colours."""
    color = QColor(background)
    ink = (.08, .08, .08) if color.lightnessF() > .5 else (.92, .95, .93)
    for bar in getattr(plotter, 'scalar_bars', {}).values():
        for name in ('GetTitleTextProperty', 'GetLabelTextProperty', 'GetAnnotationTextProperty'):
            getattr(bar, name)().SetColor(*ink)


def apply_to_plotter(plotter, preferences, *, render=True):
    """Only rendering properties change; imported geometry and selection IDs stay intact."""
    if plotter is None:
        return
    plotter.set_background(preferences.background)
    scalar_bar_contrast(plotter, preferences.background)
    performance = getattr(plotter, '_viewport_performance', None)
    if performance is not None:
        performance.configure(anti_aliasing=preferences.anti_aliasing, samples=preferences.samples,
                              interactive_edges=preferences.interactive_edges)
    if render:
        plotter.render()


class DisplaySettingsDialog(QDialog):
    def __init__(self, preferences, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Параметры отображения')
        self.setStyleSheet(DIALOG_STYLE)
        self.setMinimumWidth(530)
        self.apply_preferences = apply
        layout = QVBoxLayout(self)
        description = QLabel('Настройки применяются к слайсеру и предеформации и сохраняются после закрытия программы.')
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.theme = getattr(parent, 'engineering_theme', None)
        if self.theme is not None:
            self.theme_choice = QComboBox()
            self.theme_choice.addItem('Светлая', 'light'); self.theme_choice.addItem('Тёмная', 'dark')
            self.theme_choice.setCurrentIndex(self.theme_choice.findData(self.theme.mode))
            form.addRow('Тема интерфейса', self.theme_choice)
            self.theme.changed.connect(self.sync_theme)
        self.background = QPushButton()
        self.background.setProperty('preserveThemeColors', True)
        self.background.clicked.connect(self.choose_background)
        form.addRow('Фон рабочей сцены', self.background)
        self.antialiasing = QComboBox()
        for text, value in (('Отключено', ('none', 4)), ('FXAA — быстрое', ('fxaa', 4)),
                            ('MSAA ×4 — рекомендуется', ('msaa', 4)), ('MSAA ×8 — повышенное качество', ('msaa', 8))):
            self.antialiasing.addItem(text, value)
        self.antialiasing.setToolTip('Сглаживает ступеньки на контурах. Доступное качество MSAA зависит от видеокарты.')
        form.addRow('Сглаживание контуров', self.antialiasing)
        self.interactive_edges = QCheckBox('Скрывать сетку крупных деталей при вращении')
        self.interactive_edges.setToolTip('Рёбра треугольников возвращаются после завершения движения камеры.')
        form.addRow(self.interactive_edges)
        layout.addLayout(form)
        note = QLabel('Сглаживание улучшает изображение, но не добавляет детализацию исходной сетке. '
                      'Для круглых поверхностей CAD качество триангуляции задаётся при импорте.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply |
                                        QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttons.button(QDialogButtonBox.Ok).setText('ОК')
        self.buttons.button(QDialogButtonBox.Apply).setText('Применить')
        self.buttons.button(QDialogButtonBox.Cancel).setText('Отмена')
        self.buttons.button(QDialogButtonBox.RestoreDefaults).setText('По умолчанию')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(self.buttons)
        self.set_values(preferences)
        if self.theme is not None: self.theme_choice.currentIndexChanged.connect(self.preview_theme_background)

    def _set_background(self, color):
        self.background_color = color
        self.background.setText(f'Выбрать цвет…  {color.upper()}')
        self.background.setStyleSheet(f'QPushButton {{border-left: 24px solid {color}; padding: 5px;}}')

    def choose_background(self):
        color = QColorDialog.getColor(QColor(self.background_color), self, 'Цвет фона сцены')
        if color.isValid():
            self._set_background(color.name())

    def set_values(self, preferences):
        self._set_background(preferences.background)
        chosen = (preferences.anti_aliasing, preferences.samples if preferences.anti_aliasing == 'msaa' else 4)
        index = next(i for i in range(self.antialiasing.count())
                     if tuple(self.antialiasing.itemData(i)) == chosen)
        self.antialiasing.setCurrentIndex(index)
        self.interactive_edges.setChecked(preferences.interactive_edges)

    def preferences(self):
        aa, samples = self.antialiasing.currentData()
        return DisplayPreferences(self.background_color, aa, samples, self.interactive_edges.isChecked())

    def apply(self):
        preferences = self.preferences()
        if self.theme is not None: self.theme.set_mode(self.theme_choice.currentData())
        self.apply_preferences(preferences)
        self._set_background(preferences.background)

    def preview_theme_background(self):
        from ui_theme import THEME_COLORS
        mode = self.theme_choice.currentData()
        color = (self.theme.window.ui.display_preferences.background if mode == self.theme.mode else
                 self.theme.window.settings.value('appearance/background/' + mode, THEME_COLORS[mode]['scene']))
        self._set_background(color if QColor(color).isValid() else THEME_COLORS[mode]['scene'])

    def sync_theme(self, mode):
        self.theme_choice.setCurrentIndex(self.theme_choice.findData(mode))
        self._set_background(self.theme.window.ui.display_preferences.background)

    def restore_defaults(self):
        preferences = DisplayPreferences()
        if self.theme is not None:
            from ui_theme import SCENE
            self.theme_choice.setCurrentIndex(self.theme_choice.findData('light'))
            preferences = replace(preferences, background=SCENE)
        self.set_values(preferences)

    def accept(self):
        self.apply()
        super().accept()


class DisplaySettingsController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        buttons = window.ui.ribbon_btns
        for name, handler in (('Параметры', self.open_settings), ('Горячие клавиши', self.show_shortcuts),
                              ('Справка', self.show_help), ('О программе', self.show_about),
                              ('Проверить обновления', window.updater.manual_check)):
            buttons[name].clicked.connect(handler)

    def apply(self, preferences):
        self.window.ui.display_preferences = preferences
        for plotter in (self.window.ui.slicer_plotter, self.window.ui.plotter):
            apply_to_plotter(plotter, preferences)
        save_preferences(self.window.settings, preferences)
        display = getattr(self.window, 'display_tools', None)
        if display is not None: display.statistics.request()

    def open_settings(self):
        DisplaySettingsDialog(self.window.ui.display_preferences, self.apply, self.window).exec()

    def _show_text(self, title, html):
        dialog = QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.setStyleSheet(DIALOG_STYLE)
        dialog.resize(650, 480)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml(html)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText('Закрыть')
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_shortcuts(self):
        panel_help = ('<h3>Боковые панели</h3><p>Ручки с названиями по краям слайсера и предеформации '
                      'появляются только при скрытых панелях. У открытой панели на границе со сценой '
                      'есть тонкий захват с тремя точками. Щелчок — скрыть или открыть; '
                      'перетаскивание к сцене — вытянуть панель, к краю — свернуть. '
                      'Tab переводит фокус на ручку; Enter или пробел переключают панель. '
                      'Ширина и состояние сохраняются.</p>')
        cube_help = ('Перетаскивание куба левой кнопкой вращает сцену; '
                     'двойной щелчок по кубу возвращает изометрию. Работает в обеих сценах.<br>')
        self._show_text('Горячие клавиши и управление', f'''
            <h3>Проект</h3><p><b>Ctrl+S</b> — сохранить проект.<br>
            <b>Ctrl+Z</b> — отменить действие.<br>
            <b>Ctrl+Y</b> или <b>Ctrl+Shift+Z</b> — повторить действие.</p>
            {panel_help}
            <h3>Рабочая сцена</h3><p>Левая кнопка мыши — выбор детали или области;
            средняя кнопка — перемещение камеры; колесо — приближение и отдаление.<br>
            <b>Двойной щелчок левой кнопкой по детали в слайсере</b> — вращение вокруг центра детали.<br>
            <b>Двойной щелчок левой кнопкой по пустому месту</b> — вернуть центр вращения
            в центр плиты построения (по умолчанию). Выбор детали в списке центр не меняет.<br>
            Щелчок по грани куба — вид вдоль соответствующей оси.<br>
            {cube_help}
            <b>Зажатая правая кнопка</b> — пунктирный круг в центре сцены.<br>
            Если начать движение <b>внутри круга</b> — вращение сцены в 3D;
            <b>снаружи круга</b> — поворот по/против часовой стрелки в плоскости экрана.<br>
            Режим определяется в момент нажатия и сохраняется до отпускания кнопки.
            Работает в слайсере и предеформации, также при выборе поверхностей и измерении.<br>
            Правая кнопка без перетаскивания при выбранной детали — круговое меню действий.<br>
            Щелчок по детали выбирает её; щелчок по пустому месту снимает выбор всех деталей.<br>
            ЛКМ от пустого места — рамка выбора деталей; <b>Shift + ЛКМ</b> — добавить
            детали щелчком или рамкой, <b>Ctrl + ЛКМ</b> — переключить их выбор.<br>
            При открытой «Маркировке» рамка ЛКМ на выбранной детали задаёт область нанесения.<br>
            <b>Esc</b> — выйти из выбора поверхностей, измерения или установки поддержек.</p>
            <p>В режиме выбора поверхностей левая кнопка выбирает поверхность;
            <b>Shift</b> добавляет, <b>Ctrl</b> вычитает из выделения.
            Для вращения при выборе поверхностей и измерении используйте правую кнопку.
            <b>Alt + двойной щелчок левой кнопкой</b> меняет центр вращения слайсера
            без выхода из этих режимов.
            Режим задаётся на панели над сценой.</p>''')

    def show_help(self):
        self._show_text('Справка — Meshropractor', '''
            <h3>Слайсер</h3><p>На вкладке «Главная» импортируйте деталь. В списке деталей
            отметьте нужные в столбце «Выбр.», чтобы перемещать, вращать, копировать,
            сохранять или выгружать только выбранные модели.</p>
            <p>Панель над сценой выбирает поверхности треугольником, плоскостью или кистью.
            На вкладке «Поддержки» доступны генерация, ручная установка и просмотр областей.
            Ручные поддержки хранятся внутри своей детали.</p>
            <p>В панели «Сечения» выберите строку щелчком и изменяйте позицию ползунком.
            Сечение меняет отображение, не обрезая исходную геометрию.
            В панели «Измерения» выберите режим и укажите точки или поверхности на модели.</p>
            <h3>Предеформация</h3><p>Загрузите CAD и скан, совместите модели, постройте карту
            отклонений и задайте компенсацию. Для диагностики и лечения сетки используйте
            «Автоисправление»: проверка показывает проблемы, применение ремонта требует подтверждения.</p>
            <h3>Отображение и обновления</h3><p>В «Параметрах» доступны цвет фона, сглаживание
            и ускорение вращения. Настройки общие для обеих сцен. Проверка обновлений запускается
            автоматически при старте; кнопка «Проверить обновления» запускает её вручную.</p>''')

    def show_about(self):
        self._show_text('О программе', f'''<h2>Meshropractor {APP_VERSION}</h2>
            <p>Подготовка деталей к печати, проверка и исправление сеток,
            совмещение CAD со сканом и компенсация деформации.</p>
            <p><a href="https://github.com/B0ogie888/Meshropractor">Исходный код и документация</a><br>
            <a href="https://github.com/B0ogie888/Meshropractor/releases">Выпуски приложения</a><br>
            <a href="https://github.com/B0ogie888/Meshropractor/issues">Сообщить об ошибке</a></p>
            <p>У используемых библиотек собственные лицензии;
            отдельный движок полного исправления сеток распространяется с лицензией GPL.</p>''')
