"""Engineering shell over the shared UI controls and geometry tools.

The base class is intentionally reused: every registered ribbon command,
dialog, table and signal remains available as the application evolves.
"""
from dataclasses import replace

from PySide6.QtCore import Qt, QSize, QByteArray, QSignalBlocker
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QToolButton, QHeaderView, QButtonGroup, QSizePolicy, QTabWidget, QScrollArea,
    QGroupBox, QBoxLayout)

from ui_base import Ui_MainWindow as BaseUi, CollapsibleBox
from ribbon_layout import asset_icon, normalize_ribbon, style_engineering_commands
from ui_theme import EngineeringTheme, THEME_COLORS


class Ui_MainWindow(BaseUi):
    show_cad_ribbon_button = False

    def setupUi(self, main_window):
        super().setupUi(main_window)
        if not main_window.settings.contains('display/preferences'):
            mode = main_window.settings.value('appearance/theme', 'light')
            self.display_preferences = replace(self.display_preferences,
                background=THEME_COLORS.get(mode, THEME_COLORS['light'])['scene'])
        self.base_layout.setContentsMargins(0, 0, 0, 0)
        self.base_layout.setSpacing(0)
        self.magics_ribbon.setObjectName('EngineeringRibbon')
        self.magics_ribbon.setAttribute(Qt.WA_StyledBackground, True)
        self.magics_ribbon.setStyleSheet(self.magics_ribbon.styleSheet() + '''
            #EngineeringRibbon, #EngineeringRibbon QTabBar {background: #fafbf8;}
        ''')
        self.title_bar.setProperty('engineeringChrome', True)
        self.title_bar.setFixedHeight(48)
        self.title_bar.setStyleSheet('''
            QWidget { background: #252b2b; color: #fafbf8; }
            QLabel { background: transparent; color: #fafbf8; }
            QToolButton, QPushButton { color: #fafbf8; background: transparent;
                border: 0; padding: 5px 9px; }
            QToolButton:hover, QPushButton:hover { background: #46534e; }
        ''')
        self.menu_btn.hide()
        self.btn_close.setStyleSheet('QPushButton {color: #fafbf8; border: 0;} QPushButton:hover {background: #b64139;}')
        self.navigation = QWidget()
        self.navigation.setObjectName('EngineeringNavigation')
        nav = QHBoxLayout(self.navigation); nav.setContentsMargins(12, 4, 12, 4); nav.setSpacing(5)
        self.workspace_buttons = {}
        self.workspace_group = QButtonGroup(self.navigation)
        for key, text, group, name in (
                ('project', 'Проект', 'main', 'Загрузить проект'),
                ('slicer', 'Слайсер', 'workspace', 'slicer'),
                ('predef', 'Предеформация', 'workspace', 'predef'),
                ('reports', 'Анализ и отчёты', 'display', 'volume')):
            button = QToolButton(); button.setText(text); button.setCheckable(True)
            button.setProperty('engineeringQuietToggle', True)
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setIcon(asset_icon(group, name)); button.setIconSize(QSize(20, 20))
            button.setAccessibleName(text); button.setMinimumHeight(36)
            self.workspace_group.addButton(button); self.workspace_buttons[key] = button; nav.addWidget(button)
        nav.addStretch()
        self.new_log_button = QToolButton(); self.new_log_button.setCheckable(True)
        self.new_log_button.setText('Журнал'); self.new_log_button.setToolTip('Журнал операций')
        self.new_log_button.clicked.connect(lambda: self.log_dock.setVisible(self.new_log_button.isChecked()))
        self.log_dock.visibilityChanged.connect(self.new_log_button.setChecked)
        nav.addWidget(self.new_log_button)
        self.new_settings_button = QToolButton(); self.new_settings_button.setText('Параметры')
        self.new_settings_button.setIcon(asset_icon('settings', 'Параметры'))
        self.new_settings_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.new_settings_button.setAccessibleName('Параметры отображения')
        nav.addWidget(self.new_settings_button)
        self.inspector_button = QToolButton(); self.inspector_button.setCheckable(True)
        self.inspector_button.setIcon(asset_icon('display', 'dimensions'))
        self.inspector_button.setIconSize(QSize(20, 20))
        self.inspector_button.setToolTip('Показать / скрыть свойства выбранных деталей')
        self.inspector_button.setAccessibleName('Свойства выбранных деталей')
        nav.insertWidget(nav.indexOf(self.new_settings_button), self.inspector_button)
        self.inspector_button.clicked.connect(self.set_inspector_visible)
        self.base_layout.insertWidget(1, self.navigation)
        self.workspace_buttons['project'].clicked.connect(lambda: self.stack.setCurrentWidget(self.page_start))
        self.workspace_buttons['slicer'].clicked.connect(self.show_slicer)
        self.workspace_buttons['predef'].clicked.connect(lambda: self.stack.setCurrentWidget(self.page_predef))
        self.workspace_buttons['reports'].clicked.connect(self.show_reports)
        self.new_settings_button.clicked.connect(lambda: self.ribbon_btns['Параметры'].click())
        self.stack.currentChanged.connect(self.sync_navigation)
        self.magics_ribbon.currentChanged.connect(self.sync_navigation)

        parts = self.slicer_normal_groups[0]
        parts.toggle_button.setText('▼ Детали проекта')
        self.slicer_left_layout.removeWidget(parts)
        self.slicer_left_layout.insertWidget(0, parts, 1)
        for box in self.slicer_left_scroll.findChildren(CollapsibleBox):
            if box is not parts: box.toggle_button.setChecked(True)
        self.tbl_parts.setMinimumHeight(185)
        self.tbl_parts.setAlternatingRowColors(True)
        self.tbl_parts.verticalHeader().setDefaultSectionSize(34)
        header = self.tbl_parts.horizontalHeader()
        header.setStretchLastSection(False)
        header.moveSection(header.visualIndex(6), 0)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.resizeSection(1, 44); header.resizeSection(2, 44); header.resizeSection(5, 46)
        self.tbl_parts.horizontalHeaderItem(1).setText('Выб.')
        self.tbl_parts.horizontalHeaderItem(1).setToolTip('Выбрать деталь для операции')
        self.tbl_parts.horizontalHeaderItem(2).setText('Вид.')
        self.tbl_parts.horizontalHeaderItem(2).setToolTip('Показать / скрыть деталь')
        for table in (self.tbl_cad, self.tbl_scan, self.tbl_heat, self.tbl_res):
            table.setAlternatingRowColors(True); table.verticalHeader().setDefaultSectionSize(32)
        self.slicer_splitter.setSizes([390, 1200, 0])
        self.slicer_splitter.setStretchFactor(0, 0)
        self.slicer_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([395, 820, 330])
        self.sync_navigation()

    def init_start_page(self):
        page = QWidget(); layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 30, 36, 30); layout.setSpacing(18)
        label = QLabel('ИНЖЕНЕРНАЯ РАБОЧАЯ СРЕДА'); label.setStyleSheet('color: #5b6463; font-size: 12px;')
        title = QLabel('Meshropractor'); title.setStyleSheet('font-size: 32px; font-weight: 600; color: #252b2b;')
        description = QLabel('Подготовка деталей к печати и компенсация отклонений')
        description.setWordWrap(True)
        layout.addStretch(); layout.addWidget(label); layout.addWidget(title); layout.addWidget(description)
        grid = QGridLayout(); grid.setSpacing(16)
        entries = [
            ('btn_new_project', 'Новый проект', 'main', 'Новый проект'),
            ('btn_open_project', 'Открыть проект', 'main', 'Загрузить проект'),
            ('btn_recent_projects', 'Недавно использованные проекты', 'main', 'Сохранить проект'),
            ('btn_donate', 'Поддержать автора', 'settings', 'О программе')]
        for i, (key, text, group, name) in enumerate(entries):
            button = QToolButton(); button.setText(text); button.setIcon(asset_icon(group, name))
            button.setIconSize(QSize(32, 32)); button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setMinimumHeight(115); button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            setattr(self, key, button); grid.addWidget(button, i // 2, i % 2)
        layout.addLayout(grid)
        from app_version import APP_VERSION
        self.btn_check_updates = QPushButton(f'Проверить обновления · {APP_VERSION}')
        layout.addWidget(self.btn_check_updates, alignment=Qt.AlignLeft)
        layout.addStretch(2)
        return page

    def show_slicer(self):
        self.stack.setCurrentWidget(self.page_slicer)
        if self.magics_ribbon.tabText(self.magics_ribbon.currentIndex()) == 'АНАЛИЗ И ОТЧЕТЫ':
            self.magics_ribbon.setCurrentIndex(0)

    def show_reports(self):
        self.stack.setCurrentWidget(self.page_slicer)
        for index in range(self.magics_ribbon.count()):
            if self.magics_ribbon.tabText(index) == 'АНАЛИЗ И ОТЧЕТЫ':
                self.magics_ribbon.setCurrentIndex(index); break

    def sync_navigation(self, *args):
        page = self.stack.currentWidget()
        key = 'predef' if page is self.page_predef else 'slicer' if page is self.page_slicer else 'project'
        if key == 'slicer' and self.magics_ribbon.tabText(self.magics_ribbon.currentIndex()) == 'АНАЛИЗ И ОТЧЕТЫ':
            key = 'reports'
        self.workspace_buttons[key].setChecked(True)
        self.inspector_button.setVisible(page is self.page_slicer)

    def finish_setup(self, window):
        from part_controls import PartControls
        from parts_view import PartsBrowser
        list_page = self.tbl_parts.parentWidget()
        list_layout = list_page.layout()
        list_layout.removeWidget(self.tbl_parts)
        self.parts_browser = PartsBrowser(window, self.tbl_parts)
        list_layout.addWidget(self.parts_browser)
        self.part_controls = PartControls(window)
        list_layout.addWidget(self.part_controls)
        self.refine_sidebar()
        self.refine_predef(window)
        self.create_inspector(window)
        from panel_rails import PanelRails
        self.slicer_rails = PanelRails(window, self.slicer_splitter, 'slicer',
            {0: 'Детали проекта', 2: 'Свойства'}, {0: 350, 2: 320})
        self.predef_rails = PanelRails(window, self.main_splitter, 'predef',
            {0: 'Модели', 2: 'Параметры'}, {0: 355, 2: 415})
        self.slicer_rails.changed.connect(self.remember_inspector_size)
        self.remember_inspector_size()
        self.theme_button = QToolButton()
        self.theme_button.setFixedSize(40, 36)
        self.theme_button.setIconSize(QSize(21, 21))
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.title_layout.insertWidget(self.title_layout.indexOf(self.btn_min), self.theme_button)
        style_engineering_commands(self.magics_ribbon)
        window.engineering_theme = EngineeringTheme(window)
        self.theme_button.clicked.connect(lambda: window.engineering_theme.set_mode(
            'dark' if window.engineering_theme.mode == 'light' else 'light'))
        window.engineering_theme.changed.connect(self.update_theme_button)
        self.update_theme_button(window.engineering_theme.mode)
        self.action_save.setIcon(asset_icon('main', 'Сохранить проект'))
        for action in (self.action_save, self.action_undo, self.action_redo):
            action.setIcon(window.engineering_theme.recolor_icon(action.icon(), '#fafbf8'))
        self.navigation.setStyleSheet('''
            #EngineeringNavigation {background: #fafbf8; border-bottom: 1px solid #cdd3cf;}
            QToolButton {background: transparent; color: #252b2b; border: 0; padding: 5px 10px;}
            QToolButton:checked {border-bottom: 3px solid #e5d943; background: #f4f4e5;}
        ''')
        normalize_ribbon(self.magics_ribbon)
        # Several legacy pages rely on a palette captured by their scroll
        # viewport before theming. Give the actual panel its own light surface.
        for index in range(self.magics_ribbon.count()):
            scroll = self.magics_ribbon.widget(index)
            panel = scroll.widget()
            panel.setAutoFillBackground(True)
            panel.setPalette(window.engineering_theme.palette)
            scroll.viewport().setPalette(window.engineering_theme.palette)
        self.sync_navigation()

    def update_theme_button(self, mode):
        glyph = ('<path d="M19 15.5A8 8 0 0 1 8.5 5 8 8 0 1 0 19 15.5Z"/>' if mode == 'light' else
                 '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>')
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="none" stroke="#fafbf8" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{glyph}</g></svg>'
        pixmap = QPixmap(); pixmap.loadFromData(QByteArray(svg.encode()), 'SVG')
        self.theme_button.setIcon(QIcon(pixmap))
        text = 'Включить тёмную тему' if mode == 'light' else 'Включить светлую тему'
        self.theme_button.setToolTip(text); self.theme_button.setAccessibleName(text)

    @staticmethod
    def anchor_section(box, layout):
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box.content_layout.setAlignment(Qt.AlignTop)
        layout.setStretchFactor(box, 0)
        # The base toggle adds stretch to expanded boxes; the shell is an inspector
        # whose contents stay at the top, independent of the window height.
        box.toggle_button.toggled.connect(lambda *_: layout.setStretchFactor(box, 0))

    def refine_sidebar(self):
        """A flat project inspector instead of nested base group boxes."""
        sidebar = self.slicer_left_scroll.widget()
        sidebar.setObjectName('EngineeringSidebar')
        sidebar.setStyleSheet('')
        self.slicer_left_scroll.setMinimumWidth(350)
        self.slicer_left_layout.setContentsMargins(18, 8, 18, 16)
        self.slicer_left_layout.setSpacing(0)
        self.slicer_left_layout.setAlignment(Qt.AlignTop)
        parts = self.slicer_normal_groups[0]
        titles = {parts: 'ДЕТАЛИ ПРОЕКТА'}
        boxes = self.slicer_left_scroll.findChildren(CollapsibleBox)
        boxes = [parts] + [b for b in boxes if b is not parts]
        for number, box in enumerate(boxes, 1):
            title = titles.get(box, box.toggle_button.text().lstrip('▼▶ ').upper())
            arrow = '▶' if box.toggle_button.isChecked() else '▼'
            box.toggle_button.setText(f'{arrow}  {number:02d} / {title}')
            box.toggle_button.setStyleSheet('''QPushButton {text-align: left; background: transparent;
                color: #46564d; border: 0; border-top: 1px solid #dce2da;
                padding: 10px 0; font-size: 11px; font-weight: 600; border-radius: 0;}
                QPushButton:hover {color: #252b2b; background: #f0f2e9;}''')
            box.content_area.setStyleSheet('')
            box.content_layout.setContentsMargins(0, 0, 0, 10)
            self.anchor_section(box, self.slicer_left_layout)
        parts.toggle_button.setStyleSheet(parts.toggle_button.styleSheet().replace('border-top: 1px solid #dce2da;', 'border-top: 0;'))
        tabs = parts.findChild(QTabWidget)
        tabs.setObjectName('NewPartTabs')
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        tabs.setTabText(0, 'Детали'); tabs.setTabText(1, 'Сведения')
        tabs.setTabVisible(2, False)  # Old empty tab; real scene selector is retained.
        tabs.widget(0).layout().setContentsMargins(0, 12, 0, 0)
        tabs.widget(0).layout().setSpacing(12)
        self.lbl_part_count.hide()
        self.cb_plat.setStyleSheet('')
        self.cb_plat.setMinimumHeight(34)
        self.cb_plat.setToolTip('Активная платформа')
        self.tbl_parts.setShowGrid(False)
        sidebar.setStyleSheet('''
            #EngineeringSidebar {background: #fafbf8; border: 0;}
            QTabWidget::pane {border: 0; background: transparent;}
            QTabBar::tab {border: 0; background: transparent; color: #68776c; padding: 9px 14px;}
            QTabBar::tab:selected {color: #252b2b; border-bottom: 2px solid #b8aa20; background: transparent;}
            QLineEdit, QComboBox, QSpinBox {background: #f0f3ed; border: 0;
                border-bottom: 1px solid #ccd5ca; border-radius: 4px; padding: 7px 9px;}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {border-bottom-color: #a49819;}
            QComboBox::drop-down {border: 0; width: 20px;}
            #EngineeringPartList {background: transparent; border: 0; outline: 0; padding: 0;}
            #PartsEyebrow {font-family: "Consolas", "DejaVu Sans Mono"; font-size: 11px; color: #69796d;}
            #SelectionCount {font-size: 11px; color: #52644e; font-weight: 600;}
            #EmptyPartsTitle {font-size: 15px; font-weight: 600; color: #263b30;}
            #PartsHint {color: #69766d; font-size: 11px;}
            #PartControls {border: 0; border-top: 1px solid #dce2da;}
            #PartControls QPushButton {background: #edf1e8; border: 0; border-radius: 5px; padding: 9px 8px;}
            #PartControls QPushButton:hover {background: #e9e6bf;}
            QPushButton[quietAction="true"] {background: transparent; border: 0; color: #526553; padding: 2px 4px;}
            QPushButton[quietAction="true"]:hover {color: #252b2b; background: #e9eddf;}
            QSlider::groove:horizontal {height: 3px; border: 0; background: #d7decf;}
            QSlider::sub-page:horizontal {background: #a8ac57; border: 0;}
            QSlider::handle:horizontal {background: #556953; border: 0; width: 12px; margin: -5px 0; border-radius: 6px;}
            #PartControls QCheckBox {font-size: 11px; color: #69766d; padding-top: 3px;}
        ''')

    def refine_predef(self, window):
        from predef_panel import PredefBrowser, PredefControls
        scroll = self.left_panel.findChild(QScrollArea)
        scroll.setMinimumWidth(350)
        self.predef_left_scroll = scroll
        content = scroll.widget(); layout = content.layout()
        content.setObjectName('EngineeringSidebar')
        content.setStyleSheet(self.slicer_left_scroll.widget().styleSheet())
        layout.setContentsMargins(18, 8, 18, 16); layout.setSpacing(0); layout.setAlignment(Qt.AlignTop)
        self.predef_browsers = []
        for number, (box, table, title, hint) in enumerate((
                (self.grp_cad, self.tbl_cad, 'НОМИНАЛЬНАЯ МОДЕЛЬ / CAD', 'Загрузите STEP (CAD) или сетку STL.'),
                (self.grp_scan, self.tbl_scan, 'СКАНЫ', 'Добавьте фактическую сетку для сравнения.'),
                (self.grp_heat, self.tbl_heat, 'КАРТЫ ОТКЛОНЕНИЙ', 'Здесь появятся рассчитанные карты.'),
                (self.grp_res, self.tbl_res, 'РЕЗУЛЬТАТЫ', 'Здесь появятся модели после компенсации.')), 1):
            box.toggle_button.setText(f'▼  {number:02d} / {title}')
            box.toggle_button.setStyleSheet(self.slicer_normal_groups[0].toggle_button.styleSheet())
            box.content_area.setStyleSheet('')
            box.content_layout.setContentsMargins(0, 0, 0, 12)
            self.anchor_section(box, layout)
            index = box.content_layout.indexOf(table)
            box.content_layout.removeWidget(table)
            browser = PredefBrowser(window, table, hint)
            box.content_layout.insertWidget(index, browser)
            self.predef_browsers.append(browser)
        for label in (self.lbl_active_heatmap, self.lbl_result_quality):
            label.setObjectName('PartsHint'); label.setWordWrap(True)
        self.predef_controls = PredefControls(window, self.predef_browsers)
        layout.insertWidget(layout.count()-1, self.predef_controls)

        # The workflow keeps every original control and callback. Scrolling
        # each stage also makes its lower controls reachable on short screens.
        workflow = QWidget(); workflow.setObjectName('PredefWorkflow')
        workflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        bar = QHBoxLayout(workflow); bar.setContentsMargins(18, 8, 18, 8); bar.setSpacing(8)
        self.predef_steps = []; self.predef_step_group = QButtonGroup(workflow)
        for i, (page, title) in enumerate(((self.tab_align, 'Совмещение'), (self.tab_heatmap, 'Отклонения'),
                                         (self.tab_params, 'Деформация'), (self.tab_comp, 'Компенсация'))):
            button = QToolButton(); button.setText(f'{i+1:02d} / {title}'); button.setCheckable(True)
            button.setMinimumHeight(34); button.clicked.connect(lambda checked=False, index=i: self.tabs.setCurrentIndex(index))
            self.predef_step_group.addButton(button); self.predef_steps.append(button); bar.addWidget(button)
            page.setStyleSheet(''); page.layout().setAlignment(Qt.AlignTop)
            page.layout().setContentsMargins(14, 12, 14, 16)
            for label in page.findChildren(QLabel): label.setWordWrap(True)
            for group in page.findChildren(QGroupBox): group.setStyleSheet('')
        for stack in (self.def_stack, self.comp_stack):
            stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            for i in range(stack.count()): stack.widget(i).layout().setAlignment(Qt.AlignTop)
        # These long checkboxes were side by side and forced a horizontal
        # scrollbar. Keep both labels fully visible in the inspector column.
        for child_layout in self.tab_params.findChildren(QHBoxLayout):
            if child_layout.indexOf(self.chk_show_vectors) >= 0:
                child_layout.setDirection(QBoxLayout.TopToBottom)
                for i in reversed(range(child_layout.count())):
                    if child_layout.itemAt(i).spacerItem(): child_layout.takeAt(i)
        bar.addStretch()
        self.tabs.currentChanged.connect(lambda index: self.predef_steps[index].setChecked(True) if index >= 0 else None)
        self.tabs.setStyleSheet('QTabWidget::pane {border: 0; background: #fafbf8;}')
        pages = [(self.tabs.widget(i), self.tabs.tabText(i)) for i in range(self.tabs.count())]
        for page, title in pages:
            self.tabs.removeTab(0)
        for page, title in pages:
            stage = QScrollArea(); stage.setWidgetResizable(True); stage.setFrameShape(QScrollArea.NoFrame)
            stage.setWidget(page); self.tabs.addTab(stage, title)
        self.tabs.tabBar().hide()
        self.predef_steps[0].setChecked(True)
        self.page_predef.layout().insertWidget(0, workflow)
        self.page_predef.layout().setStretchFactor(self.main_splitter, 1)
        workflow.setStyleSheet('''#PredefWorkflow {background: #fafbf8; border-bottom: 1px solid #cdd3cf;}
            QToolButton {background: transparent; border: 0; color: #5b6463; padding: 7px 12px;}
            QToolButton:checked {background: #f4f4e5; color: #252b2b; border-bottom: 2px solid #b8aa20;}''')
        self.right_panel.setStyleSheet('''QGroupBox {border: 0; border-top: 1px solid #dce2da;
            margin-top: 16px; padding-top: 16px;} QGroupBox::title {color: #46564d; left: 0;}
            QScrollArea {background: #fafbf8; border: 0;}''')
        self.chk_icp.setText('Уточнить совмещение (ICP)')
        self.chk_icp.setToolTip('Вычислить дополнительное наилучшее соответствие (ICP)')
        self.main_splitter.setStretchFactor(0, 0); self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([355, 820, 415])
        self.page_predef.layout().setSpacing(0)
        self.right_layout.setContentsMargins(0, 0, 0, 0); self.right_layout.setSpacing(0)
        self._def_center_layout.setSpacing(0)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet('QSplitter::handle {background: #fafbf8; border: 0;}')

    def create_inspector(self, window):
        from part_inspector import PartInspector
        self._inspector_window = window
        self.inspector_panel = self.slicer_splitter.widget(2)
        self.inspector_panel.layout().setContentsMargins(0, 0, 0, 0)
        self.inspector_panel.layout().setSpacing(0)
        self.inspector_panel.setMinimumWidth(280)
        while self.slicer_tabs.count():
            page = self.slicer_tabs.widget(0); self.slicer_tabs.removeTab(0); page.deleteLater()
        self.part_inspector = PartInspector(window)
        self.inspector_scroll = QScrollArea(); self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setFrameShape(QScrollArea.NoFrame)
        self.inspector_scroll.setWidget(self.part_inspector)
        self.slicer_tabs.addTab(self.inspector_scroll, 'Свойства'); self.slicer_tabs.tabBar().hide()
        self.slicer_tabs.setStyleSheet('QTabWidget::pane {border: 0; background: #fafbf8;}')
        self.slicer_splitter.setHandleWidth(1)
        self.slicer_splitter.setStyleSheet('QSplitter::handle {background: #fafbf8; border: 0;}')
        self.slicer_splitter.setStretchFactor(2, 0)
        self.slicer_splitter.splitterMoved.connect(self.remember_inspector_size)
        self.set_inspector_visible(window.settings.value('new_ui/inspector_visible', True, type=bool))

    def set_inspector_visible(self, visible):
        if not hasattr(self, 'inspector_panel'): return
        if hasattr(self, 'slicer_rails'):
            self.slicer_rails.set_visible(2, visible)
            if visible: self.part_inspector.schedule_refresh()
            return
        settings = self._inspector_window.settings
        self.inspector_panel.setVisible(visible)
        with QSignalBlocker(self.inspector_button): self.inspector_button.setChecked(visible)
        settings.setValue('new_ui/inspector_visible', visible)
        if visible:
            sizes = self.slicer_splitter.sizes()
            width = max(280, min(600, settings.value('new_ui/inspector_width', 320, type=int)))
            self.slicer_splitter.setSizes([sizes[0] or 350, max(150, sum(sizes)-sizes[0]-width), width])
            self.part_inspector.schedule_refresh()

    def remember_inspector_size(self, *args):
        width = self.slicer_splitter.sizes()[2]
        with QSignalBlocker(self.inspector_button): self.inspector_button.setChecked(width > 0)
        self._inspector_window.settings.setValue('new_ui/inspector_visible', width > 0)
        if width > 0: self._inspector_window.settings.setValue('new_ui/inspector_width', width)

    def _ensure_slicer_plotter(self):
        super()._ensure_slicer_plotter()
        cube = getattr(getattr(self, 'workspace_tools', None), 'cube', None)
        if cube: cube.enable_drag = True

    def _ensure_def_plotter(self):
        super()._ensure_def_plotter()
        if hasattr(self, 'def_cube'): self.def_cube.enable_drag = True
