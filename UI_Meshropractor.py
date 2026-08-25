import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QCheckBox, QGroupBox, QTextEdit,
                               QScrollArea, QTabWidget, QGridLayout, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QToolBar, QStyle, QMainWindow,
                               QToolButton, QMenu, QStackedWidget, QLineEdit, QProgressBar, QFileDialog)
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap, QIcon, QAction
from pyvistaqt import QtInteractor
from assets import LOGO_B64


class Ui_MainWindow(object):
    """Класс, который отвечает ТОЛЬКО за внешний вид программы (кнопки, цвета, ползунки)"""

    def setupUi(self, main_window: QMainWindow):
        self.sliders = {}
        self.mesh_colors = {
            "CAD": "#1f77b4",
            "Scan": "#d3d3d3",
            "Result": "#2ca02c"
        }

        # === БАЗОВЫЕ НАСТРОЙКИ ОКНА ===
        main_window.setWindowTitle("DeWarp Enterprise V6.1")
        main_window.resize(1600, 900)
        main_window.setWindowFlags(Qt.FramelessWindowHint)
        main_window.setMinimumSize(800, 600)
        main_window.setMouseTracking(True)

        self.central_widget = QWidget(main_window)
        self.central_widget.setMouseTracking(True)
        self.central_widget.setObjectName("MainWidget")
        self.central_widget.setStyleSheet("#MainWidget { background-color: #2b2b2b; }")
        main_window.setCentralWidget(self.central_widget)

        self.base_layout = QVBoxLayout(self.central_widget)
        self.base_layout.setContentsMargins(5, 5, 5, 5)

        # === ВЕРХНИЙ КАСТОМНЫЙ ЗАГОЛОВОК ===
        self.title_bar = QWidget()
        self.title_bar.setCursor(Qt.ArrowCursor)
        self.title_bar.setFixedHeight(40)
        self.title_bar.setStyleSheet("background-color: #2b2b2b;")

        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(5, 0, 0, 0)

        # --- КНОПКА МЕНЮ (GOM Style) ---
        self.menu_btn = QToolButton()
        self.menu_btn.setText(" ≡ Меню ")
        self.menu_btn.setCursor(Qt.PointingHandCursor)
        self.menu_btn.setStyleSheet("""
            QToolButton { font-size: 14px; font-weight: bold; color: white; background-color: #b31b1b; border: none; padding: 5px 15px; border-radius: 3px;}
            QToolButton:hover { background-color: #e74c3c; }
            QToolButton::menu-indicator { image: none; } 
        """)
        self.menu_btn.setPopupMode(QToolButton.InstantPopup)

        self.dropdown_menu = QMenu(main_window)
        self.dropdown_menu.setStyleSheet("""
            QMenu { background-color: #333333; color: white; border: 1px solid #555; font-size: 14px; }
            QMenu::item { padding: 10px 40px; }
            QMenu::item:selected { background-color: #b31b1b; }
        """)

        # Создаем действия (Экраны)
        action_start = QAction("🏠 Старт", main_window)
        action_slicer = QAction("🔪 Слайсер", main_window)
        action_predef = QAction("🕸 Предеформация", main_window)
        action_inspect = QAction("🔍 Инспектирование", main_window)
        action_report = QAction("📄 Отчет", main_window)

        self.dropdown_menu.addActions([action_start, action_slicer, action_predef, action_inspect, action_report])
        self.menu_btn.setMenu(self.dropdown_menu)
        self.title_layout.addWidget(self.menu_btn)

        # --- ТУЛБАР (Иконки и Лого) ---
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet("border: none;")

        # Вшиваем логотип
        logo_bytes = QByteArray.fromBase64(LOGO_B64)
        logo_pixmap = QPixmap()
        logo_pixmap.loadFromData(logo_bytes)
        main_window.setWindowIcon(QIcon(logo_pixmap))

        self.logo_label = QLabel()
        self.logo_label.setPixmap(logo_pixmap.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label.setStyleSheet("padding-left: 10px; padding-right: 20px;")
        self.toolbar.addWidget(self.logo_label)

        icon_save = main_window.style().standardIcon(QStyle.SP_DialogSaveButton)
        self.action_save = QAction(icon_save, "Сохранить", main_window)
        self.toolbar.addAction(self.action_save)

        icon_undo = main_window.style().standardIcon(QStyle.SP_ArrowBack)
        self.action_undo = QAction(icon_undo, "Назад", main_window)
        self.toolbar.addAction(self.action_undo)

        icon_redo = main_window.style().standardIcon(QStyle.SP_ArrowForward)
        self.action_redo = QAction(icon_redo, "Вперед", main_window)
        self.toolbar.addAction(self.action_redo)

        self.title_layout.addWidget(self.toolbar)
        self.title_layout.addStretch()

        # --- СИСТЕМНЫЕ КНОПКИ ОКНА ---
        self.btn_min = QPushButton("—")
        self.btn_max = QPushButton("▢")
        self.btn_close = QPushButton("✕")

        for btn in (self.btn_min, self.btn_max, self.btn_close):
            btn.setFixedSize(45, 40)
            btn.setStyleSheet("QPushButton { border: none; color: white; font-size: 14px; } "
                              "QPushButton:hover { background-color: #444444; }")
            self.title_layout.addWidget(btn)

        self.btn_close.setStyleSheet("QPushButton { border: none; color: white; font-size: 14px; } "
                                     "QPushButton:hover { background-color: #e81123; }")

        self.btn_min.clicked.connect(main_window.showMinimized)
        self.btn_max.clicked.connect(
            lambda: main_window.showNormal() if main_window.isMaximized() else main_window.showMaximized())
        self.btn_close.clicked.connect(main_window.close)

        self.base_layout.addWidget(self.title_bar)

        # === МЕНЕДЖЕР РАБОЧИХ ЗОН (QStackedWidget) ===
        self.stack = QStackedWidget()
        self.base_layout.addWidget(self.stack)

        # Создаем экраны
        self.page_start = self.init_start_page()
        self.page_slicer = self.init_slicer_page()
        self.page_predef = self.init_deformation_page()
        self.page_inspect = self.create_mockup_page("Зона инспектирования (в разработке...)")
        self.page_report = self.create_mockup_page("Отчеты (в разработке...)")
        self.page_recent = self.init_recent_page()

        self.stack.addWidget(self.page_start)
        self.stack.addWidget(self.page_slicer)
        self.stack.addWidget(self.page_predef)
        self.stack.addWidget(self.page_inspect)
        self.stack.addWidget(self.page_report)
        self.stack.addWidget(self.page_recent)

        # Подключаем меню к переключению зон
        action_start.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_start))
        action_slicer.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_slicer))
        action_predef.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_predef))
        action_inspect.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_inspect))
        action_report.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_report))

        # Открываем "Старт" по умолчанию при запуске
        self.stack.setCurrentWidget(self.page_start)

    # =========================================================
    # ГЕНЕРАТОРЫ РАБОЧИХ ЗОН
    # =========================================================

    def init_start_page(self):
        """Интерфейс стартовой страницы в стиле GOM Inspect"""
        page = QWidget()
        # Светло-серый фон как в GOM, чтобы белые кнопки выделялись
        page.setStyleSheet("background-color: #f4f4f4;")

        main_layout = QVBoxLayout(page)
        main_layout.setAlignment(Qt.AlignCenter)

        # Заголовок с логотипом
        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)

        # Декодируем и загружаем наш логотип
        logo_bytes = QByteArray.fromBase64(LOGO_B64)
        logo_pixmap = QPixmap()
        logo_pixmap.loadFromData(logo_bytes)

        lbl_icon = QLabel()
        # Ставим размер 40x40 пикселей со сглаживанием
        lbl_icon.setPixmap(logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_icon.setStyleSheet("padding-right: 10px;")  # Небольшой отступ до текста

        lbl_text = QLabel("Meshropractor")
        lbl_text.setStyleSheet("font-size: 24px; color: #333; font-weight: bold;")

        title_layout.addWidget(lbl_icon)
        title_layout.addWidget(lbl_text)

        main_layout.addLayout(title_layout)
        main_layout.addSpacing(30)  # Отступ между заголовком и кнопками

        # Сетка для кнопок
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignCenter)
        grid.setSpacing(20)

        # Создаем 3 кнопки с помощью вспомогательной функции
        self.btn_new_project = self.create_big_button("📄", "Новый проект")
        self.btn_open_project = self.create_big_button("📂", "Открыть проект")
        self.btn_recent_projects = self.create_big_button("🕒", "Недавно использованные\nпроекты")

        grid.addWidget(self.btn_new_project, 0, 0)
        grid.addWidget(self.btn_open_project, 0, 1)
        grid.addWidget(self.btn_recent_projects, 0, 2)

        main_layout.addLayout(grid)
        return page

    def create_big_button(self, icon_text, title):
        """Создает большую квадратную кнопку в стиле GOM"""
        btn = QPushButton()
        btn.setFixedSize(260, 200)
        btn.setCursor(Qt.PointingHandCursor)

        # Внутренний Layout для кнопки, чтобы расположить иконку над текстом
        layout = QVBoxLayout(btn)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        icon_lbl = QLabel(icon_text)
        icon_lbl.setAlignment(Qt.AlignCenter)
        # Прозрачный фон, чтобы не перекрывать эффекты кнопки
        icon_lbl.setStyleSheet("font-size: 70px; color: #777; background: transparent; border: none;")

        text_lbl = QLabel(title)
        text_lbl.setAlignment(Qt.AlignCenter)
        text_lbl.setStyleSheet("font-size: 16px; color: #333; background: transparent; border: none;")

        layout.addWidget(icon_lbl)
        layout.addWidget(text_lbl)

        # Стили самой кнопки (белый фон, серая рамка, при наведении - красная рамка)
        btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 2px;
            }
            QPushButton:hover {
                border: 2px solid #b31b1b;
                background-color: #fafafa;
            }
        """)
        return btn

    def create_mockup_page(self, title_text):
        """Заглушка для пустых зон"""
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(title_text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 28px; color: #888; font-weight: bold;")
        layout.addWidget(label)
        return page

    def init_slicer_page(self):
        """Интерфейс зоны Слайсера (подготовка к Concept Laser)"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        title = QLabel("Подготовка задания для Concept Laser M2 (.cls)")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white; margin: 15px;")
        layout.addWidget(title)

        # Файлы
        file_group = QGroupBox("1. Геометрия")
        file_group.setStyleSheet("QGroupBox { color: white; border: 1px solid #555; padding-top: 15px; }")
        fl = QVBoxLayout()

        self.lbl_slicer_part = QLabel("Деталь: Не выбрана")
        self.lbl_slicer_part.setStyleSheet("color: #ccc;")
        self.btn_load_slicer_part = QPushButton("📂 Выбрать деталь (STL)")

        self.lbl_slicer_supp = QLabel("Поддержки: Не выбраны (Опционально)")
        self.lbl_slicer_supp.setStyleSheet("color: #ccc;")
        self.btn_load_slicer_supp = QPushButton("📂 Выбрать поддержки (STL)")

        fl.addWidget(self.lbl_slicer_part)
        fl.addWidget(self.btn_load_slicer_part)
        fl.addWidget(self.lbl_slicer_supp)
        fl.addWidget(self.btn_load_slicer_supp)
        file_group.setLayout(fl)
        layout.addWidget(file_group)

        # Настройки
        param_group = QGroupBox("2. Параметры нарезки (По аналогии с Magics)")
        param_group.setStyleSheet("QGroupBox { color: white; border: 1px solid #555; padding-top: 15px; }")
        pl = QGridLayout()

        pl.addWidget(QLabel("Толщина слоя (мм):", styleSheet="color: white;"), 0, 0)
        self.input_layer_th = QLineEdit("0.03")
        pl.addWidget(self.input_layer_th, 0, 1)

        pl.addWidget(QLabel("Размер острова X/Y (мм):", styleSheet="color: white;"), 1, 0)
        self.input_island = QLineEdit("5.0")
        pl.addWidget(self.input_island, 1, 1)

        param_group.setLayout(pl)
        layout.addWidget(param_group)

        # Кнопка старта
        self.btn_run_slice = QPushButton("🚀 Сгенерировать .CLS")
        self.btn_run_slice.setStyleSheet(
            "height: 40px; background-color: #b31b1b; color: white; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.btn_run_slice)

        self.slicer_progress = QProgressBar()
        self.slicer_progress.setValue(0)
        layout.addWidget(self.slicer_progress)

        layout.addStretch()
        return page

    def init_deformation_page(self):
        """Интерфейс зоны Предеформации (ваш оригинальный код)"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.main_splitter)

        # --- Левая панель (Дерево) ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Элементы проекта")
        self.tree.setStyleSheet("QTreeWidget { font-size: 13px; color: white; background-color: #333; }")

        self.cat_cad = QTreeWidgetItem(self.tree, ["Номинальные элементы (CAD)"])
        self.cat_scan = QTreeWidgetItem(self.tree, ["Фактические элементы (Скан)"])
        self.cat_res = QTreeWidgetItem(self.tree, ["Результаты"])
        self.tree.expandAll()
        self.left_layout.addWidget(self.tree)

        # --- Средняя панель (3D) ---
        self.plotter = QtInteractor(page)
        self.plotter.setCursor(Qt.ArrowCursor)
        self.plotter.set_background('white')
        self.plotter.add_axes()

        # --- Правая панель (Настройки) ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)

        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.plotter.interactor)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([250, 950, 400])

        # === ВКЛАДКИ НА ПРАВОЙ ПАНЕЛИ ===
        self.tabs = QTabWidget()
        # Применяем черный текст ТОЛЬКО к верхним кнопкам вкладок, а внутренности делаем белыми
        self.tabs.setStyleSheet("""
                    QTabBar::tab { color: black; background-color: #cccccc; padding: 5px 10px; }
                    QTabBar::tab:selected { background-color: #ffffff; font-weight: bold; }
                    QTabWidget::pane { border: 1px solid #555; }
                    QWidget { color: #E0E0E0; } 
                """)

        self.tab_align = QWidget()
        self.initAlignTab()
        self.tabs.addTab(self.tab_align, "Шаг 1: Совмещение (ICP)")

        self.tab_comp = QWidget()
        self.initCompTab()
        self.tabs.addTab(self.tab_comp, "Шаг 2: Компенсация (RBF)")

        self.tab_donate = QWidget()
        self.initDonateTab()
        self.tabs.addTab(self.tab_donate, "💰 Поддержка автора")

        self.right_layout.addWidget(self.tabs, stretch=6)

        # === ПАНЕЛЬ СЛОЕВ ===
        self.view_group = QGroupBox("Слои (Видимость, Цвет, Прозрачность)")
        self.view_group.setStyleSheet("color: white;")
        self.view_layout = QGridLayout()

        self.chk_view_cad = QCheckBox("CAD")
        self.chk_view_cad.setChecked(True)
        self.btn_col_cad = self.create_color_button("CAD")
        self.sld_op_cad = QSlider(Qt.Horizontal)
        self.sld_op_cad.setRange(0, 100)
        self.sld_op_cad.setValue(80)
        self.lbl_op_cad = QLabel("80%")

        self.chk_view_scan = QCheckBox("Скан")
        self.chk_view_scan.setChecked(True)
        self.btn_col_scan = self.create_color_button("Scan")
        self.sld_op_scan = QSlider(Qt.Horizontal)
        self.sld_op_scan.setRange(0, 100)
        self.sld_op_scan.setValue(80)
        self.lbl_op_scan = QLabel("80%")

        self.chk_view_res = QCheckBox("Результат")
        self.chk_view_res.setChecked(True)
        self.btn_col_res = self.create_color_button("Result")
        self.sld_op_res = QSlider(Qt.Horizontal)
        self.sld_op_res.setRange(0, 100)
        self.sld_op_res.setValue(100)
        self.lbl_op_res = QLabel("100%")

        self.view_layout.addWidget(self.chk_view_cad, 0, 0);
        self.view_layout.addWidget(self.btn_col_cad, 0, 1);
        self.view_layout.addWidget(self.sld_op_cad, 0, 2);
        self.view_layout.addWidget(self.lbl_op_cad, 0, 3)
        self.view_layout.addWidget(self.chk_view_scan, 1, 0);
        self.view_layout.addWidget(self.btn_col_scan, 1, 1);
        self.view_layout.addWidget(self.sld_op_scan, 1, 2);
        self.view_layout.addWidget(self.lbl_op_scan, 1, 3)
        self.view_layout.addWidget(self.chk_view_res, 2, 0);
        self.view_layout.addWidget(self.btn_col_res, 2, 1);
        self.view_layout.addWidget(self.sld_op_res, 2, 2);
        self.view_layout.addWidget(self.lbl_op_res, 2, 3)

        self.sld_op_cad.valueChanged.connect(lambda v: self.lbl_op_cad.setText(f"{v}%"))
        self.sld_op_scan.valueChanged.connect(lambda v: self.lbl_op_scan.setText(f"{v}%"))
        self.sld_op_res.valueChanged.connect(lambda v: self.lbl_op_res.setText(f"{v}%"))

        self.view_group.setLayout(self.view_layout)
        self.right_layout.addWidget(self.view_group, stretch=0)

        # === ПАНЕЛЬ АНАЛИЗА ===
        self.heat_group = QGroupBox("Анализ (Цветовая карта отклонений)")
        self.heat_group.setStyleSheet("color: white;")
        self.heat_layout = QVBoxLayout()
        self.row_heat = QHBoxLayout()

        self.btn_heatmap = QPushButton("🔥 Построить Heatmap (Скан vs CAD)")
        self.btn_heatmap.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.row_heat.addWidget(self.btn_heatmap)

        self.btn_clear_heat = QPushButton("Сбросить")
        self.btn_clear_heat.setStyleSheet("color: black;")
        self.row_heat.addWidget(self.btn_clear_heat)
        self.heat_layout.addLayout(self.row_heat)

        self.add_slider(self.heat_layout, "Предел градиента (± мм)", 1, 50, 10, 1, "heat_limit", divider=10.0)
        self.heat_group.setLayout(self.heat_layout)
        self.right_layout.addWidget(self.heat_group, stretch=0)

        # === КОНСОЛЬ ===
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(180)
        self.console.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas; font-size: 12px;")
        self.right_layout.addWidget(self.console, stretch=0)

        return page

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ГЕНЕРАЦИИ ИНТЕРФЕЙСА ---
    def create_color_button(self, key):
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"background-color: {self.mesh_colors[key]}; border: 1px solid #555; border-radius: 3px;")
        return btn

    def add_slider(self, layout, text, vmin, vmax, vdef, step, name, divider=1.0):
        lbl = QLabel(f"{text}: {vdef / divider}")
        lbl.setStyleSheet("color: #E0E0E0; font-weight: bold;")  # Явный светлый цвет
        sld = QSlider(Qt.Horizontal)
        sld.setMinimum(vmin)
        sld.setMaximum(vmax)
        sld.setValue(vdef)
        sld.setSingleStep(step)
        sld.valueChanged.connect(lambda val, l=lbl, t=text, d=divider: l.setText(f"{t}: {val / d}"))
        layout.addWidget(lbl)
        layout.addWidget(sld)
        self.sliders[name] = (sld, divider)

    def initAlignTab(self):
        l = QVBoxLayout(self.tab_align)
        group_files = QGroupBox("1. Базовые данные")
        fl = QVBoxLayout()
        self.btn_load_cad = QPushButton("Загрузить Исходный CAD (.stl)")
        self.btn_load_scan = QPushButton("Загрузить Скан (.stl)")
        fl.addWidget(self.btn_load_cad)
        fl.addWidget(self.btn_load_scan)
        group_files.setLayout(fl)
        l.addWidget(group_files)

        group_pts = QGroupBox("2. Вспомогательные маркеры (От локальных минимумов)")
        pt_layout = QVBoxLayout()
        self.lbl_pts = QLabel("Точек на CAD: 0 | Точек на Скане: 0")
        self.lbl_pts.setStyleSheet("font-weight: bold; color: blue;")
        pt_layout.addWidget(self.lbl_pts)

        row_btns = QHBoxLayout()
        self.btn_pick_cad = QPushButton("📍 Выбрать на CAD")
        self.btn_pick_scan = QPushButton("📍 Выбрать на Скане")
        row_btns.addWidget(self.btn_pick_cad)
        row_btns.addWidget(self.btn_pick_scan)
        pt_layout.addLayout(row_btns)

        self.btn_clear_pts = QPushButton("Сбросить маркеры")
        pt_layout.addWidget(self.btn_clear_pts)
        group_pts.setLayout(pt_layout)
        l.addWidget(group_pts)

        self.btn_run_icp = QPushButton("▶ СОВМЕСТИТЬ МОДЕЛИ (ICP)")
        self.btn_run_icp.setStyleSheet(
            "height: 50px; background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        l.addWidget(self.btn_run_icp)
        l.addStretch()

    def initCompTab(self):
        l = QVBoxLayout(self.tab_comp)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        set_widget = QWidget()
        set_layout = QVBoxLayout(set_widget)

        self.add_slider(set_layout, "Разрешение маячков (>10k для мощных ПК)", 1000, 20000, 4000, 500, "points")
        self.add_slider(set_layout, "Жесткость RBF поля", 1, 150, 50, 1, "smooth", divider=10.0)
        self.chk_remesh = QCheckBox("Использовать Умный Remesh")
        self.chk_remesh.setChecked(True)
        set_layout.addWidget(self.chk_remesh)
        self.add_slider(set_layout, "Шаг сетки Remesh (мм)", 2, 30, 10, 1, "edge_len", divider=10.0)

        adv_lbl = QLabel("--- ПРОДВИНУТЫЕ НАСТРОЙКИ ---")
        adv_lbl.setStyleSheet("color: red; font-weight: bold; margin-top: 10px;")
        set_layout.addWidget(adv_lbl)

        self.add_slider(set_layout, "Область влияния RBF (Соседей) [0 = Глобально]", 0, 2000, 300, 50, "neighbors")
        self.add_slider(set_layout, "Лимит аномалий (мм)", 5, 50, 20, 1, "limit", divider=10.0)
        self.add_slider(set_layout, "Строгость нормалей", 30, 99, 80, 1, "norm", divider=100.0)
        self.chk_anchor = QCheckBox("Якорить пустоты сканера (0мм)")
        self.chk_anchor.setChecked(True)
        set_layout.addWidget(self.chk_anchor)

        scroll.setWidget(set_widget)
        l.addWidget(scroll)

        self.btn_run_comp = QPushButton("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.btn_run_comp.setStyleSheet(
            "height: 50px; background-color: #c0392b; color: white; font-weight: bold; font-size: 14px;")
        l.addWidget(self.btn_run_comp)

        self.btn_save = QPushButton("💾 Сохранить Результат")
        self.btn_save.setEnabled(False)
        l.addWidget(self.btn_save)

    def initDonateTab(self):
        l = QVBoxLayout(self.tab_donate)
        l.setAlignment(Qt.AlignCenter)

        lbl_story = QLabel(
            "Тяжело быть инженером-конструктором в наше время...\n\nБессонные ночи перед дедлайнами, литры выпитого кофе...")
        lbl_story.setAlignment(Qt.AlignCenter)
        lbl_story.setStyleSheet("font-size: 14px; font-style: italic; color: #E0E0E0; margin-bottom: 10px;")
        l.addWidget(lbl_story)

        # --- СПЕЦИАЛЬНАЯ ФУНКЦИЯ ДЛЯ ПОИСКА ПУТИ В .EXE ---
        import sys
        import os
        def resource_path(relative_path):
            """Возвращает правильный путь к файлу при запуске из Python и из .exe"""
            try:
                # PyInstaller создает временную папку и хранит путь в _MEIPASS
                base_path = sys._MEIPASS
            except Exception:
                base_path = os.path.abspath(".")
            return os.path.join(base_path, relative_path)

        # ----------------------------------------------------

        lbl_img = QLabel()
        # Используем нашу новую функцию, чтобы найти картинку
        img_path = resource_path("f.jpeg")
        pixmap = QPixmap(img_path)

        if not pixmap.isNull():
            lbl_img.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl_img.setText(f"[Картинка не найдена:\n{img_path}]")
            lbl_img.setStyleSheet("color: #ff4444; font-weight: bold;")
        lbl_img.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl_img)

        lbl_card = QLabel("💳 Реквизиты карты: <b style='font-size: 20px; color: #ff6b6b;'>2200150959050136</b>")
        lbl_card.setAlignment(Qt.AlignCenter)
        lbl_card.setStyleSheet("margin-top: 15px; color: white;")
        lbl_card.setTextInteractionFlags(Qt.TextSelectableByMouse)
        l.addWidget(lbl_card)

    def init_recent_page(self):
        """Галерея недавно использованных проектов"""
        page = QWidget()
        page.setStyleSheet("background-color: #f4f4f4;")
        layout = QVBoxLayout(page)

        # Заголовок и кнопка "Назад"
        header_layout = QHBoxLayout()
        self.btn_back_to_start = QPushButton("⬅ Назад")
        self.btn_back_to_start.setFixedSize(120, 40)
        self.btn_back_to_start.setCursor(Qt.PointingHandCursor)
        self.btn_back_to_start.setStyleSheet(
            "background-color: #555; color: white; font-weight: bold; border-radius: 3px;")

        title = QLabel(" Недавно использованные проекты")
        title.setStyleSheet("font-size: 24px; color: #333; font-weight: bold;")

        header_layout.addWidget(self.btn_back_to_start)
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Скроллируемая область для сетки проектов
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")

        self.recent_container = QWidget()
        self.recent_layout = QGridLayout(self.recent_container)
        self.recent_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.recent_layout.setSpacing(20)

        scroll.setWidget(self.recent_container)
        layout.addWidget(scroll)

        return page