# Файл: UI_Meshropractor.py
import sys
import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QCheckBox, QGroupBox, QTextEdit,
                               QScrollArea, QTabWidget, QGridLayout, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QToolBar, QStyle, QMainWindow,
                               QToolButton, QMenu, QStackedWidget, QLineEdit, QProgressBar, QFileDialog,
                               QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QComboBox, QSpinBox)
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap, QIcon, QAction
from pyvistaqt import QtInteractor
# from assets import LOGO_B64


class Ui_MainWindow(object):
    """Класс, который отвечает ТОЛЬКО за внешний вид программы (кнопки, цвета, ползунки)"""

    def setupUi(self, main_window: QMainWindow):
        self.sliders = {}
        self.mesh_colors = {
            "CAD": "#1f77b4",
            "Scan": "#d3d3d3",
            "Result": "#2ca02c"
        }
        self.ribbon_btns = {}  # Словарь для кнопок ленты слайсера

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
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")

        logo_pixmap = QPixmap(logo_path)
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

        # Подключаем кнопку "Новый проект" ---
        self.btn_new_project.clicked.connect(self.show_new_project_dialog)

        # Открываем "Старт" по умолчанию при запуске
        self.stack.setCurrentWidget(self.page_start)

    def show_new_project_dialog(self, checked=False):
        """Открывает диалог выбора и переключает на нужный экран"""
        dialog = DialogNewProject()
        if dialog.exec():
            print(f"DEBUG: Выбрана среда -> {dialog.selected_mode}")  # Будет видно в консоли PyCharm

            if dialog.selected_mode == "slicer":
                self.stack.setCurrentWidget(self.page_slicer)
            elif dialog.selected_mode == "predef":
                self.stack.setCurrentWidget(self.page_predef)
            elif dialog.selected_mode == "inspect":
                self.stack.setCurrentWidget(self.page_inspect)
            elif dialog.selected_mode == "report":
                self.stack.setCurrentWidget(self.page_report)

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
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")  # Укажи точное имя файла

        logo_pixmap = QPixmap(logo_path)

        lbl_icon = QLabel()
        # Ставим размер 40x40 пикселей со сглаживанием
        lbl_icon.setPixmap(logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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

    def create_magics_left_panel(self):
        """Создает левую мега-панель слайсера (Отображение, Детали, Заметки, Измерения)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #2b2b2b; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Единый темный стиль для всех блоков (имитация сворачиваемых панелей Magics)
        dark_style = """
            QGroupBox { border: 1px solid #444; margin-top: 20px; font-weight: bold; color: #E0E0E0; background-color: #2b2b2b; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; top: -10px; }
            QTabWidget::pane { border: 1px solid #444; background-color: #333; }
            QTabBar::tab { background-color: #222; color: #aaa; padding: 4px 10px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }
            QTabBar::tab:selected { background-color: #333; color: white; font-weight: bold; border-top: 2px solid #b31b1b; }
            QTableWidget { background-color: #2a2a2a; color: white; border: 1px solid #444; gridline-color: #444; font-size: 11px; }
            QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 2px; font-size: 11px; }
            QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 4px 8px; border-radius: 2px; }
            QPushButton:hover { background-color: #555; border: 1px solid #777; }
        """

        # --- 1. Отображение ---
        grp_disp = QGroupBox("▼ Отображение")
        grp_disp.setStyleSheet(dark_style)
        lo_disp = QVBoxLayout(grp_disp)
        lo_disp.setContentsMargins(5, 15, 5, 5)

        tabs_disp = QTabWidget()
        tab_sec = QWidget()
        lo_sec = QVBoxLayout(tab_sec)

        # Таблица сечений
        tbl_sec = QTableWidget(5, 6)
        tbl_sec.setHorizontalHeaderLabels(["Активно", "Тип", "Отсечь", "Цвет", "Позиция", "Шаг"])
        tbl_sec.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl_sec.verticalHeader().setVisible(False)
        tbl_sec.setFixedHeight(120)
        lo_sec.addWidget(tbl_sec)

        # Кнопки и ползунок под таблицей
        h_sec = QHBoxLayout()
        h_sec.addWidget(QPushButton("Указать"))
        h_sec.addWidget(QPushButton("Выровнять"))
        btn_exp = QPushButton("Экспорт ▾")
        h_sec.addWidget(btn_exp)
        sld_sec = QSlider(Qt.Horizontal)
        h_sec.addWidget(sld_sec)
        lo_sec.addLayout(h_sec)

        tabs_disp.addTab(tab_sec, "Сечения")
        tabs_disp.addTab(QWidget(), "Срезы")
        lo_disp.addWidget(tabs_disp)
        layout.addWidget(grp_disp)

        # --- 2. Детали ---
        grp_parts = QGroupBox("▼ Детали")
        grp_parts.setStyleSheet(dark_style)
        lo_parts = QVBoxLayout(grp_parts)
        lo_parts.setContentsMargins(5, 15, 5, 5)

        tabs_parts = QTabWidget()
        tab_list = QWidget()
        lo_list = QVBoxLayout(tab_list)

        h_list_top = QHBoxLayout()
        cb_plat = QComboBox()
        cb_plat.addItem("M2_220x220_Magics")
        cb_plat.setStyleSheet("background-color: #333; color: white;")
        h_list_top.addWidget(cb_plat, stretch=1)
        h_list_top.addWidget(QLabel("Кол-во деталей: 1"))
        lo_list.addLayout(h_list_top)

        tbl_parts = QTableWidget(1, 8)
        tbl_parts.setHorizontalHeaderLabels(
            ["#", "Выбранные", "Видимые", "Затенение", "Прозр.", "Цвет", "Способ", "Название"])
        tbl_parts.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tbl_parts.verticalHeader().setVisible(False)
        tbl_parts.setFixedHeight(100)
        lo_list.addWidget(tbl_parts)

        # Имитация мелких тулбарных иконок
        lo_list.addWidget(QLabel("🔧 👁 📋 ❌ 🔄 (Инструменты работы с деталью)"))

        tabs_parts.addTab(tab_list, "Список деталей")
        tabs_parts.addTab(QWidget(), "Информация о детали")
        tabs_parts.addTab(QWidget(), "Исправление деталей - инфо")
        tabs_parts.addTab(QWidget(), "Оценка времени")
        tabs_parts.addTab(QWidget(), "Сцены")
        lo_parts.addWidget(tabs_parts)
        layout.addWidget(grp_parts)

        # --- 3. Заметки ---
        grp_notes = QGroupBox("▼ Заметки")
        grp_notes.setStyleSheet(dark_style)
        lo_notes = QVBoxLayout(grp_notes)
        lo_notes.setContentsMargins(5, 15, 5, 5)
        tabs_notes = QTabWidget()
        tabs_notes.addTab(QWidget(), "Текст")
        tabs_notes.addTab(QWidget(), "Рисунки")
        tabs_notes.addTab(QWidget(), "Приложения")
        tabs_notes.addTab(QWidget(), "Текстуры")
        lo_notes.addWidget(tabs_notes)
        layout.addWidget(grp_notes)

        # --- 4. Измерения ---
        grp_meas = QGroupBox("▼ Измерения")
        grp_meas.setStyleSheet(dark_style)
        lo_meas = QVBoxLayout(grp_meas)
        lo_meas.setContentsMargins(5, 15, 5, 5)
        tabs_meas = QTabWidget()
        tab_dist = QWidget()
        lo_dist = QVBoxLayout(tab_dist)

        lo_dist.addWidget(QLabel("📏 🟢 🟩 (Тулбар измерений)"))
        info_meas = QGroupBox("Информация о измерениях")
        info_meas.setFixedHeight(60)
        lo_dist.addWidget(info_meas)
        lo_dist.addWidget(QCheckBox("Скрыто"))

        h_meas_btn = QHBoxLayout()
        h_meas_btn.addWidget(QPushButton("Выбрать"))
        h_meas_btn.addWidget(QPushButton("Очистить измерения"))
        h_meas_btn.addWidget(QPushButton("Настройки привязки"))
        lo_dist.addLayout(h_meas_btn)

        tabs_meas.addTab(tab_dist, "Расстояние")
        tabs_meas.addTab(QWidget(), "Окружность")
        tabs_meas.addTab(QWidget(), "Угол")
        tabs_meas.addTab(QWidget(), "Инфо")
        tabs_meas.addTab(QWidget(), "Фактическая деталь")
        tabs_meas.addTab(QWidget(), "Отчеты")
        lo_meas.addWidget(tabs_meas)
        layout.addWidget(grp_meas)

        # --- 5. Исправления деталей ---
        grp_fix = QGroupBox("▼ Исправления деталей")
        grp_fix.setStyleSheet(dark_style)
        lo_fix = QVBoxLayout(grp_fix)
        lo_fix.setContentsMargins(5, 15, 5, 5)
        tabs_fix = QTabWidget()
        tabs_fix.addTab(QWidget(), "Автоисправление")
        tabs_fix.addTab(QWidget(), "Базовые")
        tabs_fix.addTab(QWidget(), "Отверстия")
        tabs_fix.addTab(QWidget(), "Треугольники")
        tabs_fix.addTab(QWidget(), "Фрагмент")
        tabs_fix.addTab(QWidget(), "Нахлёсты")
        tabs_fix.addTab(QWidget(), "Точки")
        lo_fix.addWidget(tabs_fix)
        layout.addWidget(grp_fix)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def init_slicer_page(self):
        """Интерфейс зоны Слайсера в стиле Magics (Использование \n для переноса строк на кнопках)"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- 1. ВЕРХНЯЯ ЛЕНТА ИНСТРУМЕНТОВ ---
        self.magics_ribbon = QTabWidget()
        self.magics_ribbon.setFixedHeight(95)
        self.magics_ribbon.setStyleSheet("""
            QTabWidget::pane { 
                border-top: 1px solid #444444; 
                background-color: #2b2b2b; 
            }
            QTabBar::tab { 
                background-color: #222222; 
                color: #aaaaaa; 
                padding: 8px 15px; 
                font-weight: bold; 
                border: none;
            }
            QTabBar::tab:selected { 
                background-color: #2b2b2b; 
                color: #ffffff; 
                border-bottom: 2px solid #b31b1b; 
            }
            QTabBar::tab:hover { 
                color: #ffffff; 
                background-color: #333333; 
            }
        """)

        # Используем \n для аккуратного переноса длинных названий на вторую строку
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Создать", "Дублировать", "Пакетное\nдублирование"], "Создание"), "ИНСТРУМЕНТЫ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Автоисправление", "Бормашина", "Отверстия", "Триксел"], "Лечение сетки"),
            "ИСПРАВЛЕНИЕ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Текстура 1", "Текстура 2"], "Текстурирование"), "ТЕКСТУРЫ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Перемещать", "Вращать", "Масштабировать", "Озеркалить"], "Позиционирование"),
            "РАСПОЛОЖЕНИЕ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Платформа M2", "Платформа Mlab", "Параметры стола"], "Оборудование"), "ПЛАТФОРМЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Колонны", "Решетка", "Контурные\nподдержки"], "Генерация"),
                                  "ПОДДЕРЖКИ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Heatmap", "Сравнение", "Мин/Макс\nтолщины"], "Контроль"),
                                  "АНАЛИЗ И ОТЧЕТЫ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Создание срезов\nConcept Laser"], "Concept Laser"), "СРЕЗЫ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Цвет деталей", "Прозрачность", "Отображение\nсетки"], "Визуализация"),
            "ОТОБРАЖЕНИЕ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Параметры", "Язык", "Горячие\nклавиши"], "Система"),
                                  "НАСТРОЙКИ И ПОМОЩЬ")

        layout.addWidget(self.magics_ribbon)

        # --- 2. ЦЕНТРАЛЬНЫЙ БЛОК ---
        slicer_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(slicer_splitter)

        # Левая панель: Мега-панель Magics
        slicer_left_panel = self.create_magics_left_panel()

        # Центральная зона: 3D Сцена + Вкладки сцен снизу
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.slicer_plotter = QtInteractor(center_container)
        self.slicer_plotter.setCursor(Qt.ArrowCursor)
        self.slicer_plotter.set_background('white')
        self.slicer_plotter.add_axes()
        center_layout.addWidget(self.slicer_plotter.interactor)

        # ТЕМНЫЕ вкладки переключения сценок снизу
        self.scene_tabs = QTabWidget()
        self.scene_tabs.setFixedHeight(32)
        self.scene_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #222222; color: #aaaaaa; padding: 4px 15px; font-size: 11px; border: 1px solid #3d3d3d; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; font-weight: bold; border-bottom: 2px solid #b31b1b; }
            QTabBar::tab:hover { background-color: #333333; color: #ffffff; }
        """)
        self.scene_tabs.addTab(QWidget(), "🖥 Модельная сцена (В воздухе)")
        self.scene_tabs.addTab(QWidget(), "📦 M2_220x220 (Concept Laser)")
        self.scene_tabs.addTab(QWidget(), "📦 mlab_90x90 (Concept Laser)")
        self.scene_tabs.addTab(QWidget(), "📦 M2_245x245 (Concept Laser)")

        self.scene_tabs.currentChanged.connect(self.on_scene_tab_changed)
        center_layout.addWidget(self.scene_tabs)

        # Правая панель: Шторка параметров
        slicer_right_panel = QWidget()
        slicer_right_layout = QVBoxLayout(slicer_right_panel)
        slicer_right_layout.setContentsMargins(5, 5, 5, 5)

        slicer_splitter.addWidget(slicer_left_panel)
        slicer_splitter.addWidget(center_container)
        slicer_splitter.addWidget(slicer_right_panel)
        slicer_splitter.setSizes([450, 1150, 0])

        self.slicer_tabs = QTabWidget()
        self.slicer_tabs.setStyleSheet("""
            QTabBar::tab { color: black; background-color: #cccccc; padding: 5px 10px; }
            QTabBar::tab:selected { background-color: #ffffff; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #555; }
            QWidget { color: #E0E0E0; } 
        """)

        tab_preview = QWidget()
        tpr_layout = QVBoxLayout(tab_preview)
        tpr_layout.addWidget(QLabel("Просмотр срезов .CLS", styleSheet="color: white; font-weight: bold;"))
        tpr_layout.addStretch()
        self.slicer_tabs.addTab(tab_preview, "Срезы")

        slicer_right_layout.addWidget(self.slicer_tabs)
        return page

    def create_ribbon_tab(self, button_names, category_name):
        """Создает панель с поддержкой переноса текста через \n"""
        container = QWidget()
        container.setStyleSheet("background-color: #2b2b2b;")
        h_layout = QHBoxLayout(container)
        h_layout.setAlignment(Qt.AlignLeft)
        h_layout.setContentsMargins(8, 5, 8, 5)
        h_layout.setSpacing(8)

        for name in button_names:
            btn = QPushButton(name)
            btn.setFixedSize(115, 60)
            btn.setCursor(Qt.PointingHandCursor)

            # Qt автоматически переносит строки по символу \n и центрирует текст на кнопке
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #e0e0e0;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #383838;
                    border: 1px solid #666666;
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background-color: #222222;
                    border: 1px solid #b31b1b;
                }
            """)

            clean_name = name.replace('\n', ' ')
            self.ribbon_btns[clean_name] = btn

            h_layout.addWidget(btn)

        h_layout.addStretch()
        return container

    def on_scene_tab_changed(self, index):
        """Логика смены сцены в зависимости от выбранной платформы"""
        # Индексы: 0 - Модельная, 1 - M2_220x220, 2 - Mlab_90x90, 3 - M2_245x245
        if index == 0:
            print("Переключено на: Модельная сцена (без платформы)")
        elif index == 1:
            print("Переключено на платформу: M2 220x220 мм")
        elif index == 2:
            print("Переключено на платформу: Mlab 90x90 мм")
        elif index == 3:
            print("Переключено на платформу: M2 245x245 мм")

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

        # Создаем менеджер экранов для вкладки (Настройки <-> Прогресс-бар)
        self.comp_stack = QStackedWidget()

        # --- СТРАНИЦА 1: Настройки ---
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
        self.comp_stack.addWidget(scroll)  # Добавляем скролл с настройками на индекс 0

        # --- СТРАНИЦА 2: Прогресс-бар ---
        self.progress_widget = QWidget()
        prog_layout = QVBoxLayout(self.progress_widget)
        prog_layout.setAlignment(Qt.AlignCenter)

        self.lbl_progress_status = QLabel("Расчет матрицы предеформации...\nОжидайте завершения.")
        self.lbl_progress_status.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.lbl_progress_status.setAlignment(Qt.AlignCenter)

        self.comp_progress_bar = QProgressBar()
        self.comp_progress_bar.setRange(0, 100)
        self.comp_progress_bar.setValue(0)
        self.comp_progress_bar.setFixedSize(320, 35)
        self.comp_progress_bar.setTextVisible(True)
        self.comp_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #555; border-radius: 5px; text-align: center;
                        color: white; font-weight: bold; font-size: 14px; background-color: #333;
                    }
                    QProgressBar::chunk { background-color: #c0392b; border-radius: 3px; }
                """)

        # === ДОБАВЛЯЕМ КНОПКУ ОТМЕНЫ ===
        self.btn_cancel_comp = QPushButton("❌ Отменить расчет")
        self.btn_cancel_comp.setFixedSize(180, 35)
        self.btn_cancel_comp.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_comp.setStyleSheet("""
                    QPushButton {
                        background-color: #444; color: white; border: 1px solid #555; 
                        border-radius: 4px; font-weight: bold; font-size: 13px;
                    }
                    QPushButton:hover { background-color: #c0392b; border: 1px solid #ff5555; }
                """)

        prog_layout.addWidget(self.lbl_progress_status)
        prog_layout.addSpacing(20)
        prog_layout.addWidget(self.comp_progress_bar)
        prog_layout.addSpacing(15)  # Отступ между баром и кнопкой
        prog_layout.addWidget(self.btn_cancel_comp, alignment=Qt.AlignCenter)

        self.comp_stack.addWidget(self.progress_widget)  # Добавляем прогресс на индекс 1

        # Добавляем StackedWidget на вкладку
        l.addWidget(self.comp_stack)

        # --- КНОПКИ ЗАПУСКА И СОХРАНЕНИЯ (Всегда видны внизу) ---
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

        # Вычисляем путь к картинке, используя os (не забудь добавить import os в начало файла)
        import os
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        img_path = os.path.join(base_path, "assets", "f.jpeg")

        lbl_img = QLabel()
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


class DialogExportCLS(QDialog):
    """Модальное окно параметров экспорта для Concept Laser"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Concept Laser")
        self.setFixedSize(480, 800)

        # Фирменный темный стиль
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #e0e0e0; }
            QGroupBox { border: 1px solid #555; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #ccc; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 3px; }
            QLabel, QCheckBox, QRadioButton { color: #e0e0e0; }
            QLineEdit, QComboBox, QSpinBox { background-color: #333; color: white; border: 1px solid #555; padding: 3px; }
            QTableWidget { background-color: #333; color: white; border: 1px solid #555; gridline-color: #555; }
            QHeaderView::section { background-color: #444; color: white; border: 1px solid #555; }
            QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 5px 15px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; border: 1px solid #777; }
            QPushButton:pressed { background-color: #b31b1b; }
        """)

        main_layout = QVBoxLayout(self)

        # 1. Файлы
        grp_files = QGroupBox("Файлы")
        lo_files = QVBoxLayout(grp_files)
        table_layout = QHBoxLayout()
        self.table = QTableWidget(1, 2)
        self.table.setHorizontalHeaderLabels(["Модель", "Файлы срезов"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(60)
        self.table.setItem(0, 0, QTableWidgetItem("Деталь_1"))
        self.table.setItem(0, 1, QTableWidgetItem("Job_1.cls"))
        table_layout.addWidget(self.table)
        btn_browse = QPushButton("...")
        btn_browse.setFixedSize(30, 30)
        table_layout.addWidget(btn_browse)
        lo_files.addLayout(table_layout)
        lo_files.addWidget(QLabel("Выбранный каталог"))
        lo_files.addWidget(QLineEdit("C:\\Meshropractor\\Export"))
        main_layout.addWidget(grp_files)

        # 2. Пресеты
        grp_preset = QGroupBox("Предустановленный набор параметров")
        lo_preset = QVBoxLayout(grp_preset)
        h_combo = QHBoxLayout()
        h_combo.addWidget(QLabel("Предопредел."))
        h_combo.addWidget(QComboBox())
        lo_preset.addLayout(h_combo)
        h_btns = QHBoxLayout()
        for btn_text in ["Новый", "Обновить", "Удалить", "По умолчанию"]:
            h_btns.addWidget(QPushButton(btn_text))
        lo_preset.addLayout(h_btns)
        main_layout.addWidget(grp_preset)

        # 3. Срезы модели и поддержек
        h_slices = QHBoxLayout()
        grp_mod = QGroupBox("Срезы модели")
        lo_mod = QGridLayout(grp_mod)
        lo_mod.addWidget(QLabel("Толщина среза"), 0, 0)
        lo_mod.addWidget(QLineEdit("0,025"), 0, 1)
        lo_mod.addWidget(QLabel("Компенсация луча"), 1, 0)
        lo_mod.addWidget(QLineEdit("0,000"), 1, 1)
        h_slices.addWidget(grp_mod)

        grp_sup = QGroupBox("Срезы поддержек")
        lo_sup = QGridLayout(grp_sup)
        lo_sup.addWidget(QLabel("Толщина среза"), 0, 0)
        lo_sup.addWidget(QLineEdit("0,050"), 0, 1)
        lo_sup.setAlignment(Qt.AlignTop)
        h_slices.addWidget(grp_sup)
        main_layout.addLayout(h_slices)

        # 4. Оболочка-ядро
        grp_core = QGroupBox("Оболочка-ядро")
        lo_core = QVBoxLayout(grp_core)
        h_rad = QHBoxLayout()
        r1 = QRadioButton("Без ядра");
        r1.setChecked(True)
        h_rad.addWidget(r1)
        h_rad.addWidget(QRadioButton("Без оболочки"))
        h_rad.addWidget(QRadioButton("Оболочка"))
        lo_core.addLayout(h_rad)
        h_thick = QHBoxLayout()
        h_thick.addWidget(QLabel("Толщина стенки"))
        h_thick.addWidget(QLineEdit("0,000"))
        h_thick.addStretch()
        lo_core.addLayout(h_thick)
        main_layout.addWidget(grp_core)

        # 5. Островок (Шахматка)
        grp_isl = QGroupBox("Островок")
        lo_isl = QVBoxLayout(grp_isl)
        lo_isl.addWidget(QCheckBox("Включить островки"))
        grid_isl = QGridLayout()
        grid_isl.addWidget(QLabel("X-размер"), 0, 0);
        grid_isl.addWidget(QLineEdit("5,000"), 0, 1)
        grid_isl.addWidget(QLabel("Y-размер"), 1, 0);
        grid_isl.addWidget(QLineEdit("5,000"), 1, 1)
        grid_isl.addWidget(QLabel("X-сдвиг"), 0, 2);
        grid_isl.addWidget(QLineEdit("2,500"), 0, 3)
        grid_isl.addWidget(QLabel("Y-сдвиг"), 1, 2);
        grid_isl.addWidget(QLineEdit("2,500"), 1, 3)
        grid_isl.addWidget(QLabel("Угол"), 2, 0);
        grid_isl.addWidget(QLineEdit("0,000"), 2, 1)
        lo_isl.addLayout(grid_isl)
        main_layout.addWidget(grp_isl)

        # Кнопки внизу
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_yes = QPushButton("Да")
        btn_yes.clicked.connect(self.accept)  # Закрывает диалог с успехом
        btn_yes.setFixedWidth(80)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)  # Закрывает без сохранения
        btn_cancel.setFixedWidth(80)
        btn_layout.addWidget(btn_yes)
        btn_layout.addWidget(btn_cancel)
        main_layout.addLayout(btn_layout)


class DialogNewProject(QDialog):
    """Окно выбора типа нового проекта"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Создание нового проекта")

        # Немного увеличим высоту окна под два ряда кнопок
        self.setFixedSize(450, 340)

        # --- УСТАНОВКА ЛОГОТИПА В ЗАГОЛОВОК ---
        from PySide6.QtGui import QIcon
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        self.setWindowIcon(QIcon(logo_path))
        # ----------------------------------------

        # Строгий темный стиль окна
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel { color: #e0e0e0; font-size: 14px; font-weight: bold; }
            QPushButton { 
                background-color: #333333; 
                border: 1px solid #555555; 
                border-radius: 5px; 
                color: white; 
                font-size: 14px; 
            }
            QPushButton:hover { background-color: #444444; border: 1px solid #b31b1b; }
            QPushButton:pressed { background-color: #b31b1b; color: white; }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Выберите рабочую среду для нового проекта:"), alignment=Qt.AlignCenter)
        layout.addSpacing(10)

        # Сетка 2x2 для кнопок
        grid = QGridLayout()
        grid.setSpacing(15)

        # Кнопки
        self.btn_slicer = QPushButton("🔪\nСлайсер\n(Подготовка к печати)")
        self.btn_slicer.setFixedSize(190, 110)
        self.btn_slicer.setCursor(Qt.PointingHandCursor)

        self.btn_predef = QPushButton("🕸\nПредеформация\n(Компенсация усадки)")
        self.btn_predef.setFixedSize(190, 110)
        self.btn_predef.setCursor(Qt.PointingHandCursor)

        self.btn_inspect = QPushButton("🔍\nИнспектирование\n(Сравнение и контроль)")
        self.btn_inspect.setFixedSize(190, 110)
        self.btn_inspect.setCursor(Qt.PointingHandCursor)

        self.btn_report = QPushButton("📄\nОтчет\n(Генерация документации)")
        self.btn_report.setFixedSize(190, 110)
        self.btn_report.setCursor(Qt.PointingHandCursor)

        # Размещаем кнопки в сетке: строка, столбец
        grid.addWidget(self.btn_slicer, 0, 0)
        grid.addWidget(self.btn_predef, 0, 1)
        grid.addWidget(self.btn_inspect, 1, 0)
        grid.addWidget(self.btn_report, 1, 1)

        layout.addLayout(grid)

        # Логика выбора
        # Логика выбора (используем lambda для передачи точного имени режима)
        self.selected_mode = None
        self.btn_slicer.clicked.connect(lambda: self.set_mode("slicer"))
        self.btn_predef.clicked.connect(lambda: self.set_mode("predef"))
        self.btn_inspect.clicked.connect(lambda: self.set_mode("inspect"))
        self.btn_report.clicked.connect(lambda: self.set_mode("report"))

    def set_mode(self, mode):
        """Универсальная функция сохранения выбора"""
        self.selected_mode = mode
        self.accept()