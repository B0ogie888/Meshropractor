# Файл: UI_Meshropractor.py
import sys
import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QCheckBox, QGroupBox, QTextEdit,
                               QScrollArea, QTabWidget, QGridLayout, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QToolBar, QStyle, QMainWindow,
                               QToolButton, QMenu, QStackedWidget, QLineEdit, QProgressBar, QFileDialog,
                               QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QComboBox, QSpinBox,
                               QFrame)
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap, QIcon, QAction
from pyvistaqt import QtInteractor


class CollapsibleBox(QWidget):
    """Кастомный виджет для сворачиваемых панелей (как в Magics)"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)  # False = панель развернута при старте
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet("""
            QPushButton { 
                text-align: left; font-weight: bold; background-color: #383838; 
                color: #e0e0e0; border: 1px solid #444; padding: 6px; border-radius: 2px; margin-top: 5px;
            }
            QPushButton:hover { background-color: #444444; border: 1px solid #777; }
            QPushButton:checked { background-color: #2b2b2b; color: #aaaaaa; border: 1px solid #333; }
        """)

        self.content_area = QWidget()
        self.content_area.setStyleSheet(
            ".QWidget { background-color: #2b2b2b; border: 1px solid #444; border-top: none; }")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(5)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self.on_toggle)

    def on_toggle(self, checked):
        # Если кнопка нажата (checked), скрываем контент
        self.content_area.setVisible(not checked)
        title = self.toggle_button.text()
        # Меняем стрелочку в зависимости от состояния
        if checked:
            self.toggle_button.setText(title.replace("▼", "▶"))
        else:
            self.toggle_button.setText(title.replace("▶", "▼"))

class Ui_MainWindow(object):
    """Класс, который отвечает ТОЛЬКО за внешний вид программы"""

    def setupUi(self, main_window: QMainWindow):
        print("[DEBUG] setupUi: старт", flush=True)
        self.sliders = {}
        self.mesh_colors = {
            "CAD": "#1f77b4",
            "Scan": "#d3d3d3",
            "Result": "#2ca02c"
        }
        self.ribbon_btns = {}

        # === БАЗОВЫЕ НАСТРОЙКИ ОКНА ===
        main_window.setWindowTitle("DeWarp Enterprise V6.1")
        main_window.resize(1600, 900)
        main_window.setWindowFlags(Qt.FramelessWindowHint)
        main_window.setMinimumSize(800, 600)
        main_window.setMouseTracking(True)

        self.central_widget = QWidget(main_window)
        self.central_widget.setMouseTracking(True)
        self.central_widget.setObjectName("MainWidget")

        # --- ГЛОБАЛЬНЫЙ СТИЛЬ: КРАСИМ ПОЛЗУНКИ И ЧЕКБОКСЫ ---
        main_window.setStyleSheet("""
                            #MainWidget { background-color: #2b2b2b; }

                            /* Стиль ползунков (QSlider) - спокойный красный цвет */
                            QSlider::groove:horizontal { border: 1px solid #444; height: 6px; background: #333; border-radius: 3px; }
                            QSlider::sub-page:horizontal { background: #c0392b; border-radius: 3px; }
                            QSlider::handle:horizontal { background: #ffffff; border: 1px solid #777; width: 14px; margin: -4px 0; border-radius: 7px; }
                            QSlider::handle:horizontal:hover { border: 1px solid #c0392b; background: #f0f0f0; }

                            /* Стиль галочек (QCheckBox) - темный квадрат с векторной галочкой */
                            QCheckBox { color: #e0e0e0; font-weight: bold; spacing: 8px; }
                            QCheckBox::indicator { 
                                width: 16px; 
                                height: 16px; 
                                border: 2px solid #555; 
                                border-radius: 4px; 
                                background-color: #333; 
                            }
                            QCheckBox::indicator:hover { 
                                border: 2px solid #c0392b; 
                            }
                            QCheckBox::indicator:checked { 
                                background-color: #333; 
                                border: 2px solid #c0392b; 
                                border-radius: 4px;
                                /* Используем встроенную системную галочку Qt */
                                image: url(":/qt-project.org/styles/commonstyle/images/check-16.png");
                            }
                        """)
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

        # --- КНОПКА МЕНЮ ---
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

        action_start = QAction("🏠 Старт", main_window)
        action_slicer = QAction("🔪 Слайсер", main_window)
        action_predef = QAction("🕸 Предеформация", main_window)
        action_inspect = QAction("🔍 Инспектирование", main_window)
        action_report = QAction("📄 Отчет", main_window)

        self.dropdown_menu.addActions([action_start, action_slicer, action_predef, action_inspect, action_report])
        self.menu_btn.setMenu(self.dropdown_menu)
        self.title_layout.addWidget(self.menu_btn)

        # --- ТУЛБАР ---
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet("border: none;")

        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_path, "assets", "logo.ico")

        logo_pixmap = QPixmap(logo_path)
        main_window.setWindowIcon(QIcon(logo_pixmap))

        self.logo_label = QLabel()
        if not logo_pixmap.isNull():
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

        # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ 3D-СЦЕН: Сработает только при переходе на нужную вкладку
        self.stack.currentChanged.connect(self._on_stack_current_changed)

        # Подключаем меню
        action_start.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_start))
        action_slicer.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_slicer))
        action_predef.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_predef))
        action_inspect.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_inspect))
        action_report.triggered.connect(lambda: self.stack.setCurrentWidget(self.page_report))

        self.btn_new_project.clicked.connect(self.show_new_project_dialog)

        # Стартовая страница по умолчанию
        self.stack.setCurrentWidget(self.page_start)
        print("[DEBUG] setupUi ПОЛНОСТЬЮ завершен OK", flush=True)

    def _ensure_slicer_plotter(self):
        """Лениво создает VTK-сцену слайсера"""
        if self.slicer_plotter is not None:
            return
        print("[DEBUG] Ленивое создание slicer_plotter...", flush=True)
        self.slicer_plotter = QtInteractor(self._slicer_center_container)
        self.slicer_plotter.setCursor(Qt.ArrowCursor)
        self.slicer_plotter.set_background('white')
        self.slicer_plotter.add_axes()
        self.slicer_plotter.winId()  # Заставляем Qt выделить память до скрытия
        self._slicer_center_layout.insertWidget(0, self.slicer_plotter)

    def _ensure_def_plotter(self):
        """Лениво создает VTK-сцену предеформации"""
        if self.plotter is not None:
            return
        print("[DEBUG] Ленивое создание plotter предеформации...", flush=True)
        self.plotter = QtInteractor(self._def_center_container)
        self.plotter.setCursor(Qt.ArrowCursor)
        self.plotter.set_background('white')
        self.plotter.add_axes()
        self.plotter.winId()  # Заставляем Qt выделить память до скрытия
        self._def_center_layout.addWidget(self.plotter)

    def _on_stack_current_changed(self, index):
        """Триггерит создание сцен при реальном переходе пользователя на вкладку"""
        if self.stack.widget(index) is self.page_slicer:
            self._ensure_slicer_plotter()
        elif self.stack.widget(index) is self.page_predef:
            self._ensure_def_plotter()

    def show_new_project_dialog(self, checked=False):
        dialog = DialogNewProject()
        if dialog.exec():
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
        page = QWidget()
        page.setStyleSheet("background-color: #f4f4f4;")
        main_layout = QVBoxLayout(page)
        main_layout.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)

        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_path, "assets", "logo.ico")

        logo_pixmap = QPixmap(logo_path)
        lbl_icon = QLabel()
        if not logo_pixmap.isNull():
            lbl_icon.setPixmap(logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_icon.setStyleSheet("padding-right: 10px;")

        lbl_text = QLabel("Meshropractor")
        lbl_text.setStyleSheet("font-size: 24px; color: #333; font-weight: bold;")

        title_layout.addWidget(lbl_icon)
        title_layout.addWidget(lbl_text)

        main_layout.addLayout(title_layout)
        main_layout.addSpacing(30)

        grid = QGridLayout()
        grid.setAlignment(Qt.AlignCenter)
        grid.setSpacing(20)

        self.btn_new_project = self.create_big_button("📄", "Новый проект")
        self.btn_open_project = self.create_big_button("📂", "Открыть проект")
        self.btn_recent_projects = self.create_big_button("🕒", "Недавно использованные\nпроекты")
        self.btn_donate = self.create_big_button("☕", "Поддержать\nавтора")  # <-- Новая кнопка

        grid.addWidget(self.btn_new_project, 0, 0)
        grid.addWidget(self.btn_open_project, 0, 1)
        grid.addWidget(self.btn_recent_projects, 1, 0)
        grid.addWidget(self.btn_donate, 1, 1)  # <-- Ставим её в правый нижний угол

        main_layout.addLayout(grid)
        return page

    def create_big_button(self, icon_text, title):
        btn = QPushButton()
        btn.setFixedSize(260, 200)
        btn.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(btn)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        icon_lbl = QLabel(icon_text)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 70px; color: #777; background: transparent; border: none;")

        text_lbl = QLabel(title)
        text_lbl.setAlignment(Qt.AlignCenter)
        text_lbl.setStyleSheet("font-size: 16px; color: #333; background: transparent; border: none;")

        layout.addWidget(icon_lbl)
        layout.addWidget(text_lbl)

        btn.setStyleSheet("""
            QPushButton { background-color: white; border: 1px solid #cccccc; border-radius: 2px; }
            QPushButton:hover { border: 2px solid #b31b1b; background-color: #fafafa; }
        """)
        return btn

    def create_mockup_page(self, title_text):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(title_text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 28px; color: #888; font-weight: bold;")
        layout.addWidget(label)
        return page

    def create_magics_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #2b2b2b; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(5)

        # Единый темный стиль для внутренних элементов
        dark_style = """
                    .QWidget { background-color: #2b2b2b; }
                    QTabWidget::pane { border: 1px solid #444; background-color: #333; }
                    QTabBar::tab { background-color: #222; color: #aaa; padding: 4px 10px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }
                    QTabBar::tab:selected { background-color: #333; color: white; font-weight: bold; border-top: 2px solid #b31b1b; }
                    QTableWidget { background-color: #2a2a2a; color: white; border: 1px solid #444; gridline-color: #444; font-size: 11px; }
                    QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 2px; font-size: 11px; }
                    QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 4px 8px; border-radius: 2px; font-weight: normal; }
                    QPushButton:hover { background-color: #555; border: 1px solid #777; }
                    QComboBox { background-color: #333; color: white; border: 1px solid #555; padding: 3px; }

                    /* Стилизуем чекбоксы внутри таблицы: центрируем через выравнивание ячейки
                       (см. setTextAlignment(Qt.AlignCenter) на самом item), БЕЗ жесткого сдвига -
                       margin-left фиксированной величиной "уезжал" за пределы узких колонок
                       (например "Видимые", 70px) и там чекбокс мог вообще не быть виден. */
                    QTableView::indicator {
                        width: 14px;
                        height: 14px;
                        border: 2px solid #555;
                        border-radius: 3px;
                        background-color: #333;
                    }
                    QTableView::indicator:hover {
                        border: 2px solid #c0392b;
                    }
                    /* Состояние "отмечено" показываем сплошной заливкой - раньше здесь
                       грузилась встроенная иконка Qt по внутреннему пути ресурсов, которая
                       на практике не загружалась, и отмеченный чекбокс выглядел как пустой
                       квадрат с одной лишь красной рамкой (галочки не было видно вовсе). */
                    QTableView::indicator:checked {
                        background-color: #c0392b;
                        border: 2px solid #c0392b;
                    }
                    QTableView::indicator:unchecked {
                        background-color: #333;
                        border: 2px solid #555;
                    }
                """
        content.setStyleSheet(dark_style)

        # --- 1. Отображение ---
        grp_disp = CollapsibleBox("▼ Отображение")
        tabs_disp = QTabWidget()
        tab_sec = QWidget()
        lo_sec = QVBoxLayout(tab_sec)
        tbl_sec = QTableWidget(5, 6)
        tbl_sec.setHorizontalHeaderLabels(["Активно", "Тип", "Отсечь", "Цвет", "Позиция", "Шаг"])
        tbl_sec.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl_sec.verticalHeader().setVisible(False)
        tbl_sec.setFixedHeight(120)
        lo_sec.addWidget(tbl_sec)
        h_sec = QHBoxLayout()
        h_sec.addWidget(QPushButton("Указать"))
        h_sec.addWidget(QPushButton("Выровнять"))
        h_sec.addWidget(QPushButton("Экспорт ▾"))
        h_sec.addWidget(QSlider(Qt.Horizontal))
        lo_sec.addLayout(h_sec)
        tabs_disp.addTab(tab_sec, "Сечения")
        tabs_disp.addTab(QWidget(), "Срезы")
        grp_disp.content_layout.addWidget(tabs_disp)
        layout.addWidget(grp_disp)

        # --- 2. Детали ---
        grp_parts = CollapsibleBox("▼ Детали")
        tabs_parts = QTabWidget()
        tab_list = QWidget()
        lo_list = QVBoxLayout(tab_list)
        h_list_top = QHBoxLayout()
        cb_plat = QComboBox()
        cb_plat.addItem("M2_220x220_Magics")
        cb_plat.setStyleSheet("background-color: #333; color: white;")
        h_list_top.addWidget(cb_plat, stretch=1)

        # 1. Сделали счетчик доступным извне
        self.lbl_part_count = QLabel("Кол-во деталей: 0", styleSheet="color: white;")
        h_list_top.addWidget(self.lbl_part_count)
        lo_list.addLayout(h_list_top)

        # 2. Сделали таблицу доступной и пустой по умолчанию (0 строк)
        self.tbl_parts = QTableWidget(0, 8)
        self.tbl_parts.setHorizontalHeaderLabels(
            ["#", "Выбранные ▾", "Видимые", "Затенение", "Прозр.", "Цвет", "Способ", "Название"])

        # Включаем интерактивный режим (позволяет пользователю таскать границы столбцов)
        self.tbl_parts.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_parts.horizontalHeader().setStretchLastSection(True)  # Название всегда заполняет остаток справа

        # Задаем комфортную стартовую ширину для каждого столбца в пикселях
        self.tbl_parts.horizontalHeader().resizeSection(0, 25)  # #
        self.tbl_parts.horizontalHeader().resizeSection(1, 85)  # Выбранные ▾
        self.tbl_parts.horizontalHeader().resizeSection(2, 70)  # Видимые
        self.tbl_parts.horizontalHeader().resizeSection(3, 70)  # Затенение
        self.tbl_parts.horizontalHeader().resizeSection(4, 50)  # Прозр.
        self.tbl_parts.horizontalHeader().resizeSection(5, 50)  # Цвет
        self.tbl_parts.horizontalHeader().resizeSection(6, 50)  # Способ

        self.tbl_parts.verticalHeader().setVisible(False)
        self.tbl_parts.setFixedHeight(100)
        self.tbl_parts.setSelectionBehavior(QTableWidget.SelectRows)  # Выделение строки целиком
        lo_list.addWidget(self.tbl_parts)

        lo_list.addWidget(QLabel("🔧 👁 📋 ❌ 🔄 (Инструменты работы с деталью)", styleSheet="color: white;"))
        tabs_parts.addTab(tab_list, "Список деталей")
        tabs_parts.addTab(QWidget(), "Информация о детали")
        tabs_parts.addTab(QWidget(), "Сцены")
        grp_parts.content_layout.addWidget(tabs_parts)
        layout.addWidget(grp_parts)

        # --- 3. Заметки ---
        grp_notes = CollapsibleBox("▼ Заметки")
        tabs_notes = QTabWidget()
        tabs_notes.addTab(QWidget(), "Текст")
        tabs_notes.addTab(QWidget(), "Рисунки")
        tabs_notes.addTab(QWidget(), "Приложения")
        tabs_notes.addTab(QWidget(), "Текстуры")
        grp_notes.content_layout.addWidget(tabs_notes)
        layout.addWidget(grp_notes)

        # --- 4. Измерения ---
        grp_meas = CollapsibleBox("▼ Измерения")
        tabs_meas = QTabWidget()
        tab_dist = QWidget()
        lo_dist = QVBoxLayout(tab_dist)
        lo_dist.addWidget(QLabel("📏 🟢 🟩 (Тулбар измерений)", styleSheet="color: white;"))
        info_meas = QGroupBox("Информация о измерениях")
        info_meas.setFixedHeight(60)
        info_meas.setStyleSheet(
            "QGroupBox { border: 1px solid #444; margin-top: 15px; color: #aaa; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; top: -10px; }")
        lo_dist.addWidget(info_meas)
        lo_dist.addWidget(QCheckBox("Скрыто"))
        h_meas_btn = QHBoxLayout()
        h_meas_btn.addWidget(QPushButton("Выбрать"))
        h_meas_btn.addWidget(QPushButton("Очистить"))
        lo_dist.addLayout(h_meas_btn)
        tabs_meas.addTab(tab_dist, "Расстояние")
        tabs_meas.addTab(QWidget(), "Угол")
        grp_meas.content_layout.addWidget(tabs_meas)
        layout.addWidget(grp_meas)

        # --- 5. Исправления деталей (ВОССТАНОВЛЕНО) ---
        grp_fix = CollapsibleBox("▼ Исправления деталей")
        tabs_fix = QTabWidget()
        tabs_fix.addTab(QWidget(), "Автоисправление")
        tabs_fix.addTab(QWidget(), "Базовые")
        tabs_fix.addTab(QWidget(), "Отверстия")
        tabs_fix.addTab(QWidget(), "Треугольники")
        tabs_fix.addTab(QWidget(), "Фрагмент")
        tabs_fix.addTab(QWidget(), "Нахлёсты")
        tabs_fix.addTab(QWidget(), "Точки")
        grp_fix.content_layout.addWidget(tabs_fix)
        layout.addWidget(grp_fix)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def create_main_ribbon_tab(self):
        """Создает вкладку 'ГЛАВНАЯ' с группами кнопок (Проект, Детали) и разделителем"""
        from PySide6.QtWidgets import QFrame  # На всякий случай импортируем здесь
        container = QWidget()
        container.setStyleSheet("background-color: #2b2b2b;")
        main_h_layout = QHBoxLayout(container)
        main_h_layout.setAlignment(Qt.AlignLeft)
        main_h_layout.setContentsMargins(8, 2, 8, 2)
        main_h_layout.setSpacing(15)

        def create_group(title, button_names):
            group_widget = QWidget()
            group_layout = QVBoxLayout(group_widget)
            group_layout.setAlignment(Qt.AlignBottom)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(2)

            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(5)
            for name in button_names:
                btn = QPushButton(name)
                btn.setFixedHeight(50)  # Фиксируем только высоту
                btn.setMinimumWidth(75)  # Минимальная ширина для коротких названий
                btn.setCursor(Qt.PointingHandCursor)
                btn.setStyleSheet("""
                                QPushButton { background-color: transparent; color: #e0e0e0; border: none; border-radius: 4px; font-size: 11px; font-weight: bold; padding: 0 6px; }
                                QPushButton:hover { background-color: #383838; border: 1px solid #666666; color: #ffffff; }
                                QPushButton:pressed { background-color: #222222; border: 1px solid #b31b1b; }
                            """)
                clean_name = name.replace('\n', ' ')
                self.ribbon_btns[clean_name] = btn
                btn_layout.addWidget(btn)

            group_layout.addLayout(btn_layout)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #777777; font-size: 10px; font-weight: bold;")
            group_layout.addWidget(lbl)
            return group_widget

        # Создаем группы и вертикальную линию
        group_project = create_group("Проект", ["Новый\nпроект", "Загрузить\nпроект", "Сохранить\nпроект",
                                                "Сохранить\nпроект как"])
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #555555;")
        group_parts = create_group("Детали",
                                   ["Импорт\nдетали", "Сохранить выбранные\nдетали как", "Сохранить\nвсе в папку",
                                    "Выгрузить\nдеталь"])

        main_h_layout.addWidget(group_project)
        main_h_layout.addWidget(line)
        main_h_layout.addWidget(group_parts)
        main_h_layout.addStretch()
        return container

    def init_slicer_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.magics_ribbon = QTabWidget()
        self.magics_ribbon.setFixedHeight(95)
        self.magics_ribbon.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #444444; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #222222; color: #aaaaaa; padding: 8px 15px; font-weight: bold; border: none; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; border-bottom: 2px solid #b31b1b; }
            QTabBar::tab:hover { color: #ffffff; background-color: #333333; }
        """)

        # === НОВАЯ ВКАДКА "ГЛАВНАЯ" (На 1 месте) ===
        self.magics_ribbon.addTab(self.create_main_ribbon_tab(), "ГЛАВНАЯ")

        # === ВОССТАНОВЛЕННЫЕ ОСТАЛЬНЫЕ ВКЛАДКИ ===
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
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Создание срезов\nConcept Laser"], "Concept Laser"), "СРЕЗЫ")
        self.magics_ribbon.addTab(
            self.create_ribbon_tab(["Цвет деталей", "Прозрачность", "Отображение\nсетки"], "Визуализация"),
            "ОТОБРАЖЕНИЕ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Параметры", "Язык", "Горячие\nклавиши"], "Система"),
                                  "НАСТРОЙКИ И ПОМОЩЬ")

        layout.addWidget(self.magics_ribbon)

        slicer_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(slicer_splitter)

        slicer_left_panel = self.create_magics_left_panel()

        # БАЗА ДЛЯ ЛЕНИВОГО СОЗДАНИЯ СЦЕНЫ
        self.slicer_plotter = None
        self._slicer_center_container = QWidget()
        self._slicer_center_layout = QVBoxLayout(self._slicer_center_container)
        self._slicer_center_layout.setContentsMargins(0, 0, 0, 0)
        self._slicer_center_layout.setSpacing(0)

        self.scene_tabs = QTabWidget()
        self.scene_tabs.setFixedHeight(32)
        self.scene_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #222222; color: #aaaaaa; padding: 4px 15px; font-size: 11px; border: 1px solid #3d3d3d; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; font-weight: bold; border-bottom: 2px solid #b31b1b; }
        """)
        self.scene_tabs.addTab(QWidget(), "🖥 Модельная сцена")
        self.scene_tabs.addTab(QWidget(), "📦 M2_220x220")
        self._slicer_center_layout.addWidget(self.scene_tabs)

        slicer_right_panel = QWidget()
        slicer_right_layout = QVBoxLayout(slicer_right_panel)
        slicer_right_layout.setContentsMargins(5, 5, 5, 5)

        slicer_splitter.addWidget(slicer_left_panel)
        slicer_splitter.addWidget(self._slicer_center_container)
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
        container = QWidget()
        container.setStyleSheet("background-color: #2b2b2b;")
        h_layout = QHBoxLayout(container)
        h_layout.setAlignment(Qt.AlignLeft)
        h_layout.setContentsMargins(8, 5, 8, 5)
        h_layout.setSpacing(8)

        for name in button_names:
            btn = QPushButton(name)
            btn.setFixedHeight(60)  # Фиксируем высоту панели инструментов
            btn.setMinimumWidth(75)  # Кнопка сама расширится под длинный текст
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                        QPushButton { background-color: transparent; color: #e0e0e0; border: none; border-radius: 4px; font-size: 11px; font-weight: bold; padding: 0 6px; }
                        QPushButton:hover { background-color: #383838; border: 1px solid #666666; color: #ffffff; }
                        QPushButton:pressed { background-color: #222222; border: 1px solid #b31b1b; }
                    """)
            clean_name = name.replace('\n', ' ')
            self.ribbon_btns[clean_name] = btn
            h_layout.addWidget(btn)

        h_layout.addStretch()
        return container

    def on_scene_tab_changed(self, index):
        pass

    def init_deformation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.main_splitter)

        # --- Левая панель ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Элементы проекта")
        self.tree.setStyleSheet("""
                    QTreeWidget { font-size: 13px; color: white; background-color: #333333; border: 1px solid #444; }
                    QHeaderView::section { background-color: #444444; color: white; font-weight: bold; border: 1px solid #555; padding: 4px; }
                """)

        self.cat_cad = QTreeWidgetItem(self.tree, ["Номинальные элементы (CAD)"])
        self.cat_scan = QTreeWidgetItem(self.tree, ["Фактические элементы (Скан)"])
        self.cat_res = QTreeWidgetItem(self.tree, ["Результаты"])
        self.tree.expandAll()
        self.left_layout.addWidget(self.tree)

        # --- Средняя панель (БАЗА ДЛЯ ЛЕНИВОЙ ЗАГРУЗКИ) ---
        self.plotter = None
        self._def_center_container = QWidget()
        self._def_center_layout = QVBoxLayout(self._def_center_container)
        self._def_center_layout.setContentsMargins(0, 0, 0, 0)

        # --- Правая панель ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)

        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self._def_center_container)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([250, 950, 400])

        # === ВКЛАДКИ НА ПРАВОЙ ПАНЕЛИ ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
                    QTabWidget::pane { border: 1px solid #444444; background-color: #2b2b2b; }
                    QTabBar::tab { background-color: #222222; color: #aaaaaa; padding: 8px 15px; border: 1px solid #444444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }
                    QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; font-weight: bold; border-top: 2px solid #b31b1b; }
                    QTabBar::tab:hover { background-color: #333333; color: #ffffff; }
                """)

        self.tab_align = QWidget()
        self.initAlignTab()
        self.tabs.addTab(self.tab_align, "Шаг 1: Совмещение (ICP)")

        self.tab_comp = QWidget()
        self.initCompTab()
        self.tabs.addTab(self.tab_comp, "Шаг 2: Компенсация (RBF)")

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

        return page

    def create_color_button(self, key):
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"background-color: {self.mesh_colors[key]}; border: 1px solid #555; border-radius: 3px;")
        return btn

    def add_slider(self, layout, text, vmin, vmax, vdef, step, name, divider=1.0):
        lbl = QLabel(f"{text}: {vdef / divider}")
        lbl.setStyleSheet("color: #E0E0E0; font-weight: bold;")
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

        # === ПРИМЕНЯЕМ ТЕМНЫЙ СТИЛЬ К ШАГУ 1 ===
        self.tab_align.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #cccccc; border: 1px solid #555555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #444444; color: white; border: 1px solid #555555; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #555555; border: 1px solid #777777; }
            QPushButton:pressed { background-color: #b31b1b; }
        """)

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

        # Изменили цвет на светло-синий для читаемости на темном фоне
        self.lbl_pts.setStyleSheet("font-weight: bold; color: #5dade2;")
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
            "height: 50px; background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px; border-radius: 4px;")
        l.addWidget(self.btn_run_icp)
        l.addStretch()

    def initCompTab(self):
        l = QVBoxLayout(self.tab_comp)

        # === ПРИМЕНЯЕМ ЖЕСТКИЙ ТЕМНЫЙ СТИЛЬ К ШАГУ 2 ===
        self.tab_comp.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #e0e0e0; }
            QScrollArea { border: none; background-color: #2b2b2b; }
            QCheckBox { color: #e0e0e0; font-weight: bold; }
        """)

        self.comp_stack = QStackedWidget()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        set_widget = QWidget()
        # Точка (.QWidget) гарантирует, что стиль применится только к фону, не ломая ползунки!
        set_widget.setStyleSheet(".QWidget { background-color: #2b2b2b; }")
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
        self.comp_stack.addWidget(scroll)

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
                    QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; color: white; font-weight: bold; font-size: 14px; background-color: #333; }
                    QProgressBar::chunk { background-color: #c0392b; border-radius: 3px; }
                """)

        self.btn_cancel_comp = QPushButton("❌ Отменить расчет")
        self.btn_cancel_comp.setFixedSize(180, 35)
        self.btn_cancel_comp.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_comp.setStyleSheet("""
                    QPushButton { background-color: #444; color: white; border: 1px solid #555; border-radius: 4px; font-weight: bold; font-size: 13px; }
                    QPushButton:hover { background-color: #c0392b; border: 1px solid #ff5555; }
                """)

        prog_layout.addWidget(self.lbl_progress_status)
        prog_layout.addSpacing(20)
        prog_layout.addWidget(self.comp_progress_bar)
        prog_layout.addSpacing(15)
        prog_layout.addWidget(self.btn_cancel_comp, 0, Qt.AlignCenter)

        self.comp_stack.addWidget(self.progress_widget)
        l.addWidget(self.comp_stack)

        self.btn_run_comp = QPushButton("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.btn_run_comp.setStyleSheet(
            "height: 50px; background-color: #c0392b; color: white; font-weight: bold; font-size: 14px;")
        l.addWidget(self.btn_run_comp)

        self.btn_save = QPushButton("💾 Сохранить Результат")
        self.btn_save.setEnabled(False)
        l.addWidget(self.btn_save)

    def init_recent_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: #f4f4f4;")
        layout = QVBoxLayout(page)

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Concept Laser")
        self.setFixedSize(480, 800)

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

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_yes = QPushButton("Да")
        btn_yes.clicked.connect(self.accept)
        btn_yes.setFixedWidth(80)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setFixedWidth(80)
        btn_layout.addWidget(btn_yes)
        btn_layout.addWidget(btn_cancel)
        main_layout.addLayout(btn_layout)


class DialogNewProject(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Создание нового проекта")
        self.setFixedSize(450, 340)

        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        self.setWindowIcon(QIcon(logo_path))

        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel { color: #e0e0e0; font-size: 14px; font-weight: bold; }
            QPushButton { background-color: #333333; border: 1px solid #555555; border-radius: 5px; color: white; font-size: 14px; }
            QPushButton:hover { background-color: #444444; border: 1px solid #b31b1b; }
            QPushButton:pressed { background-color: #b31b1b; color: white; }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Выберите рабочую среду для нового проекта:"), 0, Qt.AlignCenter)
        layout.addSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(15)

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

        grid.addWidget(self.btn_slicer, 0, 0)
        grid.addWidget(self.btn_predef, 0, 1)
        grid.addWidget(self.btn_inspect, 1, 0)
        grid.addWidget(self.btn_report, 1, 1)

        layout.addLayout(grid)

        self.selected_mode = None
        self.btn_slicer.clicked.connect(lambda: self.set_mode("slicer"))
        self.btn_predef.clicked.connect(lambda: self.set_mode("predef"))
        self.btn_inspect.clicked.connect(lambda: self.set_mode("inspect"))
        self.btn_report.clicked.connect(lambda: self.set_mode("report"))

    def set_mode(self, mode):
        self.selected_mode = mode
        self.accept()


class DialogDonate(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе / Поддержка")
        # Делаем окно более компактным и вертикальным
        self.setFixedSize(350, 480)

        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        self.setWindowIcon(QIcon(logo_path))

        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel { color: #e0e0e0; font-size: 14px; }
            QPushButton { background-color: #b31b1b; border: none; border-radius: 5px; color: white; font-size: 14px; font-weight: bold; padding: 8px 15px; }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:pressed { background-color: #8e1515; }
            /* Стилизуем кликабельную ссылку на GitHub */
            QLabel a { color: #5dade2; text-decoration: none; }
            QLabel a:hover { text-decoration: underline; color: #85c1e9; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(15)

        # 1. Логотип программы сверху
        lbl_logo = QLabel()
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            lbl_logo.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_logo.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_logo)

        # 2. Название и авторство
        lbl_author = QLabel("<b>Meshropractor</b><br>Автор: B0ogie888")
        lbl_author.setAlignment(Qt.AlignCenter)
        lbl_author.setStyleSheet("font-size: 18px;")
        main_layout.addWidget(lbl_author)

        # 3. Кликабельная ссылка на GitHub
        lbl_github = QLabel(
            "<a href='https://github.com/B0ogie888/Meshropractor'>Официальный репозиторий на GitHub</a>")
        lbl_github.setOpenExternalLinks(True)  # Открывать браузер при клике
        lbl_github.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_github)

        main_layout.addSpacing(15)

        # 4. Текст благодарности
        lbl_thanks = QLabel("Буду рад любой поддержке<br>на развитие проекта!")
        lbl_thanks.setAlignment(Qt.AlignCenter)
        lbl_thanks.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(lbl_thanks)

        # 5. QR-код Boosty
        qr_path = os.path.join(base_path, "assets", "qr_donate.png")
        lbl_qr = QLabel()
        qr_pixmap = QPixmap(qr_path)
        if not qr_pixmap.isNull():
            lbl_qr.setPixmap(qr_pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Заглушка, если картинка не найдется
            lbl_qr.setText("[QR-код Boosty]")
            lbl_qr.setStyleSheet("background-color: white; color: black; border: 2px dashed #777; font-weight: bold;")
            lbl_qr.setFixedSize(180, 180)

        lbl_qr.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_qr, 0, Qt.AlignCenter)

        main_layout.addSpacing(10)

        # 6. Кнопка закрытия
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setFixedSize(120, 35)
        self.btn_close.clicked.connect(self.accept)
        main_layout.addWidget(self.btn_close, 0, Qt.AlignCenter)