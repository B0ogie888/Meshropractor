# Файл: UI_Meshropractor.py
import sys
import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSlider, QCheckBox, QGroupBox, QTextEdit,
                               QScrollArea, QTabWidget, QGridLayout, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QToolBar, QStyle, QMainWindow,
                               QToolButton, QMenu, QStackedWidget, QLineEdit, QProgressBar, QFileDialog,
                               QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton, QComboBox, QSpinBox,
                               QFrame, QAbstractItemView, QStyledItemDelegate, QButtonGroup, QDoubleSpinBox)
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

        # --- УМНОЕ РАСШИРЕНИЕ ---
        # Если панель открыта (not checked), она забирает всё свободное место (stretch=1)
        # Если закрыта, отдаёт место другим (stretch=0)
        parent = self.parentWidget()
        if parent and parent.layout():
            parent.layout().setStretchFactor(self, 0 if checked else 1)


class NoEditTableWidget(QTableWidget):
    """QTableWidget с гарантированно отключенным редактированием ячеек.

    setEditTriggers(NoEditTriggers), выставленный на обычном QTableWidget,
    управляет только тем, ПОБУЖДАЕТ ли конкретное действие (клик/двойной клик/
    клавиша) начать редактирование - это настройка уровня экземпляра, и в редких
    случаях (нестандартные пути входа в редактирование в самом Qt/стиле) она может
    не сработать. Здесь же редактирование запрещено на уровне САМОГО МЕТОДА edit() -
    он всегда возвращает False, поэтому редактор не может открыться в принципе,
    независимо от того, что именно попытается его вызвать.
    """
    def edit(self, index, trigger=None, event=None):
        return False


class NoFocusDelegate(QStyledItemDelegate):
    """Делегат, который полностью блокирует визуальное выделение и фокус"""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)

        # 1. Отбираем у ячейки состояние "в фокусе" (убивает белые рамки)
        if option.state & QStyle.State_HasFocus:
            option.state &= ~QStyle.State_HasFocus

        # 2. Отбираем состояние "выделено" (УБИВАЕТ СИНИЕ ПОЛОСКИ и сглаживание шрифта)
        # Для движка отрисовки ячейка теперь ВСЕГДА выглядит как обычная невыделенная
        if option.state & QStyle.State_Selected:
            option.state &= ~QStyle.State_Selected

        # Для абсолютной надежности принудительно делаем шрифт обычным
        option.font.setBold(False)

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

        # --- ГЛОБАЛЬНЫЙ СТИЛЬ: КРАСИМ ВСЕ СИСТЕМНЫЕ МЕНЮ И ЭЛЕМЕНТЫ В ТЕМНЫЙ ---
        main_window.setStyleSheet("""
                    #MainWidget { background-color: #2b2b2b; }

                    /* Системные всплывающие меню (QMenu) по всей программе */
                    QMenu { 
                        background-color: #2b2b2b; 
                        color: #e0e0e0; 
                        border: 1px solid #444444; 
                        padding: 4px;
                    }
                    QMenu::item { 
                        padding: 6px 24px 6px 14px; 
                        border-radius: 2px;
                    }
                    QMenu::item:selected { 
                        background-color: #b31b1b; 
                        color: #ffffff; 
                    }
                    QMenu::separator { 
                        height: 1px; 
                        background: #444444; 
                        margin: 4px 6px; 
                    }

                    /* Все выпадающие списки (QComboBox) и их раскрывающиеся окна */
                    QComboBox { 
                        background-color: #333333; 
                        color: #ffffff; 
                        border: 1px solid #555555; 
                        padding: 4px; 
                        border-radius: 3px; 
                    }
                    QComboBox QAbstractItemView { 
                        background-color: #2b2b2b; 
                        color: #ffffff; 
                        selection-background-color: #b31b1b; 
                        selection-color: #ffffff; 
                        border: 1px solid #555555; 
                        outline: none;
                    }

                    /* Скроллбары (убираем белые системные полосы прокрутки) */
                    QScrollBar:vertical {
                        border: none;
                        background: #2b2b2b;
                        width: 8px;
                        margin: 0;
                    }
                    QScrollBar::handle:vertical {
                        background: #444444;
                        min-height: 20px;
                        border-radius: 4px;
                    }
                    QScrollBar::handle:vertical:hover {
                        background: #555555;
                    }
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                        height: 0px;
                    }

                    /* Ползунки (QSlider) */
                    QSlider::groove:horizontal { border: 1px solid #444; height: 6px; background: #333; border-radius: 3px; }
                    QSlider::sub-page:horizontal { background: #c0392b; border-radius: 3px; }
                    QSlider::handle:horizontal { background: #ffffff; border: 1px solid #777; width: 14px; margin: -4px 0; border-radius: 7px; }
                    QSlider::handle:horizontal:hover { border: 1px solid #c0392b; background: #f0f0f0; }

                    /* Галочки (QCheckBox) */
                    QCheckBox { color: #e0e0e0; font-weight: bold; spacing: 8px; }
                    QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #555; border-radius: 4px; background-color: #333; }
                    QCheckBox::indicator:hover { border: 2px solid #c0392b; }
                    QCheckBox::indicator:checked { background-color: #333; border: 2px solid #c0392b; border-radius: 4px; }
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

        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    def _build_standard_part_table(self):
        """Вспомогательный метод для создания пустой таблицы в стиле Слайсера"""
        tbl = QTableWidget(0, 7)
        tbl.setHorizontalHeaderLabels(["#", "Выбранные ▾", "Видимые", "Затенение", "Прозр.", "Цвет", "Название"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.horizontalHeader().resizeSection(0, 25)
        tbl.horizontalHeader().resizeSection(1, 85)
        tbl.horizontalHeader().resizeSection(2, 70)
        tbl.horizontalHeader().resizeSection(3, 70)
        tbl.horizontalHeader().resizeSection(4, 50)
        tbl.horizontalHeader().resizeSection(5, 50)
        tbl.verticalHeader().setVisible(False)
        tbl.setSelectionMode(QAbstractItemView.NoSelection)
        tbl.setFocusPolicy(Qt.NoFocus)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setItemDelegate(NoFocusDelegate(tbl))
        return tbl

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
        page.setStyleSheet("background-color: #2b2b2b;") # ТЕМНЫЙ ФОН
        main_layout = QVBoxLayout(page)
        main_layout.setAlignment(Qt.AlignCenter)

        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)

        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logo_path = os.path.join(base_path, "assets", "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_path, "assets", "logo.ico")

        logo_pixmap = QPixmap(logo_path)
        lbl_icon = QLabel()
        if not logo_pixmap.isNull():
            lbl_icon.setPixmap(logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_icon.setStyleSheet("padding-right: 10px;")

        lbl_text = QLabel("Meshropractor")
        lbl_text.setStyleSheet("font-size: 28px; color: #e0e0e0; font-weight: bold;") # СВЕТЛЫЙ ТЕКСТ

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
        icon_lbl.setStyleSheet("font-size: 70px; color: #aaaaaa; background: transparent; border: none;") # СВЕТЛАЯ ИКОНКА

        text_lbl = QLabel(title)
        text_lbl.setAlignment(Qt.AlignCenter)
        text_lbl.setStyleSheet("font-size: 16px; color: #e0e0e0; background: transparent; border: none;") # СВЕТЛЫЙ ТЕКСТ

        layout.addWidget(icon_lbl)
        layout.addWidget(text_lbl)

        # ТЕМНАЯ КНОПКА
        btn.setStyleSheet("""
            QPushButton { background-color: #333333; border: 1px solid #555555; border-radius: 5px; }
            QPushButton:hover { border: 2px solid #b31b1b; background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
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
                    QTableWidget { 
                        background-color: #2b2b2b; 
                        color: #e0e0e0; 
                        gridline-color: #444444; 
                        border: 1px solid #444444; 
                        selection-background-color: transparent; /* Убивает серый фон выделения всей строки */
                        selection-color: #e0e0e0;                /* Жестко фиксирует цвет (шрифт перестанет "жирнеть") */
                        outline: none;                           /* Добивает остатки рамок */
                    }
                    QTableWidget::item:hover { background-color: transparent; }
                    QTableWidget::item:focus { outline: none; }
                    QTableWidget::item:selected {
                        background-color: transparent; /* Убирает серую подсветку выделения */
                        color: #e0e0e0;                /* Фиксируем цвет текста */
                        border: none;                  /* Убивает синюю системную полоску */
                        outline: none;                 /* Убивает пунктирную рамку */
                        font-weight: normal;           /* Запрещает шрифту "жирнеть" */
                    }

                    QTableWidget::item:selected:active {
                        background-color: transparent; 
                        color: #e0e0e0;
                        border: none;
                    }
                    QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 2px; font-size: 11px; }
                    QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 4px 8px; border-radius: 2px; font-weight: normal; }
                    QPushButton:hover { background-color: #555; border: 1px solid #777; }
                    QComboBox { background-color: #333; color: white; border: 1px solid #555; padding: 3px; }
                    /* --- СТИЛЬ ДЛЯ ПОЛЯ ВВОДА (РЕДАКТИРОВАНИЕ НАЗВАНИЯ) --- */
                    QLineEdit { 
                        background-color: #222222; 
                        color: #ffffff; 
                        border: 1px solid #5dade2; /* Аккуратная синяя рамка фокуса */
                        padding: 0 4px;
                    }
                    
                    /* Стиль для полей "только для чтения" */
                    QLineEdit[readOnly="true"] {
                        background-color: #333333;
                        color: #aaaaaa;
                        border: 1px solid #444444;
                    }
                    
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
        layout.addWidget(grp_disp, stretch=1)

        # --- 2. Детали ---
        grp_parts = CollapsibleBox("▼ Детали")
        tabs_parts = QTabWidget()
        tab_list = QWidget()
        lo_list = QVBoxLayout(tab_list)
        h_list_top = QHBoxLayout()
        # Делаем комбобокс публичным (self) и убираем хардкод
        self.cb_plat = QComboBox()
        self.cb_plat.setStyleSheet("background-color: #333; color: white;")
        h_list_top.addWidget(self.cb_plat, stretch=1)

        # 1. Сделали счетчик доступным извне
        self.lbl_part_count = QLabel("Кол-во деталей: 0", styleSheet="color: white;")
        h_list_top.addWidget(self.lbl_part_count)
        lo_list.addLayout(h_list_top)

        # 2. Сделали таблицу доступной и пустой по умолчанию (0 строк)
        self.tbl_parts = QTableWidget(0, 7)
        self.tbl_parts.setHorizontalHeaderLabels(
            ["#", "Выбранные ▾", "Видимые", "Затенение", "Прозр.", "Цвет", "Название"])

        self.tbl_parts.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_parts.horizontalHeader().setStretchLastSection(True)

        self.tbl_parts.horizontalHeader().resizeSection(0, 25)
        self.tbl_parts.horizontalHeader().resizeSection(1, 85)
        self.tbl_parts.horizontalHeader().resizeSection(2, 70)
        self.tbl_parts.horizontalHeader().resizeSection(3, 70)
        self.tbl_parts.horizontalHeader().resizeSection(4, 50)
        self.tbl_parts.horizontalHeader().resizeSection(5, 50)

        self.tbl_parts.verticalHeader().setVisible(False)
        self.tbl_parts.setSelectionBehavior(QTableWidget.SelectRows)  # Выделение строки целиком
        # --- ФИНАЛЬНЫЙ УДАР: Полностью запрещаем выделение строк ---
        self.tbl_parts.setSelectionMode(QAbstractItemView.NoSelection)
        # --- ИСПРАВЛЕНИЕ БАГА: Убиваем синие полоски и белые рамки фокуса ---
        self.tbl_parts.setFocusPolicy(Qt.NoFocus)
        # --- ИСПРАВЛЕНИЕ БАГА (продолжение): предыдущие два фикса выше (стили :selected
        # и setFocusPolicy) лечат подсветку/фокус САМОЙ ячейки, но не помогали, потому что
        # реальная причина "жирного/крупного текста + синей полоски" была в другом: клик по
        # ячейке "Название" переводил ее в режим РЕДАКТИРОВАНИЯ - поверх ячейки всплывал
        # отдельный виджет QLineEdit со своим системным шрифтом и своей стандартной синей
        # подсветкой выделения текста, которые стилям таблицы (dark_style) не подчиняются
        # вообще, т.к. это другой виджет. Редактирование названия детали инлайн в этой
        # таблице не используется как функция - поэтому просто отключаем его целиком.
        # NoEditTriggers запрещает редактирование на уровне "что считать поводом его начать",
        # а класс NoEditTableWidget (см. выше) дополнительно запрещает его на уровне самого
        # метода edit() - редактор не откроется в принципе, каким бы путем его ни попытались
        # вызвать (одного NoEditTriggers на практике оказалось недостаточно, отсюда и вторая,
        # более жесткая защита).
        self.tbl_parts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # --- ФИНАЛЬНОЕ УБИЙСТВО СИНИХ РАМОК И ЖИРНОГО ШРИФТА ---
        self.tbl_parts.setItemDelegate(NoFocusDelegate(self.tbl_parts))
        lo_list.addWidget(self.tbl_parts)

        tabs_parts.addTab(tab_list, "Список деталей")
        # === ВКЛАДКА "ИНФОРМАЦИЯ О ДЕТАЛИ" ===
        tab_info = QWidget()
        lo_info = QVBoxLayout(tab_info)
        lo_info.setSpacing(10)

        # 1. Верхняя панель (Выбор детали)
        h_info_top = QHBoxLayout()
        h_info_top.addWidget(QLabel("Название детали"))
        self.cb_info_name = QComboBox()
        self.cb_info_name.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        h_info_top.addWidget(self.cb_info_name, stretch=1)
        self.btn_info_next = QPushButton("Далее")
        h_info_top.addWidget(self.btn_info_next)
        lo_info.addLayout(h_info_top)

        # Вспомогательная функция для создания нередактируемых полей
        def make_ro_line():
            le = QLineEdit("0.000")
            le.setReadOnly(True)
            le.setAlignment(Qt.AlignRight)
            return le

        # 2. Группа "Размеры" (Bounding Box)
        grp_dim = QGroupBox("Размеры")
        grid_dim = QGridLayout(grp_dim)
        grid_dim.addWidget(QLabel("Мин"), 0, 1)
        grid_dim.addWidget(QLabel("Макс"), 0, 2)
        grid_dim.addWidget(QLabel("Дельта"), 0, 3)

        self.le_dim_min = [make_ro_line(), make_ro_line(), make_ro_line()]
        self.le_dim_max = [make_ro_line(), make_ro_line(), make_ro_line()]
        self.le_dim_delta = [make_ro_line(), make_ro_line(), make_ro_line()]

        for i, axis in enumerate(["X", "Y", "Z"]):
            grid_dim.addWidget(QLabel(axis), i + 1, 0)
            grid_dim.addWidget(self.le_dim_min[i], i + 1, 1)
            grid_dim.addWidget(self.le_dim_max[i], i + 1, 2)
            grid_dim.addWidget(self.le_dim_delta[i], i + 1, 3)
            grid_dim.addWidget(QLabel("мм"), i + 1, 4)
        lo_info.addWidget(grp_dim)

        # 3. Группа "Информация о поверхности"
        grp_surf = QGroupBox("Информация о поверхности")
        grid_surf = QGridLayout(grp_surf)
        self.le_tris = make_ro_line();
        self.le_tris.setText("0")
        self.le_pts = make_ro_line();
        self.le_pts.setText("0")
        grid_surf.addWidget(QLabel("# Треугольников"), 0, 0)
        grid_surf.addWidget(self.le_tris, 0, 1)
        grid_surf.addWidget(QLabel("# Точек"), 0, 2)
        grid_surf.addWidget(self.le_pts, 0, 3)
        lo_info.addWidget(grp_surf)

        # 4. Группа "Параметры" (Объем и площадь)
        grp_params = QGroupBox("Параметры")
        grid_params = QGridLayout(grp_params)
        self.le_vol = make_ro_line()
        self.le_area = make_ro_line()
        grid_params.addWidget(QLabel("Объем"), 0, 0)
        grid_params.addWidget(self.le_vol, 0, 1)
        grid_params.addWidget(QLabel("мм³"), 0, 2)
        grid_params.addWidget(QLabel("Поверхность"), 1, 0)
        grid_params.addWidget(self.le_area, 1, 1)
        grid_params.addWidget(QLabel("мм²"), 1, 2)
        lo_info.addWidget(grp_params)

        lo_info.addStretch()
        tabs_parts.addTab(tab_info, "Информация о детали")
        # =====================================
        tabs_parts.addTab(QWidget(), "Сцены")
        grp_parts.content_layout.addWidget(tabs_parts)
        layout.addWidget(grp_parts, stretch=1)

        # --- 3. Заметки ---
        grp_notes = CollapsibleBox("▼ Заметки")
        tabs_notes = QTabWidget()
        tabs_notes.addTab(QWidget(), "Текст")
        tabs_notes.addTab(QWidget(), "Рисунки")
        tabs_notes.addTab(QWidget(), "Приложения")
        tabs_notes.addTab(QWidget(), "Текстуры")
        grp_notes.content_layout.addWidget(tabs_notes)
        layout.addWidget(grp_notes, stretch=1)

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
        layout.addWidget(grp_meas, stretch=1)

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
        layout.addWidget(grp_fix, stretch=1)

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

        # === ВКАДКА "ГЛАВНАЯ" (На 1 месте) ===
        self.magics_ribbon.addTab(self.create_main_ribbon_tab(), "ГЛАВНАЯ")

        # === ВОССТАНОВЛЕННЫЕ ОСТАЛЬНЫЕ ВКЛАДКИ ===
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Создать", "Дублировать", "Пакетное\nдублирование"], "Создание"), "ИНСТРУМЕНТЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Автоисправление", "Бормашина", "Отверстия", "Триксел"], "Лечение сетки"), "ИСПРАВЛЕНИЕ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Текстура 1", "Текстура 2"], "Текстурирование"), "ТЕКСТУРЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Перемещать", "Вращать", "Масштабировать", "Озеркалить"], "Позиционирование"), "РАСПОЛОЖЕНИЕ")
        # === вкладку менеджера платформ ===
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Управление\nплатформами"], "Оборудование"), "ПЛАТФОРМЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Колонны", "Решетка", "Контурные\nподдержки"], "Генерация"), "ПОДДЕРЖКИ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Heatmap", "Сравнение", "Мин/Макс\nтолщины"], "Контроль"), "АНАЛИЗ И ОТЧЕТЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Создание срезов\nConcept Laser"], "Concept Laser"), "СРЕЗЫ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Цвет деталей", "Прозрачность", "Отображение\nсетки"], "Визуализация"), "ОТОБРАЖЕНИЕ")
        self.magics_ribbon.addTab(self.create_ribbon_tab(["Параметры", "Язык", "Горячие\nклавиши"], "Система"), "НАСТРОЙКИ И ПОМОЩЬ")

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

        # --- ЛЕВАЯ ПАНЕЛЬ (В стиле Слайсера) ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)

        # Скролл-зона, как в Magics
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #2b2b2b; }")

        content = QWidget()
        content.setStyleSheet("""
            .QWidget { background-color: #2b2b2b; } 
            QTableWidget { background-color: #2b2b2b; color: #e0e0e0; gridline-color: #444444; border: 1px solid #444444; } 
            QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 2px; font-size: 11px; }
        """)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(5)

        # 1. Группа CAD
        self.grp_cad = CollapsibleBox("▼ Номинальная модель (CAD)")
        self.tbl_cad = self._build_standard_part_table()
        self.grp_cad.content_layout.addWidget(self.tbl_cad)
        content_layout.addWidget(self.grp_cad)

        # 2. Группа Скан
        self.grp_scan = CollapsibleBox("▼ Фактические модели (Скан)")
        self.tbl_scan = self._build_standard_part_table()
        self.grp_scan.content_layout.addWidget(self.tbl_scan)
        content_layout.addWidget(self.grp_scan)

        # 3. Группа Результаты
        self.grp_res = CollapsibleBox("▼ Результаты (Компенсация)")
        self.tbl_res = self._build_standard_part_table()
        self.grp_res.content_layout.addWidget(self.tbl_res)
        content_layout.addWidget(self.grp_res)

        content_layout.addStretch()
        scroll.setWidget(content)
        self.left_layout.addWidget(scroll)

        # --- Средняя панель (Сцена) ---
        self.plotter = None
        self._def_center_container = QWidget()
        self._def_center_layout = QVBoxLayout(self._def_center_container)
        self._def_center_layout.setContentsMargins(0, 0, 0, 0)

        # --- Правая панель (Настройки) ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)

        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self._def_center_container)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([450, 850, 300])  # Сделали левую панель пошире для таблиц

        # === ВКЛАДКИ НА ПРАВОЙ ПАНЕЛИ ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444444; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #222222; color: #aaaaaa; padding: 8px 10px; font-size: 11px; border: 1px solid #444444; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; font-weight: bold; border-top: 2px solid #b31b1b; }
        """)

        self.tab_align = QWidget()
        self.initAlignTab()
        self.tabs.addTab(self.tab_align, "1: Совмещение")

        self.tab_heatmap = QWidget()
        self.initHeatmapTab()
        self.tabs.addTab(self.tab_heatmap, "2: Карта отклонений")

        self.tab_params = QWidget()
        self.initParamsTab()
        self.tabs.addTab(self.tab_params, "3: Деформация")

        self.tab_comp = QWidget()
        self.initCompTab()
        self.tabs.addTab(self.tab_comp, "4: Компенсация")

        self.right_layout.addWidget(self.tabs, stretch=6)

        # ВНИМАНИЕ: Старая панель view_group (Слои) отсюда удалена!
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

        # === ПРИМЕНЯЕМ ТЕМНЫЙ СТИЛЬ ===
        self.tab_align.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #ffffff; border: 1px solid #555555; margin-top: 15px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #e0e0e0; font-weight: normal; border: none; }
            QPushButton { background-color: #444444; color: white; border: 1px solid #555555; padding: 6px; border-radius: 3px; font-weight: bold; }
            QPushButton:hover { background-color: #555555; border: 1px solid #777777; }
            QPushButton:pressed { background-color: #b31b1b; }
            QComboBox { background-color: #333333; color: #ffffff; border: 1px solid #666666; padding: 4px; border-radius: 3px; }
            QComboBox QAbstractItemView { background-color: #333333; color: #ffffff; selection-background-color: #b31b1b; }
            QCheckBox { color: #e0e0e0; }
        """)

        # 1. Элементы (САПР и Скан)
        group_files = QGroupBox("Элементы")
        fl = QGridLayout(group_files)
        fl.addWidget(QLabel("Номинальная модель (CAD):"), 0, 0)
        self.btn_load_cad = QPushButton("📁 Загрузить CAD (.stl)")
        fl.addWidget(self.btn_load_cad, 0, 1)

        fl.addWidget(QLabel("Фактическая сетка (Скан):"), 1, 0)
        self.btn_load_scan = QPushButton("📁 Загрузить Скан (.stl)")
        fl.addWidget(self.btn_load_scan, 1, 1)
        l.addWidget(group_files)

        # 2. Параметры поиска
        group_params = QGroupBox("Параметры предварительного выравнивания")
        pl = QGridLayout(group_params)

        pl.addWidget(QLabel("Время поиска:"), 0, 0)
        self.cb_search_time = QComboBox()
        self.cb_search_time.addItems(["Кратко", "Нормаль", "Длинно"])
        self.cb_search_time.setCurrentIndex(1)
        pl.addWidget(self.cb_search_time, 0, 1)

        # 3. Маркеры
        lbl_markers = QLabel("Вспомогательные маркеры (Override)")
        lbl_markers.setStyleSheet("color: #5dade2; font-weight: bold; margin-top: 5px;")
        pl.addWidget(lbl_markers, 1, 0, 1, 2)

        self.lbl_pts = QLabel("Точек на CAD: 0 | Точек на Скане: 0")
        pl.addWidget(self.lbl_pts, 2, 0, 1, 2)

        row_btns = QHBoxLayout()
        self.btn_pick_cad = QPushButton("📍 Маркер на CAD")
        self.btn_pick_scan = QPushButton("📍 Маркер на Скане")
        self.btn_clear_pts = QPushButton("Сбросить")
        row_btns.addWidget(self.btn_pick_cad)
        row_btns.addWidget(self.btn_pick_scan)
        row_btns.addWidget(self.btn_clear_pts)
        pl.addLayout(row_btns, 3, 0, 1, 2)

        # 4. Дополнительное наилучшее соответствие (ICP)
        self.chk_icp = QCheckBox("Вычислить дополнительное наилучшее соответствие (ICP)")
        self.chk_icp.setChecked(True)
        self.chk_icp.setStyleSheet("margin-top: 10px; font-weight: bold;")
        pl.addWidget(self.chk_icp, 4, 0, 1, 2)

        l.addWidget(group_params)

        # 5. Результат
        group_res = QGroupBox("Результат")
        rl = QHBoxLayout(group_res)
        rl.addWidget(QLabel("Отклонение (RMSE):"))
        rl.addStretch()
        self.lbl_rmse = QLabel("--- mm")
        self.lbl_rmse.setStyleSheet("font-weight: bold; color: #ffffff;")
        rl.addWidget(self.lbl_rmse)
        l.addWidget(group_res)

        # Кнопка запуска
        self.btn_run_icp = QPushButton("▶ ВЫПОЛНИТЬ ВЫРАВНИВАНИЕ")
        self.btn_run_icp.setCursor(Qt.PointingHandCursor)
        self.btn_run_icp.setStyleSheet(
            "height: 50px; background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; margin-top: 10px;")
        l.addWidget(self.btn_run_icp)
        l.addStretch()

    def initHeatmapTab(self):
        """Шаг 2: Карта отклонений"""
        layout = QVBoxLayout(self.tab_heatmap)
        self.heat_group = QGroupBox("Анализ (Цветовая карта отклонений)")
        self.heat_group.setStyleSheet("color: white; font-weight: bold; border: 1px solid #555; margin-top: 10px;")
        heat_layout = QVBoxLayout(self.heat_group)

        row_heat = QHBoxLayout()
        self.btn_heatmap = QPushButton("🔥 Построить Heatmap")
        self.btn_heatmap.setStyleSheet("background-color: #e67e22; color: white; padding: 10px;")
        self.btn_clear_heat = QPushButton("Сбросить")

        row_heat.addWidget(self.btn_heatmap)
        row_heat.addWidget(self.btn_clear_heat)
        heat_layout.addLayout(row_heat)

        self.add_slider(heat_layout, "Предел градиента (± мм)", 1, 50, 10, 1, "heat_limit", divider=10.0)

        # --- БЛОК ТОЧЕЧНОГО КОНТРОЛЯ (GOM CALLOUTS) ---
        row_callouts = QHBoxLayout()
        self.chk_callouts = QCheckBox("📍 Флажки отклонений (Клик)")
        self.chk_callouts.setStyleSheet("color: #5dade2; font-weight: bold;")
        self.chk_callouts.setCursor(Qt.PointingHandCursor)

        self.btn_clear_callouts = QPushButton("Очистить флажки")
        self.btn_clear_callouts.setCursor(Qt.PointingHandCursor)
        self.btn_clear_callouts.setStyleSheet(
            "background-color: #444; color: white; padding: 4px 8px; border-radius: 3px;")

        row_callouts.addWidget(self.chk_callouts)
        row_callouts.addWidget(self.btn_clear_callouts)
        heat_layout.addLayout(row_callouts)
        # ----------------------------------------------

        layout.addWidget(self.heat_group)
        layout.addStretch()

    def create_spinbox_wrapper(self, spinbox):
        """Оборачивает QSpinBox в кастомный виджет с кнопками [-] и [+] по бокам"""
        container = QWidget()
        container.setObjectName("SpinWrapper")
        container.setStyleSheet("""
            QWidget#SpinWrapper {
                background-color: #333333;
                border: 1px solid #666666;
                border-radius: 3px;
            }
            QWidget#SpinWrapper:disabled {
                background-color: #222222;
                border: 1px solid #444444;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: transparent;
                border: none;
                color: #ffffff;
                font-weight: bold;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                color: #777777;
            }
        """)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(0)

        btn_minus = QPushButton("—")
        btn_plus = QPushButton("+")

        btn_style = """
            QPushButton { 
                background: transparent; 
                color: #aaaaaa; 
                font-weight: bold; 
                border: none; 
                font-size: 14px;
            }
            QPushButton:hover { color: #ffffff; }
            QPushButton:pressed { color: #b31b1b; }
            QPushButton:disabled { color: #555555; }
        """
        btn_minus.setStyleSheet(btn_style)
        btn_plus.setStyleSheet(btn_style)
        btn_minus.setFixedSize(24, 24)
        btn_plus.setFixedSize(24, 24)
        btn_minus.setCursor(Qt.PointingHandCursor)
        btn_plus.setCursor(Qt.PointingHandCursor)

        # Отключаем системные стрелки и центрируем текст
        spinbox.setButtonSymbols(QSpinBox.NoButtons)
        spinbox.setAlignment(Qt.AlignCenter)

        # Привязываем клики к системному шагу ползунка
        btn_minus.clicked.connect(spinbox.stepDown)
        btn_plus.clicked.connect(spinbox.stepUp)

        layout.addWidget(btn_minus)
        layout.addWidget(spinbox)
        layout.addWidget(btn_plus)

        return container

    def initParamsTab(self):
        """Шаг 3: Деформация (Параметры нейросети)"""
        layout = QVBoxLayout(self.tab_params)

        grp = QGroupBox("Настройки алгоритма")
        grp.setStyleSheet("""
            QGroupBox { 
                color: #ffffff; 
                border: 1px solid #555; 
                margin-top: 15px; 
                padding-top: 15px; 
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #e0e0e0; font-weight: normal; border: none; }
            QComboBox { 
                background-color: #333333; 
                color: #ffffff; 
                border: 1px solid #666666; 
                padding: 4px;
                border-radius: 3px;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #ffffff;
                selection-background-color: #b31b1b;
                selection-color: #ffffff;
                border: 1px solid #555;
            }
            QComboBox::drop-down { border: none; }
            QComboBox:disabled { background-color: #222222; color: #777777; }
        """)

        glayout = QGridLayout(grp)
        glayout.setSpacing(10)

        # 1. Деформация
        glayout.addWidget(QLabel("Деформация:"), 0, 0)
        self.cb_def_type = QComboBox()
        self.cb_def_type.addItems(["Мягко", "Нормально", "Жестко"])
        self.cb_def_type.setCurrentIndex(1)
        glayout.addWidget(self.cb_def_type, 0, 1)

        # 2. Дискретизация
        glayout.addWidget(QLabel("Дискретизация:"), 1, 0)
        self.cb_samples = QComboBox()
        self.cb_samples.addItems(["Свое значение", "Нет", "Грубо", "Средне", "Подробно"])
        self.cb_samples.setCurrentIndex(3)
        glayout.addWidget(self.cb_samples, 1, 1)

        # 3. Облако точек (ИСПОЛЬЗУЕМ ОБЕРТКУ)
        glayout.addWidget(QLabel("Облако точек:"), 2, 0)
        self.sb_points = QSpinBox()
        self.sb_points.setRange(1000, 500000)
        self.sb_points.setSingleStep(1000)
        self.sb_points.setValue(20000)

        self.sb_points_wrapper = self.create_spinbox_wrapper(self.sb_points)
        self.sb_points_wrapper.setEnabled(False)  # Изначально заблокировано пресетом
        glayout.addWidget(self.sb_points_wrapper, 2, 1)

        self.cb_samples.currentIndexChanged.connect(self._on_sample_type_changed)

        layout.addWidget(grp)

        # 4. Переключатели визуализации
        self.chk_preview_pts = QCheckBox("Предпросмотр облака точек")
        self.chk_preview_pts.setStyleSheet("margin-top: 5px; margin-bottom: 5px; color: #ffffff; font-weight: bold;")

        self.chk_show_vectors = QCheckBox("Векторы смещения (стрелки)")
        self.chk_show_vectors.setStyleSheet("margin-top: 5px; margin-bottom: 5px; color: #f39c12; font-weight: bold;")

        h_toggle = QHBoxLayout()
        h_toggle.addWidget(self.chk_show_vectors)
        h_toggle.addStretch()
        h_toggle.addWidget(self.chk_preview_pts)
        layout.addLayout(h_toggle)

        # --- НОВЫЙ БЛОК: Кнопка запуска Деформации ---
        self.def_stack = QStackedWidget()

        page_start = QWidget()
        p0_layout = QVBoxLayout(page_start)
        self.btn_run_def = QPushButton("⚡ РАССЧИТАТЬ ДЕФОРМАЦИЮ")
        self.btn_run_def.setCursor(Qt.PointingHandCursor)
        self.btn_run_def.setStyleSheet(
            "height: 60px; background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        p0_layout.addWidget(self.btn_run_def)
        p0_layout.addStretch()
        self.def_stack.addWidget(page_start)

        page_prog = QWidget()
        p1_layout = QVBoxLayout(page_prog)
        self.lbl_def_status = QLabel("Обучение сети и симуляция усадки...")
        self.lbl_def_status.setAlignment(Qt.AlignCenter)
        self.lbl_def_status.setStyleSheet("color: white; font-weight: bold;")

        self.def_progress_bar = QProgressBar()
        self.def_progress_bar.setRange(0, 100)
        self.def_progress_bar.setFixedHeight(30)
        self.def_progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; color: white; } QProgressBar::chunk { background-color: #c0392b; }")

        self.btn_cancel_def = QPushButton("❌ Отменить")
        self.btn_cancel_def.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_def.setStyleSheet(
            "background-color: #444; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")

        p1_layout.addWidget(self.lbl_def_status)
        p1_layout.addWidget(self.def_progress_bar)
        p1_layout.addWidget(self.btn_cancel_def, 0, Qt.AlignCenter)
        p1_layout.addStretch()
        self.def_stack.addWidget(page_prog)

        layout.addWidget(self.def_stack)
        # ---------------------------------------------

        layout.addStretch()

    def _on_sample_type_changed(self, idx):
        """Блокирует/разблокирует КОНТЕЙНЕР с точками в зависимости от пресета"""
        presets = {1: 500000, 2: 5000, 3: 20000, 4: 200000}

        if idx == 0:
            self.sb_points_wrapper.setEnabled(True)  # Включаем и поле, и кнопки +/-
        else:
            self.sb_points_wrapper.setEnabled(False)  # Выключаем всё вместе
            self.sb_points.setValue(presets.get(idx, 20000))

    def initCompTab(self):
        """Шаг 4: Компенсация и Расчет"""
        layout = QVBoxLayout(self.tab_comp)

        # --- БЛОК: Масштабирование вектора (Анизотропное XY / Z) ---
        grp_factor = QGroupBox("Масштабирование результата")
        grp_factor.setStyleSheet("""
                    QGroupBox { 
                        color: #ffffff; 
                        border: 1px solid #555; 
                        margin-top: 15px; 
                        padding-top: 15px; 
                        font-weight: bold; 
                    }
                    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                    QLabel { color: #e0e0e0; font-weight: normal; border: none; }
                """)
        flayout = QVBoxLayout(grp_factor)

        self.chk_link_factor = QCheckBox("Связать XY и Z (Изотропная усадка)")
        self.chk_link_factor.setChecked(True)
        self.chk_link_factor.setCursor(Qt.PointingHandCursor)
        self.chk_link_factor.setStyleSheet("color: #5dade2; margin-bottom: 5px;")
        flayout.addWidget(self.chk_link_factor)

        grid_factors = QGridLayout()

        # Коэффициент в плоскости слоев (XY)
        grid_factors.addWidget(QLabel("Коэфф. XY (слои):"), 0, 0)
        self.sb_factor = QDoubleSpinBox()
        self.sb_factor.setRange(0.1, 5.0)
        self.sb_factor.setSingleStep(0.1)
        self.sb_factor.setValue(1.0)
        self.sb_factor_wrapper = self.create_spinbox_wrapper(self.sb_factor)
        grid_factors.addWidget(self.sb_factor_wrapper, 0, 1)

        # Коэффициент по высоте построения (Z)
        grid_factors.addWidget(QLabel("Коэфф. Z (рост):"), 1, 0)
        self.sb_factor_z = QDoubleSpinBox()
        self.sb_factor_z.setRange(0.1, 5.0)
        self.sb_factor_z.setSingleStep(0.1)
        self.sb_factor_z.setValue(1.0)
        self.sb_factor_z_wrapper = self.create_spinbox_wrapper(self.sb_factor_z)
        self.sb_factor_z_wrapper.setEnabled(False)  # Заблокировано при связанном режиме
        grid_factors.addWidget(self.sb_factor_z_wrapper, 1, 1)

        flayout.addLayout(grid_factors)
        layout.addWidget(grp_factor)
        # -------------------------------------------

        self.comp_stack = QStackedWidget()

        page_start = QWidget()
        p0_layout = QVBoxLayout(page_start)
        self.btn_run_comp = QPushButton("⚡ ЗАПУСТИТЬ КОМПЕНСАЦИЮ")
        self.btn_run_comp.setCursor(Qt.PointingHandCursor)
        self.btn_run_comp.setStyleSheet(
            "height: 60px; background-color: #c0392b; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        p0_layout.addWidget(self.btn_run_comp)
        p0_layout.addStretch()
        self.comp_stack.addWidget(page_start)

        page_prog = QWidget()
        p1_layout = QVBoxLayout(page_prog)
        self.lbl_progress_status = QLabel("Обучение сети и деформация...")
        self.lbl_progress_status.setAlignment(Qt.AlignCenter)
        self.lbl_progress_status.setStyleSheet("color: white; font-weight: bold;")
        self.comp_progress_bar = QProgressBar()
        self.comp_progress_bar.setRange(0, 100)
        self.comp_progress_bar.setFixedHeight(30)
        self.comp_progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; color: white; } QProgressBar::chunk { background-color: #c0392b; }")

        self.btn_cancel_comp = QPushButton("❌ Отменить")
        self.btn_cancel_comp.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_comp.setStyleSheet(
            "background-color: #444; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")

        p1_layout.addWidget(self.lbl_progress_status)
        p1_layout.addWidget(self.comp_progress_bar)
        p1_layout.addWidget(self.btn_cancel_comp, 0, Qt.AlignCenter)
        p1_layout.addStretch()
        self.comp_stack.addWidget(page_prog)

        layout.addWidget(self.comp_stack)

        self.btn_save = QPushButton("💾 Сохранить Результат")
        self.btn_save.setEnabled(False)
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet(
            "height: 40px; background-color: #2c3e50; color: white; font-weight: bold; border-radius: 3px;")
        layout.addWidget(self.btn_save)

    def init_recent_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: #2b2b2b;") # Темный фон
        layout = QVBoxLayout(page)

        header_layout = QHBoxLayout()
        self.btn_back_to_start = QPushButton("⬅ Назад")
        self.btn_back_to_start.setFixedSize(120, 40)
        self.btn_back_to_start.setCursor(Qt.PointingHandCursor)
        self.btn_back_to_start.setStyleSheet(
            "background-color: #444; color: white; font-weight: bold; border: 1px solid #555; border-radius: 3px;")

        title = QLabel(" Недавно использованные проекты")
        title.setStyleSheet("font-size: 24px; color: #e0e0e0; font-weight: bold;") # Светлый текст

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

        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


class DialogPlatformManager(QDialog):
    """Менеджер платформ (список машин)"""

    def __init__(self, parent=None, platforms_data=None):
        super().__init__(parent)
        self.setWindowTitle("Менеджер платформ")
        self.setFixedSize(650, 400)

        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #e0e0e0; }
            QLabel { color: #e0e0e0; font-weight: bold; }
            QTableWidget { background-color: #333333; color: white; border: 1px solid #555; gridline-color: #444; selection-background-color: #555555; outline: none; }
            QTableWidget::item { border: none; }
            QTableWidget::item:focus { border: none; outline: none; }
            QTableWidget::item:selected { border: none; color: white; background-color: #555555; }
            QHeaderView::section { background-color: #444; color: white; border: 1px solid #555; padding: 4px; font-weight: bold; }
            QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 6px 15px; border-radius: 3px; font-weight: bold; }
            QPushButton:hover { background-color: #555; border: 1px solid #777; }
            QPushButton:pressed { background-color: #b31b1b; }
            QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid #555; border-radius: 4px; background-color: #333; }
            QCheckBox::indicator:hover { border: 2px solid #c0392b; }
            QCheckBox::indicator:checked { background-color: #c0392b; border: 2px solid #c0392b; image: url(":/qt-project.org/styles/commonstyle/images/check-16.png"); }
        """)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Название", "Габариты (X×Y×Z)", "На панели"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        self.table.horizontalHeader().resizeSection(0, 40)
        self.table.horizontalHeader().resizeSection(2, 150)
        self.table.horizontalHeader().resizeSection(3, 100)

        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout.addWidget(QLabel("Доступные рабочие платформы:"))
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Создать")
        self.btn_edit = QPushButton("Редактировать")
        self.btn_delete = QPushButton("Удалить")

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_edit)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addStretch()

        self.btn_apply = QPushButton("Применить")
        self.btn_apply.setStyleSheet("background-color: #b31b1b;")
        btn_layout.addWidget(self.btn_apply)

        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.btn_add.clicked.connect(self.action_add)
        self.btn_edit.clicked.connect(self.action_edit)
        self.btn_delete.clicked.connect(self.action_delete)

        if platforms_data:
            for p in platforms_data:
                self.add_platform_row(p, p.get("is_default", False))

    def add_platform_row(self, plat_dict, is_default):
        row = self.table.rowCount()
        self.table.insertRow(row)

        item_id = QTableWidgetItem(str(row + 1))
        item_id.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, item_id)

        item_name = QTableWidgetItem(plat_dict["name"])
        # ХРАНИМ ВЕСЬ СЛОВАРЬ (С ЗОНАМИ), А НЕ ТОЛЬКО DIM
        item_name.setData(Qt.UserRole, plat_dict)
        self.table.setItem(row, 1, item_name)

        dim = plat_dict.get("dim", [220, 220, 280])
        item_dim = QTableWidgetItem(f"{dim[0]} × {dim[1]} × {dim[2]} мм")
        item_dim.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 2, item_dim)

        cb_container = QWidget()
        cb_layout = QHBoxLayout(cb_container)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setAlignment(Qt.AlignCenter)
        cb = QCheckBox()
        cb.setChecked(is_default)
        cb.setCursor(Qt.PointingHandCursor)
        cb_layout.addWidget(cb)
        self.table.setCellWidget(row, 3, cb_container)

    def action_add(self):
        dialog = DialogEditPlatform(self)
        if dialog.exec():
            new_data = dialog.get_data()
            self.add_platform_row(new_data, False)

    def action_edit(self):
        current_row = self.table.currentRow()
        if current_row < 0: return

        # Считываем полный словарь платформы из ячейки
        plat_dict = self.table.item(current_row, 1).data(Qt.UserRole)

        dialog = DialogEditPlatform(self, platform_data=plat_dict)
        if dialog.exec():
            new_data = dialog.get_data()
            self.table.item(current_row, 1).setText(new_data["name"])
            self.table.item(current_row, 1).setData(Qt.UserRole, new_data)
            self.table.item(current_row, 2).setText(
                f"{new_data['dim'][0]} × {new_data['dim'][1]} × {new_data['dim'][2]} мм")

    def action_delete(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def get_data(self):
        platforms = []
        for row in range(self.table.rowCount()):
            plat_dict = self.table.item(row, 1).data(Qt.UserRole)
            plat_dict["name"] = self.table.item(row, 1).text()

            is_default = False
            cb_container = self.table.cellWidget(row, 3)
            if cb_container:
                cb = cb_container.findChild(QCheckBox)
                if cb and cb.isChecked():
                    is_default = True

            plat_dict["is_default"] = is_default
            platforms.append(plat_dict)
        return platforms


class DialogEditPlatform(QDialog):
    """Окно создания/редактирования конкретной платформы с настройкой мертвых зон"""

    def __init__(self, parent=None, platform_data=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка платформы")
        self.resize(650, 420)

        self.zones_data = []  # Внутреннее хранилище зон
        self.current_zone_idx = -1

        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: white; }
            QLabel { color: #e0e0e0; }
            QLineEdit { background-color: #333; color: white; border: 1px solid #555; padding: 4px; }
            QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 6px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; border: 1px solid #777; }
            QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; top: -1px; }
            QTabBar::tab { background-color: #222; color: #aaa; padding: 8px 15px; border: 1px solid #444; border-top-left-radius: 3px; border-top-right-radius: 3px; }
            QTabBar::tab:selected { background-color: #2b2b2b; color: white; font-weight: bold; border-bottom: none; border-top: 2px solid #b31b1b; }
            QTableWidget { background-color: #333; color: white; border: 1px solid #555; gridline-color: #444; outline: none; }
            QTableWidget::item:focus { outline: none; }
            QTableWidget::item:selected { background-color: #b31b1b; color: white; }
            QComboBox { background-color: #333; color: white; border: 1px solid #555; padding: 3px; }
            QCheckBox { color: #e0e0e0; font-weight: bold; }
        """)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.tab_general = QWidget()
        self.init_general_tab()
        self.tabs.addTab(self.tab_general, "Общая информация")

        self.tab_zones = QWidget()
        self.init_zones_tab()
        self.tabs.addTab(self.tab_zones, "Запретные зоны")

        layout.addWidget(self.tabs)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Сохранить")
        self.btn_save.setStyleSheet("background-color: #b31b1b; font-weight: bold; padding: 6px 20px;")
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setStyleSheet("padding: 6px 20px;")

        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # === ЗАГРУЗКА ДАННЫХ ===
        if platform_data:
            self.le_name.setText(platform_data.get("name", ""))
            dim = platform_data.get("dim", [220.0, 220.0, 280.0])
            self.le_x.setText(str(dim[0]))
            self.le_y.setText(str(dim[1]))
            self.le_z.setText(str(dim[2]))

            self.chk_use_zones.setChecked(platform_data.get("use_zones", False))
            self.zones_data = platform_data.get("zones", [])
        else:
            # Стартовые зоны для новой платформы
            self.zones_data = [
                {"name": "OL", "shape": 0, "x": -90.0, "y": 90.0, "r": 5.0, "zmin": 0.100, "zmax": 0.110,
                 "full_h": False},
                {"name": "OR", "shape": 0, "x": 90.0, "y": 90.0, "r": 5.0, "zmin": 0.100, "zmax": 0.110,
                 "full_h": False},
                {"name": "UL", "shape": 0, "x": -90.0, "y": -90.0, "r": 5.0, "zmin": 0.100, "zmax": 0.110,
                 "full_h": False},
                {"name": "UR", "shape": 0, "x": 90.0, "y": -90.0, "r": 5.0, "zmin": 0.100, "zmax": 0.110,
                 "full_h": False}
            ]

        self.refresh_zones_list()

    def init_general_tab(self):
        layout = QVBoxLayout(self.tab_general)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        layout.addWidget(QLabel("Название машины/платформы:"))
        self.le_name = QLineEdit("Новая машина")
        layout.addWidget(self.le_name)

        grp_dim = QGroupBox("Габариты рабочей камеры (мм)")
        grp_dim.setStyleSheet(
            "QGroupBox { border: 1px solid #555; margin-top: 10px; color: #ccc; font-weight: bold; } QGroupBox::title { top: -8px; left: 10px; }")
        grid_dim = QGridLayout(grp_dim)

        grid_dim.addWidget(QLabel("Ось X:"), 0, 0)
        self.le_x = QLineEdit("220")
        grid_dim.addWidget(self.le_x, 0, 1)

        grid_dim.addWidget(QLabel("Ось Y:"), 1, 0)
        self.le_y = QLineEdit("220")
        grid_dim.addWidget(self.le_y, 1, 1)

        grid_dim.addWidget(QLabel("Ось Z (Высота):"), 2, 0)
        self.le_z = QLineEdit("280")
        grid_dim.addWidget(self.le_z, 2, 1)

        layout.addWidget(grp_dim)

        layout.addWidget(QLabel("Пользовательская плита построения (Опционально):"))
        h_stl = QHBoxLayout()
        self.le_stl_path = QLineEdit()
        self.le_stl_path.setPlaceholderText("Файл не выбран...")
        self.le_stl_path.setReadOnly(True)
        self.btn_browse_stl = QPushButton("Обзор...")
        h_stl.addWidget(self.le_stl_path)
        h_stl.addWidget(self.btn_browse_stl)
        layout.addLayout(h_stl)
        layout.addStretch()

    def init_zones_tab(self):
        layout = QVBoxLayout(self.tab_zones)
        layout.setContentsMargins(15, 15, 15, 15)

        self.chk_use_zones = QCheckBox("Активировать Запретные зоны")
        layout.addWidget(self.chk_use_zones)
        layout.addSpacing(10)

        h_main = QHBoxLayout()

        # === ЛЕВАЯ ПАНЕЛЬ: СПИСОК ЗОН ===
        left_layout = QVBoxLayout()
        self.table_zones = QTableWidget(0, 1)
        self.table_zones.horizontalHeader().setVisible(False)
        self.table_zones.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_zones.verticalHeader().setVisible(False)
        self.table_zones.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_zones.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_zones.itemSelectionChanged.connect(self.on_zone_select)
        left_layout.addWidget(self.table_zones)

        btn_z_layout = QHBoxLayout()
        self.btn_z_add = QPushButton("➕ Добавить")
        self.btn_z_del = QPushButton("❌ Удалить")
        self.btn_z_add.clicked.connect(self.add_zone)
        self.btn_z_del.clicked.connect(self.del_zone)
        btn_z_layout.addWidget(self.btn_z_add)
        btn_z_layout.addWidget(self.btn_z_del)
        left_layout.addLayout(btn_z_layout)
        h_main.addLayout(left_layout, stretch=1)

        # === ПРАВАЯ ПАНЕЛЬ: ПАРАМЕТРЫ ===
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 0, 0, 0)

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(QLabel("Форма"), 0, 0)
        self.cb_shape = QComboBox()
        self.cb_shape.addItems(["Цилиндрические", "Прямоугольные"])
        grid.addWidget(self.cb_shape, 0, 1, 1, 3)

        grid.addWidget(QLabel("Центр"), 1, 0)
        grid.addWidget(QLabel("X"), 1, 1, Qt.AlignRight)
        self.le_zx = QLineEdit("0.000")
        grid.addWidget(self.le_zx, 1, 2)
        grid.addWidget(QLabel("мм"), 1, 3)

        grid.addWidget(QLabel("Y"), 2, 1, Qt.AlignRight)
        self.le_zy = QLineEdit("0.000")
        grid.addWidget(self.le_zy, 2, 2)
        grid.addWidget(QLabel("мм"), 2, 3)

        grid.addWidget(QLabel("Радиус"), 3, 0)
        grid.addWidget(QLabel("R"), 3, 1, Qt.AlignRight)
        self.le_zr = QLineEdit("5.000")
        grid.addWidget(self.le_zr, 3, 2)
        grid.addWidget(QLabel("мм"), 3, 3)

        grid.addWidget(QLabel("Высота"), 4, 0)
        grid.addWidget(QLabel("Z Мин."), 4, 1, Qt.AlignRight)
        self.le_zmin = QLineEdit("0.000")
        grid.addWidget(self.le_zmin, 4, 2)
        grid.addWidget(QLabel("мм"), 4, 3)

        grid.addWidget(QLabel("Z Макс."), 5, 1, Qt.AlignRight)
        self.le_zmax = QLineEdit("0.000")
        grid.addWidget(self.le_zmax, 5, 2)
        grid.addWidget(QLabel("мм"), 5, 3)

        right_layout.addLayout(grid)

        h_chk = QHBoxLayout()
        h_chk.addSpacing(60)
        self.chk_full_h = QCheckBox("Высота всей платформы")
        h_chk.addWidget(self.chk_full_h)
        right_layout.addLayout(h_chk)
        right_layout.addStretch()
        h_main.addLayout(right_layout, stretch=2)
        layout.addLayout(h_main)

        # ПРИВЯЗЫВАЕМ СОХРАНЕНИЕ НА ЛЕТУ
        self.cb_shape.currentIndexChanged.connect(self.save_current_zone)
        self.chk_full_h.toggled.connect(self.save_current_zone)
        for le in (self.le_zx, self.le_zy, self.le_zr, self.le_zmin, self.le_zmax):
            le.textEdited.connect(self.save_current_zone)

    def refresh_zones_list(self):
        """Перерисовывает таблицу зон слева"""
        self.table_zones.blockSignals(True)
        self.table_zones.setRowCount(0)
        for zone in self.zones_data:
            row = self.table_zones.rowCount()
            self.table_zones.insertRow(row)
            self.table_zones.setItem(row, 0, QTableWidgetItem(zone["name"]))
        self.table_zones.blockSignals(False)

        if self.zones_data:
            self.table_zones.selectRow(0)

    def on_zone_select(self):
        """Заполняет поля справа при клике на зону в списке"""
        selected = self.table_zones.selectedItems()
        if not selected: return

        self.current_zone_idx = selected[0].row()
        zone = self.zones_data[self.current_zone_idx]

        # Блокируем, чтобы не сработало случайное сохранение при автозаполнении
        self._block_zone_signals(True)
        self.cb_shape.setCurrentIndex(zone.get("shape", 0))
        self.le_zx.setText(str(zone.get("x", 0.0)))
        self.le_zy.setText(str(zone.get("y", 0.0)))
        self.le_zr.setText(str(zone.get("r", 5.0)))
        self.le_zmin.setText(str(zone.get("zmin", 0.0)))
        self.le_zmax.setText(str(zone.get("zmax", 0.0)))
        self.chk_full_h.setChecked(zone.get("full_h", False))
        self._block_zone_signals(False)

    def save_current_zone(self, *args):
        """Считывает поля справа и сохраняет в память при любом изменении"""
        if self.current_zone_idx < 0 or self.current_zone_idx >= len(self.zones_data):
            return

        zone = self.zones_data[self.current_zone_idx]
        zone["shape"] = self.cb_shape.currentIndex()
        zone["full_h"] = self.chk_full_h.isChecked()

        try:
            zone["x"] = float(self.le_zx.text().replace(',', '.'))
            zone["y"] = float(self.le_zy.text().replace(',', '.'))
            zone["r"] = float(self.le_zr.text().replace(',', '.'))
            zone["zmin"] = float(self.le_zmin.text().replace(',', '.'))
            zone["zmax"] = float(self.le_zmax.text().replace(',', '.'))
        except ValueError:
            pass  # Игнорируем, если в поле вбили минус или оставили пустым при вводе

    def add_zone(self):
        """Создает новую зону по умолчанию"""
        new_zone = {
            "name": f"Зона {len(self.zones_data) + 1}",
            "shape": 0, "x": 0.0, "y": 0.0, "r": 5.0, "zmin": 0.0, "zmax": 0.0, "full_h": False
        }
        self.zones_data.append(new_zone)
        self.refresh_zones_list()
        self.table_zones.selectRow(len(self.zones_data) - 1)

    def del_zone(self):
        """Удаляет выбранную зону"""
        selected = self.table_zones.selectedItems()
        if not selected: return
        idx = selected[0].row()
        self.zones_data.pop(idx)
        self.current_zone_idx = -1
        self.refresh_zones_list()

    def _block_zone_signals(self, block):
        self.cb_shape.blockSignals(block)
        self.le_zx.blockSignals(block)
        self.le_zy.blockSignals(block)
        self.le_zr.blockSignals(block)
        self.le_zmin.blockSignals(block)
        self.le_zmax.blockSignals(block)
        self.chk_full_h.blockSignals(block)

    def get_data(self):
        """Собирает введенные данные в словарь для передачи в Менеджер"""
        name = self.le_name.text().strip() or "Без названия"
        try:
            x = float(self.le_x.text().replace(',', '.'))
            y = float(self.le_y.text().replace(',', '.'))
            z = float(self.le_z.text().replace(',', '.'))
        except ValueError:
            x, y, z = 220.0, 220.0, 280.0

        return {
            "name": name,
            "dim": [x, y, z],
            "use_zones": self.chk_use_zones.isChecked(),
            "zones": self.zones_data
        }
class DialogDonate(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе / Поддержка")
        # Делаем окно более компактным и вертикальным
        self.setFixedSize(350, 480)

        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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