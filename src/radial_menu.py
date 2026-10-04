"""Click-only radial action menu, constrained to the monitor's work area."""
from PySide6.QtCore import Qt, QPoint, QSize, QTimer
from PySide6.QtGui import QGuiApplication, QPainter, QColor
from PySide6.QtWidgets import QWidget, QToolButton, QLabel
from tool_ribbon import main_icon, tool_icon


class RadialMenu(QWidget):
    def __init__(self, window):
        super().__init__(window, Qt.Popup | Qt.FramelessWindowHint)
        self.owner = window
        self.circle_color = QColor(37, 41, 47, 235)
        self.setProperty('engineeringChrome', True)
        self.setWindowFlag(Qt.NoDropShadowWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(390, 300)
        entries = [
            ('Перемещение\nдеталей', tool_icon(3), lambda: window.run_slicer_tool('Перемещать'), (131, 3)),
            ('Вращение\nдеталей', tool_icon(4), lambda: window.run_slicer_tool('Вращать'), (257, 111)),
            ('Сохранить\nвыбранные как…', main_icon('Сохранить выбранные детали как'), window.save_selected_slicer_parts, (131, 223)),
            ('Выгрузить\nвыбранные', main_icon('Выгрузить деталь'), window.unload_slicer_part, (3, 111)),
        ]
        self.buttons = []
        self.original_icons = []
        for title, icon, callback, (x, y) in entries:
            button = QToolButton(self)
            button.setText(title)
            button.setIcon(icon)
            button.setIconSize(QSize(30, 30))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setGeometry(x, y, 130, 74)
            button.setStyleSheet('QToolButton {background: #30343a; color: white; border: 1px solid #737b85; border-radius: 18px;} QToolButton:hover, QToolButton:focus {background: #485e71; border-color: #79c6ff;}')
            button.clicked.connect(lambda checked=False, action=callback: self.activate(action))
            self.buttons.append(button)
            self.original_icons.append(icon)
        label = QLabel(f'Выбрано: {len(window.selected_slicer_rows())}\nEsc — закрыть', self)
        label.setGeometry(135, 119, 120, 62)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color: white; background: transparent;')
        self.label = label
        theme = getattr(window, 'engineering_theme', None)
        if theme is not None:
            theme.changed.connect(self.apply_theme)
            self.apply_theme(theme.mode)

    def apply_theme(self, mode):
        light = mode == 'light'
        ink = '#18241d' if light else '#e6ede7'
        surface = 'rgba(250, 251, 248, 226)' if light else 'rgba(40, 51, 44, 230)'
        hover = 'rgba(233, 230, 191, 244)' if light else 'rgba(69, 81, 54, 244)'
        border = '#9aa89c' if light else '#839680'
        self.circle_color = QColor(250, 251, 248, 155) if light else QColor(27, 34, 31, 172)
        for button, icon in zip(self.buttons, self.original_icons):
            button.setStyleSheet(f'QToolButton {{background: {surface}; color: {ink}; border: 1px solid {border}; border-radius: 18px;}} '
                f'QToolButton:hover, QToolButton:focus {{background: {hover}; border-color: {border};}}')
            button.setIcon(self.owner.engineering_theme.recolor_icon(icon, ink))
        self.label.setStyleSheet(f'color: {ink}; background: transparent;')
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.circle_color)
        painter.drawEllipse(65, 20, 260, 260)

    def popup(self, position):
        screen = QGuiApplication.screenAt(position) or QGuiApplication.primaryScreen()
        rect = screen.availableGeometry()
        origin = position - QPoint(self.width() // 2, self.height() // 2)
        origin.setX(max(rect.left(), min(origin.x(), rect.right() - self.width() + 1)))
        origin.setY(max(rect.top(), min(origin.y(), rect.bottom() - self.height() + 1)))
        self.move(origin)
        self.show()
        self.buttons[0].setFocus()

    def activate(self, callback):
        self.close()
        QTimer.singleShot(0, callback)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape: self.close()
        else: super().keyPressEvent(event)
