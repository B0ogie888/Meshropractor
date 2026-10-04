"""One height and icon size for every ribbon; editable vector assets."""
from pathlib import Path
import sys
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QFont, QFontMetrics
from PySide6.QtWidgets import QFrame, QLayout, QScrollArea, QToolButton, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QLabel

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
ICON_SIZE = 28
COMMAND_FONT_SIZE = 12


def style_engineering_commands(tabs):
    """Apply common states before the engineering theme captures canonical styles."""
    for index in range(tabs.count()):
        for button in tabs.widget(index).findChildren(QToolButton):
            if not button.text(): continue
            button.setStyleSheet(button.styleSheet() + '''
                QToolButton {background: transparent; color: #252b2b;
                    border: 1px solid transparent; border-radius: 3px;}
                QToolButton:enabled:focus {border-color: #e5d943;}
                QToolButton:enabled:hover {background: #f2efc6; color: #252b2b;
                    border: 1px solid #e5d943;}
                QToolButton:enabled:pressed, QToolButton:enabled:checked {
                    background: #e5d943; color: #252b2b; border: 1px solid #e5d943;}
                QToolButton:disabled, QToolButton:disabled:hover {
                    background: transparent; color: #89918a; border: 1px solid transparent;}
            ''')


def compact_layout(layout, *, root=False):
    """Anchor every command row to the same top edge, with compact gaps."""
    if layout is None: return
    layout.setAlignment((layout.alignment() & Qt.AlignHorizontal_Mask) | Qt.AlignTop)
    margins = layout.contentsMargins()
    layout.setContentsMargins(min(4, margins.left()), 4 if root else 0,
                              min(4, margins.right()), 0)
    if isinstance(layout, QHBoxLayout):
        layout.setSpacing(2)
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is not None and not isinstance(item.widget(), QFrame):
            item.setAlignment((item.alignment() & Qt.AlignHorizontal_Mask) | Qt.AlignTop)
        child = item.layout() or (item.widget().layout() if item.widget() else None)
        if child is not None: compact_layout(child)


def normalize_font(widget, selector, pixels):
    font = QFont(); font.setFamilies(['Segoe UI', 'DejaVu Sans'])
    font.setPixelSize(pixels); font.setWeight(QFont.Normal); font.setItalic(False)
    widget.setFont(font)
    if not widget.property('ribbonTypography'):
        widget.setStyleSheet(widget.styleSheet() + f'\n{selector} {{font-family: "Segoe UI", "DejaVu Sans"; '
            f'font-size: {pixels}px; font-weight: 400; font-style: normal;}}')
        widget.setProperty('ribbonTypography', True)
    widget.ensurePolished()


def asset_icon(group, name):
    path = ROOT / 'assets' / 'ribbon' / group / (str(name) + '.svg')
    return QIcon(str(path)) if path.is_file() else None


def size_compact_column(column):
    buttons = column.findChildren(QToolButton)
    for button in buttons: button.ensurePolished()
    height = max(22, max(QFontMetrics(button.font()).lineSpacing() + 6 for button in buttons))
    width = max(QFontMetrics(button.font()).horizontalAdvance(button.text()) + 32 for button in buttons)
    for button in buttons:
        button.setFixedSize(width, height)


def compact_button_column(buttons):
    """Three single-line commands, using the same buttons and signal connections."""
    column = QWidget(); column.setProperty('ribbonCompactColumn', True)
    column.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    layout = QVBoxLayout(column); layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1); layout.setAlignment(Qt.AlignVCenter)
    for button in buttons:
        button.setText(button.accessibleName())
        button.setProperty('ribbonCompact', True)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(16, 16))
        button.setStyleSheet(button.styleSheet() + '\nQToolButton {padding: 0 3px;}')
        layout.addWidget(button)
    size_compact_column(column)
    return column


def protect_group_spacing(panel):
    """Keep dividers in their own space even when the viewport becomes narrow."""
    for frame in panel.findChildren(QFrame):
        if frame.frameShape()!=QFrame.VLine and not (frame.minimumWidth()==frame.maximumWidth()==1):
            continue
        layout=frame.parentWidget().layout()
        if layout is None or frame.property('ribbonDividerSpacing'): continue
        index=layout.indexOf(frame)
        if index<0 or not hasattr(layout,'insertSpacing'): continue
        frame.setFixedWidth(1)
        layout.insertSpacing(index,2);layout.insertSpacing(index+2,2)
        frame.setProperty('ribbonDividerSpacing',True)
    # The scroll area's content must grow to the groups' minimum width, rather
    # than squeeze nested layouts underneath fixed-width buttons.
    layout=panel.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.invalidate();layout.activate()


def normalize_ribbon(tabs):
    from repair_ribbon import _RibbonScrollArea
    entries = [(tabs.widget(i), tabs.tabIcon(i), tabs.tabText(i)) for i in range(tabs.count())]
    large_buttons = []
    for widget, _, _ in entries:
        for button in widget.findChildren(QToolButton):
            normalize_font(button, 'QToolButton', COMMAND_FONT_SIZE)
            if not button.property('ribbonCompactPadding'):
                button.setStyleSheet(button.styleSheet() + '\nQToolButton {padding: 0 3px;}')
                button.setProperty('ribbonCompactPadding', True)
            if button.toolButtonStyle() == Qt.ToolButtonTextBesideIcon and button.text():
                metrics = QFontMetrics(button.font())
                button.setFixedHeight(max(24, metrics.lineSpacing() + 6))
                button.setFixedWidth(metrics.horizontalAdvance(button.text()) + button.iconSize().width() + 16)
            if button.toolButtonStyle() != Qt.ToolButtonTextUnderIcon: continue
            button.setIconSize(QSize(ICON_SIZE, ICON_SIZE)); button.ensurePolished()
            metrics = QFontMetrics(button.font())
            lines = button.text().split('\n')
            height = max(74, ICON_SIZE + len(lines) * metrics.lineSpacing() + 18)
            button.setFixedHeight(height)
            large_buttons.append(button)
            menu_width = 12 if button.menu() is not None else 0
            button.setFixedWidth(max(60, max(metrics.horizontalAdvance(line) for line in lines) + 14 + menu_width))
        for label in widget.findChildren(QLabel):
            normalize_font(label, 'QLabel', 10)
            label.setFixedHeight(max(16, QFontMetrics(label.font()).lineSpacing() + 2))
        for column in widget.findChildren(QWidget):
            if column.property('ribbonCompactColumn'): size_compact_column(column)
    # One- and two-line captions must not change the row's vertical position.
    command_height = max((button.height() for button in large_buttons), default=74)
    for button in large_buttons: button.setFixedHeight(command_height)
    for widget, _, _ in entries:
        panel = widget.widget() if isinstance(widget, QScrollArea) else widget
        compact_layout(panel.layout(), root=True)
        protect_group_spacing(panel)
    tabs.clear()
    for widget, icon, title in entries:
        if not isinstance(widget, QScrollArea):
            scroll = _RibbonScrollArea(); scroll.setFrameShape(QFrame.NoFrame); scroll.setWidgetResizable(True)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(widget); widget = scroll
        tabs.addTab(widget, icon, title)
    tabs.setFixedHeight(max(144, max(widget.sizeHint().height() for widget, _, _ in entries) + tabs.tabBar().sizeHint().height() + 16))
