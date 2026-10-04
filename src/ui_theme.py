"""Engineering appearance for the desktop application.

Only UI declarations are translated. Model colours, textures, VTK actors and
colour swatches are data and must never be recoloured by a theme.
"""
import re
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QEvent, Signal
from PySide6.QtGui import QColor, QPalette, QIcon, QPixmap, QPainter
from PySide6.QtWidgets import QApplication, QWidget, QAbstractButton, QDialog


BACKGROUND = '#edf0ed'
PANEL = '#fafbf8'
INK = '#252b2b'
SECONDARY = '#5b6463'
BORDER = '#cdd3cf'
ACCENT = '#e5d943'
SCENE = '#e6eae5'

LIGHT_STYLE = '''
QWidget { color: #252b2b; font-family: "Segoe UI", "DejaVu Sans"; font-size: 12px; }
QMainWindow, #MainWidget { background: #edf0ed; }
QDialog, QMenu { background: #fafbf8; }
QLabel { background: transparent; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTextBrowser,
QPlainTextEdit, QListWidget, QTreeWidget, QTableWidget {
    background: #ffffff; color: #252b2b; selection-background-color: #e5d943;
    selection-color: #252b2b; border: 1px solid #cdd3cf; border-radius: 2px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 4px; }
QSpinBox:disabled, QDoubleSpinBox:disabled { color: #89918a; background: #edf0ed; border-color: #dce1dc; }
QPushButton, QToolButton { color: #252b2b; background: #fafbf8;
    border: 1px solid #cdd3cf; border-radius: 2px; padding: 5px 8px; }
QPushButton:hover, QToolButton:hover { background: #f2efc6; border-color: #8e968c; }
QPushButton:pressed, QToolButton:pressed, QToolButton:checked,
QPushButton[primary="true"] { background: #e5d943; color: #252b2b; }
QPushButton:disabled, QToolButton:disabled, QLabel:disabled { color: #89918a; }
QToolTip { background: #252b2b; color: #fafbf8; padding: 6px; border: 0; }
QTabWidget::pane { background: #fafbf8; border: 1px solid #cdd3cf; }
QTabBar::tab { background: #edf0ed; color: #5b6463; padding: 7px 12px; }
QTabBar::tab:selected { background: #fafbf8; color: #252b2b; border-bottom: 3px solid #e5d943; }
QHeaderView::section { background: #edf0ed; color: #5b6463; border: 0;
    border-bottom: 1px solid #cdd3cf; padding: 5px; }
QTableWidget { gridline-color: #dce1dc; alternate-background-color: #f2f4f0; }
QCheckBox, QRadioButton { color: #252b2b; spacing: 6px; }
QGroupBox { border: 1px solid #cdd3cf; border-radius: 2px; margin-top: 10px; padding-top: 10px; }
QGroupBox::title { color: #5b6463; subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QSplitter::handle { background: #cdd3cf; }
QScrollBar { background: #edf0ed; }
QScrollBar::handle { background: #afb9b0; border-radius: 3px; min-height: 18px; min-width: 18px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QProgressBar { color: #252b2b; background: #e0e6dd; border: 0; text-align: center; }
QProgressBar::chunk { background: #a0b18e; }
QStatusBar { background: #fafbf8; color: #5b6463; border-top: 1px solid #cdd3cf; }
QMenu::item:selected { background: #e5d943; color: #252b2b; }
'''

_DARK = {'#111111', '#161616', '#181818', '#1e1e1e', '#202020', '#222222',
         '#242424', '#252525', '#262626', '#272727', '#282828', '#292929',
         '#2b2b2b', '#2c2c2c', '#2d2d2d', '#303030', '#333333', '#343434',
         '#353535', '#363636', '#383838', '#3a3a3a', '#3b3b3b', '#3d3d3d',
         '#3e3e3e', '#404040', '#444444', '#484848', '#4a4a4a', '#4b4b4b',
         '#505050', '#555555', '#606060', '#666666', '#777777'}
_LIGHT = {'#ffffff', '#eeeeee', '#e0e0e0', '#dddddd', '#cccccc', '#bbbbbb',
          '#aaaaaa', '#a0a0a0', '#999999', '#888888', '#d0d0d0', '#d3d3d3', '#f0f0f0'}
_ACTION = {'#2c3e50', '#34495e', '#1a252f', '#b31b1b', '#a52d2d', '#c0392b', '#e74c3c', '#922b21', '#a93226',
           '#d32f2f', '#b71c1c', '#2980b9', '#3498db', '#1f618d', '#2471a3',
           '#2e86c1', '#27ae60', '#229954', '#1e8449', '#28a745', '#e67e22',
           '#d35400', '#f39c12', '#506d38'}
_CHROME_BACKGROUNDS = {'#41464a': '#f2efc6', '#244650': ACCENT,
                       '#345365': ACCENT, '#426071': ACCENT, '#515151': BORDER,
                       '#60666b': '#afb9b0', '#8a949d': '#8e968c'}
_TOKEN = re.compile(r'#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|\b(?:white|black)\b')
_DECL = re.compile(r'(?P<property>[\w-]+)\s*:\s*(?P<value>[^;{}]+)')


def translate_style(style):
    """Map old chrome, preserving geometry, selectors, sizes and data colours."""
    def declaration(match):
        prop = match['property'].lower()
        def token(value):
            raw = value.group(0).lower()
            color = '#ffffff' if raw == 'white' else '#000000' if raw == 'black' else raw
            if len(color) == 4: color = '#' + ''.join(c * 2 for c in color[1:])
            if 'border' in prop or prop == 'gridline-color':
                if color in ('#70bddb', '#78c3ce'): return ACCENT
                if color == '#626c73': return BORDER
                return ACCENT if color in _ACTION else BORDER if color in _DARK | _LIGHT else raw
            if 'background' in prop:
                if color in _CHROME_BACKGROUNDS: return _CHROME_BACKGROUNDS[color]
                return ACCENT if color in _ACTION else PANEL if color in _DARK else raw
            if prop in ('color', 'selection-color'):
                if color == '#929ca3': return SECONDARY
                if color in _LIGHT or color == '#000000': return INK
                if color in ('#5dade2', '#85c1e9'): return '#346b91'
                if color in ('#90ee90', '#a9dfbf', '#2ecc71'): return '#33724f'
                if color == '#f39c12': return '#8e660c'
            return raw
        return match['property'] + ': ' + _TOKEN.sub(token, match['value'])
    # Pseudo states and subcontrols (QToolButton:checked, QMenu::item) are
    # selectors, not declarations. Touch only brace contents in full QSS.
    if '{' in style:
        return re.sub(r'\{([^{}]*)\}', lambda m: '{' + _DECL.sub(declaration, m[1]) + '}', style)
    return _DECL.sub(declaration, style)


def light_palette():
    palette = QPalette()
    for role, color in ((QPalette.Window, PANEL), (QPalette.WindowText, INK),
                        (QPalette.Base, '#ffffff'), (QPalette.AlternateBase, BACKGROUND),
                        (QPalette.Text, INK), (QPalette.Button, PANEL),
                        (QPalette.ButtonText, INK), (QPalette.Highlight, ACCENT),
                        (QPalette.HighlightedText, INK), (QPalette.ToolTipBase, INK),
                        (QPalette.ToolTipText, PANEL), (QPalette.PlaceholderText, SECONDARY)):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor('#89918a'))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#89918a'))
    return palette


THEME_COLORS = {
    'light': dict(panel=PANEL, ink=INK, secondary=SECONDARY, scene=SCENE,
                  row='#f3f5f0', selected='#f1f0d8', edge='#59685f'),
    'dark': dict(panel='#282f2f', ink='#e6ede7', secondary='#aebbb4', scene='#1d2825',
                 row='#303a35', selected='#424a32', edge='#aebbb4'),
}
_DARK_SURFACES = {
    '#fafbf8': '#282f2f', '#ffffff': '#1c2421', '#edf0ed': '#202626',
    '#f0f3ed': '#222d27', '#edf1e8': '#354337', '#f3f5f0': '#303a35',
    '#f2f4f0': '#303a35', '#f0f2e9': '#354337', '#e9eddf': '#424a32',
    '#f4f4e5': '#424a32', '#e9e6bf': '#4c5234', '#f2efc6': '#424a32',
    '#dce1dc': '#46534e', '#dce2da': '#46534e', '#cdd3cf': '#46534e',
    '#ccd5ca': '#46534e', '#d7decf': '#46534e', '#e0e6dd': '#354337',
    '#a0b18e': '#7e946c', '#afb9b0': '#637368', '#556953': '#c5ce9a',
}
_DARK_TEXT = {'#252b2b': '#e6ede7', '#263b30': '#e6ede7', '#46564d': '#c0cdbf',
              '#5b6463': '#aebbb4', '#68776c': '#aebbb4', '#69796d': '#aebbb4',
              '#52644e': '#c5ce9a', '#69766d': '#aebbb4', '#526553': '#c5ce9a',
              '#346b91': '#91bddd', '#33724f': '#a4c897', '#8e660c': '#e6c46b',
              '#89918a': '#87958a'}


def theme_style(canonical, mode):
    """Render canonical light chrome; never round-trip already themed colours."""
    if mode != 'dark': return canonical
    canonical = canonical.replace('/theme/up-light.svg', '/theme/up-dark.svg').replace('/theme/down-light.svg', '/theme/down-dark.svg')
    for state in ('on','off'):
        canonical=canonical.replace(f'/theme/radio-{state}-light.svg',f'/theme/radio-{state}-dark.svg')
    def block(source):
        accent = bool(re.search(r'(?<![-\w])background(?:-color)?\s*:\s*#e5d943', source))
        def declaration(match):
            prop = match['property'].lower()
            def color(token):
                value = token[0].lower()
                if prop in ('color', 'selection-color'):
                    if prop == 'selection-color' or accent: return '#252b2b' if value == INK else value
                    return _DARK_TEXT.get(value, value)
                return _DARK_SURFACES.get(value, value)
            return match['property'] + ': ' + _TOKEN.sub(color, match['value'])
        return _DECL.sub(declaration, source)
    return re.sub(r'\{([^{}]*)\}', lambda m: '{' + block(m[1]) + '}', canonical) if '{' in canonical else block(canonical)


def theme_palette(mode):
    palette = light_palette()
    if mode == 'dark':
        for role, color in ((QPalette.Window, '#282f2f'), (QPalette.WindowText, '#e6ede7'),
                (QPalette.Base, '#1c2421'), (QPalette.AlternateBase, '#303a35'),
                (QPalette.Text, '#e6ede7'), (QPalette.Button, '#282f2f'),
                (QPalette.ButtonText, '#e6ede7'), (QPalette.PlaceholderText, '#aebbb4'),
                (QPalette.ToolTipBase, '#282f2f'), (QPalette.ToolTipText, '#e6ede7')):
            palette.setColor(role, QColor(color))
    return palette


class EngineeringTheme(QObject):
    changed = Signal(str)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._changing = False
        self.icons = {}
        settings = getattr(window, 'settings', None)
        self.mode = settings.value('appearance/theme', 'light') if settings else 'light'
        if self.mode not in THEME_COLORS: self.mode = 'light'
        self.palette = theme_palette(self.mode)
        window.setPalette(self.palette)
        assets = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1])) / 'assets'
        checkmark = (assets / 'checkmark_dark.svg').as_posix()
        checkbox_style = f'QCheckBox::indicator:checked {{image: url("{checkmark}");}}'
        arrows = ''.join(f'''QSpinBox::{direction}-arrow, QDoubleSpinBox::{direction}-arrow {{
            image: url("{assets.as_posix()}/theme/{direction}-light.svg"); width: 10px; height: 6px;}}'''
            for direction in ('up', 'down'))
        arrows += '''QSpinBox, QDoubleSpinBox {padding-right: 20px;}
            QSpinBox::up-button, QDoubleSpinBox::up-button {subcontrol-origin: border;
                subcontrol-position: top right; width: 18px; border: 0; background: transparent;}
            QSpinBox::down-button, QDoubleSpinBox::down-button {subcontrol-origin: border;
                subcontrol-position: bottom right; width: 18px; border: 0; background: transparent;}'''
        radios=f'''QRadioButton {{min-height: 18px;}}
            QRadioButton::indicator {{width: 16px; height: 16px; border: none; background: transparent;
                image: url("{assets.as_posix()}/theme/radio-off-light.svg");}}
            QRadioButton::indicator:checked {{image: url("{assets.as_posix()}/theme/radio-on-light.svg");}}'''
        self.root_style = translate_style(window.styleSheet()) + LIGHT_STYLE + checkbox_style + arrows + radios
        self.style_widget(window)
        for widget in window.findChildren(QWidget): self.style_widget(widget)
        QApplication.instance().installEventFilter(self)

    def recolor_icon(self, icon, color=INK, on_color=None):
        key = icon.cacheKey(), color, on_color
        if icon.isNull(): return icon
        if key not in self.icons:
            result = QIcon()
            for size in (16, 20, 24, 28, 32, 48, 56, 64):
                pixmap = icon.pixmap(size, size)
                painted = QPixmap(pixmap.size()); painted.fill(QColor('transparent'))
                painter = QPainter(painted)
                painter.drawPixmap(0, 0, pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(painted.rect(), QColor(color)); painter.end()
                result.addPixmap(painted, QIcon.Normal, QIcon.Off)
                if on_color is not None:
                    painter = QPainter(painted); painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                    painter.fillRect(painted.rect(), QColor(on_color)); painter.end()
                    result.addPixmap(painted, QIcon.Normal, QIcon.On)
            self.icons[key] = result
        return self.icons[key]

    def belongs(self, widget):
        current = widget
        while current is not None:
            if current is self.window or current.property('engineeringThemeOwner') == id(self.window): return True
            current = current.parentWidget()
        # Some existing modal dialogs intentionally have no parent.
        active = QApplication.activeWindow()
        return active is self.window and widget.isWindow() and widget.parentWidget() is None

    def style_widget(self, widget):
        if widget.property('preserveThemeColors'): return
        current = widget
        while current is not None:
            if current.property('engineeringChrome'): return
            current = current.parentWidget()
        style = widget.styleSheet()
        source = widget.property('engineeringSourceStyle')
        if widget is self.window:
            source = self.root_style
        elif source is None or style != widget.property('engineeringAppliedStyle'):
            source = translate_style(style)
            widget.setProperty('engineeringSourceStyle', source)
        if style or widget is self.window:
            # Empty, small square buttons are model/section colour swatches.
            if isinstance(widget, QAbstractButton) and not widget.text() and widget.icon().isNull() and widget.maximumWidth() <= 40:
                return
            translated = theme_style(source, self.mode)
            if translated != style:
                changing = self._changing; self._changing = True
                try: widget.setStyleSheet(translated)
                finally: self._changing = changing
            widget.setProperty('engineeringAppliedStyle', translated)
        if widget.isWindow() or widget.autoFillBackground(): widget.setPalette(self.palette)
        if widget.isWindow():
            from native_window_theme import apply_native_caption
            colors = THEME_COLORS[self.mode]
            apply_native_caption(widget, self.mode, colors['panel'], colors['ink'])
        if isinstance(widget, QAbstractButton) and not widget.icon().isNull():
            original = widget.property('engineeringOriginalIcon')
            if original is None or widget.icon().cacheKey() != widget.property('engineeringAppliedIcon'):
                original = widget.icon(); widget.setProperty('engineeringOriginalIcon', original)
            on_color = INK if widget.isCheckable() and not widget.property('engineeringQuietToggle') else None
            recolored = self.recolor_icon(original, THEME_COLORS[self.mode]['ink'], on_color)
            widget.setProperty('engineeringAppliedIcon', recolored.cacheKey())
            if widget.icon().cacheKey() != recolored.cacheKey(): widget.setIcon(recolored)

    def set_mode(self, mode):
        if mode not in THEME_COLORS or mode == self.mode: return
        from dataclasses import replace
        from display_settings import save_preferences, apply_to_plotter
        window = self.window; old = self.mode
        preferences = window.ui.display_preferences
        window.settings.setValue('appearance/background/' + old, preferences.background)
        self.mode = mode; self.palette = theme_palette(mode)
        window.settings.setValue('appearance/theme', mode)
        background = window.settings.value('appearance/background/' + mode, THEME_COLORS[mode]['scene'])
        if not QColor(background).isValid(): background = THEME_COLORS[mode]['scene']
        window.ui.display_preferences = replace(preferences, background=background)
        self._changing = True
        try:
            window.setPalette(self.palette)
            # A parent's stylesheet repolishes descendants and resets scroll
            # viewport palettes. QApplication.allWidgets() has no tree order:
            # styling a parent last could undo an already themed viewport until
            # its next Show event. Finish parents before their children.
            def depth(widget):
                level = 0
                while widget.parentWidget() is not None:
                    level += 1; widget = widget.parentWidget()
                return level
            widgets = sorted((w for w in QApplication.allWidgets() if self.belongs(w)), key=depth)
            for widget in widgets:
                self.style_widget(widget); widget.update()
        finally: self._changing = False
        for plotter in (window.ui.slicer_plotter, window.ui.plotter):
            apply_to_plotter(plotter, window.ui.display_preferences)
        display = getattr(window, 'display_tools', None)
        if display is not None: display.statistics.request()
        save_preferences(window.settings, window.ui.display_preferences)
        self.changed.emit(mode)

    def eventFilter(self, widget, event):
        if (not self._changing and isinstance(widget, QWidget)
                and event.type() in (QEvent.Polish, QEvent.Show, QEvent.StyleChange, QEvent.WinIdChange)
                and self.belongs(widget)):
            self.style_widget(widget)
            if event.type() == QEvent.Show and widget.isWindow():
                widget.setProperty('engineeringThemeOwner', id(self.window))
                self._changing = True
                try:
                    if isinstance(widget, QDialog) and widget.styleSheet():
                        # Qt may finish caching the original QSS after our
                        # StyleChange/Polish handler translated it. Reapply the
                        # completed dialog's rules after polish, before paint.
                        # Theme toggling used to do this only on a later visit.
                        widget.setStyleSheet(widget.styleSheet())
                        self.style_widget(widget)
                    for child in widget.findChildren(QWidget): self.style_widget(child)
                finally:
                    self._changing = False
        return False
