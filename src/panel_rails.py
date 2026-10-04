"""Persistent edge grips for collapsible workspace panels."""
from PySide6.QtCore import QObject, Qt, QEvent, QTimer, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QFont
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget, QHBoxLayout, QSizePolicy


class PanelGrip(QAbstractButton):
    def __init__(self, controller, index, title, parent, *, compact=False):
        super().__init__(parent)
        self.controller, self.index, self.title = controller, index, title
        self.compact = compact
        self.setFixedWidth(8 if compact else 22)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed if compact else QSizePolicy.Expanding)
        if compact: self.setFixedHeight(50)
        self.setFocusPolicy(Qt.StrongFocus); self.setCursor(Qt.SizeHorCursor)
        self.setAccessibleName(title + ': скрыть или открыть панель')
        self.clicked.connect(lambda: controller.set_visible(index, not controller.is_open(index)))
        self.start = None; self.dragged = False

    def refresh(self):
        opened = self.controller.is_open(self.index)
        action = 'Скрыть' if opened else 'Открыть'
        self.setToolTip(f'{action}: {self.title}\nЩелчок или Enter/пробел — переключить. Потяните мышью для изменения ширины.')
        self.setAccessibleDescription(('Панель открыта' if opened else 'Панель скрыта') + '. Можно вытянуть мышью.')
        self.update()

    def paintEvent(self, event):
        from ui_theme import THEME_COLORS
        theme = getattr(self.controller.window, 'engineering_theme', None)
        colors = THEME_COLORS[theme.mode if theme else 'light']
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors['panel']))
        active = self.underMouse() or self.hasFocus() or self.isDown()
        if self.compact:
            if active: painter.fillRect(self.rect(), QColor(colors['selected']))
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(colors['ink'] if active else colors['secondary']))
            for dy in (-5, 0, 5): painter.drawEllipse(QPointF(self.width()/2, self.height()/2+dy), 1, 1)
            if self.hasFocus():
                painter.setPen(QPen(QColor(colors['edge']), 1, Qt.DotLine)); painter.setBrush(Qt.NoBrush)
                painter.drawRect(self.rect().adjusted(1,1,-1,-1))
            return
        height = min(184, self.height() - 8)
        rect = QRectF(2, (self.height()-height)/2, self.width()-4, height)
        opened = self.controller.is_open(self.index)
        painter.setPen(QPen(QColor(colors['edge']), .7))
        painter.setBrush(QColor(colors['selected'] if active else colors['row']))
        # Clipped corners and an accent tick identify a retained, hidden panel.
        painter.drawPolygon(QPolygonF([rect.topLeft()+QPointF(4,0), rect.topRight(),
            rect.bottomRight()-QPointF(0,4), rect.bottomRight()-QPointF(4,0),
            rect.bottomLeft(), rect.topLeft()+QPointF(0,4)]))
        if not opened or active:
            painter.fillRect(QRectF(rect.x(), rect.top()+28, 2, rect.height()-56), QColor('#e5d943'))
        x = self.width()/2; y = rect.top()+13
        direction = (1 if self.index == 0 else -1) * (-1 if opened else 1)
        painter.setPen(QPen(QColor(colors['ink']), 1.3))
        painter.drawPolyline(QPolygonF([QPointF(x-direction*2,y-4), QPointF(x+direction*2,y), QPointF(x-direction*2,y+4)]))
        font = QFont('Segoe UI'); font.setPixelSize(10); font.setWeight(QFont.Normal)
        painter.setFont(font); painter.save()
        painter.translate(x, rect.center().y()); painter.rotate(-90 if self.index == 0 else 90)
        painter.drawText(QRectF(-height/2+25, -8, height-50, 16), Qt.AlignCenter, self.title.upper())
        painter.restore(); painter.setPen(Qt.NoPen); painter.setBrush(QColor(colors['secondary']))
        for dx in (-2, 2):
            for dy in (0, 4, 8): painter.drawEllipse(QPointF(x+dx, rect.bottom()-18+dy), .8, .8)
        if self.hasFocus():
            painter.setPen(QPen(QColor(colors['ink']), 1, Qt.DotLine)); painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect.adjusted(2,2,-2,-2))

    def enterEvent(self, event): super().enterEvent(event); self.update()
    def leaveEvent(self, event): super().leaveEvent(event); self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton: return super().mousePressEvent(event)
        self.setFocus(Qt.MouseFocusReason); self.setDown(True)
        self.start = (event.globalPosition().x(), self.controller.splitter.sizes()[self.index])
        self.dragged = False; event.accept()

    def mouseMoveEvent(self, event):
        if self.start is None: return super().mouseMoveEvent(event)
        delta = event.globalPosition().x() - self.start[0]
        if abs(delta) >= QApplication.startDragDistance(): self.dragged = True
        if self.dragged:
            self.controller.resize_panel(self.index, round(self.start[1] + delta * (1 if self.index == 0 else -1)))
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.start is None: return super().mouseReleaseEvent(event)
        self.start = None; self.setDown(False)
        if not self.dragged: self.click()
        self.controller.sync()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter): self.click(); event.accept()
        else: super().keyPressEvent(event)


class PanelRails(QObject):
    changed = Signal()

    def __init__(self, window, splitter, workspace, titles, defaults):
        super().__init__(splitter)
        self.window, self.splitter = window, splitter
        self.keys = {0: f'new_ui/{workspace}_left', 2: f'new_ui/{workspace}_right'}
        if workspace == 'slicer': self.keys[2] = 'new_ui/inspector'
        self.widths = {i: window.settings.value(self.keys[i]+'_width', defaults[i], type=int) for i in (0,2)}
        visible = {i: window.settings.value(self.keys[i]+'_visible', True, type=bool) for i in (0,2)}
        parent = splitter.parentWidget(); layout = parent.layout()
        self.container = QWidget(parent); row = QHBoxLayout(self.container)
        row.setContentsMargins(0,0,0,0); row.setSpacing(0)
        layout.replaceWidget(splitter, self.container)
        if hasattr(layout, 'setStretchFactor'): layout.setStretchFactor(self.container, 1)
        self.grips = {i: PanelGrip(self, i, titles[i], self.container) for i in (0,2)}
        row.addWidget(self.grips[0]); row.addWidget(splitter, 1); row.addWidget(self.grips[2])
        # Open panels only need a small grip on their scene-facing separator.
        # Keep the labelled outer rails for collapsed panels.
        splitter.setHandleWidth(8)
        self.compact_grips = {}
        for i in (0,2):
            handle = splitter.handle(1 if i == 0 else 2)
            grip = PanelGrip(self, i, titles[i], handle, compact=True)
            grip.move(0, max(0, (handle.height()-grip.height())//2))
            handle.installEventFilter(self)
            self.compact_grips[i] = grip
        splitter.setCollapsible(1, False)
        for i in (0,2):
            splitter.setCollapsible(i, True)
            splitter.widget(i).show()  # Collapse by size; external support tools also use setSizes().
            splitter.widget(i).installEventFilter(self)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.sync)
        splitter.installEventFilter(self); splitter.splitterMoved.connect(self.sync)
        sizes = [self.widths[0] if visible[0] else 0, 1, self.widths[2] if visible[2] else 0]
        sizes[1] = max(1, splitter.width()-sizes[0]-sizes[2]-2*splitter.handleWidth())
        splitter.setSizes(sizes)
        self.sync()

    def is_open(self, index):
        return not self.splitter.widget(index).isHidden() and self.splitter.sizes()[index] > 0

    def set_visible(self, index, visible):
        if visible:
            self.resize_panel(index, max(self.minimum(index), self.widths[index]))
        else: self.resize_panel(index, 0)

    def minimum(self, index):
        panel = self.splitter.widget(index)
        return max(180, panel.minimumWidth(), panel.minimumSizeHint().width())

    def resize_panel(self, index, width):
        sizes = self.splitter.sizes(); total = sum(sizes)
        minimum = self.minimum(index)
        if sizes[index] > 0: self.widths[index] = sizes[index]
        if width < minimum/2: width = 0
        else:
            available = max(minimum, total - sizes[2 if index == 0 else 0] - 100)
            width = min(available, max(minimum, width))
        self.splitter.widget(index).show()
        sizes[index] = width; sizes[1] = max(1, total-sizes[0]-sizes[2])
        self.splitter.setSizes(sizes); self.sync()

    def sync(self, *args):
        for index in (0,2):
            opened = self.is_open(index)
            self.window.settings.setValue(self.keys[index]+'_visible', opened)
            if opened:
                self.widths[index] = self.splitter.sizes()[index]
                self.window.settings.setValue(self.keys[index]+'_width', self.widths[index])
            outer, compact = self.grips[index], self.compact_grips[index]
            # Do not hide a button that owns the current mouse drag. Hand over
            # to the other grip on release, preserving native mouse capture.
            focused = outer.hasFocus() or compact.hasFocus()
            outer.setVisible(not opened or outer.start is not None)
            compact.setVisible(opened or compact.start is not None)
            if focused and outer.start is None and compact.start is None:
                (compact if opened else outer).setFocus(Qt.OtherFocusReason)
            outer.refresh(); compact.refresh()
        self.changed.emit()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            for grip in self.compact_grips.values():
                if obj is grip.parentWidget(): grip.move(0, max(0, (obj.height()-grip.height())//2))
        if event.type() in (QEvent.Resize, QEvent.Show, QEvent.Hide) and hasattr(self, 'timer'):
            if not self.timer.isActive(): self.timer.start(0)
        return False
