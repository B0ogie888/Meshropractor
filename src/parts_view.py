"""Compact project browser backed by the existing part table and scene selection."""
from PySide6.QtCore import Qt, QSize, QRectF, QPointF, QAbstractListModel, QModelIndex
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import (QWidget, QListView, QStyledItemDelegate, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QStackedWidget, QCheckBox, QAbstractItemView, QSizePolicy)


class PartListModel(QAbstractListModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.entries = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.entries): return None
        entry = self.entries[index.row()]
        if role == Qt.UserRole: return entry
        if role in (Qt.DisplayRole, Qt.AccessibleTextRole): return entry['name']
        if role == Qt.ToolTipRole:
            return entry['name'] + '\n' + entry['detail'] + '\nГлаз — видимость; цветной круг — цвет. Ctrl/Shift — выбор нескольких деталей.'

    def replace(self, entries):
        if [e['row'] for e in entries] != [e['row'] for e in self.entries]:
            self.beginResetModel(); self.entries = entries; self.endResetModel()
        elif entries != self.entries:
            self.entries = entries
            if entries: self.dataChanged.emit(self.index(0), self.index(len(entries)-1))


class PartRowDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index): return QSize(280, 64)

    def paint(self, painter, option, index):
        entry = index.data(Qt.UserRole)
        if not entry: return
        painter.save(); painter.setRenderHint(QPainter.Antialiasing)
        from ui_theme import THEME_COLORS
        window = self.parent().browser.window
        colors = THEME_COLORS[getattr(getattr(window, 'engineering_theme', None), 'mode', 'light')]
        rect = QRectF(option.rect).adjusted(0, 2, -1, -2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(colors['selected'] if entry['selected'] else colors['row']))
        painter.drawRoundedRect(rect, 5, 5)
        if entry['selected']:
            painter.fillRect(QRectF(rect.x(), rect.y()+9, 3, rect.height()-18), QColor('#c4b727'))
        x, y = rect.x(), rect.y()
        box = QRectF(x+12, y+14, 14, 14)
        painter.setPen(QPen(QColor(colors['edge']), 1.2))
        painter.setBrush(QColor('#e5d943') if entry['selected'] else QColor(colors['panel']))
        painter.drawRoundedRect(box, 2, 2)
        if entry['selected']:
            path = QPainterPath(QPointF(x+15, y+21))
            path.lineTo(x+18, y+24); path.lineTo(x+23, y+18)
            painter.setPen(QPen(QColor('#252b2b'), 1.6)); painter.drawPath(path)
        font = option.font; font.setPixelSize(13); font.setWeight(font.Weight.DemiBold)
        painter.setFont(font); painter.setPen(QColor(colors['ink'] if entry['visible'] else colors['secondary']))
        title = painter.fontMetrics().elidedText(entry['name'], Qt.ElideMiddle, max(20, int(rect.width()-118)))
        painter.drawText(QRectF(x+37, y+7, rect.width()-118, 24), Qt.AlignVCenter, title)
        font.setPixelSize(11); font.setWeight(font.Weight.Normal); painter.setFont(font)
        painter.setPen(QColor(colors['secondary']))
        detail = painter.fontMetrics().elidedText(entry['detail'], Qt.ElideRight, int(rect.width()-50))
        painter.drawText(QRectF(x+37, y+32, rect.width()-50, 20), Qt.AlignVCenter, detail)
        # Visibility and material colour remain accessible on every row.
        eye = QRectF(rect.right()-62, y+14, 18, 12)
        painter.setPen(QPen(QColor(colors['ink'] if entry['visible'] else colors['secondary']), 1.4))
        painter.setBrush(Qt.NoBrush); painter.drawEllipse(eye)
        painter.drawEllipse(eye.center(), 2, 2)
        if not entry['visible']: painter.drawLine(eye.topLeft(), eye.bottomRight())
        painter.setBrush(QColor(entry['color'])); painter.setPen(QPen(QColor('#87948c'), 1))
        painter.drawEllipse(QPointF(rect.right()-22, y+20), 8, 8)
        painter.restore()


class PartListView(QListView):
    def __init__(self, browser):
        super().__init__(browser)
        self.browser = browser
        self.setObjectName('EngineeringPartList')
        self.setAccessibleName('Детали проекта: выбор, видимость и цвет')
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setCursor(Qt.PointingHandCursor)
        self.setItemDelegate(PartRowDelegate(self))
        self.anchor = None

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton: return
        index = self.indexAt(event.position().toPoint())
        if not index.isValid() or self.browser.window._busy(): return
        entry = index.data(Qt.UserRole)
        rect = self.visualRect(index)
        x = event.position().x() - rect.x()
        if x > rect.width()-40:
            table = self.browser.table
            button = table.cellWidget(entry['row'], 5).findChild(QPushButton)
            self.browser.window.flush_history()
            button.click()
            self.browser.window.flush_history('Цвет детали')
        elif x > rect.width()-76:
            table = self.browser.table
            check = table.cellWidget(entry['row'], 2).findChild(QCheckBox)
            self.browser.window.flush_history()
            check.setChecked(not check.isChecked())
            self.browser.window.flush_history('Видимость детали')
        else:
            self.select_entry(index.row(), event.modifiers(), toggle=x<32)
        self.browser.refresh()

    def select_entry(self, index, modifiers=Qt.NoModifier, toggle=False):
        if self.browser.window._busy(): return
        entries = self.model().entries
        if not 0 <= index < len(entries): return
        if modifiers & Qt.ShiftModifier and self.anchor is not None:
            start, end = sorted((min(self.anchor, len(entries)-1), index))
            rows = [e['row'] for e in entries[start:end+1]]; operation = 'add'
        else:
            rows = [entries[index]['row']]
            operation = 'toggle' if toggle or modifiers & Qt.ControlModifier else 'replace'
            self.anchor = index
        self.browser.window.workspace_tools.select_parts(rows, operation)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self.browser.select_all(); return
        if event.key() == Qt.Key_Escape:
            self.browser.select_none(); return
        if event.key() == Qt.Key_Space:
            self.select_entry(self.currentIndex().row(), toggle=True); return
        super().keyPressEvent(event)
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End):
            self.select_entry(self.currentIndex().row(), event.modifiers())


class PartsBrowser(QWidget):
    def __init__(self, window, table):
        super().__init__()
        self.window = window; self.table = table
        self.setObjectName('PartsBrowser')
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        bar = QHBoxLayout(); bar.setSpacing(8)
        self.counter = QLabel('00 / ДЕТАЛЕЙ'); self.counter.setObjectName('PartsEyebrow')
        bar.addWidget(self.counter); bar.addStretch()
        self.all_button = QPushButton('Все'); self.none_button = QPushButton('Снять')
        for b in (self.all_button, self.none_button):
            b.setProperty('quietAction', True); bar.addWidget(b)
        layout.addLayout(bar)
        self.stack = QStackedWidget(); layout.addWidget(self.stack)
        self.model = PartListModel(self)
        self.list = PartListView(self); self.list.setModel(self.model)
        self.stack.addWidget(self.list)
        self.empty = QWidget(); el = QVBoxLayout(self.empty)
        el.setContentsMargins(14, 20, 14, 20); el.setSpacing(10)
        self.empty_title = QLabel('Здесь будут ваши детали'); self.empty_title.setObjectName('EmptyPartsTitle')
        self.empty_title.setAlignment(Qt.AlignCenter)
        self.empty_text = QLabel('Перетащите STL или STEP в сцену\nили загрузите модель')
        self.empty_text.setWordWrap(True); self.empty_text.setAlignment(Qt.AlignCenter)
        self.empty_text.setObjectName('PartsHint')
        self.import_button = QPushButton('+  Загрузить модель'); self.import_button.setProperty('primary', True)
        el.addStretch(); el.addWidget(self.empty_title); el.addWidget(self.empty_text)
        el.addWidget(self.import_button); el.addStretch()
        self.stack.addWidget(self.empty); self.stack.addWidget(table)
        table.setMinimumHeight(160)
        self.all_button.clicked.connect(self.select_all); self.none_button.clicked.connect(self.select_none)
        self.import_button.clicked.connect(window.import_slicer_part)
        self.refresh()

    def select_all(self):
        if not self.window._busy():
            self.window.workspace_tools.select_parts([e['row'] for e in self.model.entries], 'add')

    def select_none(self):
        if not self.window._busy():
            self.window.workspace_tools.select_parts([e['row'] for e in self.model.entries], 'subtract')

    def set_table_mode(self, visible):
        self.stack.setCurrentWidget(self.table if visible else self.list)
        self.refresh()

    def refresh(self, *args):
        from pyvista import Color
        table_mode = self.stack.currentWidget() is self.table
        entries = []
        for row, part in enumerate(self.window.slicer_parts):
            if row >= self.table.rowCount() or self.table.isRowHidden(row): continue
            checked = self.table.cellWidget(row, 1)
            if checked is None or self.table.cellWidget(row, 5) is None: continue
            name = part['filename']
            style = self.window._style_for(self.table, row)
            mesh = part['mesh']; cad = 'cad_native' in mesh.metadata
            detail = ('BREP' if cad else 'СЕТКА') + f'  /  {len(mesh.faces):,} треуг.'.replace(',', ' ')
            if part.get('supports'): detail += f"  /  Поддержки: {len(part['supports'])}"
            entries.append(dict(row=row, name=name, detail=detail,
                selected=checked.findChild(QCheckBox).isChecked(), visible=style['is_visible'],
                color=Color(style['color']).hex_rgb))
        self.model.replace(entries)
        self.counter.setText(f'{len(entries):02d} / ДЕТАЛЕЙ')
        if not table_mode:
            self.stack.setCurrentWidget(self.list if entries else self.empty)
        self.stack.setFixedHeight(240 if table_mode else min(272, len(entries)*64+4) if entries else 146)
        busy = self.window._job is not None
        self.list.setEnabled(not busy); self.all_button.setEnabled(bool(entries) and not busy)
        self.none_button.setEnabled(bool(entries) and not busy); self.import_button.setEnabled(not busy)
