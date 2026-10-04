"""Contact sheets of the actual monochrome ribbon icons in both themes."""
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter
from PySide6.QtWidgets import QApplication
from ui_theme import EngineeringTheme, THEME_COLORS


def main():
    app = QApplication.instance() or QApplication([])
    target = ROOT / 'output/icon-audit'
    target.mkdir(parents=True, exist_ok=True)
    cache = SimpleNamespace(icons={})
    for group in sorted((ROOT / 'assets/ribbon').iterdir()):
        if not group.is_dir():
            continue
        paths = sorted(group.glob('*.svg'))
        canvas = QImage(1000, ((len(paths) + 4) // 5) * 108, QImage.Format_ARGB32)
        canvas.fill(QColor('#fafbf8'))
        painter = QPainter(canvas)
        font = QFont('Segoe UI'); font.setPixelSize(11); painter.setFont(font)
        for index, path in enumerate(paths):
            x, y = (index % 5) * 200, (index // 5) * 108
            for column, mode in enumerate(('light', 'dark')):
                colors = THEME_COLORS[mode]
                painter.fillRect(x + column * 100, y, 100, 72, QColor(colors['panel']))
                icon = EngineeringTheme.recolor_icon(cache, QIcon(str(path)), colors['ink'])
                painter.drawPixmap(x + column * 100 + 8, y + 15, icon.pixmap(28, 28))
                painter.drawPixmap(x + column * 100 + 45, y + 6, icon.pixmap(48, 48))
            painter.setPen(QColor('#252b2b'))
            painter.drawText(QRect(x + 4, y + 74, 192, 32), Qt.TextWordWrap, path.stem)
        painter.end()
        assert canvas.save(str(target / (group.name + '.png')))
    print('ICON_PREVIEWS_OK', target)


if __name__ == '__main__':
    main()
