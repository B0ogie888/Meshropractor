"""Build marking contours on the GUI thread without creating an editor window."""
import base64
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainterPath, QImage
from marking_geometry import mask_contours


def content_parameters(values, code_preview=None):
    values = dict(values)
    if values['content']=='text':
        if not values['text'].strip() or len(values['text'])>512: raise ValueError('Введите от 1 до 512 символов.')
        font = QFont(values.get('font','Arial')); font.setPixelSize(100)
        font.setBold(values['bold']); font.setItalic(values['italic']); font.setUnderline(values['underline']); font.setStrikeOut(values['strike'])
        from PySide6.QtGui import QFontMetricsF
        metrics = QFontMetricsF(font); path = QPainterPath()
        values['glyph_height'] = metrics.capHeight()
        lines = values['text'].splitlines()
        for index,line in enumerate(lines):
            width = metrics.horizontalAdvance(line)
            x = 0 if values['align']=='left' else -width if values['align']=='right' else -width/2
            y = index*metrics.height()*values['spacing']; path.addText(x,y,font,line)
            if values['underline']: path.addRect(x,y+metrics.underlinePos(),width,metrics.lineWidth())
            if values['strike']: path.addRect(x,y-metrics.strikeOutPos(),width,metrics.lineWidth())
        values['contours'] = [[(p.x(),-p.y()) for p in polygon] for polygon in path.simplified().toSubpathPolygons()]
    else:
        if values['content']=='datamatrix':
            if not values['code'] or len(values['code'].encode('utf-8'))>300: raise ValueError('Data Matrix: введите до 300 байт данных.')
            from pystrich.datamatrix import DataMatrixData, DataMatrixEncoder
            encoder = DataMatrixEncoder(DataMatrixData(values['code'],auto_encoding=True))
            mask = np.asarray(encoder.get_pilimage(cellsize=1).convert('L')) < 128
            png = encoder.get_imagedata(cellsize=4)
            if code_preview is not None: code_preview(png)
            if values['circular']: raise ValueError('Для считываемого Data Matrix выберите прямоугольную маркировку.')
        else:
            if not values['image']: raise ValueError('Загрузите рисунок.')
            image = QImage.fromData(base64.b64decode(values['image']))
            if image.isNull(): raise ValueError('Не удалось прочитать сохранённый рисунок.')
            image = image.scaled(int(values['raster_size']),int(values['raster_size']),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            # Composite alpha against white so transparent pixels never become black relief.
            rgba = image.convertToFormat(QImage.Format_RGBA8888)
            arr = np.frombuffer(rgba.constBits(),np.uint8).reshape(rgba.height(),rgba.bytesPerLine())[:,:rgba.width()*4].reshape(rgba.height(),rgba.width(),4)
            rgb = arr[:,:,:3]*(arr[:,:,3:4]/255) + 255*(1-arr[:,:,3:4]/255)
            mask = rgb.mean(2)<values['threshold']
            if values['invert_image']: mask = ~mask
        values['contours'] = mask_contours(mask)
    return values
