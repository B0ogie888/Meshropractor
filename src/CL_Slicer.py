"""Experimental CLS contours. Desktop export awaits machine/reference validation."""
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from project_store import load_mesh


def pack_int(value):
    return struct.pack('<i', int(value))


def pack_border(points, is_closed=True):
    if not points:
        return pack_int(0)
    data = bytearray(struct.pack('<i B i', 1, int(is_closed), len(points)))
    for x, y in points:
        data.extend(struct.pack('<ii', int(x), int(y)))
    return bytes(data)


def _write_borders(stream, section, scale):
    if section is not None:
        for polygon in section.discrete:
            points = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
            stream.write(b'NEW_BORDER' + pack_border(points, bool(np.allclose(polygon[0], polygon[-1]))))


def slice_stl_to_cls(part_path, support_path, cls_path, layer_height=0.03,
                     progress_callback=None, cancel_callback=None, platform_size=(220.0, 220.0)):
    if not math.isfinite(layer_height) or layer_height <= 0:
        raise ValueError("Толщина слоя должна быть положительной.")
    dimensions = np.asarray(platform_size, dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("Некорректный размер платформы.")
    part = load_mesh(part_path)
    support = load_mesh(support_path) if support_path else None
    bounds = part.bounds.copy()
    if support is not None:
        bounds[0] = np.minimum(bounds[0], support.bounds[0])
        bounds[1] = np.maximum(bounds[1], support.bounds[1])
    height = float(bounds[1, 2] - bounds[0, 2])
    if height <= 0:
        raise ValueError("Модель не имеет высоты.")
    count = math.ceil(height / layer_height)
    scale = 10000.0
    destination = Path(cls_path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            header = (
                f"CONCEPT Laser Slice File version 0004;"
                f"PSZX={dimensions[0]:.3f};PSZY={dimensions[1]:.3f};NIIO=0;IIOD=0.100;NOIO=0;OIOD=0.100;"
                f"SLTH={layer_height:.3f};BCMP=0.000;SKTH=INF;"
                f"BOUNDS=({bounds[0][0]:.3f},{bounds[0][1]:.3f},{bounds[0][2]:.3f})::"
                f"({bounds[1][0]:.3f},{bounds[1][1]:.3f},{bounds[1][2]:.3f});    "
            )
            stream.write(header.encode('ascii'))
            for layer in range(count):
                if cancel_callback and cancel_callback(): return False
                low = bounds[0, 2] + layer * layer_height
                high = min(low + layer_height, bounds[1, 2])
                z = (low + high) / 2.0
                section = part.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                supp_section = support.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1]) if support is not None else None
                stream.write(b'NEW_LAYER' + struct.pack('<iii', int(z * scale), 1, 1))
                _write_borders(stream, section, scale)
                stream.write(b'NEW_BORDER' + pack_border([]))
                stream.write(b'INC_OFFSETS' + pack_int(0) + pack_int(0))
                x, y = (dimensions * scale / 2).astype(int)
                stream.write(b'NEW_QUADRANT' + struct.pack('<iiii', -x, x, -y, y))
                stream.write(b'NEW_SKIN' + pack_int(1))
                stream.write(b'NEW_ISLAND' + struct.pack('<iiii', *(int(v * scale) for v in bounds[:, :2].T.ravel())))
                _write_borders(stream, section, scale)
                stream.write(b'NEW_CORE' + pack_int(0) + b'SUPPORT')
                _write_borders(stream, supp_section, scale)
                stream.write(b'NEW_BORDER' + pack_border([]))
                if progress_callback: progress_callback(min(99, int((layer + 1) / count * 100)))
            stream.flush()
            os.fsync(stream.fileno())
        if cancel_callback and cancel_callback(): return False
        os.replace(temporary, destination)
        if progress_callback: progress_callback(100)
        return True
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
