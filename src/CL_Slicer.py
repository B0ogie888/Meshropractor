import struct
import trimesh
import os


def pack_int(val): return struct.pack('<i', int(val))


def pack_border(points, is_closed=True):
    if not points: return pack_int(0)
    flag = 1 if is_closed else 0
    b = struct.pack('<i B i', 1, flag, len(points))
    for x, y in points:
        b += pack_int(x) + pack_int(y)
    return b


def slice_stl_to_cls(part_path, support_path, cls_path, layer_height=0.03, progress_callback=None):
    part_mesh = trimesh.load(part_path)
    supp_mesh = trimesh.load(support_path) if support_path and os.path.exists(support_path) else None

    # Габариты берем по основной детали
    bounds = part_mesh.bounds
    scale = 10000.0

    header_str = (
        f"CONCEPT Laser Slice File version 0004;"
        f"PSZX=220.000;PSZY=220.000;NIIO=0;IIOD=0.100;NOIO=0;OIOD=0.100;"
        f"SLTH={layer_height:.3f};BCMP=0.000;SKTH=INF;"
        f"BOUNDS=({bounds[0][0]:.3f},{bounds[0][1]:.3f},{bounds[0][2]:.3f})::({bounds[1][0]:.3f},{bounds[1][1]:.3f},{bounds[1][2]:.3f});    "
    )
    body = bytearray(header_str.encode('ascii'))

    z_min, z_max = bounds[0][2], bounds[1][2]
    z_current = z_min + (layer_height / 2.0)
    total_layers = int((z_max - z_min) / layer_height)
    layer_count = 0

    while z_current <= z_max:
        # Сечем деталь
        part_slice = part_mesh.section(plane_origin=[0, 0, z_current], plane_normal=[0, 0, 1])
        # Сечем поддержки (если есть)
        supp_slice = supp_mesh.section(plane_origin=[0, 0, z_current], plane_normal=[0, 0, 1]) if supp_mesh else None

        z_scaled = int(z_current * scale)
        body += b'NEW_LAYER' + struct.pack('<iii', z_scaled, 1, 1)

        # --- 1. ВНЕШНИЕ КОНТУРЫ ДЕТАЛИ ---
        if part_slice is not None:
            for polygon in part_slice.discrete:
                pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
                body += b'NEW_BORDER' + pack_border(pts, is_closed=True)
        body += b'NEW_BORDER' + pack_border([])

        # --- 2. ОСТРОВА И ЯДРО ---
        body += b'INC_OFFSETS' + pack_int(0) + pack_int(0)
        body += b'NEW_QUADRANT' + struct.pack('<iiii', -1100000, 1100000, -1100000, 1100000)
        body += b'NEW_SKIN' + pack_int(1)

        island_bounds = [
            int(bounds[0][0] * scale), int(bounds[1][0] * scale),
            int(bounds[0][1] * scale), int(bounds[1][1] * scale)
        ]
        body += b'NEW_ISLAND' + struct.pack('<iiii', *island_bounds)

        if part_slice is not None:
            for polygon in part_slice.discrete:
                pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
                body += b'NEW_BORDER' + pack_border(pts, is_closed=True)

        body += b'NEW_CORE' + pack_int(0)

        # --- 3. ПОДДЕРЖКИ ---
        body += b'SUPPORT'
        if supp_slice is not None:
            for polygon in supp_slice.discrete:
                pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
                body += b'NEW_BORDER' + pack_border(pts, is_closed=True)
        body += b'NEW_BORDER' + pack_border([])

        z_current += layer_height
        layer_count += 1

        # Обновляем прогресс-бар в интерфейсе
        if progress_callback and total_layers > 0:
            progress_callback(int((layer_count / total_layers) * 100))

    with open(cls_path, 'wb') as f:
        f.write(body)