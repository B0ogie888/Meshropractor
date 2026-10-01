"""Conservative mesh repair proposals; source meshes are never modified."""
from copy import deepcopy
import numpy as np
import trimesh
from project_store import validate_mesh, load_mesh


def inspect_mesh(mesh):
    counts = np.bincount(mesh.edges_unique_inverse)
    return dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
                boundary=int(np.count_nonzero(counts == 1)),
                nonmanifold=int(np.count_nonzero(counts > 2)),
                duplicates=int(np.count_nonzero(~mesh.unique_faces())),
                degenerate=int(np.count_nonzero(~mesh.nondegenerate_faces(height=1e-12))),
                winding=bool(mesh.is_winding_consistent), closed=bool(mesh.is_watertight))


def small_caps(mesh, diameter, max_vertices=4, cancelled=lambda: False):
    """Cap isolated, small, consistently oriented planar convex boundaries."""
    counts = np.bincount(mesh.edges_unique_inverse)
    boundary_mask = counts[mesh.edges_unique_inverse] == 1
    boundary = mesh.edges[boundary_mask]
    boundary_faces = mesh.edges_face[boundary_mask]
    incident = {}
    for (a, b), face in zip(boundary, boundary_faces):
        incident.setdefault(int(a), set()).add(int(face))
        incident.setdefault(int(b), set()).add(int(face))
    adjacent = {}
    directed = set(map(tuple, boundary.tolist()))
    for a, b in boundary:
        adjacent.setdefault(int(a), set()).add(int(b))
        adjacent.setdefault(int(b), set()).add(int(a))
    seen, caps, holes = set(), [], 0
    for first in adjacent:
        if cancelled(): raise InterruptedError('Лечение отменено')
        if first in seen: continue
        component, pending = [], [first]
        while pending:
            vertex = pending.pop()
            if vertex in seen: continue
            seen.add(vertex); component.append(vertex)
            pending.extend(adjacent[vertex] - seen)
        if not 3 <= len(component) <= max_vertices or any(len(adjacent[v]) != 2 for v in component): continue
        loop, previous, current = [first], None, first
        for _ in range(len(component) - 1):
            nxt = next(v for v in adjacent[current] if v != previous and v not in loop)
            loop.append(nxt); previous, current = current, nxt
        points = mesh.vertices[loop]
        if np.linalg.norm(points[:, None] - points[None, :], axis=2).max() > diameter: continue
        # Original boundary must have consistent winding around the entire loop.
        directions = [(loop[i], loop[(i+1) % len(loop)]) in directed for i in range(len(loop))]
        if not (all(directions) or not any(directions)): continue
        if all(directions): loop.reverse(); points = mesh.vertices[loop]
        if np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0])) <= 1e-15: continue
        if len(loop) > 3:
            normals = np.cross(np.roll(points, -1, axis=0) - points,
                               np.roll(points, -2, axis=0) - np.roll(points, -1, axis=0))
            n = normals[0]; length = np.linalg.norm(n)
            if length <= 1e-15 or np.any(normals @ n <= 0): continue
            if np.max(np.abs((points - points[0]) @ (n / length))) > 1e-6: continue
            # Every vertex must lie inside every oriented edge half-plane.
            # This also rejects self-crossing polygons with consistent local turns.
            for i in range(len(points)):
                side = np.cross(points[(i+1) % len(points)] - points[i], points - points[i]) @ (n / length)
                if np.min(side) < -1e-12: break
            else:
                side = None
            if side is not None: continue
        faces = [[loop[0], loop[i], loop[i+1]] for i in range(1, len(loop)-1)]
        existing = {tuple(sorted(f)) for f in mesh.faces[list(incident[first])]}
        if any(tuple(sorted(f)) in existing for f in faces): continue
        caps.extend(faces); holes += 1
    return caps, holes


def prepare_repair(source, kind='CAD', max_hole_mm=0.1, passes=3,
                   progress=lambda message: None, cancelled=lambda: False,
                   full=False, tolerance_mm=.05):
    if full:
        from mesh_full_repair import prepare_full_repair
        return prepare_full_repair(source, kind, passes, tolerance_mm, progress, cancelled)
    if not 1 <= passes <= 3 or not np.isfinite(max_hole_mm) or not 0 <= max_hole_mm <= 10:
        raise ValueError('Некорректные параметры лечения')
    def stage(message):
        if cancelled(): raise InterruptedError('Лечение отменено')
        progress(message)
    stage('Проверка топологии модели…')
    validate_mesh(source)
    before = inspect_mesh(source)
    stage('Сшивка совпадающих вершин и удаление дефектных граней…')
    mesh = trimesh.Trimesh(vertices=source.vertices.copy(), faces=source.faces.copy(), process=False)
    mesh.metadata = deepcopy(source.metadata)
    mesh.merge_vertices(digits_vertex=6, merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    holes, pass_results = 0, []
    if kind != 'Scan' and before['boundary'] and max_hole_mm > 0:
        for index, limit in enumerate((4, 12, 32)[:passes], 1):
            stage(f'Лечение: проход {index}/{passes}, отверстия до {max_hole_mm:g} мм и {limit} рёбер…')
            if mesh.is_watertight: break
            caps, count = small_caps(mesh, max_hole_mm, limit, cancelled)
            if caps: mesh.faces = np.vstack((mesh.faces, caps))
            holes += count
            pass_results.append(dict(pass_number=index, holes=count, faces=len(caps)))
    stage('Проверка результата лечения…')
    validate_mesh(mesh)
    after = inspect_mesh(mesh)
    changed = (before['vertices'] != after['vertices'] or before['faces'] != after['faces']
               or before['duplicates'] > 0 or before['degenerate'] > 0 or holes > 0)
    if not changed: mesh = source.copy()
    defects = (any(before[k] for k in ('nonmanifold', 'duplicates', 'degenerate'))
               or (kind != 'Scan' and before['boundary'] > 0) or not before['winding'])
    report = dict(before=before, after=after, holes=holes, changed=changed,
                  defects=defects, kind=kind, max_hole_mm=max_hole_mm, passes=pass_results)
    return mesh, report


def load_with_repair(path, linear, angle, kind, options=None,
                     progress=lambda message: None, cancelled=lambda: False):
    progress('Загрузка модели…')
    source = load_mesh(path, linear, angle)
    if cancelled(): raise InterruptedError('Загрузка отменена')
    repaired, report = prepare_repair(source, kind, **(options or {}), progress=progress, cancelled=cancelled)
    return source, repaired, report


def choose_repair(parent, report):
    """Return apply / keep / cancel. Scan boundaries are deliberately preserved."""
    from PySide6.QtWidgets import QMessageBox
    before, after = report['before'], report['after']
    dialog = QMessageBox(parent)
    dialog.setWindowTitle('Проверка и автоисправление модели')
    lines = ['Проверка завершена. Исходный файл не изменяется.', '', 'До → после исправления:']
    for key, label in [('boundary','Открытые рёбра'), ('nonmanifold','Неманифолдные рёбра'),
                       ('duplicates','Повторные грани'), ('degenerate','Вырожденные грани')]:
        lines.append(f'{label}: {before[key]} → {after[key]}')
    lines.append(f"Замкнутая сетка: {'да' if before['closed'] else 'нет'} → {'да' if after['closed'] else 'нет'}")
    lines.append(f"Согласованная ориентация граней: {'да' if after['winding'] else 'нет'}")
    if report.get('full'):
        for key, label in [('overlaps', 'Грани с нахлёстами'), ('intersections', 'Пересекающиеся грани')]:
            lines.append(f'{label}: {before[key]} → {after[key]}')
        lines.append('Полная проверка: ' + ('дефектов не найдено.' if after.get('clean') else 'остаются дефекты.'))
        if 'shape' in report:
            lines.append(f"Контроль формы по вершинам и выборке поверхности: максимум {report['shape']['max_mm']:.6f} мм.")
        if not report.get('acceptable', True): lines.append('Допуск формы превышен: применение заблокировано. Используйте мастер исправлений для настройки и проверки.')
    else:
        lines.append(f"Закрыто малых отверстий: {report['holes']}")
        lines.append('Совпадающие вершины сшиваются с точностью 0,000001 мм.')
        lines.append('Отверстия скана сохранены.' if report['kind'] == 'Scan' else
                     f"Закрываются только малые плоские выпуклые отверстия размером до {report['max_hole_mm']:g} мм.")
        for item in report.get('passes', []):
            lines.append(f"Проход {item['pass_number']}: закрыто отверстий {item['holes']}")
        lines.append('Нахлёсты и пересечения здесь не проверены. Для полного анализа откройте мастер исправлений.')
    lines.append('Результат можно отменить через Ctrl+Z.')
    dialog.setText('\n'.join(lines))
    apply = dialog.addButton('Применить исправление', QMessageBox.AcceptRole)
    apply.setEnabled(report['changed'] and report.get('acceptable', True))
    keep = dialog.addButton('Оставить исходную', QMessageBox.NoRole)
    dialog.addButton('Отмена', QMessageBox.RejectRole)
    dialog.setDefaultButton(keep)
    dialog.exec()
    return 'apply' if dialog.clickedButton() is apply else 'keep' if dialog.clickedButton() is keep else 'cancel'


def request_repair(parent, kind='CAD', importing=True):
    """Ask before *any* topology analysis. False skips, None cancels, dict opts in."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QCheckBox
    dialog = QDialog(parent)
    dialog.setWindowTitle('Проверить и подготовить лечение модели?')
    layout = QVBoxLayout(dialog)
    text = QLabel('Проверка больших сеток может занять десятки секунд или минуты.\n'
                  'Запустить проверку и подготовку лечения?\n'
                  'Перед изменением модели вы увидите отчёт и сможете отказаться.')
    layout.addWidget(text)
    form = QFormLayout(); layout.addLayout(form)
    count = QSpinBox(); count.setRange(1, 3); count.setValue(3)
    size = QDoubleSpinBox(); size.setRange(.001, 10); size.setDecimals(3); size.setValue(.1); size.setSuffix(' мм')
    form.addRow('Максимум проходов:', count)
    form.addRow('Максимальный размер отверстия:', size)
    full = QCheckBox('Полное лечение: также устранить нахлёсты и пересечения (дольше)')
    layout.addWidget(full)
    tolerance = QDoubleSpinBox(); tolerance.setRange(.001, 10); tolerance.setDecimals(3); tolerance.setValue(.05); tolerance.setSuffix(' мм')
    form.addRow('Допуск контроля формы при полном лечении:', tolerance)
    tolerance.setEnabled(False)
    def mode_changed(enabled):
        tolerance.setEnabled(enabled)
        size.setEnabled(not enabled and kind != 'Scan')
        count.setEnabled(enabled or kind != 'Scan')
        count.setMaximum(10 if enabled else 3)
    full.toggled.connect(mode_changed)
    layout.addWidget(QLabel('Полное лечение закрывает все отверстия, включая конструктивные.\n'
                            'Плотная сетка оптимизируется до ~400 тысяч граней.\n'
                            'Форма проверяется по вершинам и выборке точек поверхности.\n'
                            'Настройка плотности без импорта — в мастере исправлений.'))
    if kind == 'Scan':
        size.setEnabled(False); count.setEnabled(False)
        layout.addWidget(QLabel('В бережном режиме отверстия скана сохраняются.\n'
                                'При выборе полного лечения открытые поверхности закрываются.'))
    else:
        layout.addWidget(QLabel('Малые конструктивные отверстия также могут попасть под лечение.\n'
                                'Проходы проверяют контуры до 4, 12 и 32 рёбер; размер не увеличивается.'))
    buttons = QDialogButtonBox(); layout.addWidget(buttons)
    run = buttons.addButton('Проверить и подготовить', QDialogButtonBox.AcceptRole)
    skip = buttons.addButton('Загрузить без лечения' if importing else 'Карта без лечения', QDialogButtonBox.NoRole)
    cancel = buttons.addButton('Отмена', QDialogButtonBox.RejectRole)
    skip.setDefault(True); run.setAutoDefault(False)
    result = [None]
    def accept():
        result[0] = dict(max_hole_mm=size.value(), passes=count.value())
        if full.isChecked(): result[0].update(full=True, tolerance_mm=tolerance.value())
        dialog.accept()
    run.clicked.connect(accept)
    skip.clicked.connect(lambda: (result.__setitem__(0, False), dialog.accept()))
    cancel.clicked.connect(dialog.reject)
    dialog.exec()
    return result[0]
