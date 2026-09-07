"""Support parameters, preview and manual placement share the same geometry engine."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox, QCheckBox, QDialogButtonBox, QLabel
from background_tasks import FunctionWorker
from support_geometry import generate_supports, overhang_faces, validate, RayWorld, column

DEFAULTS = dict(angle=45., spacing=3., diameter=1., tip_diameter=.4, tip_height=1.,
                foot_diameter=2., foot_height=.5, base_z=0., only_platform=False)


class SupportDialog(QDialog):
    def __init__(self, parent, params, action, has_faces):
        super().__init__(parent)
        self.setWindowTitle(['Генерация поддержек', 'Поддержки для выбранных деталей', 'Ветвящиеся поддержки',
                             'Ручная установка поддержек', 'Области поддержек'][action])
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)
        self.fields = {}
        for key, label, low, high, suffix in [
            ('angle', 'Угол нависания от горизонтали', .1, 89.9, '°'),
            ('spacing', 'Шаг контактов', .2, 100, ' мм'), ('diameter', 'Диаметр стойки', .1, 30, ' мм'),
            ('tip_diameter', 'Диаметр контакта', .05, 30, ' мм'), ('tip_height', 'Высота сужения', .1, 30, ' мм'),
            ('foot_diameter', 'Диаметр основания', .1, 50, ' мм'), ('foot_height', 'Высота основания', .1, 30, ' мм'),
            ('base_z', 'Высота платформы Z', -100000, 100000, ' мм')]:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setValue(params[key])
            spin.setSuffix(suffix)
            self.fields[key] = spin
            self.form.addRow(label, spin)
            if action == 4 and key != 'angle': self.form.setRowVisible(spin, False)
            if action == 3 and key in ('spacing', 'angle'): self.form.setRowVisible(spin, False)
        self.only_platform = QCheckBox('Только до платформы: пропускать контакты над препятствиями')
        self.only_platform.setChecked(params['only_platform'])
        layout.addWidget(self.only_platform)
        self.only_platform.setVisible(action != 4)
        self.use_faces = QCheckBox('Только выделенные поверхности')
        self.use_faces.setEnabled(has_faces)
        self.use_faces.setChecked(has_faces)
        self.use_faces.setVisible(action != 3)
        layout.addWidget(self.use_faces)
        note = QLabel('Поддержки строятся вниз по Z. Без ограничения до платформы стойка заканчивается на ближайшей поверхности снизу. Поддержки хранятся внутри исходной детали; действие можно отменить.')
        if action == 2: note.setText(note.text() + '\nСобственный алгоритм ветвления; при препятствии используется отдельная стойка.')
        if action == 3: note.setText('После нажатия «Указать» щёлкайте по поверхности выбранной детали. Каждый щелчок создаёт стойку. Esc или инструмент выбора завершает режим.')
        if action == 4: note.setText('Красным подсвечиваются направленные вниз грани с заданным углом. Повторное нажатие кнопки предпросмотра или Esc снимает подсветку.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QLabel()
        self.error.setStyleSheet('color: #f19d81;')
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText('Показать' if action == 4 else 'Указать' if action == 3 else 'Генерировать')
        buttons.button(QDialogButtonBox.Cancel).setText('Отмена')
        buttons.accepted.connect(self.validate_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self):
        return dict({key: field.value() for key, field in self.fields.items()}, only_platform=self.only_platform.isChecked())

    def validate_accept(self):
        try: validate(self.parameters())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()


class SupportTools:
    def __init__(self, window, workspace):
        self.window, self.workspace = window, workspace
        self.params = dict(DEFAULTS)
        self.panel = None

    def refresh_panel(self):
        if self.panel and self.panel.isVisible() and not self.panel.loading: self.panel.refresh()

    def clear_preview(self):
        if self.workspace.plotter:
            for actor, _ in self.workspace.preview.values(): self.workspace.plotter.remove_actor(actor)
            self.workspace.preview.clear()
            self.workspace.plotter.render()

    def records(self):
        return [dict(row=r, mesh=self.window.slicer_parts[r]['mesh'].copy(),
                     filename=self.window.slicer_parts[r]['filename'], platform=self.window.slicer_parts[r].get('platform'))
                for r in self.workspace.visible_rows(False)]

    def open(self, action):
        if self.window._busy(): return
        if action == 3:
            if self.panel is None:
                from support_panel import SupportPanel
                self.panel = SupportPanel(self)
                self.window.ui.slicer_left_layout.insertWidget(1, self.panel)
            self.panel.open()
            return
        if action == 4 and self.workspace.preview:
            self.clear_preview()
            return
        if self.window.ui.slicer_plotter is None:
            self.window.log('Сначала загрузите детали.')
            return
        self.workspace.attach(self.window.ui.slicer_plotter)
        rows = self.workspace.visible_rows(action != 0)
        if not rows:
            self.window.log('Выберите видимые детали в текущей сцене.')
            return
        dialog = SupportDialog(self.window, self.params, action, any(self.workspace.selection.get(r) for r in rows))
        if not dialog.exec():
            dialog.deleteLater()
            return
        self.params = dialog.parameters()
        targets = {r: sorted(self.workspace.selection.get(r, set())) if dialog.use_faces.isChecked() else None for r in rows}
        dialog.deleteLater()
        self.clear_preview()
        if action == 4:
            self.preview_regions(targets)
        else:
            self.workspace.manual = None
            self.window.start_job(FunctionWorker(generate_supports, self.records(), targets, dict(self.params), action == 2), self.append_results)

    def preview_regions(self, targets):
        import numpy as np
        count = 0
        for row, selected in targets.items():
            faces = overhang_faces(self.window.slicer_parts[row]['mesh'], self.params['angle'])
            if selected is not None: faces = np.intersect1d(faces, selected)
            self.workspace.add_overlay(row, faces, self.workspace.preview, '#ed6251', 'support_region')
            count += len(faces)
        self.workspace.refresh_overlays()
        self.workspace.plotter.render()
        self.window.log(f'Области поддержек: {count} треугольников. Повторное нажатие скрывает предпросмотр.')

    def append_results(self, results):
        if not results:
            self.window.log('Нет подходящих контактов: проверьте угол, выделенные поверхности и высоту платформы.')
            return
        window = self.window
        window.flush_history()
        before = window.capture_project()
        try:
            from part_supports import make_group, remove_actors
            for result in results:
                row = result['row']
                groups = window.slicer_parts[row].setdefault('supports', [])
                group = make_group(result['mesh'], result.get('surface_faces', []), result.get('kind', 'Точечные'),
                                   result.get('params', self.params), result['contacts'])
                replace_id = result.get('replace_id')
                if replace_id:
                    index = next(i for i, item in enumerate(groups) if item['id'] == replace_id)
                    remove_actors(window, [groups[index]])
                    group['id'] = replace_id
                    groups[index] = group
                else: groups.append(group)
            window.refresh_scene_visibility()
            window.update_info_combobox()
            window.update_parts_table_filter(window.ui.scene_tabs.currentIndex())
            window.mark_dirty()
            window.flush_history('Создать поддержки')
            window.log(f"Создано контактов поддержек: {sum(r['contacts'] for r in results)}. Поддержки сохранены внутри исходных деталей.")
            self.refresh_panel()
        except Exception:
            window.restore_project(before)
            raise

    def manual_at(self, position):
        state = self.workspace.manual
        hit = self.workspace.picker(position, selected_only=False, rows=state['rows'])
        if hit is None or hit[0] not in state['rows']: return
        row, face, point = hit
        record = next(r for r in state['records'] if r['row'] == row)
        if record['mesh'].face_normals[face, 2] >= -1e-6:
            self.window.log('Для вертикальной поддержки укажите поверхность, направленную вниз. Поверните вид или используйте сечения.')
            return
        try:
            platform = record['platform']
            if platform not in state['worlds']:
                state['worlds'][platform] = RayWorld([r for r in state['records'] if r['platform'] == platform])
            bottom = state['worlds'][platform].bottom(point, self.params['base_z'], self.params['only_platform'])
            if bottom is None:
                self.window.log('В указанном месте недостаточно места или путь до платформы перекрыт.')
                return
            mesh = column(bottom, point, self.params)
            self.append_results([dict(mesh=mesh, filename=record['filename'], platform=platform, contacts=1, row=row,
                                      surface_faces=[face], kind='Точечные', params=dict(self.params))])
        except Exception as exc:
            self.window.log(f'Не удалось создать поддержку: {exc}')
