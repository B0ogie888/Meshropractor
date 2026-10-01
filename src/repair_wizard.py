"""Reviewable diagnostics/repair of every loaded project mesh."""
import json
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QMessageBox, QFileDialog, QPlainTextEdit)
from background_tasks import FunctionWorker
from mesh_diagnostics import diagnose_mesh, serializable_report


ROWS = [('faces', 'Треугольники'), ('boundary', 'Открытые рёбра'),
        ('contours', 'Замкнутые контуры отверстий'), ('branched_boundaries', 'Разветвлённые границы'),
        ('nonmanifold', 'Неманифолдные рёбра (более двух граней)'),
        ('nonmanifold_vertices', 'Неманифолдные вершины'),
        ('duplicates', 'Дубли треугольников'), ('degenerate', 'Вырожденные треугольники'),
        ('overlaps', 'Треугольники с площадными нахлёстами'),
        ('intersections', 'Пересекающиеся треугольники'), ('components', 'Связные фрагменты'),
        ('noise_components', 'Возможные фрагменты шума'), ('unreferenced', 'Неиспользуемые вершины'),
        ('winding', 'Согласованное направление граней'), ('closed', 'Замкнутая сетка'),
        ('negative_volume', 'Обратная ориентация замкнутой оболочки')]


def repair_targets(window):
    targets = []
    for row, part in enumerate(window.slicer_parts):
        targets.append(dict(label=f"Слайсер · {part.get('platform') or 'Без платформы'} · {part['filename']}",
                            kind='Part', scope='part', key=row, mesh=part['mesh']))
        from part_supports import support_mesh
        for group in part.get('supports', []):
            if not len(group['faces']): continue
            targets.append(dict(label=f"  ↳ {part['filename']} · поддержка {group['kind']} ({group['id'][:8]})",
                                kind='Part', scope='support', key=(row, group['id']), mesh=support_mesh(group)))
    for key, record in window.scene_models.items():
        if record.get('mesh') is None: continue
        targets.append(dict(label=f"Предеформация · {record['kind']} · {record.get('name', key)}",
                            kind=record['kind'], scope='model', key=key, mesh=record['mesh']))
    return targets


class RepairWizard(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.targets = repair_targets(window)
        self.source = self.repaired = self.report = self.before = None
        self.running = False
        self.setWindowTitle('Мастер исправлений · диагностика моделей')
        self.resize(800, 760)
        layout = QVBoxLayout(self)
        row = QHBoxLayout(); layout.addLayout(row)
        row.addWidget(QLabel('Текущая модель:'))
        self.models = QComboBox(); self.models.addItems([x['label'] for x in self.targets]); row.addWidget(self.models, 1)
        self.full = QCheckBox('Полный анализ: нахлёсты и пересечения'); self.full.setChecked(True); layout.addWidget(self.full)
        self.table = QTableWidget(len(ROWS), 3)
        self.table.setHorizontalHeaderLabels(['Диагностика', 'Исходная', 'После лечения'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setMinimumHeight(len(ROWS)*22+26)
        for i, (_, name) in enumerate(ROWS): self.table.setItem(i, 0, QTableWidgetItem(name))
        layout.addWidget(self.table, 1)
        self.result_label = QLabel(''); self.result_label.setWordWrap(True); layout.addWidget(self.result_label)
        options = QGroupBox('Параметры исправления'); form = QFormLayout(options); layout.addWidget(options)
        self.mode = QComboBox(); self.mode.addItems(['Полное: швы, отверстия, нахлёсты, пересечения', 'Бережное: швы и малые отверстия'])
        self.passes = QSpinBox(); self.passes.setRange(1, 10); self.passes.setValue(3)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setDecimals(3); self.tolerance.setRange(.001, 10); self.tolerance.setValue(.05); self.tolerance.setSuffix(' мм')
        self.hole = QDoubleSpinBox(); self.hole.setDecimals(3); self.hole.setRange(.001, 10); self.hole.setValue(.1); self.hole.setSuffix(' мм')
        self.optimize = QCheckBox('Оптимизировать избыточную триангуляцию'); self.optimize.setChecked(True)
        self.target_faces = QSpinBox(); self.target_faces.setRange(10000, 5000000); self.target_faces.setValue(400000); self.target_faces.setSingleStep(100000)
        form.addRow('Способ:', self.mode); form.addRow('Максимум проходов:', self.passes)
        form.addRow('Допуск контроля формы:', self.tolerance); form.addRow('Размер малых отверстий:', self.hole)
        form.addRow(self.optimize); form.addRow('Граней после оптимизации (ориентир):', self.target_faces)
        self.note = QLabel('Полное лечение закрывает все отверстия. Проверка формы: все вершины и выборка точек поверхности; это не строгая граница ошибки. Фрагменты автоматически не удаляются.')
        self.note.setWordWrap(True); layout.addWidget(self.note)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(70); layout.addWidget(self.log)
        row = QHBoxLayout(); layout.addLayout(row)
        self.analyze = QPushButton('Обновить диагностику'); row.addWidget(self.analyze)
        self.repair = QPushButton('Подготовить исправление'); row.addWidget(self.repair)
        self.cancel = QPushButton('Остановить'); row.addWidget(self.cancel)
        row = QHBoxLayout(); layout.addLayout(row)
        self.export_report = QPushButton('Сохранить отчёт…'); row.addWidget(self.export_report)
        self.export_mesh = QPushButton('Сохранить результат как…'); row.addWidget(self.export_mesh)
        self.apply = QPushButton('Применить к модели'); row.addWidget(self.apply)
        self.close_button = QPushButton('Закрыть'); row.addWidget(self.close_button)
        self.models.currentIndexChanged.connect(self.select_model)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.optimize.toggled.connect(self.mode_changed)
        self.analyze.clicked.connect(self.analyze_model)
        self.repair.clicked.connect(self.repair_model)
        self.cancel.clicked.connect(window.cancel_current_job)
        self.apply.clicked.connect(self.apply_result)
        self.export_report.clicked.connect(self.save_report)
        self.export_mesh.clicked.connect(self.save_mesh)
        self.close_button.clicked.connect(self.reject)
        self.select_model()

    def mode_changed(self):
        complete = self.mode.currentIndex() == 0
        self.passes.setMaximum(10 if complete else 3)
        self.tolerance.setEnabled(complete and not self.running)
        self.hole.setEnabled(not complete and not self.running and self.target['kind'] != 'Scan')
        self.optimize.setEnabled(complete and not self.running)
        self.target_faces.setEnabled(complete and self.optimize.isChecked() and not self.running)

    @property
    def target(self):
        return self.targets[self.models.currentIndex()]

    def select_model(self):
        if not self.targets: return
        self.source = self.target['mesh']; self.repaired = self.report = self.before = None
        self.mode.setCurrentIndex(1 if self.target['kind'] == 'Scan' else 0)
        for i in range(len(ROWS)):
            for j in (1, 2): self.table.setItem(i, j, QTableWidgetItem('—'))
        self.log.clear()
        self.result_label.setText('Выберите «Обновить диагностику» или «Подготовить исправление». Анализ больших сеток занимает несколько минут.')
        self.set_running(False)

    def set_running(self, running):
        self.running = running
        for control in (self.models, self.full, self.analyze, self.passes, self.mode, self.close_button): control.setEnabled(not running)
        self.repair.setEnabled(not running and self.target['kind'] != 'Heatmap')
        self.cancel.setEnabled(running)
        self.export_report.setEnabled(not running and self.before is not None)
        self.export_mesh.setEnabled(not running and self.repaired is not None)
        self.apply.setEnabled(not running and self.repaired is not None and self.report['changed'] and self.report.get('acceptable', True))
        self.mode_changed()

    def run(self, function, callback, *args, **kwargs):
        if self.window._busy(): return
        worker = FunctionWorker(function, *args, with_progress=True, **kwargs)
        worker.progress.connect(self.log.appendPlainText)
        worker.progress.connect(self.result_label.setText)
        worker.error.connect(self.log.appendPlainText)
        worker.finished.connect(lambda: QTimer.singleShot(0, lambda: self.set_running(False)))
        def ready(value):
            self.window._job_next = lambda: callback(value)
        if self.window.start_job(worker, ready): self.set_running(True)

    def analyze_model(self):
        self.run(diagnose_mesh, self.show_diagnostics, self.source, full=self.full.isChecked())

    def show_diagnostics(self, report):
        self.before = report; self.repaired = self.report = None
        self.fill_column(report, 1); self.fill_column({}, 2)
        self.result_label.setText('По полной проверке дефектов не найдено.' if report['clean'] else
                                 'Есть дефекты или непроверенные категории. Счётчики нахлёстов и пересечений показывают число участвующих граней.')
        self.set_running(False)

    def fill_column(self, report, column):
        for i, (key, _) in enumerate(ROWS):
            value = report.get(key)
            label = 'Не проверено' if value is None else ('Да' if value else 'Нет') if isinstance(value, bool) else f'{value:,}'.replace(',', ' ')
            item = QTableWidgetItem(label)
            if value is not None and key not in ('faces', 'components'):
                good = bool(value) if key in ('winding', 'closed') else not bool(value)
                item.setForeground(QColor('#81d49a' if good else '#f18c81'))
            self.table.setItem(i, column, item)

    def repair_model(self):
        complete = self.mode.currentIndex() == 0
        message = ('Полное лечение перестраивает дефектные участки и закрывает все отверстия, включая конструктивные и отверстия скана. '
                   if complete else 'Бережное лечение сшивает совпадающие вершины и закрывает только малые отверстия CAD. ')
        message += 'На больших моделях расчёт может занять несколько минут. После расчёта вы сможете проверить отчёт перед применением. Начать?'
        if QMessageBox.question(self, 'Подготовить лечение?', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes: return
        if complete:
            from mesh_full_repair import prepare_full_repair
            self.run(prepare_full_repair, self.show_repair, self.source, self.target['kind'], passes=self.passes.value(),
                     tolerance_mm=self.tolerance.value(), before=self.before if self.before and self.before.get('full') else None,
                     target_faces=self.target_faces.value() if self.optimize.isChecked() else 0)
        else:
            from mesh_repair import prepare_repair
            self.run(prepare_repair, self.show_repair, self.source, self.target['kind'], passes=self.passes.value(), max_hole_mm=self.hole.value())

    def show_repair(self, value):
        self.repaired, self.report = value
        self.before = self.report['before']
        self.fill_column(self.before, 1); self.fill_column(self.report['after'], 2)
        text = 'Дефекты по полной проверке устранены.' if self.report['after'].get('clean') else 'Остаются дефекты или непроверенные категории; результат не считается полностью исправным.'
        if 'shape' in self.report:
            text += f" Контроль формы: максимум {self.report['shape']['max_mm']:.6f} мм; допуск {self.report['tolerance_mm']:g} мм."
        if not self.report.get('acceptable', True): text += ' Допуск превышен: применение заблокировано. Измените способ или явно задайте другой допуск и повторите.'
        self.result_label.setText(text)
        self.set_running(False)

    def apply_result(self):
        if self.repaired is None or not self.report.get('acceptable', True): return
        message = 'Применить показанный результат к выбранной модели? Действие можно отменить через Ctrl+Z.'
        if self.target['scope'] == 'part':
            message += ' Геометрия поддержек сохранится; привязки к прежним номерам треугольников будут сброшены.'
        if QMessageBox.question(self, 'Применить исправление?', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes: return
        self.window.apply_wizard_repair(self.target, self.source, self.repaired)
        self.source = self.repaired; self.target['mesh'] = self.source
        self.repaired = None
        self.before = self.report['after']
        self.report = None
        self.fill_column(self.before, 1); self.fill_column({}, 2)
        self.result_label.setText('Исправление применено. Исходную модель можно вернуть через Ctrl+Z.')
        self.set_running(False)

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Отчёт диагностики', 'diagnostics.json', 'JSON (*.json)')
        if not path: return
        data = dict(model=self.target['label'], before=serializable_report(self.before))
        if self.report:
            data.update(after=serializable_report(self.report['after']), shape=self.report.get('shape'), passes=self.report.get('passes'))
        try: Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError as exc: QMessageBox.warning(self, 'Не удалось сохранить', str(exc))

    def save_mesh(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить исправленную копию', 'repaired.stl', 'STL (*.stl);;PLY (*.ply)')
        if not path: return
        try: self.repaired.export(path)
        except Exception as exc: QMessageBox.warning(self, 'Не удалось сохранить', str(exc))

    def reject(self):
        if self.running:
            self.window.cancel_current_job()
            self.result_label.setText('Отмена запрошена. Дождитесь остановки текущего шага.')
            return
        super().reject()
