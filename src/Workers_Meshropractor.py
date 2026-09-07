"""Qt workers: numerical work only, no scene mutations."""
import logging
from copy import deepcopy
import numpy as np
import trimesh
import pyvista as pv
from PySide6.QtCore import QThread, Signal
from ml_deformation import NativeDeformationService
from CL_Slicer import slice_stl_to_cls
from project_store import validate_mesh

# ==========================================
# ПОТОК 0: СЛАЙСИНГ (Экспорт в .CLS)
# ==========================================
class SlicerWorker(QThread):
    progress = Signal(int)
    finished_signal = Signal(object)
    error = Signal(str)

    def __init__(self, part_path, supp_path, out_path, layer_height):
        super().__init__()
        self.part_path = part_path
        self.supp_path = supp_path
        self.out_path = out_path
        self.layer_height = layer_height

    def run(self):
        try:
            slice_stl_to_cls(self.part_path, self.supp_path, self.out_path, self.layer_height,
                             progress_callback=self.progress.emit, cancel_callback=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.finished_signal.emit(self.out_path)
        except Exception as e:
            self.error.emit(f"Ошибка слайсинга: {str(e)}")


# ==========================================
# ПОТОК 1: ЖЁСТКОЕ СОВМЕЩЕНИЕ ПОВЕРХНОСТЕЙ
# ==========================================
class AlignmentThread(QThread):
    error = Signal(str)
    log_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, cad_mesh, scan_mesh, cad_pts, scan_pts, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh.copy()
        self.cad_pts = deepcopy(cad_pts)
        self.scan_pts = deepcopy(scan_pts)
        self.settings = deepcopy(settings)

    def run(self):
        from alignment import align_surfaces
        try:
            mesh, metrics = align_surfaces(self.cad_mesh, self.scan_mesh, self.cad_pts, self.scan_pts,
                                           self.settings, self.log_signal.emit, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.finished_signal.emit((mesh, metrics["rmse"]))
        except InterruptedError:
            self.log_signal.emit("Совмещение отменено.")
        except Exception as exc:
            logging.exception("Alignment failed")
            self.error.emit(f"Ошибка совмещения: {exc}")

# ==========================================
# ПОТОК 2: ML ПРЕДЕФОРМАЦИЯ (Собственная нейросеть)
# ==========================================
class CompensationThread(QThread):
    error = Signal(str)
    log_signal = Signal(str)
    finished_signal = Signal(object)
    progress_signal = Signal(int)

    def __init__(self, cad_mesh, scan_mesh, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh.copy()
        self.settings = deepcopy(settings)

    def log(self, text):
        self.log_signal.emit(text)

    def run(self):
        try:
            self.log("\n=== ЗАПУСК ML ПРЕДЕФОРМАЦИИ (Своя нейросеть) ===")
            validate_mesh(self.cad_mesh)
            validate_mesh(self.scan_mesh)
            self.progress_signal.emit(5)

            # Вызываем НАШ алгоритм из файла ml_deformation.py
            deformer = NativeDeformationService()

            if self.isInterruptionRequested():
                self.log("\n[i] Расчет отменен.")
                return

            self.log("Конвертация геометрии (Trimesh -> PyVista)...")
            faces_cad = np.pad(self.cad_mesh.faces, ((0, 0), (1, 0)), constant_values=3)
            pv_cad = pv.PolyData(self.cad_mesh.vertices, faces_cad)

            faces_scan = np.pad(self.scan_mesh.faces, ((0, 0), (1, 0)), constant_values=3)
            pv_scan = pv.PolyData(self.scan_mesh.vertices, faces_scan)

            def progress_cb(percent):
                self.progress_signal.emit(int(percent))

            def log_cb(msg):
                self.log(msg)

            # Запуск НАШЕЙ нейросети с передачей коллбека отмены
            self.log("Обучение модели и деформация...")
            pv_result = deformer.create_deformed_model(
                source_mesh=pv_cad,
                target_mesh=pv_scan,
                max_dev=self.settings.get('limit', 5.0),
                factor=self.settings.get('factor', 1.0),
                deformation_type=self.settings.get('def_type', 1),
                is_compensation=self.settings.get('is_comp', True),
                progress_callback=progress_cb,
                log_callback=log_cb,
                cancel_callback=self.isInterruptionRequested,
                sample_count=self.settings.get('points', 20000),
                min_coverage=self.settings.get('min_coverage', 0.3),
                seed=self.settings.get('seed', 42)
            )

            # Если расчет прерван пользователем — корректно выходим без падения
            if pv_result is None or self.isInterruptionRequested():
                self.log("\n[i] Расчет остановлен. Контекст GPU очищен.")
                return

            self.progress_signal.emit(99)
            self.log("Обратная конвертация (PyVista -> Trimesh)...")

            faces_result = pv_result.faces.reshape(-1, 4)[:, 1:]
            result_trimesh = trimesh.Trimesh(vertices=pv_result.points, faces=faces_result, process=False)

            # Сохраняем векторное поле в метаданные меша для отображения стрелок
            if "Deformation_Vectors" in pv_result.point_data:
                result_trimesh.metadata["vectors"] = np.asarray(pv_result.point_data["Deformation_Vectors"]).copy()
                result_trimesh.metadata["vector_origins"] = np.asarray(pv_cad.points).copy()
                result_trimesh.metadata["support_distance"] = np.asarray(pv_result.point_data["Support_Distance"]).copy()
                result_trimesh.metadata["settings"] = deepcopy(self.settings)
                result_trimesh.metadata["quality"] = deformer.last_quality

            self.progress_signal.emit(100)
            self.log("\n=== ГОТОВО! МОДЕЛЬ УСПЕШНО ДЕФОРМИРОВАНА ===")
            self.finished_signal.emit(result_trimesh)

        except Exception as e:
            logging.exception("Deformation failed")
            self.error.emit(f"Ошибка расчёта: {e}")
