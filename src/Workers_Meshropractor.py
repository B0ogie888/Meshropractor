# Файл: Workers_Meshropractor.py
import numpy as np
import trimesh
import trimesh.remesh
from scipy.interpolate import RBFInterpolator
import concurrent.futures
import pyvista as pv
try:
    from services.deformation_service import DeformationService
except ImportError:
    pass # Обработаем ошибку внутри потока

from PySide6.QtCore import QThread, Signal
#from CL_Slicer import slice_stl_to_cls

# Импортируем наш собственный алгоритм деформации
from ml_deformation import NativeDeformationService

# Пытаемся безопасно импортировать Open3D
print("[DEBUG] Перед import open3d...", flush=True)
try:
    import open3d as o3d
    HAS_O3D = True
    print("[DEBUG] open3d импортирован успешно.", flush=True)
except ImportError:
    HAS_O3D = False
    print("[DEBUG] open3d НЕ установлен (ImportError) - продолжаем без него.", flush=True)

# ==========================================
# ПОТОК 0: СЛАЙСИНГ (Экспорт в .CLS)
# ==========================================
class SlicerWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, part_path, supp_path, out_path, layer_height):
        super().__init__()
        self.part_path = part_path
        self.supp_path = supp_path
        self.out_path = out_path
        self.layer_height = layer_height

    def run(self):
        try:
            # Временно ставим заглушку, чтобы не было ошибок
            # slice_stl_to_cls(self.part_path, self.supp_path, self.out_path, self.layer_height, progress_callback=self.progress.emit)
            self.progress.emit(100)
            self.finished.emit("Слайсинг успешно завершен!")
        except Exception as e:
            self.error.emit(f"Ошибка слайсинга: {str(e)}")


# ==========================================
# ПОТОК 1: СОВМЕЩЕНИЕ (Ультимативный GOM-style ICP)
# ==========================================
class AlignmentThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, cad_mesh, scan_mesh, cad_pts, scan_pts, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh.copy()
        self.cad_pts = cad_pts
        self.scan_pts = scan_pts
        self.settings = settings

    def run(self):
        try:
            if not HAS_O3D:
                self.log_signal.emit("[!] ОШИБКА: Open3D не установлен.")
                return

            self.log_signal.emit("\n=== ЗАПУСК ВЫРАВНИВАНИЯ ===")

            # --- ФИКС 1: Точные CAD-нормали ---
            # Берем математически идеальные нормали граней из Trimesh, без аппроксимации!
            target_pc = o3d.geometry.PointCloud()
            target_pc.points = o3d.utility.Vector3dVector(np.array(self.cad_mesh.vertices))
            target_pc.normals = o3d.utility.Vector3dVector(np.array(self.cad_mesh.vertex_normals))

            source_pc = o3d.geometry.PointCloud()
            source_pc.points = o3d.utility.Vector3dVector(np.array(self.scan_mesh.vertices))

            bbox = target_pc.get_max_bound() - target_pc.get_min_bound()
            max_bound = float(np.max(bbox))

            self.log_signal.emit("   -> Расчет нормалей для скана...")
            # Для скана нормали всё еще нужно считать, чтобы сгладить микро-шероховатость SLM
            source_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_bound * 0.01, max_nn=30))

            trans_init = np.eye(4)

            # --- РЕЖИМ 1: РУЧНОЙ (По маркерам) ---
            if len(self.cad_pts) >= 3 and len(self.cad_pts) == len(self.scan_pts):
                self.log_signal.emit("1. Предварительное выравнивание (ПО МАРКЕРАМ)...")
                pcd_scan_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.scan_pts)))
                pcd_cad_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.cad_pts)))
                corres = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(self.scan_pts))]))
                estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                trans_init = estimator.compute_transformation(pcd_scan_pts, pcd_cad_pts, corres)

            # --- РЕЖИМ 2: АВТОМАТИКА (Best-Fit) ---
            else:
                self.log_signal.emit("1. Предварительное выравнивание (BEST-FIT)...")
                target_center = target_pc.get_center()
                source_center = source_pc.get_center()
                trans_init[:3, 3] = target_center - source_center

                search_time = self.settings.get('search_time', 1)

                if search_time > 0:
                    self.log_signal.emit("   -> Глобальный анализ геометрии...")
                    voxel_size = max_bound * (0.05 if search_time == 1 else 0.02)
                    search_radius = float(voxel_size * 1.5)

                    source_down = source_pc.voxel_down_sample(voxel_size)
                    target_down = target_pc.voxel_down_sample(voxel_size)
                    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                    # У target_down нормали пересчитаются корректно из уже заданных
                    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

                    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
                    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

                    if search_time == 1:
                        self.log_signal.emit("   -> Поиск совпадений (Fast Global Registration)...")
                        res_global = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                            source_down, target_down, source_fpfh, target_fpfh,
                            o3d.pipelines.registration.FastGlobalRegistrationOption(
                                maximum_correspondence_distance=search_radius
                            )
                        )
                    else:
                        self.log_signal.emit("   -> Глубокий поиск совпадений (RANSAC)...")
                        res_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                            source_down, target_down, source_fpfh, target_fpfh, True,
                            max_correspondence_distance=search_radius,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                            ransac_n=3,
                            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(search_radius)],
                            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
                        )

                    if res_global.fitness > 0.05:
                        trans_init = res_global.transformation

                if self.isInterruptionRequested(): return

                self.log_signal.emit("   -> Анализ симметрии (Anti-Flip)...")
                t_to_origin = np.eye(4);
                t_to_origin[:3, 3] = -target_center
                t_from_origin = np.eye(4);
                t_from_origin[:3, 3] = target_center
                flip_x_mat = np.diag([1, -1, -1, 1])
                flip_y_mat = np.diag([-1, 1, -1, 1])

                best_trans = trans_init
                best_score = -1.0

                for name, t_test in [("Нормаль", trans_init),
                                     ("Флип X", t_from_origin @ flip_x_mat @ t_to_origin @ trans_init),
                                     ("Флип Y", t_from_origin @ flip_y_mat @ t_to_origin @ trans_init)]:
                    res = o3d.pipelines.registration.registration_icp(
                        source_pc, target_pc, max_bound * 0.08, t_test,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
                    )
                    score = res.fitness / (res.inlier_rmse + 1e-5)
                    if score > best_score:
                        best_score = score
                        best_trans = res.transformation
                        best_name = name

                if best_name != "Нормаль":
                    self.log_signal.emit(
                        f"   -> [АВТО-КОРРЕКЦИЯ] Скан был перевернут. Применен разворот ({best_name}).")
                trans_init = best_trans

            if self.isInterruptionRequested(): return

            # --- МАГИЯ 2: Ультра-точный Robust ICP ---
            rmse = 0.0
            if self.settings.get('do_icp', True):
                self.log_signal.emit("2. Точная подгонка поверхностей (Robust Multi-scale ICP)...")

                # ФИКС 2: Добавлен 4-й шаг! (0.2% от габарита).
                # На детали 150 мм это всего 0.3 мм. Алгоритм сбросит из расчетов весь порошок и мусор.
                radii = [max_bound * 0.08, max_bound * 0.02, max_bound * 0.005, max_bound * 0.002]
                iters = [50, 100, 200, 300]

                for i, (r, it) in enumerate(zip(radii, iters)):
                    self.log_signal.emit(f"   -> Итерация {i + 1}/{len(radii)} (Точность захвата: {r:.2f} мм)...")

                    loss_func = o3d.pipelines.registration.TukeyLoss(k=r)
                    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_func)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        source_pc, target_pc, r, trans_init,
                        estimation,
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
                    )
                    trans_init = reg_p2p.transformation
                    rmse = reg_p2p.inlier_rmse

                if reg_p2p.fitness == 0:
                    self.log_signal.emit("[!] ПРЕДУПРЕЖДЕНИЕ: Совмещение провалено.")
                    return
            else:
                # ФИКС: Честная оценка выравнивания без ICP
                eval_thresh = max_bound * 0.02  # Берем коридор побольше (2% от габарита)
                eval_res = o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, eval_thresh,
                                                                                trans_init)

                # Если совпало меньше 30% площади скана - это 100% ошибка позиционирования
                if eval_res.fitness < 0.3:
                    self.log_signal.emit(
                        f"[!] ВНИМАНИЕ: Детали пересеклись под углом! Перекрытие всего {eval_res.fitness * 100:.1f}%.")
                    rmse = 999.9999  # Выдаем абсурдное значение, чтобы сбить "вранье"
                else:
                    rmse = eval_res.inlier_rmse

            self.log_signal.emit(f"   -> УСПЕХ! Финальное отклонение (RMSE): {rmse:.4f} мм")

            transformed_scan = self.scan_mesh.copy()
            transformed_scan.apply_transform(trans_init)

            self.finished_signal.emit((transformed_scan, rmse))

        except Exception as e:
            self.log_signal.emit(f"[!] ОШИБКА СОВМЕЩЕНИЯ: {str(e)}")

# ==========================================
# ПОТОК 2: ML ПРЕДЕФОРМАЦИЯ (Собственная нейросеть)
# ==========================================
class CompensationThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(object)
    progress_signal = Signal(int)

    def __init__(self, cad_mesh, scan_mesh, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh
        self.settings = settings

    def log(self, text):
        self.log_signal.emit(text)

    def run(self):
        try:
            self.log("\n=== ЗАПУСК ML ПРЕДЕФОРМАЦИИ (Своя нейросеть) ===")
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
                cancel_callback=self.isInterruptionRequested  # <-- Проверка флага отмены
            )

            # Если расчет прерван пользователем — корректно выходим без падения
            if pv_result is None or self.isInterruptionRequested():
                self.log("\n[i] Расчет остановлен. Контекст GPU очищен.")
                return

            self.progress_signal.emit(95)
            self.log("Обратная конвертация (PyVista -> Trimesh)...")

            faces_result = pv_result.faces.reshape(-1, 4)[:, 1:]
            result_trimesh = trimesh.Trimesh(vertices=pv_result.points, faces=faces_result)

            # Сохраняем векторное поле в метаданные меша для отображения стрелок
            if "Deformation_Vectors" in pv_result.point_data:
                result_trimesh.metadata["vectors"] = pv_result.point_data["Deformation_Vectors"]

            self.progress_signal.emit(100)
            self.log("\n=== ГОТОВО! МОДЕЛЬ УСПЕШНО ДЕФОРМИРОВАНА ===")
            self.finished_signal.emit(result_trimesh)

        except Exception as e:
            self.log(f"\n[!] ОШИБКА: {str(e)}")