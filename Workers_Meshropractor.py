# Файл: workers.py
import numpy as np
import trimesh
import trimesh.remesh
from scipy.interpolate import RBFInterpolator
import concurrent.futures

from PySide6.QtCore import QThread, Signal

# Пытаемся безопасно импортировать Open3D
try:
    import open3d as o3d

    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ==========================================
# ПОТОК 1: СОВМЕЩЕНИЕ (ICP - Iterative Closest Point)
# ==========================================
class AlignmentThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, cad_mesh, scan_mesh, cad_pts, scan_pts):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh.copy()
        self.cad_pts = cad_pts
        self.scan_pts = scan_pts

    def run(self):
        try:
            if not HAS_O3D:
                self.log_signal.emit("[!] ОШИБКА: Open3D не установлен. Совмещение невозможно.")
                return

            self.log_signal.emit("\n=== ЗАПУСК ICP СОВМЕЩЕНИЯ ===")
            source_pc = o3d.geometry.PointCloud()
            source_pc.points = o3d.utility.Vector3dVector(np.array(self.scan_mesh.vertices))

            target_pc = o3d.geometry.PointCloud()
            target_pc.points = o3d.utility.Vector3dVector(np.array(self.cad_mesh.vertices))

            trans_init = np.eye(4)

            # Грубое совмещение
            if len(self.cad_pts) >= 3 and len(self.cad_pts) == len(self.scan_pts):
                self.log_signal.emit("2. Грубое позиционирование (ПО МАРКЕРАМ)...")
                pcd_scan_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.scan_pts)))
                pcd_cad_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.cad_pts)))
                corres = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(self.scan_pts))]))
                estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                trans_init = estimator.compute_transformation(pcd_scan_pts, pcd_cad_pts, corres)
            else:
                self.log_signal.emit("2. Грубое позиционирование (ПО ЦЕНТРАМ МАСС)...")
                trans_init[:3, 3] = target_pc.get_center() - source_pc.get_center()

            # Тонкое совмещение
            self.log_signal.emit("3. Тонкое совмещение ICP (до 2000 итераций)...")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pc, target_pc, 5.0, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )

            self.log_signal.emit(f"   -> УСПЕХ! Точность наложения: {reg_p2p.fitness:.4f}")
            self.log_signal.emit("4. Применение матрицы к исходной сетке...")

            transformed_scan = self.scan_mesh.copy()
            transformed_scan.apply_transform(reg_p2p.transformation)
            self.finished_signal.emit(transformed_scan)

        except Exception as e:
            self.log_signal.emit(f"[!] ОШИБКА ICP: {str(e)}")


# ==========================================
# ПОТОК 2: ПРЕДЕФОРМАЦИЯ (RBF)
# ==========================================
class CompensationThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, cad_mesh, scan_mesh, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh
        self.settings = settings

    def log(self, text, replace=False):
        self.log_signal.emit(f"{'REPLACE_FLAG' if replace else ''}{text}")

    def run(self):
        try:
            if not HAS_O3D:
                self.log("\n[!] ОШИБКА: Open3D не установлен.")
                return

            self.log("\n=== ЗАПУСК ПРЕДЕФОРМАЦИИ (RBF) ===")
            cad_mesh, scan_mesh, s = self.cad_mesh, self.scan_mesh, self.settings

            # 1. Ремешинг
            if s["use_remesh"]:
                self.log(f"1. Умный Ремешинг (Шаг: {s['edge_len']} мм)...")
                v, f = trimesh.remesh.subdivide_to_size(cad_mesh.vertices, cad_mesh.faces, max_edge=s['edge_len'])
                cad_mesh = trimesh.Trimesh(vertices=v, faces=f)

            # 2. Маячки
            self.log(f"2. Установка маячков ({s['points']} шт.)...")
            ctrl_cad_verts, face_indices = trimesh.sample.sample_surface(cad_mesh, s['points'])
            ctrl_cad_normals = np.array(cad_mesh.face_normals[face_indices])

            # 3. Raycasting
            self.log("3. Аппаратная трассировка (Open3D Tensor Engine)...")
            scan_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(scan_mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.array(scan_mesh.faces, dtype=np.int32))
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(scan_tmesh)

            def get_hits_o3d(origins, directions, task_name):
                self.log(f"   -> {task_name}...")
                rays = np.hstack((origins, directions)).astype(np.float32)
                ans = scene.cast_rays(o3d.core.Tensor(rays))
                t_hit = ans['t_hit'].numpy()
                tri_idx = ans['primitive_ids'].numpy()
                valid = np.isfinite(t_hit)
                locs = origins[valid] + directions[valid] * t_hit[valid][:, None]
                return locs, np.where(valid)[0], tri_idx[valid]

            locs_out, ray_idx_out, tri_idx_out = get_hits_o3d(ctrl_cad_verts, ctrl_cad_normals, "Поиск утолщений")
            locs_in, ray_idx_in, tri_idx_in = get_hits_o3d(ctrl_cad_verts, -ctrl_cad_normals, "Поиск усадки")

            # 4. Анализ попаданий
            def process_hits(ray_origins, cad_normals, locs, index_ray, index_tri, mesh, strictness):
                if len(locs) == 0: return {}
                hit_distances = np.linalg.norm(locs - ray_origins[index_ray], axis=1)
                hit_normals = mesh.face_normals[index_tri]
                dots = np.sum(cad_normals[index_ray] * hit_normals, axis=1)
                valid_mask = (hit_distances < s['limit'] + 1.0) & (dots > strictness)

                best_hits = {}
                for r_idx, loc, dist in zip(index_ray[valid_mask], locs[valid_mask], hit_distances[valid_mask]):
                    if r_idx not in best_hits or dist < best_hits[r_idx][1]:
                        best_hits[r_idx] = (loc, dist)
                return best_hits

            hits_out = process_hits(ctrl_cad_verts, ctrl_cad_normals, locs_out, ray_idx_out, tri_idx_out, scan_mesh,
                                    s['norm'])
            hits_in = process_hits(ctrl_cad_verts, ctrl_cad_normals, locs_in, ray_idx_in, tri_idx_in, scan_mesh,
                                   s['norm'])

            self.log("   -> Анализ пустот и аномалий...")
            final_ctrl_cad, final_error_vectors = [], []

            for i in range(s['points']):
                best_loc, best_dist = None, float('inf')
                if i in hits_out and hits_out[i][1] < best_dist: best_loc, best_dist = hits_out[i]
                if i in hits_in and hits_in[i][1] < best_dist: best_loc, best_dist = hits_in[i]

                if best_loc is not None:
                    error_vec = best_loc - ctrl_cad_verts[i]
                    if np.linalg.norm(error_vec) > s['limit']:
                        final_ctrl_cad.append(ctrl_cad_verts[i])
                        final_error_vectors.append(np.zeros(3))
                    else:
                        final_ctrl_cad.append(ctrl_cad_verts[i])
                        final_error_vectors.append(error_vec)
                elif s['anchor']:
                    final_ctrl_cad.append(ctrl_cad_verts[i])
                    final_error_vectors.append(np.zeros(3))

            # 5. Математика RBF
            self.log("4. Расчет RBF-матрицы деформации...")
            if s['neighbors'] > 0:
                self.log(f"   -> Режим: Локальный (Влияние на {s['neighbors']} точек)")
                rbf = RBFInterpolator(np.array(final_ctrl_cad), np.array(final_error_vectors),
                                      kernel='thin_plate_spline', smoothing=s['smooth'], neighbors=s['neighbors'])
            else:
                self.log("   -> Режим: Глобальный (Максимальная гладкость, требует много ОЗУ!)")
                rbf = RBFInterpolator(np.array(final_ctrl_cad), np.array(final_error_vectors),
                                      kernel='thin_plate_spline', smoothing=s['smooth'])

            # 6. МНОГОПОТОЧНАЯ ДЕФОРМАЦИЯ
            self.log("5. Пакетная деформация (Многопоточный режим)...")
            cad_verts_all = np.array(cad_mesh.vertices)
            compensated_verts = np.zeros_like(cad_verts_all)
            chunk_size = 50000

            def process_rbf_chunk(start_idx):
                end_idx = min(start_idx + chunk_size, len(cad_verts_all))
                chunk = cad_verts_all[start_idx:end_idx]
                return start_idx, end_idx, chunk - (rbf(chunk) * 1.0)

            starts = list(range(0, len(cad_verts_all), chunk_size))
            total_chunks = len(starts)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_rbf_chunk, s) for s in starts]
                for future in concurrent.futures.as_completed(futures):
                    start_idx, end_idx, result_chunk = future.result()
                    compensated_verts[start_idx:end_idx] = result_chunk
                    completed += 1
                    self.log(f"   -> Прогресс: {int((completed / total_chunks) * 100)}%", replace=True)

            cad_mesh.vertices = compensated_verts
            self.log("\n=== ГОТОВО! МОДЕЛЬ УСПЕШНО ДЕФОРМИРОВАНА ===")
            self.finished_signal.emit(cad_mesh)

        except Exception as e:
            self.log(f"\n[!] ОШИБКА: {str(e)}")