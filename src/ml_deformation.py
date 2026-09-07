import math
from copy import deepcopy
import trimesh
from scipy.spatial import cKDTree
from geometry_analysis import validate_deformation, sample_surface
from typing import Callable, Optional, Sequence, Union

import numpy as np
import open3d as o3d
import pyvista as pv
import torch
import torch.nn as nn
import torch.optim as optim


class FourierFeatureEncoding(nn.Module):
    """
    Positional Encoding (Fourier Features) для снятия спектрального смещения (spectral bias).
    Позволяет компактным MLP улавливать высокочастотные деформации геометрии.
    """
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_freqs

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        features = [p] if self.include_input else []
        for freq in self.freq_bands:
            features.append(torch.sin(p * freq * math.pi))
            features.append(torch.cos(p * freq * math.pi))
        return torch.cat(features, dim=-1)


# 1. МЯГКО (Soft) - Высокое сглаживание шумов сканирования
class SoftNetwork(nn.Module):
    def __init__(self, num_freqs: int = 2):
        super().__init__()
        self.encoding = FourierFeatureEncoding(num_freqs)
        self.net = nn.Sequential(
            nn.Linear(self.encoding.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(x))


# 2. НОРМАЛЬНО (Medium) - Баланс точности и плавности
class MediumNetwork(nn.Module):
    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.encoding = FourierFeatureEncoding(num_freqs)
        self.net = nn.Sequential(
            nn.Linear(self.encoding.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(x))


# 3. ЖЕСТКО (Hard) - Максимальная детализация для сложного профиля лопаток
class HardNetwork(nn.Module):
    def __init__(self, num_freqs: int = 8):
        super().__init__()
        self.encoding = FourierFeatureEncoding(num_freqs)
        self.net = nn.Sequential(
            nn.Linear(self.encoding.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(x))


_DEFAULT_WEIGHT_DECAY = {0: 0.0, 1: 1e-7, 2: 1e-6}
_DEFAULT_SMOOTHNESS_WEIGHT = {0: 0.0, 1: 0.0, 2: 0.0005}


class NativeDeformationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_deviations_o3d(self, source_mesh, target_mesh, max_dev, log_callback=None, cancel_callback=None):
        # Open3D CPU raycasting works with standard wheels; PyTorch training may use CUDA.
        if log_callback:
            log_callback("   -> Open3D Raycasting: CPU; расчёт лучей пакетами")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(target_mesh.points, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(target_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32))))
        origins = np.asarray(source_mesh.points, dtype=np.float32)
        normals = np.asarray(source_mesh.point_normals, dtype=np.float32)
        out_chunks, in_chunks = [], []
        for start in range(0, len(origins), 100_000):
            if cancel_callback and cancel_callback():
                return np.empty((0, 3)), np.empty((0, 3))
            points, directions = origins[start:start + 100_000], normals[start:start + 100_000]
            out_chunks.append(scene.cast_rays(o3d.core.Tensor(np.hstack((points, directions))))['t_hit'].numpy())
            in_chunks.append(scene.cast_rays(o3d.core.Tensor(np.hstack((points, -directions))))['t_hit'].numpy())
        hit_out, hit_in = np.concatenate(out_chunks), np.concatenate(in_chunks)
        mask_out = np.isfinite(hit_out) & (hit_out < max_dev)
        mask_in = np.isfinite(hit_in) & (hit_in < max_dev)

        both_valid = mask_out & mask_in
        closer_out = both_valid & (hit_out < hit_in)
        closer_in = both_valid & ~(hit_out < hit_in)

        use_out = (mask_out & ~mask_in) | closer_out
        use_in = (mask_in & ~mask_out) | closer_in
        confirmed = use_out | use_in

        deviations = np.zeros_like(origins, dtype=np.float32)
        deviations[use_out] = normals[use_out] * hit_out[use_out].reshape(-1, 1)
        deviations[use_in] = -normals[use_in] * hit_in[use_in].reshape(-1, 1)

        valid_points = origins[confirmed].copy()
        deviations = deviations[confirmed]

        return valid_points, deviations

    def _predict_in_batches(self, model: nn.Module, all_pts_tensor, batch_size: int, cancel_callback=None) -> np.ndarray:
        """Батчинг предсказания с ускорением AMP и защитой от переполнения VRAM."""
        model.eval()
        chunks = []
        n_points = all_pts_tensor.shape[0]
        use_cuda = (self.device.type == "cuda")

        with torch.no_grad():
            for start in range(0, n_points, batch_size):
                if cancel_callback and cancel_callback(): return None
                end = min(start + batch_size, n_points)
                batch = torch.as_tensor(all_pts_tensor[start:end], dtype=torch.float32, device=self.device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    batch_pred = model(batch)
                chunks.append(batch_pred.to(dtype=torch.float32).cpu())

        return torch.cat(chunks, dim=0).numpy()

    @staticmethod
    def _smoothness_penalty(model: nn.Module, xb: torch.Tensor, pred_original: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Штраф за резкие градиенты без повторного прямого прохода."""
        perturbation = torch.randn_like(xb) * epsilon
        pred_perturbed = model(xb + perturbation)
        local_grad = (pred_perturbed - pred_original) / epsilon
        return torch.mean(local_grad ** 2)

    @staticmethod
    def build_deformation_glyphs(mesh: pv.PolyData, deviations: np.ndarray, stride: int = 50, scale_factor: float = 1.0) -> pv.PolyData:
        """
        Генерирует векторное поле стрелок (Glyphs) для визуализации направления усадки/компенсации.
        """
        pts = mesh.points[::stride]
        vecs = deviations[::stride]
        cloud = pv.PolyData(pts)
        cloud["vectors"] = vecs
        cloud["magnitude"] = np.linalg.norm(vecs, axis=1)
        arrows = cloud.glyph(orient="vectors", scale="magnitude", factor=scale_factor)
        return arrows

    def create_deformed_model(self, source_mesh: pv.PolyData, target_mesh: pv.PolyData,
                              max_dev: float = 5.0,
                              factor: Union[float, Sequence[float], np.ndarray] = 1.0,
                              deformation_type: int = 1,
                              is_compensation: bool = True,
                              progress_callback: Optional[Callable[[int], None]] = None,
                              log_callback: Optional[Callable[[str], None]] = None,
                              cancel_callback: Optional[Callable[[], bool]] = None,
                              repair_mesh: bool = True,
                              train_batch_size: int = 16384,
                              predict_batch_size: int = 100_000,
                              early_stop_patience: int = 60,
                              early_stop_min_delta: float = 1e-6,
                              weight_decay: Optional[float] = None,
                              smoothness_weight: Optional[float] = None,
                              smoothness_epsilon: float = 0.01,
                              positional_encoding_freqs: Optional[int] = None,
                              sample_count: int = 20000,
                              min_coverage: float = 0.3,
                              seed: int = 42,
                              epochs: int = 600,
                              target_rmse: float = 0.005) -> Optional[pv.PolyData]:
        """
        Главный пайплайн предеформации и компенсации:
        - Поддержка безопасной отмены (cancel_callback)
        - Анизотропный фактор деформации (factor: float или [Fx, Fy, Fz])
        - Автоматическая валидация и исправление топологии сетки (repair_mesh)
        """
        if max_dev <= 0 or not np.isfinite(max_dev) or not 0 < min_coverage <= 1:
            raise ValueError("Предел отклонений должен быть положительным, покрытие — от 0 до 100%.")
        if sample_count < 0 or epochs < 1 or train_batch_size < 1 or predict_batch_size < 1:
            raise ValueError("Некорректные параметры дискретизации или обучения.")
        factors = np.asarray(factor, dtype=float)
        if factors.size not in (1, 3) or not np.isfinite(factors).all():
            raise ValueError("Коэффициент должен быть числом или тройкой конечных чисел.")
        torch.manual_seed(seed)
        self.last_quality = {}
        if log_callback:
            if self.device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                log_callback(f"[Аппаратное ускорение] GPU: {gpu_name} (Свободно VRAM: {vram_free:.2f} / {vram_total:.2f} ГБ)")
            else:
                log_callback("[Аппаратное ускорение] CUDA недоступна. Расчет выполняется на CPU.")

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        if log_callback:
            log_callback("1. Поиск точных отклонений (Open3D Raycasting)...")
        if progress_callback:
            progress_callback(10)

        sampling_mesh = sample_surface(source_mesh, sample_count, seed)
        train_pts, dev_vectors = self._compute_deviations_o3d(sampling_mesh, target_mesh, max_dev, log_callback, cancel_callback)

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        if len(train_pts) < 10:
            raise ValueError("Слишком мало точек пересечения. Проверьте первичное совмещение моделей.")

        total_cad_points = len(sampling_mesh.points)
        coverage_pct = 100.0 * len(train_pts) / max(total_cad_points, 1)
        if log_callback:
            log_callback(f"Найдено {len(train_pts)} точек с подтвержденным отклонением "
                         f"из {total_cad_points} точек CAD (покрытие: {coverage_pct:.1f}%).")
        if progress_callback:
            progress_callback(25)

        if coverage_pct / 100.0 < min_coverage:
            raise ValueError(f"Недостаточное покрытие: {coverage_pct:.1f}%, требуется {min_coverage:.1%}. Проверьте совмещение и предел поиска.")
        self.last_quality = {"coverage_percent": coverage_pct, "sample_count": total_cad_points, "confirmed_count": len(train_pts), "seed": seed}
        pts_mean = np.mean(train_pts, axis=0)
        pts_scale = np.max(np.abs(train_pts - pts_mean)) + 1e-5

        X_norm = (train_pts - pts_mean) / pts_scale
        Y_norm = dev_vectors * 10.0

        if weight_decay is None:
            weight_decay = _DEFAULT_WEIGHT_DECAY.get(deformation_type, 1e-6)
        if smoothness_weight is None:
            smoothness_weight = _DEFAULT_SMOOTHNESS_WEIGHT.get(deformation_type, 0.0)

        network_kwargs = {}
        if positional_encoding_freqs is not None:
            network_kwargs["num_freqs"] = positional_encoding_freqs

        if deformation_type == 0:
            model = SoftNetwork(**network_kwargs).to(self.device)
            if log_callback: log_callback("Выбрана 'Мягкая' нейросеть (Высокое сглаживание)")
        elif deformation_type == 2:
            model = HardNetwork(**network_kwargs).to(self.device)
            if log_callback:
                log_callback(f"Выбрана 'Жесткая' нейросеть (Точное копирование). "
                             f"Smoothness-штраф: {smoothness_weight:.3f}")
        else:
            model = MediumNetwork(**network_kwargs).to(self.device)
            if log_callback: log_callback("Выбрана 'Нормальная' нейросеть (Баланс)")

        order = np.random.default_rng(seed).permutation(len(X_norm))
        split = max(1, int(len(order) * 0.1))
        val_indices, train_indices = order[:split], order[split:]
        X_val = torch.tensor(X_norm[val_indices], dtype=torch.float32, device=self.device)
        Y_val = torch.tensor(Y_norm[val_indices], dtype=torch.float32, device=self.device)
        X = torch.tensor(X_norm[train_indices], dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y_norm[train_indices], dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]

        batch_size = max(1, min(train_batch_size, n_samples))
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
        criterion = nn.MSELoss()

        use_cuda = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        best_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        if log_callback:
            log_callback(f"Обучение нейросети (AMP: {'Вкл' if use_cuda else 'Выкл'}, Батч: {batch_size})...")

        model.train()
        for epoch in range(epochs):
            if cancel_callback and cancel_callback():
                if log_callback: log_callback("[!] Расчет прерван пользователем на этапе обучения.")
                return None

            epoch_loss_sum = 0.0
            perm = torch.randperm(n_samples, device=self.device)

            for i in range(0, n_samples, batch_size):
                if cancel_callback and cancel_callback(): return None
                idx = perm[i:i + batch_size]
                xb = X[idx]
                yb = Y[idx]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    if smoothness_weight > 0:
                        loss = loss + smoothness_weight * self._smoothness_penalty(model, xb, pred, smoothness_epsilon)

                if not torch.isfinite(loss):
                    raise ValueError("Обучение потеряло численную устойчивость. Результат не создан.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_sum += loss.item() * xb.size(0)

            scheduler.step()
            epoch_loss = epoch_loss_sum / n_samples

            if epoch % 25 == 0:
                if progress_callback:
                    progress = 30 + int((epoch / epochs) * 55)
                    progress_callback(progress)
                if log_callback:
                    log_callback(f"   -> Эпоха {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

            model.eval()
            with torch.no_grad():
                validation_loss = criterion(model(X_val), Y_val).item()
            model.train()
            if not math.isfinite(validation_loss):
                raise ValueError("Некорректная ошибка на контрольной выборке.")
            if best_loss - validation_loss > early_stop_min_delta:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if math.sqrt(validation_loss) / 10.0 <= target_rmse or epochs_without_improvement >= early_stop_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.last_quality["validation_rmse_mm"] = math.sqrt(best_loss) / 10.0
        if log_callback:
            log_callback(f"Контрольная RMSE по компонентам: {self.last_quality['validation_rmse_mm']:.4f} мм")

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        if log_callback:
            log_callback("3. Генерация сглаженного поля деформации для узлов сетки...")
        if progress_callback:
            progress_callback(90)

        all_pts_norm = (source_mesh.points - pts_mean) / pts_scale
        smooth_deviations = self._predict_in_batches(model, all_pts_norm, predict_batch_size, cancel_callback)
        if smooth_deviations is None: return None
        smooth_deviations = smooth_deviations / 10.0
        if not np.isfinite(smooth_deviations).all():
            raise ValueError("Поле деформации содержит некорректные значения.")

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        # Анизотропный фактор деформации [Fx, Fy, Fz]
        if isinstance(factor, (list, tuple, np.ndarray)):
            factor_arr = np.asarray(factor, dtype=np.float32).reshape(1, 3)
        else:
            factor_arr = float(factor)

        if is_compensation:
            final_points = source_mesh.points - (smooth_deviations * factor_arr)
        else:
            final_points = source_mesh.points + (smooth_deviations * factor_arr)

        result_mesh = source_mesh.copy()
        result_mesh.points = final_points

        # Сохраняем векторное поле в саму сетку для последующей визуализации стрелками
        result_mesh["Deformation_Vectors"] = final_points - source_mesh.points
        result_mesh["Support_Distance"] = cKDTree(train_pts).query(source_mesh.points, workers=-1)[0]
        self.last_quality["max_support_distance_mm"] = float(np.max(result_mesh["Support_Distance"]))
        self.last_quality["max_displacement_mm"] = float(np.linalg.norm(final_points - source_mesh.points, axis=1).max())

        if repair_mesh:
            if cancel_callback and cancel_callback(): return None
            if log_callback: log_callback("Проверка вырожденных треугольников и самопересечений...")
            validate_deformation(source_mesh, result_mesh)
            if cancel_callback and cancel_callback(): return None
            result_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, split_vertices=False)

        if progress_callback:
            progress_callback(100)
        return result_mesh
