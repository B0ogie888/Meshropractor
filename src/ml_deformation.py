import math
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

    def _get_o3d_device(self):
        """Определяет доступность CUDA для тензорного ядра Open3D без вывода C++ предупреждений."""
        if self.device.type == "cuda":
            try:
                # Проверка сборки Open3D без провоцирования исключений в C++ ядре
                if hasattr(o3d.core, 'cuda') and o3d.core.cuda.is_available():
                    return o3d.core.Device("CUDA:0")
            except Exception:
                pass
        return o3d.core.Device("CPU:0")

    def _compute_deviations_o3d(self, source_mesh: pv.PolyData, target_mesh: pv.PolyData, max_dev: float, log_callback=None):
        """
        Векторизованная аппаратная трассировка лучей с автоматическим выбором GPU/CPU.
        """
        o3d_dev = self._get_o3d_device()
        if log_callback:
            dev_type = "NVIDIA CUDA" if o3d_dev.get_type() == o3d.core.Device.DeviceType.CUDA else "CPU (Многопоточный)"
            log_callback(f"   -> Open3D Raycasting: модуль запущен на {dev_type}")

        try:
            tgt_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(target_mesh.points, dtype=np.float32), device=o3d_dev),
                o3d.core.Tensor(np.array(target_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32), device=o3d_dev)
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tgt_tmesh)

            origins = source_mesh.points.astype(np.float32)
            normals = source_mesh.point_normals.astype(np.float32)

            rays_out = np.hstack((origins, normals)).astype(np.float32)
            rays_in = np.hstack((origins, -normals)).astype(np.float32)

            hit_out = scene.cast_rays(o3d.core.Tensor(rays_out, device=o3d_dev))['t_hit'].to(o3d.core.Device("CPU:0")).numpy()
            hit_in = scene.cast_rays(o3d.core.Tensor(rays_in, device=o3d_dev))['t_hit'].to(o3d.core.Device("CPU:0")).numpy()
        except Exception as e:
            if log_callback:
                log_callback(f"   -> [Fallback CPU] Переключение Raycasting на CPU из-за: {e}")
            cpu_dev = o3d.core.Device("CPU:0")
            tgt_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(target_mesh.points, dtype=np.float32), device=cpu_dev),
                o3d.core.Tensor(np.array(target_mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32), device=cpu_dev)
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tgt_tmesh)
            origins = source_mesh.points.astype(np.float32)
            normals = source_mesh.point_normals.astype(np.float32)
            hit_out = scene.cast_rays(o3d.core.Tensor(np.hstack((origins, normals)).astype(np.float32), device=cpu_dev))['t_hit'].numpy()
            hit_in = scene.cast_rays(o3d.core.Tensor(np.hstack((origins, -normals)).astype(np.float32), device=cpu_dev))['t_hit'].numpy()

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

    def _predict_in_batches(self, model: nn.Module, all_pts_tensor: torch.Tensor, batch_size: int) -> np.ndarray:
        """Батчинг предсказания с ускорением AMP и защитой от переполнения VRAM."""
        model.eval()
        chunks = []
        n_points = all_pts_tensor.shape[0]
        use_cuda = (self.device.type == "cuda")

        with torch.no_grad():
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    batch_pred = model(all_pts_tensor[start:end])
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
                              early_stop_rel_threshold: float = 0.01,
                              weight_decay: Optional[float] = None,
                              smoothness_weight: Optional[float] = None,
                              smoothness_epsilon: float = 0.01,
                              positional_encoding_freqs: Optional[int] = None) -> Optional[pv.PolyData]:
        """
        Главный пайплайн предеформации и компенсации:
        - Поддержка безопасной отмены (cancel_callback)
        - Анизотропный фактор деформации (factor: float или [Fx, Fy, Fz])
        - Автоматическая валидация и исправление топологии сетки (repair_mesh)
        """
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

        train_pts, dev_vectors = self._compute_deviations_o3d(source_mesh, target_mesh, max_dev, log_callback)

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        if len(train_pts) < 10:
            raise ValueError("Слишком мало точек пересечения. Проверьте первичное совмещение моделей.")

        total_cad_points = len(source_mesh.points)
        coverage_pct = 100.0 * len(train_pts) / max(total_cad_points, 1)
        if log_callback:
            log_callback(f"Найдено {len(train_pts)} точек с подтвержденным отклонением "
                         f"из {total_cad_points} точек CAD (покрытие: {coverage_pct:.1f}%).")
        if progress_callback:
            progress_callback(25)

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

        X = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y_norm, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]

        batch_size = max(1, min(train_batch_size, n_samples))
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
        criterion = nn.MSELoss()

        use_cuda = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        best_loss = float("inf")
        epochs_without_improvement = 0
        initial_loss = None
        epochs = 600

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
                idx = perm[i:i + batch_size]
                xb = X[idx]
                yb = Y[idx]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    if smoothness_weight > 0:
                        loss = loss + smoothness_weight * self._smoothness_penalty(model, xb, pred, smoothness_epsilon)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_sum += loss.item() * xb.size(0)

            scheduler.step()
            epoch_loss = epoch_loss_sum / n_samples

            if initial_loss is None:
                initial_loss = epoch_loss

            if epoch % 25 == 0:
                if progress_callback:
                    progress = 30 + int((epoch / epochs) * 55)
                    progress_callback(progress)
                if log_callback:
                    log_callback(f"   -> Эпоха {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

            if initial_loss > 0 and epoch_loss < initial_loss * early_stop_rel_threshold:
                if log_callback:
                    log_callback(f"   -> Достигнута сходимость: Loss снижен до 1% от начального (Эпоха {epoch}).")
                break

            if best_loss - epoch_loss > early_stop_min_delta:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    if log_callback:
                        log_callback(f"   -> Early Stopping: плато градиента {early_stop_patience} эпох подряд (Эпоха {epoch}).")
                    break

        if cancel_callback and cancel_callback():
            if log_callback: log_callback("[!] Расчет отменен пользователем.")
            return None

        if log_callback:
            log_callback("3. Генерация сглаженного поля деформации для узлов сетки...")
        if progress_callback:
            progress_callback(90)

        all_pts_norm = (source_mesh.points - pts_mean) / pts_scale
        all_pts_tensor = torch.tensor(all_pts_norm, dtype=torch.float32, device=self.device)

        smooth_deviations = self._predict_in_batches(model, all_pts_tensor, predict_batch_size) / 10.0

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
        result_mesh["Deformation_Vectors"] = smooth_deviations

        # Валидация нормалей без разрушения топологии и индексов вершин
        if repair_mesh:
            try:
                result_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True,
                                            auto_orient_normals=True)
            except Exception as e:
                if log_callback:
                    log_callback(f"   -> Предупреждение при расчете нормалей: {e}")

        if progress_callback:
            progress_callback(100)
        return result_mesh