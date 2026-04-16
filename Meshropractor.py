import sys
import numpy as np  # Библиотека для быстрых математических вычислений (матрицы, векторы)
import trimesh  # Библиотека для работы с 3D-сетками (загрузка STL, поиск нормалей)
import trimesh.remesh  # Модуль для перестроения сетки (умножения полигонов)
from scipy.interpolate import RBFInterpolator  # Математическое ядро для расчета плавной деформации

# Пытаемся безопасно импортировать Open3D
# Если на компьютере его нет, программа не вылетит, а просто пометит HAS_O3D = False
try:
    import open3d as o3d

    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# ==========================================
# ИМПОРТЫ ИНТЕРФЕЙСА (PySide6 - LGPL Лицензия)
# ==========================================
# Импортируем все необходимые "строительные блоки" для окон, кнопок и ползунков
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QSlider, QCheckBox, QFileDialog,
                               QGroupBox, QTextEdit, QScrollArea, QTabWidget, QGridLayout,
                               QSplitter, QColorDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QColor

# Импортируем движки для 3D-графики на экране
import pyvista as pv  # Обертка над VTK для красивой отрисовки
from pyvistaqt import QtInteractor  # Специальный виджет, встраивающий 3D-окно PyVista внутрь интерфейса PySide6


# ==========================================
# ПОТОК 1: СОВМЕЩЕНИЕ (ICP - Iterative Closest Point)
# ==========================================
# Мы наследуемся от QThread. Если запустить тяжелый расчет в главном потоке,
# окно программы "зависнет" и Windows напишет "Не отвечает". QThread решает это, считая всё в фоне.
class AlignmentThread(QThread):
    # Сигналы — это способ общения фонового потока с главным окном.
    # Поток не имеет права сам менять текст в окне, он может только "прокричать" (emit) сигнал.
    log_signal = Signal(str)  # Сигнал для передачи текста в консоль
    finished_signal = Signal(object)  # Сигнал для передачи готовой 3D-модели после завершения

    def __init__(self, cad_mesh, scan_mesh, cad_pts, scan_pts):
        super().__init__()
        # Обязательно делаем копии (.copy()), чтобы случайно не сломать оригинальные модели в памяти
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh.copy()
        self.cad_pts = cad_pts
        self.scan_pts = scan_pts

    def run(self):
        """Этот метод запускается автоматически при вызове thread.start()"""
        try:
            if not HAS_O3D:
                self.log_signal.emit("[!] ОШИБКА: Open3D не установлен. Совмещение невозможно.")
                return

            self.log_signal.emit("\n=== ЗАПУСК ICP СОВМЕЩЕНИЯ ===")

            # Конвертируем наши сетки (Trimesh) в Облака Точек (PointCloud) для Open3D
            # ICP работает именно с точками, а не с полигонами
            source_pc = o3d.geometry.PointCloud()
            source_pc.points = o3d.utility.Vector3dVector(np.array(self.scan_mesh.vertices))

            target_pc = o3d.geometry.PointCloud()
            target_pc.points = o3d.utility.Vector3dVector(np.array(self.cad_mesh.vertices))

            # trans_init - это матрица трансформации 4x4 (перемещение + вращение). Изначально единичная (без изменений).
            trans_init = np.eye(4)

            # --- ШАГ 1: ГРУБОЕ СОВМЕЩЕНИЕ ---
            # Если пользователь вручную поставил маркеры (минимум 3 штуки для определения плоскости в 3D)
            if len(self.cad_pts) >= 3 and len(self.cad_pts) == len(self.scan_pts):
                self.log_signal.emit("2. Грубое позиционирование (ПО ВЫБРАННЫМ МАРКЕРАМ)...")
                # Создаем микро-облака точек только из маркеров
                pcd_scan_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.scan_pts)))
                pcd_cad_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self.cad_pts)))

                # Указываем, что точка 0 на скане равна точке 0 на CAD, 1=1, 2=2 и т.д.
                corres = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(self.scan_pts))]))
                estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                # Вычисляем матрицу, которая наложит маркеры друг на друга
                trans_init = estimator.compute_transformation(pcd_scan_pts, pcd_cad_pts, corres)
            else:
                # АВТОПИЛОТ: Если маркеров нет, просто совмещаем центры масс обеих деталей
                self.log_signal.emit("2. Грубое позиционирование (по центрам масс)...")
                if len(self.cad_pts) > 0:
                    self.log_signal.emit("   [i] Выбрано меньше 3 точек. Включен Автопилот.")
                # Вычисляем разницу между центрами и записываем её в блок перемещения матрицы (столбец 3)
                trans_init[:3, 3] = target_pc.get_center() - source_pc.get_center()

            # --- ШАГ 2: ТОНКОЕ СОВМЕЩЕНИЕ (АЛГОРИТМ ICP) ---
            self.log_signal.emit("3. Тонкое совмещение ICP (до 2000 итераций)...")
            threshold = 5.0  # Максимальное расстояние (мм) на котором алгоритм ищет соседние точки

            # Запускаем. Алгоритм будет двигать Скан к CAD, пока они не слипнутся максимально плотно
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pc, target_pc, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )

            # Fitness показывает процент точек, которые совпали идеально (от 0.0 до 1.0)
            fitness = reg_p2p.fitness
            self.log_signal.emit(f"   -> УСПЕХ! Точность наложения (Fitness): {fitness:.4f}")

            # --- ШАГ 3: ПРИМЕНЕНИЕ РЕЗУЛЬТАТА ---
            self.log_signal.emit("4. Применение матрицы к исходной сетке...")
            transformed_scan = self.scan_mesh.copy()
            # Физически сдвигаем вершины скана по вычисленной матрице
            transformed_scan.apply_transform(reg_p2p.transformation)

            # Отправляем готовую деталь в главное окно
            self.finished_signal.emit(transformed_scan)

        except Exception as e:
            self.log_signal.emit(f"[!] ОШИБКА ICP: {str(e)}")


# ==========================================
# ПОТОК 2: ПРЕДЕФОРМАЦИЯ (RBF - Radial Basis Function)
# ==========================================
class CompensationThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(object)

    def __init__(self, cad_mesh, scan_mesh, settings):
        super().__init__()
        self.cad_mesh = cad_mesh.copy()
        self.scan_mesh = scan_mesh
        self.settings = settings  # Словарь со всеми настройками ползунков из интерфейса

    def log(self, text, replace=False):
        """Удобная функция-обертка. Если replace=True, она перезапишет последнюю строку в консоли (для процентов загрузки)"""
        self.log_signal.emit(f"{'REPLACE_FLAG' if replace else ''}{text}")

    def run(self):
        try:
            if not HAS_O3D:
                self.log("\n[!] ОШИБКА: Open3D не установлен. Аппаратное ускорение недоступно.")
                return

            self.log("\n=== ЗАПУСК ПРЕДЕФОРМАЦИИ (RBF) ===")
            cad_mesh, scan_mesh, s = self.cad_mesh, self.scan_mesh, self.settings

            # --- 1. РЕМЕШИНГ (Дробление полигонов) ---
            # CAD модели часто на торцах состоят из огромных плоских треугольников. RBF не сможет их согнуть плавно.
            # Поэтому мы дробим большие треугольники на маленькие (с шагом edge_len).
            if s["use_remesh"]:
                self.log(f"1. Умный Ремешинг (Шаг: {s['edge_len']} мм)...")
                v, f = trimesh.remesh.subdivide_to_size(cad_mesh.vertices, cad_mesh.faces, max_edge=s['edge_len'])
                cad_mesh = trimesh.Trimesh(vertices=v, faces=f)

            # --- 2. УСТАНОВКА МАЯЧКОВ ---
            # Алгоритм случайным, но равномерным образом раскидывает N точек по всей площади CAD-детали
            self.log(f"2. Установка маячков слежения ({s['points']} шт.)...")
            ctrl_cad_verts, face_indices = trimesh.sample.sample_surface(cad_mesh, s['points'])
            # Запоминаем нормали (направление "наружу") для каждой точки, чтобы знать, куда стрелять
            ctrl_cad_normals = np.array(cad_mesh.face_normals[face_indices])

            # --- 3. ЛАЗЕРНАЯ ТРАССИРОВКА (Raycasting) ---
            self.log("3. Аппаратная лазерная трассировка (Open3D C++ Engine)...")

            # Загружаем скан в тензорную сцену Intel
            scan_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(scan_mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.array(scan_mesh.faces, dtype=np.int32))
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(scan_tmesh)

            def get_hits_o3d(origins, directions, task_name):
                """Функция выстреливает пучок лучей из CAD в сторону Скана"""
                self.log(f"   -> {task_name}: Идет выстрел лучами...")
                rays = np.hstack((origins, directions)).astype(np.float32)
                # Выстрел! Вазилиновое дрисло
                ans = scene.cast_rays(o3d.core.Tensor(rays))

                t_hit = ans['t_hit'].numpy()  # Дистанция до попадания
                tri_idx = ans['primitive_ids'].numpy()  # В какой треугольник попали

                # Фильтруем лучи, которые улетели в пустоту (не попали в скан)
                valid = np.isfinite(t_hit)

                # Вычисляем точные X,Y,Z координаты попадания по формуле: Точка + Вектор * Дистанция
                locs = origins[valid] + directions[valid] * t_hit[valid][:, None]
                ray_idx = np.where(valid)[0]  # Номера успешных лучей

                self.log(f"   -> {task_name}: 100% Успешно", replace=True)
                return locs, ray_idx, tri_idx[valid]

            # Стреляем наружу (поиск распухания детали)
            locs_out, ray_idx_out, tri_idx_out = get_hits_o3d(ctrl_cad_verts, ctrl_cad_normals, "Поиск утолщений")
            # Стреляем внутрь (инвертируем нормаль через минус) - ищем усадку
            locs_in, ray_idx_in, tri_idx_in = get_hits_o3d(ctrl_cad_verts, -ctrl_cad_normals, "Поиск усадки")

            # --- 4. АНАЛИЗ ПОПАДАНИЙ ---
            def process_hits(ray_origins, cad_normals, locs, index_ray, index_tri, mesh, strictness):
                """Фильтрует попадания, отбрасывая фальшивые (когда луч пробил деталь насквозь)"""
                if len(locs) == 0: return {}
                hit_distances = np.linalg.norm(locs - ray_origins[index_ray], axis=1)
                hit_normals = mesh.face_normals[index_tri]

                # Математическое скалярное произведение (Dot Product).
                # Оно проверяет, смотрят ли CAD и Скан в одну сторону. Если нет - луч пробил стену насквозь.
                dots = np.sum(cad_normals[index_ray] * hit_normals, axis=1)

                # Маска: Дистанция не больше лимита И поверхности смотрят в одну сторону
                valid_mask = (hit_distances < s['limit'] + 1.0) & (dots > strictness)

                best_hits = {}
                for r_idx, loc, dist in zip(index_ray[valid_mask], locs[valid_mask], hit_distances[valid_mask]):
                    # Если один луч попал в два места (например в складку), берем ближайшее
                    if r_idx not in best_hits or dist < best_hits[r_idx][1]:
                        best_hits[r_idx] = (loc, dist)
                return best_hits

            # Получаем чистые, отфильтрованные данные
            hits_out = process_hits(ctrl_cad_verts, ctrl_cad_normals, locs_out, ray_idx_out, tri_idx_out, scan_mesh,
                                    s['norm'])
            hits_in = process_hits(ctrl_cad_verts, ctrl_cad_normals, locs_in, ray_idx_in, tri_idx_in, scan_mesh,
                                   s['norm'])

            self.log("   -> Анализ пустот и аномалий...")
            # Массивы для будущей математической матрицы
            final_ctrl_cad = []
            final_error_vectors = []
            anchored_count, capped_count = 0, 0

            # Проходим по всем нашим 4000 (или сколько задали) маячкам
            for i in range(s['points']):
                best_loc, best_dist = None, float('inf')

                # Ищем, с какой стороны отклонение было ближе - усадка или утолщение
                if i in hits_out and hits_out[i][1] < best_dist:
                    best_loc, best_dist = hits_out[i]
                if i in hits_in and hits_in[i][1] < best_dist:
                    best_loc, best_dist = hits_in[i]

                # Если луч вообще во что-то попал
                if best_loc is not None:
                    # Вектор ошибки = Координата на Скане МИНУС Координата на CAD
                    error_vec = best_loc - ctrl_cad_verts[i]

                    # Если отклонение слишком огромное (например кусок поддержки или мусор), обрезаем его в ноль
                    if np.linalg.norm(error_vec) > s['limit']:
                        final_ctrl_cad.append(ctrl_cad_verts[i])
                        final_error_vectors.append(np.zeros(3))
                        capped_count += 1
                    else:
                        # Все хорошо, записываем вектор деформации
                        final_ctrl_cad.append(ctrl_cad_verts[i])
                        final_error_vectors.append(error_vec)
                else:
                    # Если луч улетел в бесконечность (в скане дырка).
                    # Якорь = принудительно зафиксировать CAD в этой точке, чтобы он не улетел вслед за дыркой.
                    if s['anchor']:
                        final_ctrl_cad.append(ctrl_cad_verts[i])
                        final_error_vectors.append(np.zeros(3))  # Нулевой вектор деформации
                        anchored_count += 1

            # --- 5. МАТЕМАТИКА RBF ---
            self.log("4. Расчет RBF-матрицы деформации...")

            # RBFInterpolator решает огромную систему уравнений и создает функцию-искривитель
            if s['neighbors'] > 0:
                # ЛОКАЛЬНЫЙ РЕЖИМ (Сверхбыстрый). Использует KD-деревья, чтобы искать только соседей.
                self.log(f"   -> Режим: Локальный (Влияние на {s['neighbors']} точек)")
                rbf = RBFInterpolator(
                    np.array(final_ctrl_cad), np.array(final_error_vectors),
                    kernel='thin_plate_spline', smoothing=s['smooth'], neighbors=s['neighbors']
                )
            else:
                # ГЛОБАЛЬНЫЙ РЕЖИМ. Матрица может достигать миллиардов ячеек, но дает идеальную гладкость.
                self.log("   -> Режим: Глобальный (Максимальная гладкость, ВНИМАНИЕ: Требует много ОЗУ!)")
                rbf = RBFInterpolator(
                    np.array(final_ctrl_cad), np.array(final_error_vectors),
                    kernel='thin_plate_spline', smoothing=s['smooth']
                )

            # --- 6. ПРИМЕНЕНИЕ ДЕФОРМАЦИИ ---
            self.log("5. Пакетная деформация геометрии CAD...")
            cad_verts_all = np.array(cad_mesh.vertices)
            compensated_verts = np.zeros_like(cad_verts_all)
            chunk_size = 50000

            # Деформируем вершины детали не все разом (чтобы не кончилась ОЗУ), а порциями по 50к штук
            for i in range(0, len(cad_verts_all), chunk_size):
                end = min(i + chunk_size, len(cad_verts_all))
                chunk_verts = cad_verts_all[i:end]

                # Главная формула программы: Новая Вершина = Старая Вершина МИНУС Вектор ошибки
                # Мы вычитаем ошибку, создавая контр-деформацию (предыскажение)
                compensated_verts[i:end] = chunk_verts - (rbf(chunk_verts) * 1.0)

                self.log(f"   -> Прогресс: {int((end / len(cad_verts_all)) * 100)}%", replace=True)

            # Сохраняем новые изогнутые вершины обратно в сетку
            cad_mesh.vertices = compensated_verts
            self.log("\n=== ГОТОВО! МОДЕЛЬ УСПЕШНО ДЕФОРМИРОВАНА ===")
            self.finished_signal.emit(cad_mesh)

        except Exception as e:
            self.log(f"\n[!] ОШИБКА: {str(e)}")


# ==========================================
# ГЛАВНЫЙ ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (GUI)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # Инициализируем базовый класс QMainWindow
        self.setWindowTitle("DeWarp Enterprise V6.1")
        self.setGeometry(100, 100, 1600, 900)  # (X_экрана, Y_экрана, Ширина, Высота)

        # Переменные для хранения загруженных 3D-моделей
        self.cad_mesh = None
        self.scan_mesh = None
        self.result_mesh = None

        # Актеры - это визуальные объекты на 3D-сцене.
        # Храним их в словаре, чтобы легко скрывать/показывать.
        self.actors = {"CAD": None, "Scan": None, "Result": None, "Heatmap": None}
        self.sliders = {}  # Словарь для хранения всех ползунков

        # Цвета по умолчанию (в HEX-формате, удобном для веб и UI)
        self.mesh_colors = {
            "CAD": "#1f77b4",  # Синий
            "Scan": "#d3d3d3",  # Серый
            "Result": "#2ca02c"  # Зеленый
        }

        # Переменные для системы ручного выбора маркеров на экране
        self.pick_mode = None
        self.cad_pts = []
        self.scan_pts = []
        self.pt_actors = []  # Красные и желтые шарики

        self.initUI()  # Запускаем сборку интерфейса

    def initUI(self):
        """Функция собирает окно из виджетов, как из кубиков Lego"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Главный Layout (Макет). QVBoxLayout - вертикальный (элементы идут сверху вниз).
        base_layout = QVBoxLayout(main_widget)
        base_layout.setContentsMargins(0, 0, 0, 0)  # Убираем белые рамки по краям окна

        # Сплиттер - это разделительная полоса, которую можно двигать мышкой
        self.main_splitter = QSplitter(Qt.Horizontal)
        base_layout.addWidget(self.main_splitter)

        # --- ЛЕВАЯ ПАНЕЛЬ (3D Окно) ---
        self.plotter = QtInteractor(self)  # Окно PyVista
        self.plotter.set_background('white')
        self.plotter.add_axes()  # Добавляем стрелочки X, Y, Z в левый нижний угол
        self.plotter.add_key_event('space', self.on_space_pressed)  # Биндим пробел
        self.main_splitter.addWidget(self.plotter.interactor)

        # --- ПРАВАЯ ПАНЕЛЬ (Интерфейс) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        self.main_splitter.addWidget(right_panel)

        # Изначально отдаем 1100 пикселей под 3D-окно, и 500 под панель
        self.main_splitter.setSizes([1100, 500])

        # --- ВКЛАДКИ (Tabs) ---
        self.tabs = QTabWidget()

        self.tab_align = QWidget()
        self.initAlignTab()  # Заполняем вкладку 1
        self.tabs.addTab(self.tab_align, "Шаг 1: Совмещение (ICP)")

        self.tab_comp = QWidget()
        self.initCompTab()  # Заполняем вкладку 2
        self.tabs.addTab(self.tab_comp, "Шаг 2: Компенсация (RBF)")

        self.tab_donate = QWidget()
        self.initDonateTab()  # Заполняем вкладку 3
        self.tabs.addTab(self.tab_donate, "💰 Поддержка автора")

        right_layout.addWidget(self.tabs, stretch=6)  # stretch=6 означает "займи 6 частей свободного места"

        # --- ПАНЕЛЬ СЛОЕВ ---
        # QGroupBox - это рамка с заголовком
        view_group = QGroupBox("Слои (Видимость, Цвет, Прозрачность)")
        view_layout = QGridLayout()  # Сетка. Элементы расставляются по координатам (строка, столбец)

        # Создаем элементы для управления CAD
        self.chk_view_cad = QCheckBox("CAD")
        self.chk_view_cad.setChecked(True)
        self.btn_col_cad = self.create_color_button("CAD")
        self.sld_op_cad = QSlider(Qt.Horizontal)
        self.sld_op_cad.setRange(0, 100)
        self.sld_op_cad.setValue(80)
        self.lbl_op_cad = QLabel("80%")

        # Для Скана
        self.chk_view_scan = QCheckBox("Скан")
        self.chk_view_scan.setChecked(True)
        self.btn_col_scan = self.create_color_button("Scan")
        self.sld_op_scan = QSlider(Qt.Horizontal)
        self.sld_op_scan.setRange(0, 100)
        self.sld_op_scan.setValue(80)
        self.lbl_op_scan = QLabel("80%")

        # Для Результата
        self.chk_view_res = QCheckBox("Результат")
        self.chk_view_res.setChecked(True)
        self.btn_col_res = self.create_color_button("Result")
        self.sld_op_res = QSlider(Qt.Horizontal)
        self.sld_op_res.setRange(0, 100)
        self.sld_op_res.setValue(100)
        self.lbl_op_res = QLabel("100%")

        # Расставляем их в сетку: addWidget(виджет, строка, колонка)
        view_layout.addWidget(self.chk_view_cad, 0, 0);
        view_layout.addWidget(self.btn_col_cad, 0, 1);
        view_layout.addWidget(self.sld_op_cad, 0, 2);
        view_layout.addWidget(self.lbl_op_cad, 0, 3)
        view_layout.addWidget(self.chk_view_scan, 1, 0);
        view_layout.addWidget(self.btn_col_scan, 1, 1);
        view_layout.addWidget(self.sld_op_scan, 1, 2);
        view_layout.addWidget(self.lbl_op_scan, 1, 3)
        view_layout.addWidget(self.chk_view_res, 2, 0);
        view_layout.addWidget(self.btn_col_res, 2, 1);
        view_layout.addWidget(self.sld_op_res, 2, 2);
        view_layout.addWidget(self.lbl_op_res, 2, 3)

        # Подключаем Сигналы (События). Когда юзер нажимает галку -> вызывается функция update_visibility
        self.chk_view_cad.stateChanged.connect(self.update_visibility)
        self.chk_view_scan.stateChanged.connect(self.update_visibility)
        self.chk_view_res.stateChanged.connect(self.update_visibility)

        # lambda - это анонимная мини-функция. Она позволяет передать параметры в функцию при сигнале
        self.sld_op_cad.valueChanged.connect(lambda v: self.update_opacity("CAD", v, self.lbl_op_cad))
        self.sld_op_scan.valueChanged.connect(lambda v: self.update_opacity("Scan", v, self.lbl_op_scan))
        self.sld_op_res.valueChanged.connect(lambda v: self.update_opacity("Result", v, self.lbl_op_res))

        view_group.setLayout(view_layout)
        right_layout.addWidget(view_group, stretch=0)  # stretch=0 - не растягивать, сжать до минимума

        # --- ПАНЕЛЬ АНАЛИЗА (ТЕПЛОВАЯ КАРТА) ---
        heat_group = QGroupBox("Анализ (Цветовая карта отклонений)")
        heat_layout = QVBoxLayout()
        row_heat = QHBoxLayout()  # Горизонтальный макет для кнопок

        self.btn_heatmap = QPushButton("🔥 Построить Heatmap (Скан vs CAD)")
        self.btn_heatmap.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")  # CSS-стили
        self.btn_heatmap.clicked.connect(self.generate_heatmap)
        row_heat.addWidget(self.btn_heatmap)

        self.btn_clear_heat = QPushButton("Сбросить")
        self.btn_clear_heat.clicked.connect(self.clear_heatmap)
        row_heat.addWidget(self.btn_clear_heat)

        heat_layout.addLayout(row_heat)

        # Используем функцию-генератор ползунков
        self.add_slider(heat_layout, "Предел градиента (± мм)", 1, 50, 10, 1, "heat_limit", divider=10.0)
        self.sliders["heat_limit"][0].valueChanged.connect(self.update_heatmap_limit)

        heat_group.setLayout(heat_layout)
        right_layout.addWidget(heat_group, stretch=0)

        # --- ЧЕРНАЯ КОНСОЛЬ ВНИЗУ ---
        self.console = QTextEdit()
        self.console.setReadOnly(True)  # Юзер не может туда печатать ведь он даже не гражданин
        self.console.setFixedHeight(180)
        # Стилизуем под хакерский терминал
        self.console.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas; font-size: 12px;")
        right_layout.addWidget(self.console, stretch=0)

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ИНТЕРФЕЙСА ---

    def create_color_button(self, key):
        """Создает маленькую квадратную кнопку для выбора цвета"""
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.PointingHandCursor)  # Меняем курсор на "пальчик"
        # Красим кнопку стартовым цветом из словаря mesh_colors
        btn.setStyleSheet(f"background-color: {self.mesh_colors[key]}; border: 1px solid #555; border-radius: 3px;")
        btn.clicked.connect(lambda: self.pick_color(key, btn))
        return btn

    def pick_color(self, key, btn):
        """Открывает системную палитру Windows и меняет цвет 3D-модели"""
        initial_color = QColor(self.mesh_colors[key])
        color = QColorDialog.getColor(initial_color, self, f"Выберите цвет для {key}")

        if color.isValid():
            # Запоминаем цвет в формате "#FF0000" для кнопки
            hex_color = color.name()
            self.mesh_colors[key] = hex_color
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555; border-radius: 3px;")

            if self.actors[key]:
                # ВАЖНО: Ядро VTK (написанное на C++) принимает цвета только как дробные числа от 0.0 до 1.0!
                # Поэтому мы конвертируем HEX в RGB-float. Иначе у программы пробитие.
                r, g, b = color.redF(), color.greenF(), color.blueF()
                self.actors[key].GetProperty().SetColor(r, g, b)
                self.plotter.render()  # Принудительно заставляем экран перерисоваться

    def add_slider(self, layout, text, vmin, vmax, vdef, step, name, divider=1.0):
        """Умная функция для создания ползунков.
        PySide6 слайдеры понимают только целые числа (int).
        Поэтому если нам нужно число 0.5, мы передаем divider=10, и слайдер ходит от 1 до 10, а на экран выводится 1/10 = 0.1"""
        lbl = QLabel(f"{text}: {vdef / divider}")
        sld = QSlider(Qt.Horizontal)
        sld.setMinimum(vmin)
        sld.setMaximum(vmax)
        sld.setValue(vdef)
        sld.setSingleStep(step)

        # Сигнал: при движении ползунка -> обновить текст в метке над ним
        sld.valueChanged.connect(lambda val, l=lbl, t=text, d=divider: l.setText(f"{t}: {val / d}"))

        layout.addWidget(lbl)
        layout.addWidget(sld)

        # Сохраняем и сам ползунок, и его делитель в словарь, чтобы потом легко извлекать значения
        self.sliders[name] = (sld, divider)

    # ... (Остальные функции initAlignTab, initCompTab, initDonateTab стандартны и занимаются расстановкой виджетов)
    def initAlignTab(self):
        l = QVBoxLayout(self.tab_align)
        group_files = QGroupBox("1. Базовые данные")
        fl = QVBoxLayout()
        self.btn_load_cad = QPushButton("Загрузить Исходный CAD (.stl)")
        self.btn_load_cad.clicked.connect(self.load_cad)
        self.btn_load_scan = QPushButton("Загрузить Скан (.stl)")
        self.btn_load_scan.clicked.connect(self.load_scan)
        fl.addWidget(self.btn_load_cad)
        fl.addWidget(self.btn_load_scan)
        group_files.setLayout(fl)
        l.addWidget(group_files)

        group_pts = QGroupBox("2. Вспомогательные маркеры (От локальных минимумов)")
        pt_layout = QVBoxLayout()
        self.lbl_pts = QLabel("Точек на CAD: 0 | Точек на Скане: 0")
        self.lbl_pts.setStyleSheet("font-weight: bold; color: blue;")
        pt_layout.addWidget(self.lbl_pts)

        row_btns = QHBoxLayout()
        self.btn_pick_cad = QPushButton("📍 Выбрать на CAD")
        self.btn_pick_cad.clicked.connect(self.start_pick_cad)
        self.btn_pick_scan = QPushButton("📍 Выбрать на Скане")
        self.btn_pick_scan.clicked.connect(self.start_pick_scan)
        row_btns.addWidget(self.btn_pick_cad)
        row_btns.addWidget(self.btn_pick_scan)
        pt_layout.addLayout(row_btns)

        self.btn_clear_pts = QPushButton("Сбросить маркеры")
        self.btn_clear_pts.clicked.connect(self.clear_picks)
        pt_layout.addWidget(self.btn_clear_pts)
        group_pts.setLayout(pt_layout)
        l.addWidget(group_pts)

        self.btn_run_icp = QPushButton("▶ СОВМЕСТИТЬ МОДЕЛИ (ICP)")
        self.btn_run_icp.setStyleSheet(
            "height: 50px; background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_run_icp.clicked.connect(self.run_icp)
        l.addWidget(self.btn_run_icp)
        l.addStretch()

    def initCompTab(self):
        l = QVBoxLayout(self.tab_comp)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        set_widget = QWidget()
        set_layout = QVBoxLayout(set_widget)

        self.add_slider(set_layout, "Разрешение маячков (>10k для мощных ПК)", 1000, 20000, 4000, 500, "points")
        self.add_slider(set_layout, "Жесткость RBF поля", 1, 150, 50, 1, "smooth", divider=10.0)
        self.chk_remesh = QCheckBox("Использовать Умный Remesh")
        self.chk_remesh.setChecked(True)
        set_layout.addWidget(self.chk_remesh)
        self.add_slider(set_layout, "Шаг сетки Remesh (мм)", 2, 30, 10, 1, "edge_len", divider=10.0)

        adv_lbl = QLabel("--- ПРОДВИНУТЫЕ НАСТРОЙКИ ---")
        adv_lbl.setStyleSheet("color: red; font-weight: bold; margin-top: 10px;")
        set_layout.addWidget(adv_lbl)

        self.add_slider(set_layout, "Область влияния RBF (Соседей) [0 = Глобально]", 0, 2000, 300, 50, "neighbors")
        self.add_slider(set_layout, "Лимит аномалий (мм)", 5, 50, 20, 1, "limit", divider=10.0)
        self.add_slider(set_layout, "Строгость нормалей", 30, 99, 80, 1, "norm", divider=100.0)
        self.chk_anchor = QCheckBox("Якорить пустоты сканера (0мм)")
        self.chk_anchor.setChecked(True)
        set_layout.addWidget(self.chk_anchor)

        scroll.setWidget(set_widget)
        l.addWidget(scroll)

        self.btn_run_comp = QPushButton("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.btn_run_comp.setStyleSheet(
            "height: 50px; background-color: #c0392b; color: white; font-weight: bold; font-size: 14px;")
        self.btn_run_comp.clicked.connect(self.run_comp)
        l.addWidget(self.btn_run_comp)

        self.btn_save = QPushButton("💾 Сохранить Результат")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_result)
        l.addWidget(self.btn_save)

    def initDonateTab(self):
        l = QVBoxLayout(self.tab_donate)
        l.setAlignment(Qt.AlignCenter)

        lbl_story = QLabel(
            "Тяжело быть инженером-конструктором в наше время...\n\nБессонные ночи перед дедлайнами, литры выпитого кофе, вечная борьба с допусками\nи попытки натянуть кривой оптический скан на идеальную CAD-модель.\n\nЕсли эта программа сэкономила вам пару часов сна, нервные клетки или уберегла\nцелую партию деталей от брака на производстве — помогите собрату по цеху!\nЛюбая копеечка пойдет на развитие полезного софта (и на успокоительное).")
        lbl_story.setAlignment(Qt.AlignCenter)
        lbl_story.setStyleSheet("font-size: 14px; font-style: italic; color: #E0E0E0; margin-bottom: 10px;")
        l.addWidget(lbl_story)

        lbl_img = QLabel()
        pixmap = QPixmap("f.jpeg")
        if not pixmap.isNull():
            lbl_img.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl_img.setText("[Картинка f.jpeg не найдена в папке с программой]")
            lbl_img.setStyleSheet("color: #ff4444; font-weight: bold;")  # Ярко-красный
        lbl_img.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl_img)

        lbl_card = QLabel("💳 Реквизиты карты: <b style='font-size: 20px; color: #ff6b6b;'>2200150959050136</b>")
        lbl_card.setAlignment(Qt.AlignCenter)
        lbl_card.setStyleSheet("margin-top: 15px;")
        lbl_card.setTextInteractionFlags(Qt.TextSelectableByMouse)
        l.addWidget(lbl_card)

        lbl_thanks = QLabel("Спасибо за пользование софтом!")
        lbl_thanks.setAlignment(Qt.AlignCenter)
        lbl_thanks.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2ecc71; margin-bottom: 20px; margin-top: 5px;")
        l.addWidget(lbl_thanks)

        lbl_contact = QLabel(
            "Предложения по улучшению и баг-репорты можно направлять на почту:\n"
            "<a href='mailto:theboogie888@gmail.com' style='color: #4da6ff;'>theboogie888@gmail.com</a>"
        )
        lbl_contact.setAlignment(Qt.AlignCenter)
        lbl_contact.setStyleSheet("font-size: 12px; color: #AAAAAA;")
        lbl_contact.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        lbl_contact.setOpenExternalLinks(True)
        l.addWidget(lbl_contact)

    # --- ЛОГИКА ВЗАИМОДЕЙСТВИЯ С 3D-ОКНОМ ---

    def log(self, text, replace=False):
        """Писатель в консоль. Умеет удалять последнюю строчку, чтобы цифры % бежали на одном месте."""
        if text.startswith("REPLACE_FLAG"):
            cursor = self.console.textCursor()
            cursor.movePosition(cursor.End)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text.replace("REPLACE_FLAG", ""))
        else:
            self.console.append(text)
        # Прокручиваем скроллбар в самый низ, чтобы видеть последние логи
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def trimesh_to_pyvista(self, tmesh):
        """Преобразует математическую сетку Trimesh в визуальную сетку PyVista.
        PyVista требует особого формата полигонов: [количество_вершин, в1, в2, в3].
        Так как у нас все полигоны треугольные, мы просто добавляем цифру 3 в начало каждого лица."""
        faces = np.pad(tmesh.faces, ((0, 0), (1, 0)), constant_values=3)
        return pv.PolyData(tmesh.vertices, faces)

    def show_mesh(self, key, mesh):
        """Главная функция отрисовки деталей в окне"""
        pv_mesh = self.trimesh_to_pyvista(mesh)

        # Если актер с таким именем уже есть на сцене - удаляем его перед добавлением нового
        if self.actors[key]:
            self.plotter.remove_actor(self.actors[key])

        # Читаем прозрачность из интерфейса
        if key == "CAD":
            op = self.sld_op_cad.value() / 100.0
        elif key == "Scan":
            op = self.sld_op_scan.value() / 100.0
        elif key == "Result":
            op = self.sld_op_res.value() / 100.0
        else:
            op = 0.8

        # add_mesh возвращает ссылку на актера (VTK-объект), которую мы сохраняем в словарь
        self.actors[key] = self.plotter.add_mesh(pv_mesh, color=self.mesh_colors[key], opacity=op,
                                                 show_edges=(key == "Result"))
        self.actors[key].pickable = True  # Разрешаем тыкать в эту деталь курсором (для маркеров)
        self.plotter.reset_camera()  # Центрируем камеру на детали
        self.update_visibility()

    def update_visibility(self):
        """Включает/выключает видимость актеров в зависимости от галочек"""
        if self.actors["CAD"]: self.actors["CAD"].SetVisibility(self.chk_view_cad.isChecked())
        if self.actors["Scan"]: self.actors["Scan"].SetVisibility(self.chk_view_scan.isChecked())
        if self.actors["Result"]: self.actors["Result"].SetVisibility(self.chk_view_res.isChecked())

    def update_opacity(self, key, value, label):
        """Меняет прозрачность объекта на лету (Alpha-канал)"""
        label.setText(f"{value}%")
        if self.actors[key]:
            self.actors[key].GetProperty().SetOpacity(value / 100.0)
            self.plotter.render()

    def load_cad(self):
        # QFileDialog открывает стандартное окно проводника Windows
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить CAD", "", "STL Files (*.stl)")
        if path:
            self.log(f"> Загружен CAD: {path.split('/')[-1]}")
            self.cad_mesh = trimesh.load(path)
            self.show_mesh("CAD", self.cad_mesh)

    def load_scan(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить Скан", "", "STL Files (*.stl)")
        if path:
            self.log(f"> Загружен Скан: {path.split('/')[-1]}")
            self.scan_mesh = trimesh.load(path)
            self.show_mesh("Scan", self.scan_mesh)

    # --- ЛОГИКА МАРКЕРОВ И "ЛУЧЕМЕТА" ИЗ КАМЕРЫ ---

    def start_pick_cad(self):
        if not self.cad_mesh:
            self.log("[!] Сначала загрузите CAD!")
            return
        self.pick_mode = 'CAD'
        # Автоматически прячем Скан, чтобы случайно не кликнуть по нему
        self.chk_view_cad.setChecked(True)
        self.chk_view_scan.setChecked(False)
        self.sld_op_cad.setValue(100)
        self.log("\n[РЕЖИМ CAD] Наведите курсор на деталь и нажмите ПРОБЕЛ (Spacebar).")

    def start_pick_scan(self):
        if not self.scan_mesh:
            self.log("[!] Сначала загрузите Скан!")
            return
        self.pick_mode = 'Scan'
        self.chk_view_cad.setChecked(False)
        self.chk_view_scan.setChecked(True)
        self.sld_op_scan.setValue(100)
        self.log("\n[РЕЖИМ СКАНА] Наведите курсор на деталь и нажмите ПРОБЕЛ (Spacebar).")

    def on_space_pressed(self):
        """Эта функция срабатывает каждый раз, когда мы жмем Пробел на клавиатуре"""
        if not self.pick_mode: return

        try:
            import vtk  # Импортируем ядро визуализатора
            # Получаем координаты мышки на 2D-экране (X, Y)
            pos = self.plotter.interactor.GetEventPosition()

            # Создаем "Пикер" (CellPicker). Он пускает луч из камеры перпендикулярно экрану вглубь 3D-сцены
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

            # Если луч во что-то воткнулся (Actor)
            actor = picker.GetActor()
            if actor:
                # Получаем точные 3D координаты (X, Y, Z) точки попадания луча
                point = picker.GetPickPosition()
                self.place_marker(point)
        except Exception as e:
            self.log(f"[!] Ошибка лучемета: {str(e)}")

    def place_marker(self, point):
        """Рисует шарик в указанных 3D координатах"""
        # Вычисляем размер шарика пропорционально размеру детали
        radius = self.cad_mesh.scale * 0.015 if self.cad_mesh else 1.0

        if self.pick_mode == 'CAD':
            self.cad_pts.append(point)
            # Рисуем сферу (Sphere)
            actor = self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color='red')
            actor.pickable = False  # Запрещаем кликать по самому шарику (иначе он перекроет деталь)
            self.pt_actors.append(actor)
            self.log(f"📍 CAD-точка {len(self.cad_pts)} установлена.")
        elif self.pick_mode == 'Scan':
            self.scan_pts.append(point)
            actor = self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color='yellow')
            actor.pickable = False
            self.pt_actors.append(actor)
            self.log(f"📍 Скан-точка {len(self.scan_pts)} установлена.")

        self.lbl_pts.setText(f"Точек на CAD: {len(self.cad_pts)} | Точек на Скане: {len(self.scan_pts)}")

    def clear_picks(self):
        self.cad_pts.clear()
        self.scan_pts.clear()
        self.pick_mode = None
        # Удаляем все шарики со сцены
        for actor in self.pt_actors:
            self.plotter.remove_actor(actor)
        self.pt_actors.clear()
        self.lbl_pts.setText("Точек на CAD: 0 | Точек на Скане: 0")
        self.log("Все выбранные точки сброшены.")

    # --- ЗАПУСК ПОТОКОВ РАСЧЕТА ---

    def run_icp(self):
        if not self.cad_mesh or not self.scan_mesh:
            self.log("[!] ОШИБКА: Загрузите обе модели!")
            return

        if len(self.cad_pts) > 0 and len(self.cad_pts) != len(self.scan_pts):
            self.log("[!] ОШИБКА: Количество точек на CAD и Скане должно совпадать!")
            return

        self.pick_mode = None
        self.btn_run_icp.setEnabled(False)  # Блокируем кнопку, чтобы не нажали дважды
        self.btn_run_icp.setText("⏳ ИДЕТ СОВМЕЩЕНИЕ...")

        # Создаем экземпляр нашего класса-потока, передаем ему данные
        self.align_thread = AlignmentThread(self.cad_mesh, self.scan_mesh, self.cad_pts, self.scan_pts)

        # Подключаем сигналы от потока к функциям главного окна
        self.align_thread.log_signal.connect(self.log)  # Если поток прокричит лог -> вызвать функцию self.log
        self.align_thread.finished_signal.connect(self.on_icp_done)  # Если закончил -> вызвать on_icp_done

        # Запускаем поток (Windows выделяет под него отдельное ядро процессора)
        self.align_thread.start()

    def on_icp_done(self, aligned_scan):
        """Эта функция сработает только когда поток AlignmentThread завершит работу"""
        self.scan_mesh = aligned_scan  # Заменяем старый скан на совмещенный
        self.show_mesh("Scan", self.scan_mesh)

        self.chk_view_cad.setChecked(True)
        self.chk_view_scan.setChecked(True)
        self.sld_op_cad.setValue(40)  # Делаем CAD полупрозрачным, чтобы видеть как скан в него влез
        self.sld_op_scan.setValue(100)

        self.btn_run_icp.setEnabled(True)
        self.btn_run_icp.setText("▶ СОВМЕСТИТЬ МОДЕЛИ (ICP)")
        self.clear_picks()
        # Автоматически переключаем юзера на Вкладку 2
        self.tabs.setCurrentIndex(1)
        self.log("\n>>> Модели совмещены. Перейдите к настройкам предеформации (Шаг 2).")

    # --- ЛОГИКА ТЕПЛОВОЙ КАРТЫ (HEATMAP) ---
    # Мы используем принцип "Неразрушающего редактирования" (Non-destructive workflow).
    # Мы не красим оригинальный Скан. Мы создаем его копию (Фантом), красим ее и показываем поверх оригинала.

    def generate_heatmap(self):
        if not self.cad_mesh or not self.scan_mesh:
            self.log("[!] ОШИБКА: Загрузите CAD и Скан (желательно после совмещения).")
            return

        self.log("\n>>> Расчет цветовой карты (Open3D Tensor Engine)...")
        self.btn_heatmap.setEnabled(False)

        try:
            # 1. Загружаем CAD как математическое препятствие (Стену)
            cad_tmesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.array(self.cad_mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.array(self.cad_mesh.faces, dtype=np.int32))
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(cad_tmesh)

            # 2. Вычисляем знаковую дистанцию (Signed Distance).
            # Алгоритм берет каждую точку Скана и измеряет кратчайшее расстояние до "Стены" (CAD).
            # Если точка внутри CAD - дистанция с минусом (Усадка). Если снаружи - с плюсом (Наплыв).
            query_points = o3d.core.Tensor(np.array(self.scan_mesh.vertices, dtype=np.float32))
            signed_dists = scene.compute_signed_distance(query_points).numpy()

            # 3. Создаем Фантомный слой
            pv_heatmap = self.trimesh_to_pyvista(self.scan_mesh)
            # Прикрепляем массив вычисленных дистанций к сетке (PyVista умеет с этим работать)
            pv_heatmap['Deviation'] = signed_dists

            # Если старый фантом уже есть на сцене - удаляем его
            if self.actors["Heatmap"]:
                self.plotter.remove_actor(self.actors["Heatmap"])

            # Выключаем галочки у оригиналов, чтобы они не просвечивали через тепловую карту
            self.chk_view_cad.setChecked(False)
            self.chk_view_scan.setChecked(False)

            limit = self.sliders["heat_limit"][0].value() / self.sliders["heat_limit"][1]

            # 4. Рендерим Фантом с градиентной шкалой
            # noinspection PyArgumentList (Прячем предупреждение IDE, т.к. **kwargs передаются в VTK)
            self.actors["Heatmap"] = self.plotter.add_mesh(
                pv_heatmap,
                scalars='Deviation',  # Какое поле данных использовать для цвета
                cmap='turbo',  # Название цветовой схемы (Радуга GOM Inspect)
                clim=[-limit, limit],  # Пределы шкалы
                show_scalar_bar=True,  # Показать легенду
                scalar_bar_args={  # Настройки вертикальной легенды справа
                    'title': 'Отклонение (мм)', 'color': 'black', 'vertical': True,
                    'position_x': 0.88, 'position_y': 0.05, 'height': 0.9, 'width': 0.08,
                    'title_font_size': 18, 'label_font_size': 14, 'fmt': '%1.3f'
                }
            )

            self.plotter.reset_camera()
            self.log(f"✅ Готово! Красный = Наплыв металла, Синий = Усадка.")

        except Exception as e:
            self.log(f"[!] Ошибка: {str(e)}")
        finally:
            self.btn_heatmap.setEnabled(True)

    def update_heatmap_limit(self):
        """Эта функция вызывается 100 раз в секунду, когда юзер двигает ползунок градиента.
        Вместо того чтобы заново считать дистанции, мы просто на лету меняем параметр `scalar_range` в видеопамяти."""
        if self.actors.get("Heatmap") and hasattr(self.actors["Heatmap"].mapper, 'dataset'):
            limit = self.sliders["heat_limit"][0].value() / self.sliders["heat_limit"][1]
            self.actors["Heatmap"].mapper.scalar_range = [-limit, limit]
            self.plotter.render()

    def clear_heatmap(self):
        """Возврат в стандартный режим"""
        # Убиваем Фантом (Шкала удалится автоматически вместе с ним)
        if self.actors.get("Heatmap"):
            self.plotter.remove_actor(self.actors["Heatmap"])
            self.actors["Heatmap"] = None

        # Просто включаем видимость базовых слоев
        self.chk_view_scan.setChecked(True)
        self.chk_view_cad.setChecked(True)
        self.log("Отображение сброшено в базовый режим.")

    def run_comp(self):
        if not self.cad_mesh or not self.scan_mesh:
            return
        self.btn_run_comp.setEnabled(False)
        self.btn_run_comp.setText("⏳ ИДЕТ РАСЧЕТ МАТРИЦ...")

        # Собираем все текущие значения ползунков в один словарь (settings)
        settings = {
            "points": int(self.sliders["points"][0].value() / self.sliders["points"][1]),
            "smooth": float(self.sliders["smooth"][0].value() / self.sliders["smooth"][1]),
            "use_remesh": self.chk_remesh.isChecked(),
            "edge_len": float(self.sliders["edge_len"][0].value() / self.sliders["edge_len"][1]),
            "limit": float(self.sliders["limit"][0].value() / self.sliders["limit"][1]),
            "norm": float(self.sliders["norm"][0].value() / self.sliders["norm"][1]),
            "anchor": self.chk_anchor.isChecked(),
            "neighbors": int(self.sliders["neighbors"][0].value() / self.sliders["neighbors"][1])
        }

        # Создаем и запускаем поток предеформации
        self.comp_thread = CompensationThread(self.cad_mesh, self.scan_mesh, settings)
        self.comp_thread.log_signal.connect(self.log)
        self.comp_thread.finished_signal.connect(self.on_comp_done)
        self.comp_thread.start()

    def on_comp_done(self, result_mesh):
        """Сработает, когда отработает RBF-компенсация"""
        self.result_mesh = result_mesh
        self.show_mesh("Result", self.result_mesh)
        self.sld_op_scan.setValue(0)  # Прячем старый скан
        self.btn_run_comp.setEnabled(True)
        self.btn_run_comp.setText("⚡ ЗАПУСТИТЬ ПРЕДЕФОРМАЦИЮ")
        self.btn_save.setEnabled(True)  # Включаем кнопку Сохранить

    def save_result(self):
        """Экспорт сетки в STL-файл"""
        if self.result_mesh:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить Результат", "Compensated_Part.stl",
                                                    "STL Files (*.stl)")
            if path:
                self.result_mesh.export(path)
                self.log(f"✅ Файл успешно сохранен: {path}")

# ==========================================
# ТОЧКА ВХОДА В ПРОГРАММУ
 # ==========================================
# В Питоне этот блок защищает код от случайного выполнения, если кто-то другой попытается импортировать этот файл
if __name__ == '__main__':
    # QApplication - Главный объект, который управляет всем приложением (цикл событий Windows)
    app = QApplication(sys.argv)

    # Создаем наше главное окно из класса
    window = MainWindow()

    # По умолчанию окна скрыты, их нужно принудительно показать
    window.show()

    # Запускаем бесконечный цикл приложения. Пока юзер не нажмет крестик, программа не закроется.
    sys.exit(app.exec())