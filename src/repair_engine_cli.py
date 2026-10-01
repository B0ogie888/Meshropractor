# SPDX-License-Identifier: GPL-3.0-or-later
# Standalone command-line adapter for PyMeshFix / MeshFix. See licenses/repair-engine.
"""Independent mesh repair executable. Input/output: NumPy vertex/face archives."""
import argparse
import json
import time
import numpy as np


def main():
    import pymeshfix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--passes', type=int, default=3, choices=range(1, 11))
    parser.add_argument('--target-faces', type=int, default=0,
                        help='Optional quadric simplification before repair; 0 preserves input density.')
    args = parser.parse_args()
    def emit(message):
        print(json.dumps({'message': message}, ensure_ascii=True), flush=True)
    emit('Сшивка сетки и согласование связности…')
    data = np.load(args.input, allow_pickle=False)
    vertices = np.ascontiguousarray(data['vertices'], dtype=np.float64)
    faces = np.ascontiguousarray(data['faces'], dtype=np.int32)
    data.close()
    if args.target_faces > 0 and len(faces) > args.target_faces:
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkFiltersCore import vtkQuadricDecimation
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
        emit(f'Оптимизация избыточной триангуляции: {len(faces)} → около {args.target_faces} граней…')
        points = vtkPoints(); points.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtkCellArray()
        cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces), 3), faces)).ravel(), deep=True))
        poly = vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
        decimate = vtkQuadricDecimation(); decimate.SetInputData(poly)
        decimate.SetTargetReduction(1-args.target_faces/len(faces)); decimate.VolumePreservationOn()
        decimate.Update()
        result = decimate.GetOutput()
        vertices = np.asarray(vtk_to_numpy(result.GetPoints().GetData()), dtype=np.float64).copy()
        faces = np.asarray(vtk_to_numpy(result.GetPolys().GetData()).reshape(-1, 4)[:, 1:], dtype=np.int32).copy()
        emit(f'После оптимизации: {len(faces)} граней. Сшивка и ремонт…')
        del poly, points, cells, decimate, result
    engine = pymeshfix.PyTMesh()
    engine.set_quiet(True)
    engine.load_array(vertices, faces)
    records = []
    for i in range(args.passes):
        start = time.monotonic()
        emit(f'Проход {i+1}/{args.passes}: закрытие отверстий…')
        engine.fill_small_boundaries()
        emit(f'Проход {i+1}/{args.passes}: устранение нахлёстов и пересечений…')
        clean = engine.clean(max_iters=3, inner_loops=3)
        records.append(dict(pass_number=i+1, clean=bool(clean), boundaries=engine.n_boundaries,
                            seconds=round(time.monotonic()-start, 2)))
        emit(f'Проход {i+1} завершён; открытых контуров: {engine.n_boundaries}.')
        if clean and engine.n_boundaries == 0:
            break
    vertices, faces = engine.return_arrays()
    np.savez(args.output, vertices=vertices, faces=faces, passes=json.dumps(records))
    emit('Ремонт завершён. Независимая проверка результата…')


if __name__ == '__main__':
    main()
