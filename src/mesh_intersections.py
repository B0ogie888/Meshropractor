"""Bounded-memory triangle collision checks, including coplanar area overlaps.

The R-tree is only a broad phase. Exact triangle geometry is tested afterwards;
ordinary common vertices/edges and zero-area contact are not defects here.
All distances and the absolute numerical tolerance are in model units (mm).
"""
import numpy as np


def triangle_contacts(a, b, tolerance=1e-9):
    """Return (coplanar area overlap, transverse intersection) per triangle pair."""
    na = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
    nb = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
    valid = (la > 1e-18) & (lb > 1e-18)
    na /= np.maximum(la[:, None], 1e-300)
    nb /= np.maximum(lb[:, None], 1e-300)
    da = np.einsum('nij,nj->ni', a-b[:, :1], nb)
    db = np.einsum('nij,nj->ni', b-a[:, :1], na)
    coplanar = valid & (np.abs(da).max(axis=1) <= tolerance) & (np.abs(db).max(axis=1) <= tolerance)
    overlaps = np.zeros(len(a), dtype=bool)
    ids = np.flatnonzero(coplanar)
    if len(ids):
        # Project to a local 2-D frame via in-plane separating axes. Positive
        # interval overlap on every edge normal excludes adjacent triangles.
        aa, bb, nn = a[ids], b[ids], na[ids]
        accepted = np.ones(len(ids), dtype=bool)
        for triangles in (aa, bb):
            for edge in range(3):
                axis = np.cross(nn, triangles[:, (edge+1) % 3]-triangles[:, edge])
                axis /= np.maximum(np.linalg.norm(axis, axis=1)[:, None], 1e-300)
                pa = np.einsum('nij,nj->ni', aa-aa[:, :1], axis)
                pb = np.einsum('nij,nj->ni', bb-aa[:, :1], axis)
                accepted &= np.minimum(pa.max(axis=1), pb.max(axis=1)) - np.maximum(pa.min(axis=1), pb.min(axis=1)) > tolerance
        overlaps[ids] = accepted
    crossing = np.zeros(len(a), dtype=bool)
    direction = np.cross(na, nb)
    length = np.linalg.norm(direction, axis=1)
    possible = valid & ~coplanar & (length > 1e-12)
    possible &= (da.min(axis=1) <= tolerance) & (da.max(axis=1) >= -tolerance)
    possible &= (db.min(axis=1) <= tolerance) & (db.max(axis=1) >= -tolerance)
    ids = np.flatnonzero(possible)
    if len(ids):
        axis = direction[ids] / length[ids, None]
        origin = a[ids, :1]
        def interval(triangles, distances):
            ts = np.einsum('nij,nj->ni', triangles-origin, axis)
            on = np.abs(distances) <= tolerance
            low = np.where(on, ts, np.inf).min(axis=1)
            high = np.where(on, ts, -np.inf).max(axis=1)
            for i in range(3):
                j = (i+1) % 3
                opposite = (distances[:, i] * distances[:, j]) < 0
                divisor = distances[:, i]-distances[:, j]
                alpha = np.divide(distances[:, i], divisor, out=np.zeros(len(ids)), where=opposite)
                cut = ts[:, i] + alpha*(ts[:, j]-ts[:, i])
                low = np.minimum(low, np.where(opposite, cut, np.inf))
                high = np.maximum(high, np.where(opposite, cut, -np.inf))
            return low, high
        al, ah = interval(a[ids], da[ids])
        bl, bh = interval(b[ids], db[ids])
        crossing[ids] = np.minimum(ah, bh)-np.maximum(al, bl) > tolerance
    return overlaps, crossing


def find_intersections(mesh, progress=lambda message: None, cancelled=lambda: False, tolerance=1e-9):
    from rtree.index import Index, Property
    n = len(mesh.faces)
    low, high = np.empty((n, 3)), np.empty((n, 3))
    for start in range(0, n, 100000):
        if cancelled(): raise InterruptedError('Диагностика отменена')
        tri = mesh.vertices[mesh.faces[start:start+100000]]
        low[start:start+len(tri)], high[start:start+len(tri)] = tri.min(axis=1), tri.max(axis=1)
    progress('Построение пространственного индекса треугольников…')
    tree = Index((np.arange(n, dtype=np.int64), low, high), properties=Property(dimension=3))
    overlaps, intersections = np.zeros(n, bool), np.zeros(n, bool)
    overlap_pairs = intersection_pairs = 0
    last_percent = -1
    try:
        for start in range(0, n, 2048):
            if cancelled(): raise InterruptedError('Диагностика отменена')
            stop = min(n, start+2048)
            js, counts = tree.intersection_v(low[start:stop]-tolerance, high[start:stop]+tolerance)
            ii = np.repeat(np.arange(start, stop), counts.astype(np.int64))
            mask = js > ii
            ii, js = ii[mask], js[mask]
            # Bound the numerical narrow phase even for a dense broad-phase batch.
            for offset in range(0, len(ii), 50000):
                a, b = ii[offset:offset+50000], js[offset:offset+50000]
                fa, fb = mesh.faces[a], mesh.faces[b]
                shared = (fa[:, :, None] == fb[:, None, :]).any(axis=2).sum(axis=1)
                ov, ix = triangle_contacts(mesh.vertices[fa], mesh.vertices[fb], tolerance)
                ix &= shared < 2
                overlaps[a[ov]] = True; overlaps[b[ov]] = True
                intersections[a[ix]] = True; intersections[b[ix]] = True
                overlap_pairs += int(ov.sum()); intersection_pairs += int(ix.sum())
            percent = int(stop * 100 / max(n, 1))
            if percent >= last_percent + 5:
                progress(f'Нахлёсты и пересечения: {percent}%')
                last_percent = percent
    finally:
        tree.close()
    return dict(overlaps=int(overlaps.sum()), intersections=int(intersections.sum()),
                overlap_pairs=overlap_pairs, intersection_pairs=intersection_pairs,
                overlap_faces=np.flatnonzero(overlaps), intersection_faces=np.flatnonzero(intersections))
