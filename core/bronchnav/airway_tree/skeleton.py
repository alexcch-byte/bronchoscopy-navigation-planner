"""Airway mask -> centerline tree.

Steps: crop -> distance transform (radius) -> 3D thinning -> 26-neighbor voxel graph ->
minimum spanning tree (drops staircase triangles and loops) -> spur pruning.

Points are stored in the *grid frame*: x, y, z in mm = (index_x, index_y, index_z) * spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from skimage.morphology import skeletonize


@dataclass
class AirwayTree:
    pts_mm: np.ndarray  # (N, 3) x, y, z in grid-frame mm
    radius_mm: np.ndarray  # (N,) local lumen radius (distance transform)
    adj: csr_matrix  # symmetric, weights = edge length in mm
    spacing_mm: float
    root: int  # open proximal end of the trachea (wide endpoint farthest from the tree centroid)
    dt: np.ndarray  # cropped distance transform (z, y, x) in mm
    dt_offset_zyx: tuple[int, int, int]

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.adj.indptr)

    @property
    def junctions(self) -> np.ndarray:
        return np.nonzero(self.degree >= 3)[0]

    @property
    def endpoints(self) -> np.ndarray:
        return np.nonzero(self.degree == 1)[0]

    def total_length_mm(self) -> float:
        return float(self.adj.sum() / 2.0)

    def sample_radius(self, pts_mm: np.ndarray) -> np.ndarray:
        """Trilinear-sample the distance transform (mm) at grid-frame points (M, 3)."""
        idx_xyz = pts_mm / self.spacing_mm
        zyx = idx_xyz[:, ::-1] - np.array(self.dt_offset_zyx)[None, :]
        return ndi.map_coordinates(self.dt, zyx.T, order=1, mode="constant", cval=0.0)


def _neighbor_offsets():
    return [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) > (0, 0, 0)
    ]


def _voxel_graph(skel: np.ndarray, spacing: float):
    coords = np.argwhere(skel)
    n = len(coords)
    idxmap = np.full(np.array(skel.shape) + 2, -1, dtype=np.int64)
    idxmap[tuple((coords + 1).T)] = np.arange(n)
    rows, cols, w = [], [], []
    for off in _neighbor_offsets():
        nb = idxmap[tuple((coords + 1 + np.array(off)).T)]
        ok = nb >= 0
        rows.append(np.nonzero(ok)[0])
        cols.append(nb[ok])
        w.append(np.full(ok.sum(), np.linalg.norm(off) * spacing))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return coords, g


def _prune_spurs(adj: csr_matrix, radius: np.ndarray, spur_factor: float, min_spur_mm: float, iters: int):
    """Remove short terminal branches (skeleton spurs). Returns boolean keep-mask."""
    n = adj.shape[0]
    alive = np.ones(n, dtype=bool)
    a = adj.tolil()
    nbrs = [list(r) for r in a.rows]
    wts = [list(r) for r in a.data]
    for _ in range(iters):
        deg = np.array([sum(alive[k] for k in nbrs[i]) if alive[i] else 0 for i in range(n)])
        changed = False
        for e in np.nonzero((deg == 1) & alive)[0]:
            if not alive[e]:
                continue
            chain, length = [e], 0.0
            prev, cur = -1, e
            junction = -1
            while True:
                nxt = [(k, w) for k, w in zip(nbrs[cur], wts[cur]) if alive[k] and k != prev]
                if not nxt:
                    break  # isolated segment: keep
                if len(nxt) >= 2:
                    junction = cur
                    chain.pop()  # `cur` is the junction itself
                    break
                k, w = nxt[0]
                length += w
                prev, cur = cur, k
                chain.append(cur)
                if deg[cur] == 1:
                    break  # reached another endpoint: whole segment, keep
            if junction >= 0:
                limit = max(min_spur_mm, spur_factor * float(radius[junction]))
                if length < limit:
                    alive[chain] = False
                    changed = True
        if not changed:
            break
    return alive


def build_tree(
    mask: np.ndarray,
    spacing_mm: float,
    seed_zyx: tuple[int, int, int],
    spur_factor: float = 1.5,
    min_spur_mm: float = 2.0,
    prune_iters: int = 4,
) -> AirwayTree:
    objs = ndi.find_objects(mask.astype(np.uint8))
    if not objs:
        raise ValueError("Empty airway mask")
    box = objs[0]
    margin = 2
    sl = tuple(
        slice(max(0, s.start - margin), min(dim, s.stop + margin)) for s, dim in zip(box, mask.shape)
    )
    sub = mask[sl]
    offset = tuple(s.start for s in sl)

    dt = ndi.distance_transform_edt(sub, sampling=spacing_mm).astype(np.float32)
    skel = skeletonize(sub).astype(bool)

    # keep the skeleton component nearest the seed
    lab, nlab = ndi.label(skel, structure=np.ones((3, 3, 3)))
    if nlab == 0:
        raise ValueError("Skeletonization produced no voxels")
    if nlab > 1:
        pts = np.argwhere(skel)
        seed_local = np.array(seed_zyx) - np.array(offset)
        d = np.linalg.norm(pts - seed_local, axis=1)
        keep = lab[tuple(pts[np.argmin(d)])]
        sizes = ndi.sum(skel, lab, index=np.arange(1, nlab + 1))
        # prefer the component containing the seed-nearest voxel unless it is tiny
        if sizes[keep - 1] < 0.2 * sizes.max():
            keep = int(np.argmax(sizes)) + 1
        skel = lab == keep

    coords, g = _voxel_graph(skel, spacing_mm)
    radius = dt[tuple(coords.T)]
    # Spanning tree on a radius-aware cost, not plain length. Thinning leaves several parallel strands in
    # wide or irregular regions (AeroPath case 1 around the carina); a length-only MST can tie the trachea
    # to the rest of the tree through thin wall-hugging nodes (0.8 mm radius in a 6 mm wide lumen), which a
    # scope-size limit then cuts. Cost = length / (min radius + 0.5)^2 makes the tree run through the thick
    # central strands. Path lengths still use the true Euclidean edge lengths.
    gc = g.tocoo()
    r_min = np.minimum(radius[gc.row], radius[gc.col])
    cost = coo_matrix((gc.data / (r_min + 0.5) ** 2, (gc.row, gc.col)), shape=g.shape).tocsr()
    tree_c = minimum_spanning_tree(cost).tocoo()
    lengths = np.asarray(g[tree_c.row, tree_c.col]).ravel()
    mst = coo_matrix((lengths, (tree_c.row, tree_c.col)), shape=g.shape).tocsr()
    mst = mst + mst.T

    alive = _prune_spurs(mst, radius, spur_factor, min_spur_mm, prune_iters)
    sel = np.nonzero(alive)[0]
    adj = mst[sel][:, sel].tocsr()
    # drop anything disconnected after pruning (keep largest component)
    ncomp, comp = connected_components(adj, directed=False)
    if ncomp > 1:
        big = np.argmax(np.bincount(comp))
        sel2 = np.nonzero(comp == big)[0]
        sel = sel[sel2]
        adj = adj[sel2][:, sel2].tocsr()
    coords = coords[sel]
    radius = radius[sel]

    glob_zyx = coords + np.array(offset)
    pts_mm = glob_zyx[:, ::-1].astype(np.float64) * spacing_mm
    seed_mm = np.array(seed_zyx[::-1], dtype=np.float64) * spacing_mm
    # Root = the open proximal end of the trachea. Two simpler rules fail on real CTs:
    #  - nearest endpoint to the seed: the seed search window is 80 mm, so the seed can sit deep in the
    #    trachea and thin distal tips can be closer to it than the tracheal tip;
    #  - widest endpoint: a short thick stub of central airway can be wider than the trachea's skeleton tip
    #    (AeroPath case 1: an 8.4 mm stub ~50 mm below the real tip).
    # So: keep endpoints at least half as wide as the widest (the trachea and any thick stubs, never
    # distal tips), take those farthest from the centroid of the whole tree (the tracheal tip lies far
    # from the bulk of the lung airways), and break near-ties by distance to the seed.
    deg = np.diff(adj.indptr)
    cand = np.nonzero(deg == 1)[0]
    if cand.size == 0:
        cand = np.arange(len(pts_mm))
    wide = cand[radius[cand] >= 0.5 * radius[cand].max()]
    d_cent = np.linalg.norm(pts_mm[wide] - pts_mm.mean(axis=0), axis=1)
    far = wide[d_cent >= 0.9 * d_cent.max()]
    root = int(far[np.argmin(np.linalg.norm(pts_mm[far] - seed_mm, axis=1))])
    return AirwayTree(
        pts_mm=pts_mm,
        radius_mm=radius.astype(np.float64),
        adj=adj,
        spacing_mm=spacing_mm,
        root=root,
        dt=dt,
        dt_offset_zyx=offset,
    )
