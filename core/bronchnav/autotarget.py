"""Stand-in targets for testing on CTs without nodule annotations (e.g. AeroPath).

Pick a distal endpoint of the airway tree and place the target a few mm beyond it, inside the lung mask.
This is a test point in the lung parenchyma, not a nodule.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .airway_tree import AirwayTree


def choose_target(
    tree: AirwayTree,
    lungs: np.ndarray,
    spacing_mm: float,
    pctl: float = 0.7,
    ahead_mm: float = 15.0,
) -> tuple[int, np.ndarray, float, float]:
    """Returns (endpoint node, target in grid-frame mm, geodesic distance of the endpoint, mm placed ahead).

    `pctl` ranks endpoints by geodesic distance from the root (0 = nearest, 1 = farthest). Endpoints whose
    forward point is not inside `lungs` ((z, y, x) bool) are skipped, scanning outward from the requested rank.
    """
    dist, pred = dijkstra(tree.adj, directed=False, indices=tree.root, return_predecessors=True)
    ends = tree.endpoints
    ends = ends[np.isfinite(dist[ends])]
    if ends.size == 0:
        raise RuntimeError("airway tree has no endpoints")
    order = ends[np.argsort(dist[ends])]
    start = min(len(order) - 1, max(0, int(len(order) * pctl)))
    # scan outward from the requested rank: start, start+1, start-1, start+2, ...
    ranks = [start]
    for k in range(1, len(order)):
        if start + k < len(order):
            ranks.append(start + k)
        if start - k >= 0:
            ranks.append(start - k)
    shape_xyz = np.array(lungs.shape[::-1])
    for idx in ranks:
        e = int(order[idx])
        node, walked = e, 0.0  # direction: from a node ~6 mm back along the path to the endpoint
        while walked < 6.0 and pred[node] >= 0:
            walked += float(np.linalg.norm(tree.pts_mm[node] - tree.pts_mm[pred[node]]))
            node = int(pred[node])
        d = tree.pts_mm[e] - tree.pts_mm[node]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        d = d / n
        for ahead in (ahead_mm, 0.7 * ahead_mm, 0.5 * ahead_mm):
            p = tree.pts_mm[e] + d * ahead
            ix = np.round(p / spacing_mm).astype(int)  # x, y, z
            if np.all(ix >= 0) and np.all(ix < shape_xyz) and lungs[ix[2], ix[1], ix[0]]:
                return e, p, float(dist[e]), float(ahead)
    raise RuntimeError("no target inside the lung found")
