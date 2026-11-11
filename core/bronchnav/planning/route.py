"""Route planning on the airway centerline tree (Phase 1: nearest eligible airway point).

All points are grid-frame mm (x, y, z). Phase 3 replaces exit selection with full scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import dijkstra

from ..airway_tree import AirwayTree

# Skeleton radii (distance transform) are only accurate to about a voxel, so a node counts as passable
# for the scope when its diameter is at least this fraction of the requested minimum. The exit point
# itself must meet the full minimum.
PASSABLE_FACTOR = 0.8

# Multiples of the base smoothing sigma (2.5 mm) tried by the adaptive smoothing: 2.5, 5, 8, 12.5 mm.
SMOOTH_SCALES = (1.0, 2.0, 3.2, 5.0)


@dataclass
class Route:
    ok: bool
    message: str
    origin_node: int
    exit_node: int
    points_mm: np.ndarray  # (M, 3) smoothed camera path, ~uniform step
    radius_mm: np.ndarray  # (M,) airway lumen radius along the route (centerline radius, not wall clearance)
    tangents: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3) parallel-transported "up"
    s_mm: np.ndarray  # (M,) arclength
    bifurcations: list[dict] = field(default_factory=list)  # {"s_mm", "index", "point_mm"}
    target_mm: np.ndarray | None = None
    target_radius_mm: float = 0.0
    gap_mm: float = 0.0  # extraluminal distance from route end to target surface
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    clearance_mm: np.ndarray | None = None  # (M,) camera distance to the nearest airway wall


def nearest_node(tree: AirwayTree, pt_mm: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(tree.pts_mm - np.asarray(pt_mm)[None, :], axis=1)))


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.floor(s[-1] / step)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(s_new, s, poly[:, k]) for k in range(3)], axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _parallel_transport(T: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 1.0, 0.0])
    if abs(T[0] @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n0 = ref - (ref @ T[0]) * T[0]
    N = np.zeros_like(T)
    N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(T)):
        n = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
        norm = np.linalg.norm(n)
        N[i] = n / norm if norm > 1e-9 else N[i - 1]
    return N


def _trim_start_hook(
    pts: np.ndarray, max_trim_mm: float = 25.0, angle_deg: float = 60.0, ref_mm: tuple[float, float] = (30.0, 50.0)
) -> int:
    """Index of the first centerline point to keep.

    Thinning an open tube end can leave a short hooked tail (AeroPath case 22: an 8 mm sideways dogleg of the
    skeleton, radius growing 1.7 to 7.6 mm over ~16 mm, before the real tracheal axis). If the first points head more than
    `angle_deg` away from the route's main direction (taken between `ref_mm` along the path), drop them, never
    more than `max_trim_mm`. Gentle curvature is left alone.
    """
    n = len(pts)
    if n < 8:
        return 0
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    a, b = np.searchsorted(s, ref_mm[0]), np.searchsorted(s, ref_mm[1])
    if b >= n:  # short route: use whatever lies beyond the trim window
        a, b = min(n - 3, np.searchsorted(s, max_trim_mm)), n - 1
    ref = pts[b] - pts[a]
    if np.linalg.norm(ref) < 1e-6:
        return 0
    ref /= np.linalg.norm(ref)
    k = 3
    i = 0
    while i < n - 2 * k - 1 and s[i] < max_trim_mm:
        t = pts[min(i + k, n - 1)] - pts[max(i - k, 0)]
        nt = np.linalg.norm(t)
        if nt > 1e-9 and np.degrees(np.arccos(np.clip((t / nt) @ ref, -1.0, 1.0))) <= angle_deg:
            break
        i += 1
    return i if s[i] <= max_trim_mm else 0


def _walk_back(pred: np.ndarray, start: int, goal: int) -> list[int]:
    path = [goal]
    while path[-1] != start:
        p = int(pred[path[-1]])
        if p < 0:
            raise ValueError("No connected path from origin to the chosen exit point.")
        path.append(p)
    return path[::-1]


def _smooth_path(
    tree: AirwayTree,
    path: list[int],
    step_mm: float = 0.5,
    smooth_sigma_mm: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample and smooth the centerline path, keeping it inside the airway lumen."""
    raw = tree.pts_mm[path]
    res = _resample(raw, step_mm)
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))])
    s_res = np.linspace(0.0, s_raw[-1], len(res))  # `res` is uniform in arclength by construction
    r_raw = np.interp(s_res, s_raw, tree.radius_mm[path])

    # Adaptive smoothing. A centerline turns sharply where it passes through a junction (40-60 degrees within
    # a few mm on AeroPath), but a scope does not have to follow the exact centerline: in a wide airway there
    # is room to cut the corner. Smooth at several scales and, at every point, keep the smoothest version
    # that stays well inside the lumen (clearance >= half the centerline radius). Narrow airways keep the
    # small scale because the wider ones would leave the lumen there.
    sig = smooth_sigma_mm / step_mm
    floor = np.maximum(0.5 * r_raw, 0.4)
    cands = [gaussian_filter1d(res, sig * m, axis=0, mode="nearest") for m in SMOOTH_SCALES]
    sm = cands[0].copy()
    for cand in cands[1:]:
        good = tree.sample_radius(cand) >= floor
        sm[good] = cand[good]
    sm = gaussian_filter1d(sm, 1.0 / step_mm, axis=0, mode="nearest")  # blend the switches between scales
    for _ in range(4):  # pull points that left the lumen back toward the raw centerline
        bad = tree.sample_radius(sm) < np.maximum(0.5 * r_raw, 0.4)
        if not bad.any():
            break
        sm[bad] = res[bad]
        sm[0], sm[-1] = res[0], res[-1]
        sm = gaussian_filter1d(sm, 1.0 / step_mm, axis=0, mode="nearest")
    bad = tree.sample_radius(sm) < np.maximum(0.5 * r_raw, 0.4)  # final hard fallback, no further smoothing
    sm[bad] = res[bad]
    sm[0], sm[-1] = res[0], res[-1]
    sm = _resample(sm, step_mm)  # smoothing contracts bends: restore uniform spacing before derivatives

    seg = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return sm, s, s_raw


def _compute_metrics(
    tree: AirwayTree,
    path: list[int],
    sm: np.ndarray,
    s: np.ndarray,
    s_raw: np.ndarray,
    target_mm: np.ndarray,
    target_radius_mm: float,
    step_mm: float = 0.5,
    min_bend_radius_limit_mm: float = 12.0,
) -> tuple[dict, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], float]:
    """Compute geometric metrics, frames, and warnings along the smoothed path."""
    warnings: list[str] = []
    T = _unit(np.gradient(sm, axis=0))
    T = _unit(gaussian_filter1d(T, 1.0 / step_mm, axis=0, mode="nearest"))
    N = _parallel_transport(T)

    s_len = max(float(s[-1]), 1e-6)
    s_raw_len = max(float(s_raw[-1]), 1e-6)

    # Lumen radius comes from the centerline; wall clearance (distance transform at the camera) is separate.
    radius = np.interp(s / s_len * s_raw[-1], s_raw, tree.radius_mm[path])
    clearance = tree.sample_radius(sm)
    inside = float(np.mean(clearance > 0.2))

    # Bifurcations along the path (interior junction nodes only, located by arclength so order is monotonic)
    bif = []
    deg = tree.degree
    for k in range(1, len(path) - 1):
        node = path[k]
        if deg[node] < 3:
            continue
        s_exp = s_raw[k] / s_raw_len * s[-1]
        lo = min(int(np.searchsorted(s, s_exp - 6.0)), len(s) - 1)
        hi = max(int(np.searchsorted(s, s_exp + 6.0)), lo + 1)
        i = lo + int(np.argmin(np.linalg.norm(sm[lo:hi] - tree.pts_mm[node][None, :], axis=1)))
        min_sep = max(2.0, 1.5 * float(tree.radius_mm[node]))
        if bif and abs(s[i] - bif[-1]["s_mm"]) < min_sep:
            continue
        bif.append({"s_mm": float(s[i]), "index": i, "point_mm": sm[i].tolist()})

    # Curvature (1/mm) -> minimum bend radius
    dT = np.linalg.norm(np.gradient(T, axis=0), axis=1) / step_mm
    min_bend_r = float(1.0 / max(dT.max(), 1e-6))
    if min_bend_r < min_bend_radius_limit_mm:
        warnings.append(
            f"Minimum bend radius is {min_bend_r:.1f} mm, which is below the limit of "
            f"{min_bend_radius_limit_mm:.1f} mm; bronchoscope may encounter excessive resistance."
        )

    end = sm[-1]
    gap = max(0.0, float(np.linalg.norm(target_mm - end)) - target_radius_mm)
    if gap > 20.0:
        warnings.append(
            f"Route ends {gap:.1f} mm from the nodule surface; the remaining path is outside the airway "
            "(extraluminal) and needs another approach."
        )
    if inside < 0.99:
        warnings.append(f"Only {inside * 100:.1f}% of the smoothed path lies inside the airway lumen.")

    metrics = {
        "length_mm": float(s[-1]),
        "min_radius_mm": float(radius.min()),
        "min_diameter_mm": float(2 * radius.min()),
        "min_wall_clearance_mm": float(clearance.min()),
        "n_bifurcations": len(bif),
        "min_bend_radius_mm": min_bend_r,
        "fraction_inside_lumen": inside,
        "exit_to_target_center_mm": float(np.linalg.norm(target_mm - end)),
        "gap_to_nodule_surface_mm": gap,
    }
    return metrics, warnings, radius, clearance, T, N, bif, gap


def plan_route(
    tree: AirwayTree,
    target_mm: np.ndarray,
    target_radius_mm: float,
    origin_mm: np.ndarray | None = None,
    min_diameter_mm: float = 3.0,
    step_mm: float = 0.5,
    smooth_sigma_mm: float = 2.5,
    min_bend_radius_limit_mm: float = 12.0,
    force_exit_mm: np.ndarray | None = None,
) -> Route:
    """`force_exit_mm`: end the route at the airway point nearest this position (grid-frame mm) instead of the
    automatic choice. This is how an operator overrides the planner: in an airway tree there is exactly one
    path to any point, so choosing a different airway to end in is choosing a different route."""
    target_mm = np.asarray(target_mm, dtype=np.float64)
    warnings: list[str] = []
    n_nodes = len(tree.pts_mm)

    origin = tree.root if origin_mm is None else nearest_node(tree, origin_mm)
    if origin_mm is not None:
        d0 = float(np.linalg.norm(tree.pts_mm[origin] - np.asarray(origin_mm)))
        if d0 > 5.0:
            warnings.append(f"Origin is {d0:.1f} mm from the nearest airway centerline point; snapped to it.")

    d_all = np.linalg.norm(tree.pts_mm - target_mm[None, :], axis=1)

    # The scope must fit the whole way: route only through passable nodes, and the exit must be reachable.
    passable = tree.radius_mm * 2.0 >= PASSABLE_FACTOR * min_diameter_mm
    passable[origin] = True
    keep = np.nonzero(passable)[0]
    sub = tree.adj[keep][:, keep].tocsr()
    o_sub = int(np.searchsorted(keep, origin))
    dist_sub, pred_sub = dijkstra(sub, directed=False, indices=o_sub, return_predecessors=True)

    eligible = np.zeros(n_nodes, dtype=bool)
    eligible[keep[np.isfinite(dist_sub)]] = True
    eligible &= tree.radius_mm * 2.0 >= min_diameter_mm
    eligible[origin] = False

    constrained = bool(eligible.any())
    if force_exit_mm is not None:
        exit_node = nearest_node(tree, force_exit_mm)
        if exit_node == origin:
            raise ValueError("The chosen end point is the start of the route; pick a point further along an airway.")
        in_sub = bool(passable[exit_node]) and bool(np.isfinite(dist_sub[int(np.searchsorted(keep, exit_node))]))
        constrained = in_sub
        warnings.append("End point chosen by the operator (not the automatic choice).")
        if not in_sub:
            warnings.append(
                f"This end point cannot be reached with a {min_diameter_mm:.1f} mm scope: part of the way is narrower. "
                "The path is shown anyway."
            )
    else:
        if not constrained:
            warnings.append(
                f"No airway reachable from the origin admits a {min_diameter_mm:.1f} mm scope; "
                "ignoring the scope-size limit."
            )
            eligible = np.ones(n_nodes, dtype=bool)
            eligible[origin] = False
        exit_node = int(np.argmin(np.where(eligible, d_all, np.inf)))
        nearest_any = int(np.argmin(d_all))
        if constrained and nearest_any != exit_node:
            warnings.append(
                f"Closest airway point ({d_all[nearest_any]:.1f} mm from target) cannot be reached with a "
                f"{min_diameter_mm:.1f} mm scope; routing to the closest reachable point ({d_all[exit_node]:.1f} mm)."
            )

    if constrained:
        goal_sub = int(np.searchsorted(keep, exit_node))
        path = [int(keep[k]) for k in _walk_back(pred_sub, o_sub, goal_sub)]
    else:
        _, pred = dijkstra(tree.adj, directed=False, indices=origin, return_predecessors=True)
        path = _walk_back(pred, origin, exit_node)
    if len(path) < 2:
        raise ValueError("Origin and exit point are the same node; nothing to fly through.")
    if origin_mm is None:  # a user-chosen origin is respected; only the automatic start is cleaned up
        cut = _trim_start_hook(tree.pts_mm[path])
        if cut > 0 and len(path) - cut >= 2:
            trimmed = float(np.sum(np.linalg.norm(np.diff(tree.pts_mm[path[: cut + 1]], axis=0), axis=1)))
            path = path[cut:]
            warnings.append(
                f"Dropped the first {trimmed:.0f} mm of the centerline: a skeleton hook at the tracheal tip "
                "pointed away from the route."
            )

    sm, s, s_raw = _smooth_path(tree, path, step_mm=step_mm, smooth_sigma_mm=smooth_sigma_mm)
    (
        metrics,
        m_warnings,
        radius,
        clearance,
        T,
        N,
        bif,
        gap,
    ) = _compute_metrics(
        tree,
        path,
        sm,
        s,
        s_raw,
        target_mm,
        target_radius_mm,
        step_mm=step_mm,
        min_bend_radius_limit_mm=min_bend_radius_limit_mm,
    )
    warnings.extend(m_warnings)
    # How well the scope tip points at the nodule from the end of the route (0 = straight at it).
    to_t = target_mm - sm[-1]
    nt = float(np.linalg.norm(to_t))
    metrics["approach_angle_deg"] = float(np.degrees(np.arccos(np.clip(T[-1] @ (to_t / nt), -1.0, 1.0)))) if nt > 1e-6 else 0.0

    return Route(
        ok=True,
        message="ok",
        origin_node=int(origin),
        exit_node=int(exit_node),
        points_mm=sm,
        radius_mm=radius,
        tangents=T,
        normals=N,
        s_mm=s,
        bifurcations=bif,
        target_mm=target_mm,
        target_radius_mm=float(target_radius_mm),
        gap_mm=gap,
        warnings=warnings,
        metrics=metrics,
        clearance_mm=clearance,
    )


def exit_candidates(
    tree: AirwayTree,
    target_mm: np.ndarray,
    target_radius_mm: float,
    origin_mm: np.ndarray | None = None,
    min_diameter_mm: float = 3.0,
    n: int = 6,
    search_mm: float = 45.0,
    min_sep_mm: float = 10.0,
) -> list[dict]:
    """Ranked alternative end points for the route (grid-frame mm), to help an operator choose by eye.

    Candidates are airway centerline points within `search_mm` of the nodule centre. For each: the gap left to
    the nodule surface (outside the airway), the route length, the smallest lumen diameter on the way, the
    approach angle (how far the direction of travel at that point is from pointing at the nodule) and whether
    a `min_diameter_mm` scope can get there. Lower score is better: gap + 0.25 * angle + a penalty when the
    scope does not fit. Points closer than `min_sep_mm` to a better one are dropped, so the list shows
    genuinely different airways, not neighbours on the same branch.
    """
    target_mm = np.asarray(target_mm, dtype=np.float64)
    origin = tree.root if origin_mm is None else nearest_node(tree, origin_mm)
    dist, pred = dijkstra(tree.adj, directed=False, indices=origin, return_predecessors=True)
    d_t = np.linalg.norm(tree.pts_mm - target_mm[None, :], axis=1)
    near = np.nonzero((d_t <= search_mm) & np.isfinite(dist) & (np.arange(len(d_t)) != origin))[0]
    scored = []
    for node in near:
        path = [int(node)]
        while path[-1] != origin and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        min_d = float(2.0 * tree.radius_mm[path].min())
        prev = int(pred[node])
        direction = tree.pts_mm[node] - tree.pts_mm[prev] if prev >= 0 else np.zeros(3)
        nd = float(np.linalg.norm(direction))
        to_t = target_mm - tree.pts_mm[node]
        nt = float(np.linalg.norm(to_t))
        angle = float(np.degrees(np.arccos(np.clip((direction / nd) @ (to_t / nt), -1.0, 1.0)))) if nd > 1e-9 and nt > 1e-9 else 90.0
        reachable = min_d >= PASSABLE_FACTOR * min_diameter_mm and tree.radius_mm[node] * 2.0 >= min_diameter_mm
        gap = max(0.0, float(d_t[node]) - float(target_radius_mm))
        score = gap + 0.25 * angle + (0.0 if reachable else 15.0)
        scored.append((score, int(node), gap, float(dist[node]), min_d, angle, bool(reachable)))
    scored.sort()
    out: list[dict] = []
    for score, node, gap, length, min_d, angle, reachable in scored:
        if any(np.linalg.norm(tree.pts_mm[node] - tree.pts_mm[o["node"]]) < min_sep_mm for o in out):
            continue
        out.append(
            {
                "node": node,
                "point_mm": tree.pts_mm[node].tolist(),
                "gap_mm": round(gap, 1),
                "route_length_mm": round(length, 1),
                "min_diameter_mm": round(min_d, 2),
                "approach_angle_deg": round(angle, 1),
                "reachable": reachable,
                "score": round(score, 1),
            }
        )
        if len(out) >= n:
            break
    return out
