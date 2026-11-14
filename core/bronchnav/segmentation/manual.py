"""Operator-drawn airways: add a branch the segmentation missed.

The operator clicks along a thin airway on the CT, starting on or next to an airway that was already found.
The clicks become a polyline, which is painted into the airway mask as a tube of a stated radius. The radius is
the operator's own statement; it is not measured from the scan. Everything built from the edited mask (tree,
route, mesh, video) therefore describes a branch the software did not detect, and callers must say so.

Points are physical LPS mm, converted with the mask's `Grid`.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from ..geometry import Grid

MAX_POINTS = 60
MAX_LENGTH_MM = 120.0
MAX_BRIDGE_MM = 8.0
RADIUS_RANGE_MM = (0.4, 4.0)


def add_airway(
    mask: np.ndarray,
    grid: Grid,
    points_phys,
    radius_mm: float,
    max_bridge_mm: float = MAX_BRIDGE_MM,
) -> tuple[np.ndarray, dict]:
    """Return (new mask, info). `mask` is (z, y, x) bool on `grid`; the input is not modified.

    The first point must lie within `max_bridge_mm` of the existing airway mask; the gap is bridged so the new
    branch is connected. Raises ValueError (with a message fit to show the operator) on bad input."""
    pts = np.asarray(points_phys, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("The new airway needs points with three finite coordinates each.")
    if not (2 <= len(pts) <= MAX_POINTS):
        raise ValueError(f"The new airway needs between 2 and {MAX_POINTS} points (it has {len(pts)}).")
    lo, hi = RADIUS_RANGE_MM
    if not (np.isfinite(radius_mm) and lo <= radius_mm <= hi):
        raise ValueError(f"The airway radius must be between {lo} and {hi} mm.")
    sp = float(grid.spacing_mm)
    shape = np.array(mask.shape)  # z, y, x
    mm = grid.points_to_grid(pts)  # grid-frame mm: x, y, z
    idx = mm / sp  # voxel index x, y, z
    if (idx < -0.5).any() or (idx > (shape[::-1] - 0.5)).any():
        raise ValueError("A point of the new airway lies outside the CT volume.")

    # connect the first point to the nearest voxel of the existing airway
    reach = int(np.ceil(max_bridge_mm / sp)) + 1
    c = np.clip(np.round(idx[0][::-1]).astype(int), 0, shape - 1)  # z, y, x
    a, b = np.maximum(c - reach, 0), np.minimum(c + reach + 1, shape)
    window = mask[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    near = np.argwhere(window)
    if len(near) == 0:
        raise ValueError("Start the new airway on, or within a few mm of, an airway that was already found.")
    near_mm = (near + a)[:, ::-1] * sp
    d = np.linalg.norm(near_mm - mm[0][None, :], axis=1)
    k = int(np.argmin(d))
    bridge_mm = float(d[k])
    if bridge_mm > max_bridge_mm:
        raise ValueError(
            f"The first point is {bridge_mm:.1f} mm from the nearest airway; start within {max_bridge_mm:.0f} mm of one."
        )
    poly = mm if bridge_mm <= 0.5 * sp else np.vstack([near_mm[k], mm])

    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    length_mm = float(seg.sum())
    if length_mm > MAX_LENGTH_MM:
        raise ValueError(f"The new airway is {length_mm:.0f} mm long; the limit is {MAX_LENGTH_MM:.0f} mm.")
    if length_mm < 3.0:
        raise ValueError("The new airway is shorter than 3 mm; click further along it.")

    # densify so consecutive stamped voxels touch, then grow the stamped line into a tube of the given radius
    dense = [poly[0][None, :]]
    for p0, p1, s in zip(poly[:-1], poly[1:], seg):
        n = max(1, int(np.ceil(s / (0.5 * sp))))
        dense.append(p0 + (p1 - p0) * (np.arange(1, n + 1) / n)[:, None])
    zyx = np.clip(np.round(np.vstack(dense)[:, ::-1] / sp).astype(int), 0, shape - 1)
    pad = int(np.ceil(radius_mm / sp)) + 2
    lo_i, hi_i = np.maximum(zyx.min(axis=0) - pad, 0), np.minimum(zyx.max(axis=0) + pad + 1, shape)
    crop = tuple(slice(l, h) for l, h in zip(lo_i, hi_i))
    stamp = np.zeros(tuple(hi_i - lo_i), dtype=bool)
    stamp[tuple((zyx - lo_i).T)] = True
    tube = ndi.distance_transform_edt(~stamp, sampling=sp) <= radius_mm

    out = mask.astype(bool).copy()
    before = int(out.sum())
    out[crop] |= tube
    added = int(out.sum()) - before
    return out, {
        "length_mm": round(length_mm, 1),
        "radius_mm": float(radius_mm),
        "bridge_mm": round(bridge_mm, 1),
        "added_ml": round(added * sp**3 / 1000.0, 3),
        "points": int(len(pts)),
    }
