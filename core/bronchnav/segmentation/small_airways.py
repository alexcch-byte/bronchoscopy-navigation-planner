"""Extend an airway mask into thin distal branches that a single global threshold cannot reach.

A global HU threshold stops before the thin airways: their lumen is smaller than a voxel, so partial volume
lifts the centre of the lumen well above the HU of air, and by the time the threshold is high enough to
include them it also floods the lung (a leak). This module instead grows outward from the existing mask
through voxels that look like a *dark tube* (two strong curvatures across the lumen, little curvature along
it, measured with a Hessian at a few small scales) and are not brighter than `hu_max`. Lung parenchyma
is not tubular, so the growth cannot flood it, and the growth is limited to `reach_mm` from the starting mask
and to a bounded added volume.

Arrays are (z, y, x) in HU with isotropic `spacing_mm`. The work runs on the GPU through CuPy when one is
usable (see `bronchnav.accel`) and on the CPU otherwise; both give the same mask.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage as ndi

from .. import accel

DEFAULT_SIGMAS_MM = (0.75, 1.0)


def _eig_sym3(xp, a11, a22, a33, a12, a13, a23):
    """Eigenvalues, largest first, of symmetric 3x3 matrices given as component arrays (closed form)."""
    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * (a12 * a12 + a13 * a13 + a23 * a23)
    p = xp.sqrt(p2 / 6.0)
    safe = xp.where(p > 1e-9, p, 1.0)
    det_b = (b11 * (b22 * b33 - a23 * a23) - a12 * (a12 * b33 - a23 * a13) + a13 * (a12 * a23 - b22 * a13)) / (safe**3)
    phi = xp.arccos(xp.clip(det_b / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * xp.cos(phi)
    e3 = q + 2.0 * p * xp.cos(phi + 2.0 * math.pi / 3.0)
    return e1, 3.0 * q - e1 - e3, e3


def _tube_device(hu, spacing_mm: float, sigmas_mm, axial_ratio: float):
    xp, ndx = accel.backend()
    x = xp.asarray(hu, dtype=xp.float32)
    best = xp.zeros(x.shape, dtype=xp.float32)
    for sg in sigmas_mm:
        s = sg / spacing_mm

        def d2(a, b):
            order = [0, 0, 0]
            order[a] += 1
            order[b] += 1
            return ndx.gaussian_filter(x, s, order=order, mode="nearest")

        l1, l2, l3 = _eig_sym3(xp, d2(0, 0), d2(1, 1), d2(2, 2), d2(0, 1), d2(0, 2), d2(1, 2))
        ok = (l2 > 0) & (l1 > 0) & (xp.abs(l3) < axial_ratio * xp.maximum(l2, 1e-3))
        best = xp.maximum(best, xp.where(ok, (s * s) * l2, 0.0).astype(xp.float32))
    return best


def tube_measure(hu: np.ndarray, spacing_mm: float, sigmas_mm=DEFAULT_SIGMAS_MM, axial_ratio: float = 0.6) -> np.ndarray:
    """Dark-tube strength in HU (scale-normalised curvature across the lumen), 0 where not tube-like.

    For a dark tube the Hessian has two large positive eigenvalues (HU rises away from the axis) and one near
    zero (along the axis). The strength is sigma^2 * (second largest eigenvalue), kept only where the axial
    eigenvalue is small compared with it. The maximum over scales is returned (a NumPy array).
    """
    return accel.to_host(_tube_device(hu, spacing_mm, sigmas_mm, axial_ratio))


def _bbox(xp, m):
    box = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = xp.nonzero(m.any(axis=other))[0]
        box.append((int(idx[0]), int(idx[-1]) + 1))
    return box


class Prepared:
    """The expensive, parameter-free part of the extension (crop, tube measure, smoothed HU), reusable across
    different thresholds. Its arrays live on the GPU when one is in use."""

    def __init__(self, hu, spacing_mm, mask, reach_mm, sigmas_mm):
        xp, ndx = accel.backend()
        self.spacing_mm, self.reach_mm = spacing_mm, reach_mm
        self.layers_max = max(1, int(np.ceil(reach_mm / spacing_mm)))
        st26 = ndi.generate_binary_structure(3, 3)
        near = accel.dilate(ndx, xp.asarray(mask), st26, self.layers_max)
        pad = int(np.ceil(3.0 / spacing_mm)) + 2
        bb = _bbox(xp, near)
        self.box = tuple(slice(max(0, lo - pad), min(n, hi + pad)) for (lo, hi), n in zip(bb, hu.shape))
        sub = xp.asarray(hu[self.box], dtype=xp.float32)
        self.m0, self.region = xp.asarray(mask[self.box]), near[self.box]
        self.tube = _tube_device(sub, spacing_mm, sigmas_mm, 0.6)
        self.smooth = ndx.gaussian_filter(sub, sigmas_mm[0] / spacing_mm)


def prepare(hu: np.ndarray, spacing_mm: float, mask: np.ndarray, reach_mm: float = 30.0, sigmas_mm=DEFAULT_SIGMAS_MM) -> Prepared:
    return Prepared(hu, spacing_mm, mask, reach_mm, sigmas_mm)


def extend_into_small_airways(
    hu: np.ndarray,
    spacing_mm: float,
    mask: np.ndarray,
    reach_mm: float = 30.0,
    tube_min_hu: float = 50.0,
    hu_max: float = -550.0,
    sigmas_mm=DEFAULT_SIGMAS_MM,
    max_added_ml: float = 40.0,
    thicken: bool = True,
    thicken_hu: float | None = -700.0,
    prepared: Prepared | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (new mask, info). The input mask is never reduced: the result is a superset of it.

    Growth is 26-connected through candidate voxels (tube measure >= `tube_min_hu` and smoothed HU below
    `hu_max`), one voxel layer at a time, for at most reach_mm / spacing layers. If more than `max_added_ml`
    would be added, the growth is cut back to the layers that fit (the outermost layers are the least reliable).
    Pass `prepared` (from `prepare`) to reuse the tube measure across several parameter settings.
    """
    voxel_ml = spacing_mm**3 / 1000.0
    if not mask.any():
        return mask.copy(), {"added_ml": 0.0, "layers": 0, "capped": False}
    xp, ndx = accel.backend()
    st26 = ndi.generate_binary_structure(3, 3)
    st6 = ndi.generate_binary_structure(3, 1)
    p = prepared or Prepared(hu, spacing_mm, mask, reach_mm, sigmas_mm)
    box, m0, tube, smooth = p.box, p.m0, p.tube, p.smooth
    cand = (tube >= tube_min_hu) & (smooth < hu_max) & p.region & ~m0

    cur = m0.copy()
    layer_of = xp.zeros(m0.shape, dtype=xp.int16)
    layers = 0
    for k in range(1, p.layers_max + 1):
        ring = accel.dilate(ndx, cur, st26) & cand & ~cur
        if not bool(ring.any()):
            break
        cur |= ring
        layer_of[ring] = k
        layers = k
    added = cur & ~m0
    capped = False
    if float(added.sum()) * voxel_ml > max_added_ml:  # keep the inner layers only
        capped = True
        order = xp.sort(layer_of[added])
        keep_n = int(max_added_ml / voxel_ml)
        k_cut = int(order[min(keep_n, int(order.size) - 1)])
        cur = m0 | (added & (layer_of < k_cut))
        added = cur & ~m0
        layers = max(0, k_cut - 1)
    if thicken and bool(added.any()):
        # the ridge response is about one voxel wide; give the new branches a lumen-like width without
        # opening anything that is not already air-like
        limit = hu_max + 60.0 if thicken_hu is None else thicken_hu
        cur = cur | (accel.dilate(ndx, added, st6) & (smooth < limit))
        added = cur & ~m0
    out = mask.copy()
    out[box] = accel.to_host(cur)
    return out, {"added_ml": round(float(added.sum()) * voxel_ml, 2), "layers": int(layers), "capped": bool(capped)}
