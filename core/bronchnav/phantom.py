"""Synthetic chest-CT phantom with a branching airway tree and a free-standing nodule.

Purpose: develop and regression-test the pipeline without real patient data.
This is NOT anatomically accurate. The trachea opens at the highest z index (superior end, like a CT
that starts at the neck); z increases toward the head, matching LPS.

Frame while building: x, y, z in mm with z measured from the trachea opening downward. The final array
(z, y, x) is flipped along z so the trachea ends up at the high-z (superior) end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

SPACING_XYZ = (0.7, 0.7, 1.0)
SIZE_XYZ = (200, 200, 200)  # voxels -> 140 x 140 x 200 mm


@dataclass
class PhantomInfo:
    image: sitk.Image
    nodule_center_mm: tuple[float, float, float]  # physical (x, y, z), mm
    nodule_radius_mm: float
    airway_truth: np.ndarray  # (z, y, x) bool lumen ground truth
    nodule_hu: float = 30.0


def _paint_capsule(mask, grids, a, b, r):
    """Set mask where voxel centers are within r mm of segment a-b. grids = (X, Y, Z) 1-D mm coords."""
    gx, gy, gz = grids
    lo = np.minimum(a, b) - r - 1.0
    hi = np.maximum(a, b) + r + 1.0
    sl = []
    for g, l, h in ((gz, lo[2], hi[2]), (gy, lo[1], hi[1]), (gx, lo[0], hi[0])):
        idx = np.nonzero((g >= l) & (g <= h))[0]
        if idx.size == 0:
            return
        sl.append(slice(idx[0], idx[-1] + 1))
    Z, Y, X = np.meshgrid(gz[sl[0]], gy[sl[1]], gx[sl[2]], indexing="ij")
    p = np.stack([X, Y, Z], axis=-1)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    closest = a + t[..., None] * ab
    d = np.linalg.norm(p - closest, axis=-1)
    mask[tuple(sl)] |= d <= r


def _rotate(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


@dataclass(frozen=True)
class NoduleSpec:
    """A nodule to draw into the phantom: centre in physical LPS mm, radius in mm, density in HU."""

    center_mm: tuple[float, float, float]
    radius_mm: float = 5.0
    hu: float = 30.0


def make_phantom(
    seed: int = 0,
    noise_hu: float = 20.0,
    generations: int = 5,
    leak: bool = False,
    nodule: NoduleSpec | None = None,
) -> PhantomInfo:
    """`nodule`: draw this nodule instead of the built-in one. It must lie entirely inside a lung and clear of the
    airways, otherwise ValueError. The rest of the image (airways, noise) is identical either way."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = SIZE_XYZ
    sx, sy, sz = SPACING_XYZ
    gx, gy, gz = np.arange(nx) * sx, np.arange(ny) * sy, np.arange(nz) * sz
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")

    # Body and lungs
    body = ((X - 70) / 66) ** 2 + ((Y - 70) / 52) ** 2 <= 1.0
    lungs = np.zeros_like(body)
    lung_centers = [(36.0, 72.0, 115.0), (104.0, 72.0, 115.0)]
    lung_axes = (28.0, 40.0, 68.0)
    for cx, cy, cz in lung_centers:
        lungs |= ((X - cx) / lung_axes[0]) ** 2 + ((Y - cy) / lung_axes[1]) ** 2 + ((Z - cz) / lung_axes[2]) ** 2 <= 1.0
    lungs &= body

    # Airway tree (lumen)
    lumen = np.zeros_like(body)
    grids = (gx, gy, gz)

    def in_lung(p, shrink=0.88):
        for cx, cy, cz in lung_centers:
            q = ((p[0] - cx) / (lung_axes[0] * shrink)) ** 2 + ((p[1] - cy) / (lung_axes[1] * shrink)) ** 2 + (
                (p[2] - cz) / (lung_axes[2] * shrink)
            ) ** 2
            if q <= 1.0:
                return True
        return False

    trachea_top = np.array([70.0, 62.0, 0.0])
    carina = np.array([70.0, 62.0, 72.0])
    _paint_capsule(lumen, grids, trachea_top, carina, 8.0)

    ends: list[np.ndarray] = []

    def grow(start, direction, length, radius, gen):
        end = start + direction * length
        _paint_capsule(lumen, grids, start, end, radius)
        if gen >= generations or radius < 1.3:
            ends.append(end)
            return
        axis = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis) < 1e-3:
            axis = np.array([1.0, 0.0, 0.0])
        for sign in (-1.0, 1.0):
            for _ in range(30):
                ang = np.deg2rad(rng.uniform(22, 38)) * sign
                roll = rng.uniform(0, np.pi)
                ax = _rotate(axis, direction, roll)
                d2 = _rotate(direction, ax, ang)
                d2 = d2 + np.array([0.0, 0.0, 0.15])  # drift downward (+z)
                d2 /= np.linalg.norm(d2)
                child_len = length * rng.uniform(0.75, 0.9)
                if in_lung(end + d2 * child_len):
                    grow(end, d2, child_len, radius * 0.76, gen + 1)
                    break
            else:
                ends.append(end)

    # Main bronchi toward each lung
    for cx in (36.0, 104.0):
        target = np.array([cx, 66.0, 105.0])
        d = target - carina
        L = float(np.linalg.norm(d))
        grow(carina, d / L, L * 0.55, 5.5, 1)

    # Nodule: free-standing sphere in the right-side lung (x ~ 104), away from the airways
    best = None
    dist_to_lumen = ndi.distance_transform_edt(~lumen, sampling=(sz, sy, sx))
    cand = lungs & (dist_to_lumen > 9.0) & (X > 90)
    idx = np.argwhere(cand)
    if idx.size == 0:
        raise RuntimeError("phantom: no room for nodule")
    k = idx[rng.integers(0, len(idx))]  # always drawn, so the noise below is identical with a custom nodule
    best = np.array([gx[k[2]], gy[k[1]], gz[k[0]]])
    nod_r, nod_hu = 5.0, 30.0
    if nodule is not None:
        cx, cy, cz = nodule.center_mm
        best = np.array([cx + 70.0, cy + 70.0, 99.0 - cz])  # physical -> build frame (z counted from the trachea opening)
        nod_r, nod_hu = float(nodule.radius_mm), float(nodule.hu)
    nod_mask = ((X - best[0]) ** 2 + (Y - best[1]) ** 2 + (Z - best[2]) ** 2) <= nod_r**2
    if nodule is not None:
        near_airway = ndi.binary_dilation(lumen, iterations=3)  # lumen + wall + a 1 voxel margin
        if not nod_mask.any() or not lungs[nod_mask].all() or near_airway[nod_mask].any():
            raise ValueError("The nodule must lie entirely inside a lung and clear of the airways.")

    # Assemble HU
    hu = np.full(body.shape, -1000.0, dtype=np.float32)
    hu[body] = 40.0
    hu[lungs] = -850.0
    # airway wall: ~1 mm tissue ring around the lumen inside the lungs
    wall = ndi.binary_dilation(lumen, iterations=2) & ~lumen
    if leak and ends:
        # open the wall around one branch tip so lumen air connects to the parenchyma
        e = ends[0]
        hole = ((X - e[0]) ** 2 + (Y - e[1]) ** 2 + (Z - e[2]) ** 2) <= 4.0**2
        wall &= ~hole
    hu[wall & lungs] = 20.0
    hu[nod_mask] = nod_hu
    hu[lumen] = -1000.0
    hu += rng.normal(0.0, noise_hu, size=hu.shape).astype(np.float32)
    hu = ndi.gaussian_filter(hu, sigma=0.6)  # partial volume / reconstruction blur
    # a bit of "bone" so the intensity range looks like a CT
    hu[(X < 12) & body & (Z > 20)] = 600.0
    hu = np.clip(hu, -1024, 1500)

    # Anatomical orientation: z increases toward the head, so the trachea opens at the HIGH z index.
    hu = np.ascontiguousarray(hu[::-1])
    lumen = np.ascontiguousarray(lumen[::-1])

    img = sitk.GetImageFromArray(hu.astype(np.int16))
    img.SetSpacing(SPACING_XYZ)
    # Place it in LPS-ish physical space (origin offset so physical != index)
    img.SetOrigin((-70.0, -70.0, -100.0))
    if nodule is not None:
        center_phys = tuple(float(v) for v in nodule.center_mm)
    else:
        center_phys = tuple(
            float(v) for v in img.TransformIndexToPhysicalPoint(
                (int(round(best[0] / sx)), int(round(best[1] / sy)), nz - 1 - int(round(best[2] / sz)))
            )
        )
    return PhantomInfo(img, center_phys, nod_r, lumen, nod_hu)
