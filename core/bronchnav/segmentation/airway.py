"""Classical airway segmentation: trachea seed + threshold-ladder region growing with leak control.

Baseline for Phase 1. Reaches the larger airways; Phase 2 replaces/augments it with a learned model.
Arrays are (z, y, x) in HU. Spacing is isotropic `spacing_mm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

AIR_THRESHOLDS_HU = tuple(range(-950, -799, 25))  # -950 ... -800


@dataclass
class AirwaySegmentation:
    mask: np.ndarray  # (z, y, x) bool
    seed_zyx: tuple[int, int, int]
    threshold_hu: int
    volumes_ml: dict[int, float] = field(default_factory=dict)
    leak_detected: bool = False
    notes: list[str] = field(default_factory=list)


def _best_seed_in_slices(hu: np.ndarray, spacing_mm: float, zs, min_ring_tissue: float):
    """Best (score, z, y, x) round, enclosed, body-internal air blob over the given slice indices."""
    nz, ny, nx = hu.shape
    px_area = spacing_mm**2
    ring_px = max(2, int(round(3.0 / spacing_mm)))  # ~3 mm wall ring
    best = None
    for z in zs:
        air = hu[z] < -900
        lab, n = ndi.label(air)
        if n == 0:
            continue
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            # skip blobs touching the image border (outside air)
            if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == ny or sl[1].stop == nx:
                continue
            comp = lab[sl] == i
            area = comp.sum() * px_area
            if not (60.0 <= area <= 700.0):  # trachea cross-section ~ 100-400 mm^2
                continue
            aspect = max(h, w) / max(1, min(h, w))
            fill = comp.sum() / float(h * w)
            if aspect > 1.8 or fill < 0.55:
                continue
            # Must be wrapped in soft tissue (rejects air pockets between arm/torso/table and outside air)
            pad = ring_px + 1
            y0, x0 = max(0, sl[0].start - pad), max(0, sl[1].start - pad)
            y1, x1 = min(ny, sl[0].stop + pad), min(nx, sl[1].stop + pad)
            full = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            full[sl[0].start - y0 : sl[0].stop - y0, sl[1].start - x0 : sl[1].stop - x0] = comp
            ring = ndi.binary_dilation(full, iterations=ring_px) & ~ndi.binary_dilation(full, iterations=1)
            if ring.sum() == 0 or float((hu[z, y0:y1, x0:x1][ring] > -500).mean()) < min_ring_tissue:
                continue
            # Seed = deepest interior pixel (the centroid can fall outside an indented lumen)
            dt = ndi.distance_transform_edt(comp)
            iy, ix = np.unravel_index(np.argmax(dt), comp.shape)
            cy, cx = iy + sl[0].start, ix + sl[1].start
            mid_penalty = abs(cx - nx / 2) / nx  # trachea is near the midline
            score = fill - aspect * 0.1 - mid_penalty
            if best is None or score > best[0]:
                best = (score, z, int(cy), int(cx))
    return best


def find_trachea_seed(
    hu: np.ndarray,
    spacing_mm: float,
    search_mm: float = 80.0,
    prefer: str = "high",
    min_ring_tissue: float = 0.6,
    allow_fallback: bool = False,
) -> tuple[int, int, int]:
    """Find a voxel inside the trachea. Returns (z, y, x).

    Looks for a round, enclosed air blob wrapped in soft tissue, near the midline, in the first
    `search_mm` of the volume at one end. `prefer="high"` (default) tries the high-z end first: after
    `load_volume` (LPS) that is the superior end, where the trachea enters the scan; abdominal gas
    sits at the other end. When `prefer="high"`, raises ValueError if no seed is found in the superior
    slices rather than silently falling back to the inferior end (which risks latching onto bowel gas),
    unless `allow_fallback=True`. `prefer="low"` searches the inferior end first.
    """
    nz = hu.shape[0]
    n = max(3, min(nz, int(round(search_mm / spacing_mm))))
    high = range(nz - 1, nz - 1 - n, -1)
    low = range(0, n)
    primary = high if prefer == "high" else low
    best = _best_seed_in_slices(hu, spacing_mm, primary, min_ring_tissue)
    if best is not None:
        return best[1], best[2], best[3]
    if allow_fallback:
        secondary = low if prefer == "high" else high
        best = _best_seed_in_slices(hu, spacing_mm, secondary, min_ring_tissue)
        if best is not None:
            return best[1], best[2], best[3]
    end_name = "superior" if prefer == "high" else "inferior"
    raise ValueError(
        f"No trachea seed found in the {end_name} {int(round(search_mm))} mm of the volume; "
        f"check scan coverage or use prefer='{'low' if prefer == 'high' else 'high'}'."
    )


def segment_airway(
    hu: np.ndarray,
    spacing_mm: float,
    seed_zyx: tuple[int, int, int] | None = None,
    thresholds: tuple[int, ...] = AIR_THRESHOLDS_HU,
    leak_factor: float = 2.0,
    max_volume_ml: float = 400.0,
    min_leak_jump_ml: float = 30.0,
    connectivity: int = 1,
    min_trusted_ml: float = 10.0,
    early_leak_ml: float = 100.0,
) -> AirwaySegmentation:
    """Grow the airway from the trachea with the highest threshold that does not leak.

    For each threshold t (ascending) take the connected component of (hu < t) containing the seed.
    A leak is declared when the volume exceeds `max_volume_ml`, or when it both grows by more than
    `leak_factor`-times the previous step's volume AND gains more than `min_leak_jump_ml` mL.
    The absolute-jump condition stops genuine early growth (small trachea-only volume doubling as
    distal airways cross the threshold) from being mistaken for a leak. A parenchymal leak adds
    hundreds of mL at once. The previous threshold's component is returned on a leak.

    `connectivity` is the scipy structuring-element rank for labeling: 1 = face neighbours (6),
    2 = faces+edges (18), 3 = faces+edges+corners (26). Default 1: on AeroPath (27 CTs) 26-connected
    labeling lets a single diagonal noise voxel bridge an airway wall into the lung (one case became a
    1,126 mL leak at the first threshold) and gave lower Dice (0.806 vs 0.834) and more false positive
    volume (1.6 vs 0.3 mL). Higher connectivity can follow thin diagonal airways but is less leak-safe.

    While the previous region is smaller than `min_trusted_ml` it is only a fragment of the trachea
    (thick slices or an uncalibrated lumen can leave just a few mL at the first threshold), so a gain
    there is real airway, not a leak, unless the new volume exceeds `early_leak_ml`.
    """
    if seed_zyx is None:
        seed_zyx = find_trachea_seed(hu, spacing_mm)
    voxel_ml = spacing_mm**3 / 1000.0
    notes: list[str] = []

    # Restrict work to a box that tightly contains the first candidate component's neighborhood? Use full volume;
    # labeling is O(N) and the ladder has only a handful of steps.
    prev_mask = None
    prev_t = None
    prev_vol = None
    volumes: dict[int, float] = {}
    leak = False
    for t in thresholds:
        cand = hu < t
        if not cand[seed_zyx]:
            # Seed not below this threshold (shouldn't happen at -950 for a valid seed); skip.
            continue
        lab, _ = ndi.label(cand, structure=ndi.generate_binary_structure(3, connectivity))
        comp = lab == lab[seed_zyx]
        vol = float(comp.sum()) * voxel_ml
        volumes[t] = vol
        if vol > max_volume_ml and prev_mask is None:
            raise ValueError(
                f"Region growing already exceeds {max_volume_ml:.0f} mL ({vol:.0f} mL) at the lowest threshold "
                f"({t} HU): the airway leaks into the lung. Needs a different seed or a learned model."
            )
        if prev_vol is None:
            is_leak = False
        elif prev_vol < min_trusted_ml:
            is_leak = vol > min(early_leak_ml, max_volume_ml)
        else:
            is_leak = vol > max_volume_ml or (vol > leak_factor * prev_vol and vol - prev_vol > min_leak_jump_ml)
        if is_leak:
            leak = True
            notes.append(
                f"Leak at {t} HU: {prev_vol:.1f} mL -> {vol:.1f} mL; using {prev_t} HU result."
            )
            break
        prev_mask, prev_t, prev_vol = comp, t, vol

    if prev_mask is None:
        raise ValueError("Region growing failed: seed is not in air at any tested threshold.")

    # fill pinholes/noise in the lumen
    mask = ndi.binary_closing(prev_mask, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    mask = np.stack([ndi.binary_fill_holes(mask[z]) for z in range(mask.shape[0])], axis=0)
    if not leak and prev_t == thresholds[-1]:
        notes.append("No leak detected up to the highest threshold; airway may still be incomplete.")
    return AirwaySegmentation(
        mask=mask,
        seed_zyx=tuple(int(v) for v in seed_zyx),
        threshold_hu=int(prev_t),
        volumes_ml=volumes,
        leak_detected=leak,
        notes=notes,
    )
