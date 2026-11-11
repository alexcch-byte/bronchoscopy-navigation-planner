"""Rough lung mask from thresholds, for placing test targets on CTs that come without a lung label.

Not a segmentation to trust: it is good enough to put a stand-in target "somewhere inside the lung".
Arrays are (z, y, x) in HU.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def rough_lung_mask(hu: np.ndarray, downsample: int = 2, min_fraction: float = 0.3) -> np.ndarray:
    """Air-like voxels (< -400 HU) that are not connected to the air outside the body, keeping the large
    connected pieces (the two lungs).

    Outside air is the air-like component that touches the lateral volume faces (x or y border); the trachea
    opens at the top of the scan (z border) and stays connected to the lungs, not to the outside. Works on a
    `downsample`-times coarser copy and repeats the result back, so it is fast on 200M-voxel CTs.
    Bowel gas and the trachea may be included; the lungs dominate.
    """
    f = max(1, int(downsample))
    if f > 1:
        # block MAX, not striding: thin tissue walls (a few mm between a lung and the outside air) must survive
        nz, ny, nx = (s // f * f for s in hu.shape)
        small = hu[:nz, :ny, :nx].reshape(nz // f, f, ny // f, f, nx // f, f).max(axis=(1, 3, 5))
    else:
        small = hu
    air = small < -400
    # Opening cuts one-voxel partial-volume bridges across a thin chest wall (skin/fat/rib a few mm thick),
    # which otherwise join lung air to the air outside the body.
    core = ndi.binary_opening(air, iterations=1)
    lab, n = ndi.label(core)
    if n == 0:
        raise ValueError("no air-like region: is this a chest CT in HU?")
    outside = np.unique(np.concatenate([lab[:, 0, :].ravel(), lab[:, -1, :].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    sizes[np.isin(np.arange(1, n + 1), outside)] = 0  # drop outside air
    if sizes.max() == 0:
        raise ValueError("all air-like voxels touch the volume border: cropped scan?")
    keep_ids = np.nonzero(sizes >= min_fraction * sizes.max())[0] + 1
    lungs = np.isin(lab, keep_ids)
    lungs = ndi.binary_dilation(lungs, iterations=2) & air  # regrow what the opening trimmed, inside the air mask
    if f > 1:
        lungs = np.repeat(np.repeat(np.repeat(lungs, f, axis=0), f, axis=1), f, axis=2)
        full = np.zeros(hu.shape, dtype=bool)
        sz, sy, sx = (min(a, b) for a, b in zip(lungs.shape, hu.shape))
        full[:sz, :sy, :sx] = lungs[:sz, :sy, :sx]
        lungs = full
    return lungs & (hu < -400)
