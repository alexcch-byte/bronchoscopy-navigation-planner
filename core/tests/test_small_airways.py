"""Thin-branch extension of the airway mask (segmentation/small_airways.py), on synthetic volumes."""

import numpy as np
import scipy.ndimage as ndi

from bronchnav.segmentation import segment_airway
from bronchnav.segmentation.small_airways import extend_into_small_airways, tube_measure

SP = 1.0


def _lung_with_trunk_and_branch(seed=0, branch_hu=-900.0, branch_r=2.0, branch_len=28):
    """Noisy lung (-850 HU) with a wide vertical trunk (r = 6, -1000 HU) and a thinner horizontal branch (r = 2 mm,
    segmental size) that leaves its wall. The branch lumen is partial-volumed (-900 HU) and has a brighter wall ring
    (-300 HU), as a real airway does. Returns (hu, trunk mask, branch lumen mask). Gaussian blur imitates the PSF."""
    rng = np.random.default_rng(seed)
    n = 80
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    hu = -850.0 + rng.normal(0, 8.0, (n, n, n))
    trunk = ((y - 40) ** 2 + (x - 25) ** 2) <= 6.0**2
    along = (x >= 25) & (x < 25 + 6 + branch_len)
    rr = (z - 40) ** 2 + (y - 40) ** 2
    lumen = (rr <= branch_r**2) & along
    wall = (rr <= (branch_r + 1.4) ** 2) & along
    if branch_r > 0:
        hu[wall] = -300.0
        hu[lumen] = branch_hu
    hu[trunk] = -1000.0
    hu = ndi.gaussian_filter(hu, 0.5)
    return hu.astype(np.float32), trunk, lumen & ~trunk


def _axis(shape=(80, 80, 80)):
    """The branch's centre line, past the junction with the trunk."""
    ax = np.zeros(shape, bool)
    ax[40, 40, 34:56] = True
    return ax


def test_tube_measure_is_high_on_a_dark_tube_and_low_in_noisy_lung():
    hu, trunk, branch = _lung_with_trunk_and_branch()
    t = tube_measure(hu, SP)
    lung = ~ndi.binary_dilation(trunk | branch, iterations=8)
    assert np.median(t[_axis()]) > 1.5 * np.percentile(t[lung], 99)


def test_extension_follows_the_thin_branch_without_flooding_the_lung():
    hu, trunk, branch = _lung_with_trunk_and_branch()
    base = trunk.copy()
    out, info = extend_into_small_airways(hu, SP, base, reach_mm=40.0)
    assert (out | base).sum() == out.sum() and (base & ~out).sum() == 0  # superset: nothing removed
    added = out & ~base
    assert (out & _axis()).sum() >= 0.9 * _axis().sum()  # the branch's centre line is covered
    stray = (added & ~ndi.binary_dilation(branch, iterations=3)).sum()
    assert stray < 0.25 * max(1, added.sum())  # the added voxels are along the branch, not in the parenchyma
    assert info["added_ml"] < 5.0 and not info["capped"]


def test_pure_noisy_lung_next_to_a_trunk_gains_almost_nothing():
    hu, trunk, _ = _lung_with_trunk_and_branch(branch_len=0, branch_r=0.0)
    out, info = extend_into_small_airways(hu, SP, trunk, reach_mm=40.0)
    assert info["added_ml"] < 0.3


def test_growth_is_capped_and_the_cap_keeps_the_inner_layers():
    hu, trunk, branch = _lung_with_trunk_and_branch()
    out, info = extend_into_small_airways(hu, SP, trunk, reach_mm=40.0, max_added_ml=0.01, thicken=False)
    assert info["capped"] and 0 < info["added_ml"] <= 0.01 + 1e-6
    assert (trunk & ~out).sum() == 0


def test_empty_mask_is_returned_unchanged():
    hu, _, _ = _lung_with_trunk_and_branch()
    out, info = extend_into_small_airways(hu, SP, np.zeros(hu.shape, bool))
    assert not out.any() and info["added_ml"] == 0.0


def test_segment_airway_option_only_adds_airway_and_is_off_by_default():
    hu, trunk, _ = _lung_with_trunk_and_branch()
    hu[trunk] = -1000.0
    hu[:6, :, :] = -1000.0  # open the top so a trachea-like seed exists
    seed = (40, 40, 25)
    plain = segment_airway(hu, SP, seed_zyx=seed)
    assert plain.small_airways_ml == 0.0
    ext = segment_airway(hu, SP, seed_zyx=seed, small_airways=True)
    assert (plain.mask & ~ext.mask).sum() == 0
    assert ext.small_airways_ml >= 0.0
    assert any("Thin-branch extension" in n for n in ext.notes)
