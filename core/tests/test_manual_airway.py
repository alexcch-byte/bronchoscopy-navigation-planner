"""Operator-drawn airways (bronchnav.segmentation.manual)."""

import numpy as np
import pytest
import scipy.ndimage as ndi

from bronchnav.geometry import Grid
from bronchnav.segmentation.manual import add_airway

SP = 0.75


def _trachea():
    """A 8 mm radius vertical tube (along z) in a 60^3 voxel box; axis through voxel (30, 30)."""
    z, y, x = np.mgrid[0:60, 0:60, 0:60]
    return ((x - 30) ** 2 + (y - 30) ** 2) * SP**2 <= 8.0**2


@pytest.fixture()
def grid():
    return Grid(origin=np.zeros(3), direction=np.eye(3), spacing_mm=SP)


def _mm(x, y, z):
    return [x * SP, y * SP, z * SP]


def test_adds_a_connected_branch_and_keeps_the_input(grid):
    mask = _trachea()
    before = mask.copy()
    pts = [_mm(36, 30, 30), _mm(44, 30, 30), _mm(54, 30, 30)]   # starts at the tube wall, runs out sideways
    out, info = add_airway(mask, grid, pts, 1.0)
    assert (mask == before).all()                                 # the input mask is not modified
    assert (out & mask).sum() == mask.sum() and out.sum() > mask.sum()
    assert ndi.label(out, structure=np.ones((3, 3, 3)))[1] == 1  # one connected piece
    assert out[30, 30, 52] and not out[30, 45, 52]               # the new tube is there, and thin
    assert info["length_mm"] > 10 and info["added_ml"] > 0 and info["bridge_mm"] < 1.0
    # tube radius: about 1.0 mm across the branch (a voxel is 0.75 mm)
    width = out[30, :, 52].sum() * SP
    assert 1.2 <= width <= 3.0


def test_bridges_a_small_gap_but_refuses_a_distant_start(grid):
    mask = _trachea()
    near = [_mm(41, 30, 30), _mm(54, 30, 30)]   # 8 voxels = 6 mm of gap beyond the wall at x=40.7
    out, info = add_airway(mask, grid, near, 1.0)
    assert ndi.label(out, structure=np.ones((3, 3, 3)))[1] == 1
    far = [_mm(56, 30, 30), _mm(58, 30, 30), _mm(59, 30, 30)]
    with pytest.raises(ValueError, match="nearest airway|already found"):
        add_airway(mask, grid, far, 1.0)


@pytest.mark.parametrize("pts, radius, text", [
    ([_mm(36, 30, 30)], 1.0, "between 2 and"),
    ([_mm(36, 30, 30), _mm(37, 30, 30)], 1.0, "shorter than 3 mm"),
    ([_mm(36, 30, 30), _mm(54, 30, 30)], 9.0, "radius must be"),
    ([_mm(36, 30, 30), _mm(54, 30, 90)], 1.0, "outside the CT volume"),
    ([_mm(36, 30, 30), [float("nan"), 0, 0]], 1.0, "finite"),
])
def test_rejects_bad_input(grid, pts, radius, text):
    with pytest.raises(ValueError, match=text):
        add_airway(_trachea(), grid, pts, radius)
