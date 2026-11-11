import json

import numpy as np
import pytest
import SimpleITK as sitk

from bronchnav.airway_tree import build_tree
from bronchnav.geometry import Grid
from bronchnav.io import resample_isotropic
from bronchnav.phantom import make_phantom
from bronchnav.pipeline import route_to_dict, run_plan
from bronchnav.planning import plan_route
from bronchnav.segmentation import find_trachea_seed, segment_airway

SP = 0.75


def _truth_on_grid(p, img):
    t = sitk.GetImageFromArray(p.airway_truth.astype(np.uint8))
    t.CopyInformation(p.image)
    r = sitk.Resample(t, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(r).astype(bool)


def _dice(a, b):
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


@pytest.fixture(scope="module")
def case():
    p = make_phantom()
    img = resample_isotropic(p.image, SP)
    hu = sitk.GetArrayFromImage(img)
    seg = segment_airway(hu, SP)
    tree = build_tree(seg.mask, SP, seg.seed_zyx)
    grid = Grid.from_image(img)
    return p, img, hu, seg, tree, grid


def test_segmentation_matches_truth_without_leak(case):
    p, img, hu, seg, tree, grid = case
    assert not seg.leak_detected
    truth = _truth_on_grid(p, img)
    assert _dice(seg.mask, truth) > 0.85
    assert (seg.mask & truth).sum() / seg.mask.sum() > 0.99  # precision: no spill into parenchyma


def test_leak_is_detected_and_contained():
    p = make_phantom(leak=True)
    img = resample_isotropic(p.image, SP)
    seg = segment_airway(sitk.GetArrayFromImage(img), SP)
    assert seg.leak_detected
    assert seg.mask.sum() * SP**3 / 1000.0 < 60.0  # mL; a real leak would be > 1000 mL
    assert _dice(seg.mask, _truth_on_grid(p, img)) > 0.8


def _tube_volume(with_block: bool):
    """1 mm cube: tissue with a tube along x. Segment A (x<30) is -1000 HU, segment B (x>=30) is -935 HU,
    so B only joins the airway at the -925 threshold (volume ~3x without any leak).
    Optionally a -910 HU block touching the tube joins at -900 (a real ~180 mL leak)."""
    hu = np.full((100, 100, 100), 40.0, dtype=np.float32)  # (z, y, x)
    zz, yy, xx = np.meshgrid(np.arange(100), np.arange(100), np.arange(100), indexing="ij")
    tube = (yy - 50) ** 2 + (zz - 50) ** 2 <= 25
    hu[tube & (xx < 30)] = -1000.0
    hu[tube & (xx >= 30)] = -935.0
    if with_block:
        hu[(xx >= 60) & (yy >= 55) & (yy < 100)] = -910.0
    return hu


def test_early_airway_growth_is_not_mistaken_for_a_leak():
    hu = _tube_volume(with_block=False)
    seg = segment_airway(hu, 1.0, seed_zyx=(50, 50, 10))
    assert not seg.leak_detected
    assert seg.threshold_hu == -800  # reached the top of the ladder
    assert seg.volumes_ml[-925] > 2.0 * seg.volumes_ml[-950]  # the ratio alone would have flagged it


def _fragment_then_tube(radius_vox: int):
    """2 mm voxels. A short -1000 HU fragment (~2 mL) joins a long -935 HU tube at the -925 threshold."""
    hu = np.full((100, 100, 100), 40.0, dtype=np.float32)
    zz, yy, xx = np.meshgrid(np.arange(100), np.arange(100), np.arange(100), indexing="ij")
    r2 = (yy - 50) ** 2 + (zz - 50) ** 2
    hu[(r2 <= 9) & (xx < 10)] = -1000.0
    hu[(r2 <= radius_vox**2) & (xx >= 10)] = -935.0
    return hu


def test_small_first_fragment_does_not_count_as_leak_baseline():
    seg = segment_airway(_fragment_then_tube(5), 2.0, seed_zyx=(50, 50, 5))  # grows 2 -> ~58 mL
    assert not seg.leak_detected
    assert seg.volumes_ml[-925] > 40.0 and seg.threshold_hu == -800


def test_huge_jump_from_a_small_fragment_is_still_a_leak():
    seg = segment_airway(_fragment_then_tube(8), 2.0, seed_zyx=(50, 50, 5))  # grows 2 -> ~145 mL
    assert seg.leak_detected and seg.threshold_hu == -950


def test_default_connectivity_does_not_leak_through_a_corner_contact():
    """Two air blocks that touch only at a corner: 26-connected labeling merges them (a ~1.7 L leak),
    the 6-connected default keeps the seed's airway on its own."""
    hu = np.full((100, 100, 100), 40.0, dtype=np.float32)
    hu[5:25, 5:25, 5:25] = -1000.0  # the airway, 64 mL at 2 mm voxels (kept off the volume border)
    hu[25:85, 25:85, 25:85] = -1000.0  # a lung-sized pocket sharing only the corner voxel contact
    seg = segment_airway(hu, 2.0, seed_zyx=(15, 15, 15))
    assert not seg.leak_detected
    assert abs(seg.mask.sum() * 8e-3 - 64.0) < 1.0
    with pytest.raises(ValueError, match="leaks"):
        segment_airway(hu, 2.0, seed_zyx=(15, 15, 15), connectivity=3)


def test_large_absolute_jump_is_still_a_leak():
    hu = _tube_volume(with_block=True)
    seg = segment_airway(hu, 1.0, seed_zyx=(50, 50, 10))
    assert seg.leak_detected
    assert seg.threshold_hu == -925
    assert seg.mask.sum() / 1000.0 < 30.0  # mL, the block (~180 mL) stayed out


def test_trachea_seed_prefers_superior_end_with_fallback():
    p = make_phantom()
    img = resample_isotropic(p.image, SP)
    hu = sitk.GetArrayFromImage(img)
    z, _, _ = find_trachea_seed(hu, SP)
    assert z > hu.shape[0] - 55  # anatomical phantom: trachea at the high-z (superior) end
    # without fallback, inverted scan raises ValueError (no silent latching onto inferior end)
    with pytest.raises(ValueError, match="superior"):
        find_trachea_seed(hu[::-1].copy(), SP)
    # with explicit fallback or prefer="low", the seed is found at the low end
    z2, _, _ = find_trachea_seed(hu[::-1].copy(), SP, allow_fallback=True)
    assert z2 < 55
    z3, _, _ = find_trachea_seed(hu[::-1].copy(), SP, prefer="low")
    assert z3 < 55


def _body_with_two_air_blobs():
    hu = np.full((120, 100, 100), 40.0, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
    blob = (yy - 50) ** 2 + (xx - 50) ** 2 <= 8**2  # ~200 mm^2 round air column
    hu[90:, blob] = -1000.0  # trachea at the superior (high-z) end
    hu[:30, blob] = -1000.0  # "stomach gas" at the inferior end
    return hu


def test_seed_prefers_trachea_over_inferior_gas():
    hu = _body_with_two_air_blobs()
    assert find_trachea_seed(hu, 1.0)[0] >= 80
    assert find_trachea_seed(hu, 1.0, prefer="low")[0] < 40


def test_seed_rejects_air_pocket_outside_the_body():
    hu = np.full((60, 100, 100), -1000.0, dtype=np.float32)  # all outside air
    yy, xx = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
    rr = np.hypot(yy - 50, xx - 50)
    hu[:, (rr > 8) & (rr <= 9)] = 40.0  # thin skin-like loop enclosing a pocket of air
    with pytest.raises(ValueError):
        find_trachea_seed(hu, 1.0)


def test_leak_at_the_first_threshold_is_an_error_not_a_result():
    hu = np.full((100, 100, 100), 40.0, dtype=np.float32)
    zz, yy, xx = np.meshgrid(np.arange(100), np.arange(100), np.arange(100), indexing="ij")
    hu[(xx >= 20) & (xx < 80) & (yy >= 5) & (yy < 95)] = -1000.0  # ~4300 mL of "air" at 2 mm voxels
    with pytest.raises(ValueError, match="leaks"):
        segment_airway(hu, 2.0, seed_zyx=(50, 50, 50))


def test_tree_is_a_clean_binary_tree(case):
    *_, tree, _ = case
    assert tree.total_length_mm() > 300
    # a binary tree with one root endpoint: endpoints = junctions + 2
    assert len(tree.endpoints) == len(tree.junctions) + 2
    assert np.all(tree.degree <= 3)


def test_route_reaches_near_target_and_stays_in_lumen(case):
    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    r = plan_route(tree, target, p.nodule_radius_mm)
    assert r.ok
    assert r.metrics["fraction_inside_lumen"] >= 0.99
    assert r.metrics["n_bifurcations"] >= 3
    assert r.metrics["gap_to_nodule_surface_mm"] < 25.0
    # monotone arclength, unit tangents, frames orthogonal
    assert np.all(np.diff(r.s_mm) > 0)
    assert np.allclose(np.linalg.norm(r.tangents, axis=1), 1.0, atol=1e-6)
    assert np.abs(np.einsum("ij,ij->i", r.tangents, r.normals)).max() < 1e-6
    # it starts at the top of the trachea (small z) and moves toward the nodule
    assert np.linalg.norm(r.points_mm[-1] - target) < np.linalg.norm(r.points_mm[0] - target)


def test_tree_root_is_the_tracheal_tip_even_if_the_seed_is_deep(case):
    p, _, _, seg, _, grid = case
    nz = seg.mask.shape[0]
    deep = (nz - 1 - 60, int(round(62 / SP)), int(round(70 / SP)))  # ~45 mm below the trachea opening
    assert seg.mask[deep]
    tree = build_tree(seg.mask, SP, deep)
    assert tree.radius_mm[tree.root] > 5.0  # a wide endpoint, not a distal tip
    assert tree.pts_mm[tree.root][2] > tree.pts_mm[:, 2].max() - 6.0  # at the superior end


def test_root_ignores_a_thick_side_stub():
    """A long trachea-like tube with a short, thick, blunt stub off its middle (a central-airway lump).
    The stub's skeleton endpoint is nearly as wide as the tube but is not the tracheal tip."""
    mask = np.zeros((140, 60, 60), dtype=bool)  # (z, y, x), 1 mm voxels
    zz, yy, xx = np.meshgrid(np.arange(140), np.arange(60), np.arange(60), indexing="ij")
    mask |= ((yy - 30) ** 2 + (xx - 30) ** 2 <= 8**2) & (zz >= 10) & (zz < 130)  # main tube along z
    mask |= ((zz - 70) ** 2 + (yy - 30) ** 2 <= 6**2) & (xx >= 30) & (xx < 46)  # thick stub along +x
    # make the bottom end a thin pair of branches so only the top is a wide tube end
    mask[10:30, :, :] &= ((yy[10:30] - 30) ** 2 + (xx[10:30] - 30) ** 2 <= 2**2) | ((yy[10:30] - 30) ** 2 + (xx[10:30] - 22) ** 2 <= 2**2)
    seed = (100, 30, 30)  # deep in the tube, nearer the stub than the tube's top is
    tree = build_tree(mask, 1.0, seed)
    z_root, x_root = tree.pts_mm[tree.root][2], tree.pts_mm[tree.root][0]
    assert z_root > 115 and abs(x_root - 30) < 4  # the top of the tube, not the stub at z=70, x=46


def test_rough_lung_mask_covers_the_lungs_and_not_outside_air_or_tissue():
    """A simple chest at 2 mm voxels: elliptical body, two lungs with a 12 mm chest wall around them.
    (The phantom's lungs sit within ~3 mm of its skin, which is not a fair test.)"""
    from bronchnav.segmentation import rough_lung_mask

    nz, ny, nx = 80, 100, 120
    zz, yy, xx = np.meshgrid(np.arange(nz) * 2.0, np.arange(ny) * 2.0, np.arange(nx) * 2.0, indexing="ij")
    body = ((xx - 120) / 100) ** 2 + ((yy - 100) / 70) ** 2 <= 1.0
    hu = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    hu[body] = 40.0
    lungs_true = np.zeros_like(body)
    for cx in (70.0, 170.0):
        lungs_true |= ((xx - cx) / 40) ** 2 + ((yy - 100) / 52) ** 2 + ((zz - 80) / 60) ** 2 <= 1.0
    lungs_true &= body
    hu[lungs_true] = -850.0
    hu = np.clip(hu + np.random.default_rng(0).normal(0, 15, hu.shape).astype(np.float32), -1024, 1500)

    lungs = rough_lung_mask(hu)
    assert lungs.shape == hu.shape and lungs.dtype == bool
    assert np.all(hu[lungs] < -400)
    inter = (lungs & lungs_true).sum()
    assert 2 * inter / (lungs.sum() + lungs_true.sum()) > 0.93  # Dice against the true lungs
    assert not (lungs & ~body).any()  # outside air stays out
    assert not lungs[hu > -100].any()  # soft tissue stays out


@pytest.mark.parametrize("order", ["feet_first", "head_first"])
def test_image_stack_of_the_phantom_detects_slice_order_and_segments(tmp_path, order):
    """Export the phantom as 8-bit lung-window PNGs in either slice order; auto-detection must recover the
    anatomical orientation, and the airway grown from the result must match the original's."""
    imageio = pytest.importorskip("imageio.v2")
    from bronchnav.io import load_image_stack

    ph = make_phantom()
    hu0 = sitk.GetArrayFromImage(ph.image)  # (z, y, x), z up (trachea at the high end)
    c, w = -600.0, 1500.0
    gray = (np.clip((hu0 - (c - w / 2)) / w, 0, 1) * 255).astype(np.uint8)
    idx = range(hu0.shape[0]) if order == "feet_first" else range(hu0.shape[0] - 1, -1, -1)
    for k, z in enumerate(idx):
        imageio.imwrite(tmp_path / f"IM{k + 1:04d}.png", gray[z])
    img, warns = load_image_stack(tmp_path, 0.7, 1.0, first_slice="auto")
    arr = sitk.GetArrayFromImage(img)
    assert arr.shape == hu0.shape
    air = hu0 < -900
    assert np.abs(arr[air] - hu0[air]).mean() < 60  # same orientation as the original (flipped would be huge)
    assert any("detected" in x for x in warns)
    seg = segment_airway(sitk.GetArrayFromImage(resample_isotropic(img, SP)), SP)
    assert 15.0 < seg.mask.sum() * SP**3 / 1000.0 < 60.0  # the phantom's airway is ~26 mL


def test_autotarget_places_a_point_beyond_a_distal_tip(case):
    from bronchnav.autotarget import choose_target

    _, _, _, seg, tree, _ = case
    lungs = np.ones(seg.mask.shape, dtype=bool)
    e, tgt, geo, ahead = choose_target(tree, lungs, SP, pctl=0.7, ahead_mm=15.0)
    assert e in set(tree.endpoints.tolist()) and ahead == 15.0 and geo > 50.0
    assert abs(np.linalg.norm(tgt - tree.pts_mm[e]) - 15.0) < 1e-6
    # a lung mask that excludes everything cannot yield a target
    with pytest.raises(RuntimeError):
        choose_target(tree, np.zeros_like(lungs), SP)


def test_phantom_nodule_can_be_placed_sized_and_given_a_density():
    from bronchnav.phantom import NoduleSpec

    default = make_phantom()
    old_c = np.array(default.nodule_center_mm)
    # a lung-side point well away from the airways: the left lung, mid height
    spec = NoduleSpec(center_mm=(-34.0, 0.0, 10.0), radius_mm=7.0, hu=-300.0)  # a part-solid nodule
    ph = make_phantom(nodule=spec)
    img = ph.image

    def hu_at(p):
        idx = img.TransformPhysicalPointToIndex(tuple(float(v) for v in p))
        return float(sitk.GetArrayFromImage(img)[idx[2], idx[1], idx[0]])

    assert ph.nodule_center_mm == (-34.0, 0.0, 10.0) and ph.nodule_radius_mm == 7.0 and ph.nodule_hu == -300.0
    assert abs(hu_at((-34.0, 0.0, 10.0)) - (-300.0)) < 40  # drawn at the requested place with the requested density
    assert hu_at((-34.0 - 12.0, 0.0, 10.0)) < -700  # lung just outside the 7 mm radius (away from the main bronchus)
    assert hu_at(tuple(old_c)) < -700  # the built-in nodule is gone from its old place
    # everything else in the image is identical: same airways and same noise
    a, b = sitk.GetArrayFromImage(default.image), sitk.GetArrayFromImage(img)
    far = np.ones_like(a, dtype=bool)
    for c, r in ((old_c, 12.0), (np.array(spec.center_mm), 14.0)):
        idx = default.image.TransformPhysicalPointToIndex(tuple(float(v) for v in c))
        zz, yy, xx = np.ogrid[: a.shape[0], : a.shape[1], : a.shape[2]]
        far &= ((xx - idx[0]) * 0.7) ** 2 + ((yy - idx[1]) * 0.7) ** 2 + ((zz - idx[2]) * 1.0) ** 2 > r**2
    assert np.array_equal(a[far], b[far])
    assert np.array_equal(default.airway_truth, ph.airway_truth)


def test_phantom_nodule_placements_that_do_not_fit_are_rejected():
    from bronchnav.phantom import NoduleSpec

    for bad in (
        NoduleSpec(center_mm=(0.0, 0.0, 0.0), radius_mm=5.0),  # the mediastinum, between the lungs
        NoduleSpec(center_mm=(0.0, 0.0, 200.0), radius_mm=5.0),  # outside the volume
        NoduleSpec(center_mm=(60.0, 40.0, -37.0), radius_mm=15.0),  # sticks out of the lung
        NoduleSpec(center_mm=(0.0, -8.0, 50.0), radius_mm=4.0),  # on the trachea / main airway
    ):
        with pytest.raises(ValueError, match="inside a lung"):
            make_phantom(nodule=bad)


def test_operator_can_force_a_different_end_point(case):
    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    auto = plan_route(tree, target, p.nodule_radius_mm)
    # pick an endpoint on a different branch than the automatic one
    others = [e for e in tree.endpoints if e != auto.exit_node and np.linalg.norm(tree.pts_mm[e] - tree.pts_mm[auto.exit_node]) > 30]
    assert others
    forced = plan_route(tree, target, p.nodule_radius_mm, force_exit_mm=tree.pts_mm[others[0]], min_diameter_mm=1.0)
    assert forced.exit_node == others[0] and forced.exit_node != auto.exit_node
    assert any("chosen by the operator" in w for w in forced.warnings)
    assert np.linalg.norm(forced.points_mm[-1] - tree.pts_mm[others[0]]) < 1.0  # the path ends where it was told
    assert forced.metrics["fraction_inside_lumen"] >= 0.99
    # the other metrics are there for comparing routes
    for k in ("length_mm", "min_diameter_mm", "min_bend_radius_mm", "gap_to_nodule_surface_mm", "approach_angle_deg"):
        assert k in forced.metrics and k in auto.metrics
    assert 0.0 <= auto.metrics["approach_angle_deg"] <= 180.0
    # the start of the route is not a valid end
    with pytest.raises(ValueError, match="start of the route"):
        plan_route(tree, target, p.nodule_radius_mm, force_exit_mm=tree.pts_mm[tree.root])
    # an end the scope cannot reach is allowed but flagged
    narrow = plan_route(tree, target, p.nodule_radius_mm, force_exit_mm=tree.pts_mm[others[0]], min_diameter_mm=9.0)
    assert any("cannot be reached" in w for w in narrow.warnings)


def test_exit_candidates_are_ranked_distinct_and_described(case):
    from bronchnav.planning import exit_candidates

    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    cands = exit_candidates(tree, target, p.nodule_radius_mm, search_mm=80.0, n=5)
    assert 1 <= len(cands) <= 5
    assert [c["score"] for c in cands] == sorted(c["score"] for c in cands)  # best first
    pts = np.array([c["point_mm"] for c in cands])
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            assert np.linalg.norm(pts[a] - pts[b]) >= 10.0  # different airways, not neighbours
    for c in cands:
        assert {"node", "point_mm", "gap_mm", "route_length_mm", "min_diameter_mm", "approach_angle_deg", "reachable"} <= set(c)
        assert c["gap_mm"] >= 0 and 0 <= c["approach_angle_deg"] <= 180 and c["route_length_mm"] > 0
    # every candidate can be turned into a route
    r = plan_route(tree, target, p.nodule_radius_mm, force_exit_mm=pts[0], min_diameter_mm=1.0)
    assert r.ok


def test_route_respects_scope_size(case):
    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    wide = plan_route(tree, target, p.nodule_radius_mm, min_diameter_mm=3.0)
    big = plan_route(tree, target, p.nodule_radius_mm, min_diameter_mm=8.0)  # only the large airways
    assert big.metrics["min_diameter_mm"] >= 6.0  # never squeezes through a narrower airway
    assert big.metrics["length_mm"] < wide.metrics["length_mm"]
    assert big.metrics["gap_to_nodule_surface_mm"] > wide.metrics["gap_to_nodule_surface_mm"]
    assert big.warnings  # tells the user the closest airway point is out of reach
    # lumen size and wall clearance are reported separately
    assert wide.clearance_mm is not None and "min_wall_clearance_mm" in wide.metrics


def test_route_starts_at_top_of_trachea_and_bifurcations_are_ordered(case):
    p, _, _, seg, tree, grid = case
    r = plan_route(tree, grid.points_to_grid(np.array(p.nodule_center_mm)), p.nodule_radius_mm)
    assert tree.pts_mm[tree.root][2] > tree.pts_mm[:, 2].max() - 6.0  # root is at the superior tip
    s = [b["s_mm"] for b in r.bifurcations]
    assert s == sorted(s) and len(set(s)) == len(s)
    assert 0 < s[0] and s[-1] < r.s_mm[-1]  # origin/exit nodes are not reported as bifurcations


def test_route_json_roundtrip(case, tmp_path):
    p, img, _, seg, tree, grid = case
    from bronchnav.pipeline import PlanResult

    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    r = plan_route(tree, target, p.nodule_radius_mm)
    d = route_to_dict(PlanResult(img, grid, seg, tree, r), "phantom", None)
    d2 = json.loads(json.dumps(d))
    assert d2["target"]["center_mm"] == pytest.approx(list(p.nodule_center_mm), abs=1e-2)
    assert len(d2["path"]["points_mm"]) == len(r.points_mm)
    assert d2["intended_use"].startswith("Planning aid")
    assert d2["grid_origin_mm"] == pytest.approx(grid.origin.tolist(), abs=1e-4)
    assert np.allclose(d2["grid_direction"], grid.direction, atol=1e-4)


def test_end_to_end_without_render(tmp_path):
    p = make_phantom()
    ct = tmp_path / "ct.mha"
    sitk.WriteImage(p.image, str(ct))
    res = run_plan(ct, np.array(p.nodule_center_mm), p.nodule_radius_mm, out_dir=tmp_path / "out",
                   render=False, log=lambda *_: None)
    assert (tmp_path / "out" / "route.json").exists()
    assert (tmp_path / "out" / "airway_mask.nii.gz").exists()
    assert res.route.metrics["fraction_inside_lumen"] >= 0.99


def test_mesh_faces_the_lumen_and_frames_survive_odd_inputs(case):
    pytest.importorskip("vtk")
    from vtk.util import numpy_support as ns

    from bronchnav.render import VBRenderer, mask_to_polydata

    p, _, _, seg, tree, grid = case
    pd = mask_to_polydata(seg.mask, SP)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData())
    nrm = ns.vtk_to_numpy(pd.GetPointData().GetNormals())
    zmax = pts[:, 2].max()
    sel = (np.hypot(pts[:, 0] - 70, pts[:, 1] - 62) < 10) & (pts[:, 2] < zmax - 10) & (pts[:, 2] > zmax - 60)
    inward = np.stack([70 - pts[sel, 0], 62 - pts[sel, 1], np.zeros(sel.sum())], 1)
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    assert sel.sum() > 500
    assert (nrm[sel] * inward).sum(1).mean() > 0.8  # normals point into the lumen (toward the axis)

    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    r = plan_route(tree, target, p.nodule_radius_mm)
    vb = VBRenderer(seg.mask, SP, r, size=(161, 121), approach_blend_mm=12.0)  # odd size, blend on
    assert vb.size == (160, 120)
    for s in (-5.0, 0.0, float(r.s_mm[-1]), float(r.s_mm[-1]) + 5.0):  # out-of-range s is clamped
        f = vb.frame(s)
        assert f.shape == (120, 160, 3) and np.isfinite(f).all()


def test_render_stills_and_video(tmp_path):
    pytest.importorskip("vtk")
    pytest.importorskip("imageio")
    p = make_phantom()
    ct = tmp_path / "ct.mha"
    sitk.WriteImage(p.image, str(ct))
    res = run_plan(ct, np.array(p.nodule_center_mm), p.nodule_radius_mm, out_dir=tmp_path / "out",
                   size=(160, 120), speed_mm_s=60.0, log=lambda *_: None)
    out = tmp_path / "out"
    stills = sorted((out / "stills").glob("*.png"))
    assert len(stills) == len(res.route.bifurcations) + 1
    assert (out / "airway_mesh.ply").stat().st_size > 1000
    assert (out / "airway_mesh.glb").stat().st_size > 1000
    assert res.outputs["mesh_glb"] == "airway_mesh.glb"
    assert (out / "flythrough.mp4").stat().st_size > 1000
    import imageio.v2 as imageio

    img = imageio.imread(stills[0])
    assert img.shape == (120, 160, 3) and img.mean() > 10  # not a black frame


def test_bend_radius_warning(case):
    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    base = plan_route(tree, target, p.nodule_radius_mm)
    bend = base.metrics["min_bend_radius_mm"]  # the warning must appear exactly when the limit exceeds this
    r1 = plan_route(tree, target, p.nodule_radius_mm, min_bend_radius_limit_mm=bend + 1.0)
    assert any("bend radius" in w for w in r1.warnings)
    r2 = plan_route(tree, target, p.nodule_radius_mm, min_bend_radius_limit_mm=bend - 1.0)
    assert not any("bend radius" in w for w in r2.warnings)


def test_start_hook_is_trimmed_but_gentle_curves_are_not():
    from bronchnav.planning.route import _trim_start_hook

    step = 0.75
    hook = [(x * step, 0.0, 0.0) for x in range(10)]  # 7 mm sideways dogleg ...
    z0 = hook[-1]
    down = [(z0[0], 0.0, -(k + 1) * step) for k in range(70)]  # ... then 52 mm straight down
    pts = np.array(hook + down)
    cut = _trim_start_hook(pts)
    assert 6 <= cut <= 11  # cut at the corner
    # a gentle arc (45 degrees over 50 mm) is left alone
    th = np.linspace(0.0, np.deg2rad(45), 80)
    arc = np.stack([np.sin(th) * 60.0, np.zeros_like(th), -np.cos(th) * 60.0], axis=1)
    assert _trim_start_hook(arc) == 0
    # a straight route is left alone
    assert _trim_start_hook(np.stack([np.zeros(60), np.zeros(60), -np.arange(60) * step], axis=1)) == 0


def test_adaptive_smoothing_cuts_junction_corners_but_stays_in_the_lumen(case):
    """The route through a junction is smoothed at the widest scale that keeps clearance, so turns are gentler
    than the fixed 2.5 mm scale gave, without leaving the airway."""
    from bronchnav.planning import route as route_mod

    p, _, _, seg, tree, grid = case
    target = grid.points_to_grid(np.array(p.nodule_center_mm))
    adaptive = plan_route(tree, target, p.nodule_radius_mm)
    old = route_mod.SMOOTH_SCALES
    try:
        route_mod.SMOOTH_SCALES = (1.0,)  # the previous behaviour: one fixed scale
        fixed = plan_route(tree, target, p.nodule_radius_mm)
    finally:
        route_mod.SMOOTH_SCALES = old
    assert adaptive.metrics["min_bend_radius_mm"] >= fixed.metrics["min_bend_radius_mm"] - 1e-6
    assert adaptive.metrics["fraction_inside_lumen"] >= 0.99
    assert adaptive.metrics["min_wall_clearance_mm"] > 0.3


def test_sample_radius_out_of_bounds_clamping(case):
    *_, tree, _ = case
    out_of_bounds = np.array([[-1000.0, -1000.0, -1000.0], [5000.0, 5000.0, 5000.0]])
    sampled = tree.sample_radius(out_of_bounds)
    assert np.allclose(sampled, 0.0)


def test_mesh_physical_coordinate_alignment(tmp_path):
    pytest.importorskip("vtk")
    p = make_phantom()
    ct = tmp_path / "ct.mha"
    sitk.WriteImage(p.image, str(ct))
    res = run_plan(
        ct,
        np.array(p.nodule_center_mm),
        p.nodule_radius_mm,
        out_dir=tmp_path / "out",
        render=True,
        video=False,
        log=lambda *_: None,
    )
    out = tmp_path / "out"
    ply_file = out / "airway_mesh.ply"
    glb_file = out / "airway_mesh.glb"
    assert ply_file.exists() and ply_file.stat().st_size > 1000
    assert glb_file.exists() and glb_file.stat().st_size > 1000

    import vtk

    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(ply_file))
    reader.Update()
    poly = reader.GetOutput()
    bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    # Phantom image physical range is [-70, 70] in x/y, and [-100, 100] in z
    # The mesh must lie within these physical bounds with negative min bounds
    assert bounds[0] < 0.0 and bounds[1] > 0.0
    assert bounds[2] < 0.0 and bounds[3] > 0.0
    assert bounds[4] < 0.0 and bounds[5] > 0.0

    # Route physical points must be fully contained within the airway mesh physical bounds (with 1 mm tolerance)
    route_json = json.loads((out / "route.json").read_text(encoding="utf-8"))
    pts_phys = np.array(route_json["path"]["points_mm"])
    assert pts_phys[:, 0].min() >= bounds[0] - 1.0 and pts_phys[:, 0].max() <= bounds[1] + 1.0
    assert pts_phys[:, 1].min() >= bounds[2] - 1.0 and pts_phys[:, 1].max() <= bounds[3] + 1.0
    assert pts_phys[:, 2].min() >= bounds[4] - 1.0 and pts_phys[:, 2].max() <= bounds[5] + 1.0

    # Verify GLB header and readability
    glb_data = glb_file.read_bytes()
    assert glb_data[:4] == b"glTF"
    greader = vtk.vtkGLTFReader()
    greader.SetFileName(str(glb_file))
    greader.Update()
    assert greader.GetOutput() is not None


def test_mask_to_polydata_empty_raises():
    pytest.importorskip("vtk")
    from bronchnav.render import mask_to_polydata

    empty_mask = np.zeros((20, 20, 20), dtype=bool)
    with pytest.raises(ValueError, match="Cannot generate mesh from an empty airway mask"):
        mask_to_polydata(empty_mask, spacing_mm=1.0)


def test_cli_plan_coordinates_and_options(tmp_path):
    from bronchnav.cli import main

    p = make_phantom()
    ct = tmp_path / "phantom.mha"
    sitk.WriteImage(p.image, str(ct))

    # 1. Negative coordinates: --target -60.2 -1.4 -37.0
    out1 = tmp_path / "out1"
    rc1 = main([
        "plan",
        str(ct),
        "--target", "-60.2", "-1.4", "-37.0",
        "--out", str(out1),
        "--no-render",
    ])
    assert rc1 == 0
    assert (out1 / "route.json").exists()
    d1 = json.loads((out1 / "route.json").read_text(encoding="utf-8"))
    assert d1["target"]["center_mm"] == pytest.approx([-60.2, -1.4, -37.0], abs=1e-2)

    # 2. Comma-delimited coordinates: --target "-60.2, -1.4, -37.0"
    out2 = tmp_path / "out2"
    rc2 = main([
        "plan",
        str(ct),
        "--target", "-60.2, -1.4, -37.0",
        "--out", str(out2),
        "--no-render",
    ])
    assert rc2 == 0
    assert (out2 / "route.json").exists()
    d2 = json.loads((out2 / "route.json").read_text(encoding="utf-8"))
    assert d2["target"]["center_mm"] == pytest.approx([-60.2, -1.4, -37.0], abs=1e-2)

    # 3. Exposed parameters: --min-bend-radius, --fov, --leak-factor, --min-leak-jump
    out3 = tmp_path / "out3"
    rc3 = main([
        "plan",
        str(ct),
        "--target", "-60.2", "-1.4", "-37.0",
        "--min-bend-radius", "10.0",
        "--fov", "95.0",
        "--leak-factor", "2.5",
        "--min-leak-jump", "35.0",
        "--out", str(out3),
        "--no-render",
    ])
    assert rc3 == 0
    assert (out3 / "route.json").exists()


def test_vb_renderer_camera_roll(case):
    p, img, hu, seg, tree, grid = case
    tgt_grid = grid.points_to_grid(p.nodule_center_mm)
    route = plan_route(tree, tgt_grid, p.nodule_radius_mm)
    from bronchnav.render import VBRenderer

    vb0 = VBRenderer(seg.mask, SP, route, size=(128, 96), camera_roll_deg=0.0)
    vb90 = VBRenderer(seg.mask, SP, route, size=(128, 96), camera_roll_deg=90.0)
    f0 = vb0.frame(5.0)
    f90 = vb90.frame(5.0)
    assert f0.shape == (96, 128, 3)
    assert f90.shape == (96, 128, 3)
    diff = np.abs(f0.astype(float) - f90.astype(float)).mean()
    assert diff > 1.0
