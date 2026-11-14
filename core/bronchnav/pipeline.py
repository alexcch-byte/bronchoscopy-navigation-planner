"""End-to-end Phase 1 pipeline: CT -> airway -> tree -> route -> (stills, video, mesh) + route.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import __version__
from .airway_tree import AirwayTree, build_tree
from .geometry import Grid
from .io import load_volume, resample_isotropic
from .planning import Route, plan_route
from .segmentation import AirwaySegmentation, segment_airway

INTENDED_USE = "Planning aid only. Not for clinical use."


@dataclass
class PlanResult:
    image: sitk.Image
    grid: Grid
    segmentation: AirwaySegmentation
    tree: AirwayTree
    route: Route
    load_warnings: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def route_payload(r: Route, g: Grid, tree: AirwayTree, extra_warnings: list[str] | None = None) -> dict:
    """The parts of route.json that describe one route (physical LPS mm): target, exit, metrics, warnings,
    bifurcations and the path. Used for the main route and for operator-edited alternatives."""
    return {
        "target": {
            "center_mm": None if r.target_mm is None else _r(g.points_to_physical(r.target_mm)),
            "target_lps_mm": None if r.target_mm is None else g.point_to_physical(r.target_mm).tolist(),
            "radius_mm": r.target_radius_mm,
        },
        "exit": {"point_mm": _r(g.points_to_physical(tree.pts_mm[r.exit_node]))},
        "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.metrics.items()},
        "warnings": list(extra_warnings or []) + r.warnings,
        "bifurcations": [
            {"s_mm": round(b["s_mm"], 2), "point_mm": _r(g.points_to_physical(np.array(b["point_mm"])))}
            for b in r.bifurcations
        ],
        "path": {
            "step_mm": 0.5,
            "points_mm": _r(g.points_to_physical(r.points_mm)),
            "radius_mm": _r(r.radius_mm),
            "tangent": _r(g.vectors_to_physical(r.tangents), 4),
        },
    }


def rebuild_from_job(mask_img: sitk.Image, base_route: dict):
    """Rebuild what route editing needs from a finished job's airway mask and route.json.

    Returns (grid, mask, tree, origin_grid, target_grid, target_radius_mm). The route origin is the first point
    of the job's own path, so edited routes start exactly where the automatic one did."""
    grid = Grid.from_image(mask_img)
    mask = sitk.GetArrayFromImage(mask_img).astype(bool)
    origin_grid = grid.points_to_grid(np.array(base_route["path"]["points_mm"][0], dtype=float))
    seed = tuple(int(np.clip(round(v / grid.spacing_mm), 0, n - 1)) for v, n in zip(origin_grid[::-1], mask.shape))
    tree = build_tree(mask, grid.spacing_mm, seed)
    target_grid = grid.points_to_grid(np.array(base_route["target"]["center_mm"], dtype=float))
    return grid, mask, tree, origin_grid, target_grid, float(base_route["target"]["radius_mm"])


def added_airway_warning(info: dict) -> str:
    return (
        f"Includes an airway added by the operator ({info['length_mm']:.0f} mm long, radius {info['radius_mm']:.1f} mm "
        "as entered). The software did not detect it and its width is not measured from the scan."
    )


def route_with_added_airway(mask_img: sitk.Image, base_route: dict, points_phys, radius_mm: float, min_diameter_mm: float):
    """Paint an operator-drawn airway into a finished job's mask, rebuild the tree and route to its far end.

    Returns (new mask image, grid, mask array, tree, route, info, target radius). Raises ValueError with a message
    fit for the operator (bad points, too far from an airway, branch too short or thin to survive tree pruning)."""
    from .planning.route import nearest_node
    from .segmentation.manual import add_airway

    grid0 = Grid.from_image(mask_img)
    new_mask, info = add_airway(sitk.GetArrayFromImage(mask_img).astype(bool), grid0, points_phys, radius_mm)
    new_img = sitk.GetImageFromArray(new_mask.astype(np.uint8))
    new_img.CopyInformation(mask_img)
    grid, mask, tree, origin_grid, target_grid, target_radius = rebuild_from_job(new_img, base_route)
    end_grid = grid.points_to_grid(np.asarray(points_phys[-1], dtype=float))
    tip = float(np.linalg.norm(tree.pts_mm[nearest_node(tree, end_grid)] - end_grid))
    if tip > max(2.5, radius_mm + 1.5):
        raise ValueError(
            "The new airway is too short or too thin to keep as a branch. Click further along it "
            "or increase its radius."
        )
    route = plan_route(
        tree, target_grid, target_radius, origin_mm=origin_grid, min_diameter_mm=min_diameter_mm, force_exit_mm=end_grid
    )
    return new_img, grid, mask, tree, route, info, target_radius


def render_outputs(
    out: Path,
    mask: np.ndarray,
    spacing_mm: float,
    route: Route,
    grid: Grid,
    size: tuple[int, int] = (640, 480),
    fov_deg: float = 100.0,
    camera_roll_deg: float = 0.0,
    video: bool = True,
    speed_mm_s: float = 8.0,
    log=print,
) -> dict[str, str]:
    """Mesh (PLY, GLB), stills and optionally the fly-through video for a route; returns the outputs map."""
    from .render import VBRenderer

    outputs: dict[str, str] = {}
    log("rendering virtual bronchoscopy")
    vb = VBRenderer(mask, spacing_mm, route, size=size, fov_deg=fov_deg, camera_roll_deg=camera_roll_deg)
    vb.export_mesh(out / "airway_mesh.ply", grid=grid)
    outputs["mesh"] = "airway_mesh.ply"
    vb.export_glb(out / "airway_mesh.glb", grid=grid)
    outputs["mesh_glb"] = "airway_mesh.glb"
    stills = vb.render_stills(out / "stills")
    outputs["stills"] = "stills/ (%d images)" % len(stills)
    if video:
        vb.render_video(
            out / "flythrough.mp4", speed_mm_s=speed_mm_s, progress=lambda k, n: log(f"  video frame {k}/{n}")
        )
        outputs["video"] = "flythrough.mp4"
    return outputs


def route_to_dict(res: PlanResult, source: str, origin_mm_phys: np.ndarray | None) -> dict:
    g, r, seg, tree = res.grid, res.route, res.segmentation, res.tree
    d = {
        "schema": "bronchnav.route/0.1",
        "software": {"name": "bronchnav", "version": __version__},
        "intended_use": INTENDED_USE,
        "input": source,
        "frame": "physical LPS mm (ITK convention); points are ordered origin -> end of airway route",
        "grid_spacing_mm": g.spacing_mm,
        "grid_origin_mm": g.origin.tolist(),
        "grid_direction": g.direction.tolist(),
        "segmentation": {
            "method": "classical threshold-ladder region growing",
            "threshold_hu": seg.threshold_hu,
            "leak_detected": seg.leak_detected,
            "volumes_ml": {str(k): round(v, 2) for k, v in seg.volumes_ml.items()},
            "notes": seg.notes,
        },
        "tree": {
            "n_nodes": int(len(tree.pts_mm)),
            "total_length_mm": round(tree.total_length_mm(), 1),
            "n_junctions": int(len(tree.junctions)),
            "n_endpoints": int(len(tree.endpoints)),
        },
        "origin": {
            "requested_mm": None if origin_mm_phys is None else _r(origin_mm_phys),
            "snapped_mm": _r(g.points_to_physical(tree.pts_mm[r.origin_node])),
        },
    }
    d.update(route_payload(r, g, tree, res.load_warnings))
    return d


def run_plan(
    ct_path: str | Path,
    target_mm_phys: np.ndarray,
    target_radius_mm: float = 5.0,
    origin_mm_phys: np.ndarray | None = None,
    out_dir: str | Path | None = None,
    spacing_mm: float = 0.75,
    min_diameter_mm: float = 3.0,
    min_bend_radius_limit_mm: float = 12.0,
    leak_factor: float = 2.0,
    min_leak_jump_ml: float = 30.0,
    fov_deg: float = 100.0,
    camera_roll_deg: float = 0.0,
    render: bool = True,
    video: bool = True,
    size: tuple[int, int] = (640, 480),
    speed_mm_s: float = 8.0,
    log=print,
    image_stack: dict | None = None,
    small_airways: bool = True,
) -> PlanResult:
    """`image_stack`: when `ct_path` is a folder of JPEG/PNG/... slices instead of DICOM, pass the keyword
    arguments of `bronchnav.io.load_image_stack` (pixel_spacing_mm, slice_spacing_mm, window_center, ...)."""
    log(f"loading {ct_path}")
    if image_stack is not None:
        from .io import load_image_stack

        image, load_warnings = load_image_stack(ct_path, **image_stack)
    else:
        image, load_warnings = load_volume(ct_path)
    for w in load_warnings:
        log(f"  warning: {w}")
    image = resample_isotropic(image, spacing_mm)
    grid = Grid.from_image(image)
    hu = sitk.GetArrayFromImage(image)

    log("segmenting airway")
    seg = segment_airway(hu, spacing_mm, leak_factor=leak_factor, min_leak_jump_ml=min_leak_jump_ml, small_airways=small_airways)
    for n in seg.notes:
        log(f"  note: {n}")
    log(f"  threshold {seg.threshold_hu} HU, volume {seg.volumes_ml[seg.threshold_hu]:.1f} mL")

    log("building airway tree")
    tree = build_tree(seg.mask, spacing_mm, seg.seed_zyx)
    log(f"  {len(tree.junctions)} junctions, {len(tree.endpoints)} endpoints, {tree.total_length_mm():.0f} mm")

    tgt_grid = grid.points_to_grid(np.asarray(target_mm_phys, dtype=np.float64))
    org_grid = None if origin_mm_phys is None else grid.points_to_grid(np.asarray(origin_mm_phys, dtype=np.float64))
    log("planning route")
    route = plan_route(
        tree,
        tgt_grid,
        target_radius_mm,
        origin_mm=org_grid,
        min_diameter_mm=min_diameter_mm,
        min_bend_radius_limit_mm=min_bend_radius_limit_mm,
    )
    res = PlanResult(image, grid, seg, tree, route, load_warnings)
    for w in route.warnings:
        log(f"  warning: {w}")
    log(
        f"  route {route.metrics['length_mm']:.0f} mm, {route.metrics['n_bifurcations']} bifurcations, "
        f"gap to nodule {route.gap_mm:.1f} mm"
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        mask_img = sitk.GetImageFromArray(seg.mask.astype(np.uint8))
        mask_img.CopyInformation(image)
        sitk.WriteImage(mask_img, str(out / "airway_mask.nii.gz"))
        res.outputs["airway_mask"] = "airway_mask.nii.gz"

        (out / "route.json").write_text(
            json.dumps(route_to_dict(res, str(ct_path), origin_mm_phys), indent=1), encoding="utf-8"
        )
        res.outputs["route"] = "route.json"

        if render:
            res.outputs.update(
                render_outputs(out, seg.mask, spacing_mm, route, grid, size=size, fov_deg=fov_deg,
                               camera_roll_deg=camera_roll_deg, video=video, speed_mm_s=speed_mm_s, log=log)
            )
        log(f"outputs in {out}")
    return res
