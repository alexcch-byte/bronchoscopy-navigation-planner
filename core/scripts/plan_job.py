"""One planning job for the local test console (called by server/app/main.py as a subprocess).

  python -I core/scripts/plan_job.py --source aeropath --case 20 --out <dir> --target-mode auto
  python -I core/scripts/plan_job.py --source phantom --out <dir>
  python -I core/scripts/plan_job.py --source upload --ct-path <file or image folder> [--stack-json <file>] --out <dir>
  python -I core/scripts/plan_job.py --source edit --base-job-dir <finished job dir> --end X Y Z --out <dir>

`--stack-json` holds the keyword arguments of `bronchnav.io.load_image_stack` and marks `--ct-path` as a folder
of JPEG/PNG/... slices (no DICOM). Writes the usual `plan` outputs into --out plus `job_result.json`.
Progress goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bronchnav.airway_tree import build_tree
from bronchnav.autotarget import choose_target
from bronchnav.geometry import Grid
from bronchnav.io import load_image_stack, load_volume, resample_isotropic
from bronchnav.phantom import make_phantom
from bronchnav.pipeline import (
    added_airway_warning, rebuild_from_job, render_outputs, route_payload, route_with_added_airway, run_plan,
)
from bronchnav.planning import plan_route
from bronchnav.segmentation import rough_lung_mask, segment_airway

SP = 0.75


def log(msg: str) -> None:
    print(msg, flush=True)


def _lungs_on_grid(path: Path, like: sitk.Image) -> np.ndarray:
    lab = sitk.DICOMOrient(sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8), "LPS")
    out = sitk.Resample(lab, like, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(out).astype(bool)


def run_edit(args, out: Path) -> dict:
    """A new run whose route ends where the operator chose (physical LPS mm), rendered like any other run."""
    base = Path(args.base_job_dir)
    route0 = json.loads((base / "route.json").read_text(encoding="utf-8"))
    base_result = json.loads((base / "job_result.json").read_text(encoding="utf-8"))
    log("rebuilding the airway tree from the saved mask")
    img = sitk.ReadImage(str(base / "airway_mask.nii.gz"))
    added = None
    extra: list[str] = []
    if getattr(args, "add_airway_json", None):
        spec = json.loads(Path(args.add_airway_json).read_text(encoding="utf-8"))
        log("adding the operator's airway to the mask")
        img, grid, mask, tree, route, added, radius = route_with_added_airway(
            img, route0, spec["points_mm"], float(spec["radius_mm"]), args.min_diameter
        )
        extra.append(added_airway_warning(added))
        log(f"  added {added['length_mm']:.0f} mm of airway ({added['added_ml']:.2f} mL)")
        sitk.WriteImage(img, str(out / "airway_mask.nii.gz"))
    else:
        grid, mask, tree, origin_grid, target_grid, radius = rebuild_from_job(img, route0)
        route = plan_route(
            tree, target_grid, radius, origin_mm=origin_grid, min_diameter_mm=args.min_diameter,
            force_exit_mm=grid.points_to_grid(np.array(args.end, dtype=float)),
        )
        shutil.copy(base / "airway_mask.nii.gz", out / "airway_mask.nii.gz")
    for w in route.warnings:
        log(f"  warning: {w}")
    log(f"  route {route.metrics['length_mm']:.0f} mm, gap to nodule {route.gap_mm:.1f} mm")
    d = dict(route0)
    d.update(route_payload(route, grid, tree, extra))
    d["edit"] = {"base_job": base.name, "end_mm": [round(float(v), 1) for v in args.end]}
    if added:
        d["edit"]["added_airway"] = {**added, "points_mm": spec["points_mm"]}
        d["tree"] = {
            "n_nodes": int(len(tree.pts_mm)), "total_length_mm": round(tree.total_length_mm(), 1),
            "n_junctions": int(len(tree.junctions)), "n_endpoints": int(len(tree.endpoints)),
        }
    (out / "route.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    outputs = {"airway_mask": "airway_mask.nii.gz", "route": "route.json"}
    outputs.update(render_outputs(out, mask, grid.spacing_mm, route, grid, camera_roll_deg=getattr(args, "camera_roll", 0.0), video=not args.no_video, log=log))
    return {
        "ok": True,
        "base_source": base_result.get("source"),
        "name": f"Edited path of {base_result.get('name') or base_result.get('source')}",
        "target": base_result.get("target"),
        "target_radius_mm": radius,
        "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in route.metrics.items()},
        "warnings": extra + route.warnings,
        "outputs": outputs,
        "segmentation": base_result.get("segmentation"),
        "tree": d["tree"] if added else base_result.get("tree"),
        "edit": d["edit"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("phantom", "aeropath", "upload", "edit"), required=True)
    ap.add_argument("--base-job-dir", help="edit: the finished job whose route is being changed")
    ap.add_argument("--end", type=float, nargs=3, metavar=("X", "Y", "Z"), help="edit: new end point, physical LPS mm")
    ap.add_argument("--add-airway-json", help="edit: JSON {points_mm, radius_mm} of an airway the operator drew")
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--nodule", type=float, nargs=3, metavar=("X", "Y", "Z"), help="phantom: custom nodule center LPS mm")
    ap.add_argument("--nodule-radius", type=float, default=5.0, help="phantom: custom nodule radius mm")
    ap.add_argument("--nodule-hu", type=float, default=30.0, help="phantom: custom nodule density HU")
    ap.add_argument("--aeropath-root", default=str(Path.home() / "datasets" / "aeropath" / "extracted" / "AeroPath"))
    ap.add_argument("--ct-path", help="upload: a volume file (.nii.gz, .mha, .nrrd, ...) or a folder of images")
    ap.add_argument("--stack-json", help="upload: JSON file with image-stack options (marks ct-path as an image folder)")
    ap.add_argument("--display-name", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-mode", choices=("auto", "manual"), default="auto")
    ap.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="physical LPS mm (manual)")
    ap.add_argument("--percentile", type=float, default=0.7)
    ap.add_argument("--ahead-mm", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--min-diameter", type=float, default=3.0)
    ap.add_argument("--camera-roll", type=float, default=0.0, help="camera roll / rotation angle in degrees for fly-through video and stills")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result: dict = {
        "ok": False,
        "source": args.source,
        "case": args.case if args.source == "aeropath" else None,
        "name": args.display_name or None,
    }
    try:
        info: dict = {}
        if args.source == "edit":
            if not (args.base_job_dir and args.end):
                raise ValueError("edit needs --base-job-dir and --end X Y Z")
            result.update(run_edit(args, out))
            result["seconds"] = round(time.time() - t0, 1)
            (out / "job_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            log("done")
            return 0
        stack = json.loads(Path(args.stack_json).read_text(encoding="utf-8")) if args.stack_json else None
        if stack:
            result["image_stack"] = stack
        extra_warnings: list[str] = []
        if args.source == "phantom":
            log("building synthetic phantom")
            if args.nodule:
                from bronchnav.phantom import NoduleSpec

                spec = NoduleSpec(center_mm=tuple(args.nodule), radius_mm=args.nodule_radius, hu=args.nodule_hu)
                ph = make_phantom(nodule=spec)
                log(f"custom nodule: center={spec.center_mm}, radius={spec.radius_mm} mm, hu={spec.hu} HU")
            else:
                ph = make_phantom()
            ct = out / "phantom.mha"
            sitk.WriteImage(ph.image, str(ct))
            if args.target_mode == "manual" and args.target:
                target = np.array(args.target, dtype=float)
                info["target"] = "manual"
            else:
                target = np.array(ph.nodule_center_mm, dtype=float)
                info["target"] = "custom phantom nodule" if args.nodule else "phantom nodule"
            radius = ph.nodule_radius_mm if args.target_mode == "auto" else args.radius
        else:
            radius = args.radius
            lungs_from = None
            if args.source == "aeropath":
                d = Path(args.aeropath_root) / str(args.case)
                ct = d / f"{args.case}_CT_HR.nii.gz"
                if not ct.exists():
                    raise FileNotFoundError(ct)
                lungs_from = d / f"{args.case}_CT_HR_label_lungs.nii.gz"
            else:
                if not args.ct_path:
                    raise ValueError("upload needs --ct-path")
                ct = Path(args.ct_path)
                if not ct.exists():
                    raise FileNotFoundError(ct)
            if args.target_mode == "manual":
                if not args.target:
                    raise ValueError("manual target needs --target X Y Z")
                target = np.array(args.target, dtype=float)
                info["target"] = "manual"
            else:
                log("finding a stand-in target (segment, tree, lung mask)")
                if stack:
                    image, _ = load_image_stack(ct, **stack)
                else:
                    image, _ = load_volume(ct)
                image = resample_isotropic(image, SP)
                grid = Grid.from_image(image)
                hu = sitk.GetArrayFromImage(image)
                if lungs_from is not None:
                    lungs = _lungs_on_grid(lungs_from, image)
                    how = "auto: parenchymal point beyond a distal airway tip (not a nodule), lung label from AeroPath"
                else:
                    lungs = rough_lung_mask(hu)
                    how = "auto: point beyond a distal airway tip inside an approximate threshold lung mask (not a nodule)"
                seg = segment_airway(hu, SP, small_airways=True)  # the same mask run_plan will use
                tree = build_tree(seg.mask, SP, seg.seed_zyx)
                e, tgt_grid, geo, ahead = choose_target(tree, lungs, SP, args.percentile, args.ahead_mm)
                target = grid.points_to_physical(tgt_grid)
                info.update(target=how, percentile=args.percentile, ahead_mm=ahead, endpoint_geodesic_mm=round(geo, 1))
                del hu, lungs, image, seg, tree

        info["target_mm"] = [round(float(v), 1) for v in target]
        log(f"target (physical LPS mm): {info['target_mm']}")
        res = run_plan(
            ct,
            target,
            target_radius_mm=float(radius),
            out_dir=out,
            spacing_mm=SP,
            min_diameter_mm=args.min_diameter,
            camera_roll_deg=args.camera_roll,
            video=not args.no_video,
            log=log,
            image_stack=stack,
        )
        result.update(
            ok=True,
            target=info,
            target_radius_mm=float(radius),
            metrics={k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.route.metrics.items()},
            warnings=extra_warnings + res.load_warnings + res.route.warnings,
            outputs=res.outputs,
            segmentation={
                "threshold_hu": res.segmentation.threshold_hu,
                "leak_detected": res.segmentation.leak_detected,
                "notes": res.segmentation.notes,
            },
            tree={
                "junctions": int(len(res.tree.junctions)),
                "endpoints": int(len(res.tree.endpoints)),
                "length_mm": round(res.tree.total_length_mm(), 1),
            },
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=6)
        log("ERROR: " + result["error"])
    result["seconds"] = round(time.time() - t0, 1)
    (out / "job_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("done" if result["ok"] else "failed")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
