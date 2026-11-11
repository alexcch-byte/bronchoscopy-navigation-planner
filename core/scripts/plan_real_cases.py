"""Run route planning + virtual bronchoscopy on real AeroPath cases.

AeroPath has no nodule annotations, so a stand-in target is placed in the lung parenchyma just beyond
a distal airway branch: pick a distal tree endpoint, step a few mm further along its direction, and
require the point to be inside the ground-truth lung mask. The target is a test point, not a nodule.

Usage (python -I so the dataset folder is never on sys.path):
  python -I core/scripts/plan_real_cases.py --root <AeroPath folder> --cases 20 16 1 --out <out dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from bronchnav.airway_tree import build_tree
from bronchnav.autotarget import choose_target
from bronchnav.geometry import Grid
from bronchnav.io import load_volume, resample_isotropic
from bronchnav.pipeline import run_plan
from bronchnav.segmentation import segment_airway

SP = 0.75


def _label_on_grid(path: Path, like: sitk.Image) -> np.ndarray:
    lab = sitk.DICOMOrient(sitk.Cast(sitk.ReadImage(str(path)) > 0, sitk.sitkUInt8), "LPS")
    return sitk.GetArrayFromImage(sitk.Resample(lab, like, sitk.Transform(), sitk.sitkNearestNeighbor, 0)).astype(bool)


def run_case(root: Path, case: int, out: Path, pctl: float, ahead: float, video: bool) -> dict:
    d = root / str(case)
    ct = d / f"{case}_CT_HR.nii.gz"
    t0 = time.time()
    image, _ = load_volume(ct)
    image = resample_isotropic(image, SP)
    grid = Grid.from_image(image)
    hu = sitk.GetArrayFromImage(image)
    lungs = _label_on_grid(d / f"{case}_CT_HR_label_lungs.nii.gz", image)
    truth = _label_on_grid(d / f"{case}_CT_HR_label_airways.nii.gz", image)
    seg = segment_airway(hu, SP)
    tree = build_tree(seg.mask, SP, seg.seed_zyx)
    e, tgt_grid, geo, ahead_used = choose_target(tree, lungs, SP, pctl, ahead)
    # how far is the target from the TRUE airway (the predicted tree is incomplete)?
    ix = np.round(tgt_grid / SP).astype(int)
    box = tuple(slice(max(0, i - 40), i + 41) for i in ix[::-1])
    dt_truth = ndi.distance_transform_edt(~truth[box], sampling=SP)
    local = tuple(i - s.start for i, s in zip(ix[::-1], box))
    target_phys = grid.points_to_physical(tgt_grid)
    del hu, lungs, truth, image

    res = run_plan(
        ct,
        target_phys,
        target_radius_mm=6.0,
        out_dir=out / f"case{case}",
        spacing_mm=SP,
        video=video,
        log=lambda *_: None,
    )
    m = res.route.metrics
    return {
        "case": case,
        "target_phys_mm": [round(float(v), 1) for v in target_phys],
        "target_ahead_of_endpoint_mm": ahead_used,
        "target_dist_to_true_airway_mm": round(float(dt_truth[local]), 1),
        "endpoint_geodesic_mm": round(geo, 1),
        "tree": {"junctions": int(len(tree.junctions)), "endpoints": int(len(tree.endpoints))},
        "route": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()},
        "warnings": res.route.warnings,
        "outputs": res.outputs,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentile", type=float, default=0.7, help="distal endpoint rank (0=nearest, 1=farthest)")
    ap.add_argument("--ahead-mm", type=float, default=15.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for c in args.cases:
        print(f"--- case {c}", flush=True)
        try:
            r = run_case(Path(args.root), c, out, args.percentile, args.ahead_mm, not args.no_video)
        except Exception as e:  # keep going on the other cases
            import traceback

            r = {"case": c, "error": traceback.format_exc(limit=4)}
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
        (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
