"""Evaluate classical airway segmentation on AeroPath against the provided airway masks.

Usage (use -I so the dataset folder is never on sys.path):
  python -I core/scripts/eval_aeropath.py --root $HOME/datasets/aeropath/extracted/AeroPath \
      --cases 1 2 3 --out data/eval_aeropath.json

Per case: geometry and warnings, seed inside the ground-truth airway?, threshold/volume/leak flag,
Dice, tree-length detected (fraction of ground-truth centerline voxels within 2 voxels of the
prediction), false-positive volume (prediction farther than 2 voxels from the ground truth, mL),
centerline tree stats, and wall time. Dataset files are read as data only.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
from skimage.morphology import skeletonize

from bronchnav.airway_tree import build_tree
from bronchnav.io import load_volume, resample_isotropic
from bronchnav.segmentation import segment_airway


def _load_label(path: Path, like: sitk.Image) -> np.ndarray:
    lab = sitk.ReadImage(str(path))
    lab = sitk.DICOMOrient(sitk.Cast(lab > 0, sitk.sitkUInt8), "LPS")
    out = sitk.Resample(lab, like, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(out).astype(bool)


def eval_case(root: Path, case: int, spacing: float, seg_kwargs: dict | None = None, smooth_mm: float = 0.0) -> dict:
    d = root / str(case)
    ct, label = d / f"{case}_CT_HR.nii.gz", d / f"{case}_CT_HR_label_airways.nii.gz"
    res: dict = {"case": case}
    t0 = time.time()
    image, warnings = load_volume(ct)
    res["size_xyz"] = list(image.GetSize())
    res["spacing_mm"] = [round(s, 3) for s in image.GetSpacing()]
    res["load_warnings"] = warnings
    image = resample_isotropic(image, spacing)
    hu = sitk.GetArrayFromImage(image)
    res["hu_min_max"] = [float(hu.min()), float(hu.max())]
    if smooth_mm > 0:  # optional denoising before thresholding
        hu = ndi.gaussian_filter(hu, sigma=smooth_mm / spacing)
    truth = _load_label(label, image)
    res["truth_ml"] = round(float(truth.sum()) * spacing**3 / 1000.0, 2)
    res["t_load_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    try:
        seg = segment_airway(hu, spacing, **(seg_kwargs or {}))
    except Exception as e:  # report and continue with the other cases
        res["error"] = f"segmentation: {type(e).__name__}: {e}"
        res["seed_failed"] = "seed" in str(e).lower()
        return res
    res["t_seg_s"] = round(time.time() - t0, 1)
    pred = seg.mask
    res.update(
        seed_zyx=list(seg.seed_zyx),
        seed_in_truth=bool(truth[seg.seed_zyx]),
        threshold_hu=seg.threshold_hu,
        leak_detected=seg.leak_detected,
        pred_ml=round(float(pred.sum()) * spacing**3 / 1000.0, 2),
        volumes_ml={str(k): round(v, 1) for k, v in seg.volumes_ml.items()},
        notes=seg.notes,
    )
    inter = float((pred & truth).sum())
    res["dice"] = round(2 * inter / max(1.0, float(pred.sum() + truth.sum())), 4)
    res["recall_vox"] = round(inter / max(1.0, float(truth.sum())), 4)
    res["precision_vox"] = round(inter / max(1.0, float(pred.sum())), 4)

    # work in the bounding box of truth | pred to keep this cheap
    both = pred | truth
    box = ndi.find_objects(both.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - 4), min(n, s.stop + 4)) for s, n in zip(box, both.shape))
    p, t = pred[sl], truth[sl]
    skel_t = skeletonize(t).astype(bool)
    near_pred = ndi.binary_dilation(p, iterations=2)
    res["tree_length_detected"] = round(float((skel_t & near_pred).sum()) / max(1.0, float(skel_t.sum())), 4)
    near_truth = ndi.binary_dilation(t, iterations=2)
    res["false_positive_ml"] = round(float((p & ~near_truth).sum()) * spacing**3 / 1000.0, 2)

    t0 = time.time()
    try:
        tree = build_tree(pred, spacing, seg.seed_zyx)
        res["tree"] = {
            "junctions": int(len(tree.junctions)),
            "endpoints": int(len(tree.endpoints)),
            "length_mm": round(tree.total_length_mm(), 1),
            "root_z_is_top": bool(tree.pts_mm[tree.root][2] > tree.pts_mm[:, 2].max() - 6.0),
        }
    except Exception as e:
        res["tree_error"] = f"{type(e).__name__}: {e}"
    res["t_tree_s"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the numbered case folders")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(1, 28)))
    ap.add_argument("--spacing", type=float, default=0.75)
    ap.add_argument("--out", default="eval_aeropath.json")
    ap.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=None,
                    help="labeling connectivity for region growing (default: the library default)")
    ap.add_argument("--step", type=int, default=None, help="threshold ladder step in HU (default 25)")
    ap.add_argument("--smooth-mm", type=float, default=0.0, help="Gaussian sigma (mm) applied before thresholding")
    args = ap.parse_args()

    seg_kwargs: dict = {}
    if args.connectivity is not None:
        seg_kwargs["connectivity"] = args.connectivity
    if args.step is not None:
        seg_kwargs["thresholds"] = tuple(range(-950, -799, args.step))

    root, results = Path(args.root), []
    for c in args.cases:
        print(f"--- case {c}", flush=True)
        try:
            r = eval_case(root, c, args.spacing, seg_kwargs, args.smooth_mm)
        except Exception:
            r = {"case": c, "error": traceback.format_exc(limit=3)}
        results.append(r)
        keys = ("dice", "recall_vox", "precision_vox", "tree_length_detected", "false_positive_ml", "leak_detected", "error")
        print({k: r[k] for k in keys if k in r}, flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")  # save as we go
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
