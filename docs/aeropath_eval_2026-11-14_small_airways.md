# AeroPath evaluation with the thin-airway extension (2026-11-14)

The baseline in `aeropath_eval_2026-11-10.md` (single global threshold with leak control) reached about 55% of the true airway centerline. This run adds `segmentation/small_airways.py`: after the threshold result, it grows outward through voxels that look like a dark tube (a Hessian tube measure at 0.75 and 1.0 mm) that are darker than -550 HU, up to 30 mm from the existing mask and at most 40 mL, then widens the new branches by one voxel inside lumen-like voxels (smoothed HU below -700). The result is always a superset of the baseline mask. Produced by `core/scripts/eval_aeropath.py --small-airways` on the GPU path (CuPy, RTX 4070); raw per-case results are in `aeropath_eval_2026-11-13_small_airways.json`. The CPU path gives identical masks.

Setup is otherwise the baseline's: 27 AeroPath CTAs, 0.75 mm isotropic grid, 6-connected labeling, the same metrics. Parameters (`tube_min_hu` 50, `hu_max` -550, `thicken_hu` -700) were tuned on cases 5, 10, 12 and 20 and then run on all cases, so those four are not held-out; the other 22 are, in the sense that nothing was tuned on them.

## Summary (26 of 27 cases; case 9 still fails by design, see the baseline report)

| Metric | Baseline mean | With extension mean | Baseline median | With extension median |
|---|---|---|---|---|
| Centerline detected | 0.551 | **0.705** | 0.566 | **0.723** |
| Dice | 0.861 | 0.870 | 0.869 | 0.872 |
| Recall (voxels) | 0.783 | 0.838 | 0.806 | 0.862 |
| Precision (voxels) | 0.966 | 0.915 | 0.988 | 0.920 |
| False positive (mL) | 0.32 | 1.14 | 0.00 | 0.64 |

- Centerline reach improved on **26 of 26** cases (+0.154 on average; case 10 went from 0.44 to 0.80, case 21 from 0.75 to 0.97).
- Dice improved on 14 of 26 cases and is flat on average (0.861 to 0.870).
- The cost is precision: the added branches are a little wider than AeroPath's thin ground-truth masks. Mean precision fell from 0.966 to 0.915, and false-positive volume (added voxels more than 2 voxels from the truth) rose from 0.32 to 1.14 mL, which is small next to 20-100 mL of airway. The largest is case 18 at 10.6 mL (it already had 7.8 mL of false positive in the baseline, precision 0.775).
- Segmentation step time (GPU): 2.6 s per case on average, against 4.5 s for the baseline on the CPU.

## Per case

| Case | Dice | Centerline | Precision | FP mL | Added mL |
|---|---|---|---|---|---|
| 1 | 0.807 -> 0.833 | 0.548 -> 0.734 | 0.998 -> 0.935 | 4.41 | +13.1 |
| 2 | 0.907 -> 0.921 | 0.660 -> 0.735 | 0.996 -> 0.988 | 0.10 | +2.9 |
| 3 | 0.906 -> 0.877 | 0.527 -> 0.661 | 0.955 -> 0.854 | 1.39 | +6.1 |
| 4 | 0.891 -> 0.880 | 0.580 -> 0.711 | 0.982 -> 0.885 | 0.65 | +7.4 |
| 5 | 0.826 -> 0.868 | 0.742 -> 0.848 | 0.997 -> 0.983 | 0.23 | +2.8 |
| 6 | 0.831 -> 0.869 | 0.424 -> 0.509 | 0.999 -> 0.990 | 0.06 | +2.0 |
| 7 | 0.902 -> 0.888 | 0.569 -> 0.670 | 0.952 -> 0.877 | 0.68 | +5.3 |
| 8 | 0.888 -> 0.867 | 0.387 -> 0.591 | 0.955 -> 0.845 | 0.26 | +4.4 |
| 10 | 0.734 -> 0.821 | 0.437 -> 0.795 | 1.000 -> 0.984 | 0.34 | +11.7 |
| 11 | 0.871 -> 0.871 | 0.692 -> 0.735 | 0.995 -> 0.949 | 1.43 | +2.8 |
| 12 | 0.858 -> 0.881 | 0.432 -> 0.581 | 1.000 -> 0.979 | 0.83 | +4.1 |
| 13 | 0.899 -> 0.872 | 0.563 -> 0.704 | 0.951 -> 0.847 | 1.06 | +6.1 |
| 14 | 0.879 -> 0.897 | 0.617 -> 0.676 | 1.000 -> 0.999 | 0.00 | +1.7 |
| 15 | 0.807 -> 0.844 | 0.457 -> 0.600 | 0.999 -> 0.995 | 0.00 | +2.5 |
| 16 | 0.913 -> 0.908 | 0.620 -> 0.782 | 0.942 -> 0.889 | 0.29 | +8.0 |
| 17 | 0.887 -> 0.884 | 0.580 -> 0.720 | 0.969 -> 0.901 | 1.13 | +7.5 |
| 18 | 0.796 -> 0.777 | 0.374 -> 0.576 | 0.775 -> 0.701 | 10.60 | +8.5 |
| 19 | 0.859 -> 0.870 | 0.369 -> 0.565 | 0.908 -> 0.873 | 0.04 | +7.2 |
| 20 | 0.935 -> 0.923 | 0.552 -> 0.772 | 0.968 -> 0.905 | 0.23 | +4.9 |
| 21 | 0.839 -> 0.879 | 0.749 -> 0.965 | 1.000 -> 0.979 | 0.82 | +6.3 |
| 22 | 0.909 -> 0.897 | 0.585 -> 0.726 | 0.966 -> 0.891 | 0.29 | +8.0 |
| 23 | 0.900 -> 0.864 | 0.648 -> 0.771 | 0.877 -> 0.795 | 0.91 | +4.1 |
| 24 | 0.859 -> 0.878 | 0.610 -> 0.741 | 1.000 -> 0.982 | 0.62 | +3.3 |
| 25 | 0.826 -> 0.868 | 0.576 -> 0.734 | 0.995 -> 0.965 | 0.78 | +5.2 |
| 26 | 0.782 -> 0.837 | 0.511 -> 0.691 | 0.997 -> 0.970 | 0.54 | +8.4 |
| 27 | 0.866 -> 0.850 | 0.510 -> 0.734 | 0.947 -> 0.840 | 1.86 | +13.1 |

## Findings

1. **The thin airways are not dark enough to threshold, but they are tube-shaped.** In case 5 the airway voxels the baseline missed had a median smoothed HU near -600 (partial volume against the airway wall), so no threshold below the leak point reaches them. A tube test gated by connectivity to the existing tree does, because lung parenchyma is not tubular and cannot be flooded.
2. **Stricter tube test with a looser HU limit worked better** than the reverse (tube measure 50 and -550 HU, against 15 and -600 at first, which added 17-33 mL of false positive).
3. **Widening was the main precision lever.** A one-voxel centerline chain scored lower Dice than AeroPath's wider masks; widening it only inside voxels darker than -700 HU kept most of the recall and cut the false positives.
4. **It still stops at about 70% of the centerline.** Likely causes (not tested separately): branches too faint for the tube test on a 0.75 mm grid, or beyond the 30 mm reach limit. Going deeper is a job for the learned model (Phase 2).
5. **Case 9 (emphysema) is unchanged**: the extension needs a valid base mask, and the base segmentation rejects it. The public page now explains that instead of a bare "the run failed".

## Limitations

Parameters were tuned on a handful of these same cases. AeroPath masks are tight, so precision against them penalises branches that are a voxel wider than the truth (false positives measured 2 voxels away from the truth stay small, which supports that, but I did not review the added airways by eye). Not validated on any other dataset or scanner, and not for clinical use.
