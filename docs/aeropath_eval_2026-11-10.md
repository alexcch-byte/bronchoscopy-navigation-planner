# AeroPath baseline evaluation (2026-11-10)

Classical segmentation (`bronchnav.segmentation.segment_airway`: trachea seed + threshold-ladder region growing with leak control) against the AeroPath ground-truth airway masks. Produced by `core/scripts/eval_aeropath.py`; raw per-case results are in `aeropath_eval_2026-11-10.json`.

## Setup

- Data: AeroPath v1 (Zenodo 10069289, CC BY 4.0), 27 contrast-enhanced CTAs, MD5 verified. 0.5-1.25 mm slices, 0.68-0.76 mm in-plane.
- Working grid: 0.75 mm isotropic. Threshold ladder -950 to -800 HU in 25 HU steps. Labeling connectivity 1 (6-neighbour), which is now also the library default (it was 26-neighbour when this was first run; see Findings).
- Leak rule: volume jump over 2x AND over 30 mL; a region under 10 mL (a trachea fragment) only counts as leaking above 100 mL; a region over 400 mL is always a leak.
- Metrics: Dice and voxel recall/precision against the truth; tree length detected = fraction of the truth centerline voxels within 2 voxels of the prediction; false positive = predicted volume farther than 2 voxels from the truth (mL).

## Summary (26 of 27 cases ran)

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Dice | 0.861 | 0.869 | 0.734 | 0.935 |
| Recall (voxels) | 0.783 | 0.806 | 0.579 | 0.924 |
| Precision (voxels) | 0.966 | 0.988 | 0.775 | 1.000 |
| Centerline detected | 0.551 | 0.566 | 0.369 | 0.749 |
| False positive (mL) | 0.323 | 0.000 | 0.000 | 7.800 |

Leak guard fired on 23 of 26 cases. Seed was inside the true airway in 26 of 26.

Case 9: **ValueError: Region growing already exceeds 400 mL (1246 mL) at the lowest threshold (-950 HU): the airway leaks into the lung. Needs a different seed or a learned model.**

## Per case

| Case | Slice mm | Stop HU | Dice | Recall | Precision | Centerline | FP mL | Truth mL | Pred mL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | -925 | 0.807 | 0.677 | 0.998 | 0.548 | 0.1 | 105.8 | 71.8 |
| 2 | 0.5 | -850 | 0.907 | 0.832 | 0.996 | 0.660 | 0.0 | 77.3 | 64.6 |
| 3 | 0.5 | -900 | 0.906 | 0.862 | 0.955 | 0.527 | 0.0 | 39.8 | 35.9 |
| 4 | 0.5 | -900 | 0.891 | 0.817 | 0.982 | 0.580 | 0.0 | 47.1 | 39.1 |
| 5 | 0.5 | -850 | 0.826 | 0.706 | 0.997 | 0.742 | 0.0 | 33.8 | 23.9 |
| 6 | 0.5 | -850 | 0.831 | 0.711 | 0.999 | 0.424 | 0.0 | 28.8 | 20.5 |
| 7 | 0.5 | -900 | 0.902 | 0.857 | 0.952 | 0.569 | 0.0 | 42.7 | 38.4 |
| 8 | 0.5 | -900 | 0.888 | 0.829 | 0.955 | 0.387 | 0.0 | 23.9 | 20.7 |
| 9 | | error | | | | | | | |
| 10 | 1.25 | -900 | 0.734 | 0.579 | 1.000 | 0.437 | 0.0 | 86.0 | 49.9 |
| 11 | 0.5 | -800 | 0.871 | 0.775 | 0.995 | 0.692 | 0.0 | 39.4 | 30.6 |
| 12 | 0.5 | -875 | 0.858 | 0.752 | 1.000 | 0.432 | 0.0 | 63.0 | 47.4 |
| 13 | 0.5 | -900 | 0.899 | 0.852 | 0.951 | 0.563 | 0.3 | 36.7 | 32.8 |
| 14 | 0.5 | -800 | 0.879 | 0.784 | 1.000 | 0.617 | 0.0 | 54.3 | 42.6 |
| 15 | 0.5 | -800 | 0.807 | 0.677 | 0.999 | 0.457 | 0.0 | 41.2 | 27.9 |
| 16 | 0.5 | -925 | 0.913 | 0.886 | 0.942 | 0.620 | 0.0 | 76.3 | 71.8 |
| 17 | 0.5 | -900 | 0.887 | 0.818 | 0.969 | 0.580 | 0.0 | 63.3 | 53.5 |
| 18 | 0.5 | -900 | 0.796 | 0.818 | 0.775 | 0.374 | 7.8 | 44.4 | 46.8 |
| 19 | 0.499 | -850 | 0.859 | 0.815 | 0.908 | 0.369 | 0.0 | 74.5 | 66.9 |
| 20 | 0.999 | -875 | 0.935 | 0.905 | 0.968 | 0.552 | 0.0 | 45.4 | 42.4 |
| 21 | 0.5 | -925 | 0.839 | 0.722 | 1.000 | 0.749 | 0.0 | 68.4 | 49.4 |
| 22 | 0.997 | -950 | 0.909 | 0.860 | 0.966 | 0.585 | 0.0 | 65.6 | 58.4 |
| 23 | 0.499 | -875 | 0.900 | 0.924 | 0.877 | 0.648 | 0.0 | 29.8 | 31.3 |
| 24 | 0.499 | -825 | 0.859 | 0.753 | 1.000 | 0.610 | 0.0 | 59.7 | 45.0 |
| 25 | 1.0 | -900 | 0.826 | 0.706 | 0.995 | 0.576 | 0.1 | 47.9 | 34.0 |
| 26 | 1.245 | -875 | 0.782 | 0.643 | 0.997 | 0.511 | 0.0 | 73.9 | 47.7 |
| 27 | 0.499 | -925 | 0.866 | 0.798 | 0.947 | 0.510 | 0.0 | 71.3 | 60.1 |

## Variants tried

| Variant | Cases | Dice | Recall | Precision | Centerline | Notes |
|---|---|---|---|---|---|---|
| 26-connected, 25 HU ladder | 25 of 27 ran | 0.806 | 0.738 | 0.948 | 0.535 | Cases 9 and 16 error at the first threshold; case 8 precision 0.39 |
| 6-connected, 25 HU ladder | 26 of 27 ran | 0.834 | 0.761 | 0.966 | 0.534 | Case 16 fixed (a diagonal noise voxel bridged the wall), case 8 fixed |
| 6-connected + case 10 rule fix (this report) | 26 of 27 ran | 0.861 | 0.783 | 0.966 | 0.551 | Case 10 Dice 0.03 -> 0.73 |
| 6-connected, 10 HU ladder (10-case subset) | 10 | 0.751 | 0.657 | 0.976 | 0.491 | Same as the 25 HU baseline on that subset (0.748 / 0.655 / 0.976 / 0.490): step size is not the limit |
| 6-connected + Gaussian 0.6 mm (10-case subset) | 10 | 0.783 | 0.656 | 0.982 | 0.410 | Worse centerline (0.49 -> 0.41): blurs thin airways. Rescued case 10, a noise problem |

## Findings

1. The pipeline runs on real contrast CTs end to end in about 10 s per case, with the seed inside the true airway every time. Precision is high (median 0.99): it rarely invents airway.
2. It is shallow. Only about 55% of the truth centerline is reached, because one global threshold leaks into the lung (jumps of 100-3000 mL against 24-105 mL of true airway) before the distal airways cross it. The leaks are real, not false alarms. A finer ladder does not help and global smoothing hurts.
3. Case 9 (emphysema): at -950 HU the lung tissue itself is air-like (1,245 mL, 4% of it true airway). A threshold method cannot handle this; it fails loudly with a clear error, which is the intended behaviour.
4. 26-connected labeling lets a single diagonal noise voxel bridge an airway wall: case 16 goes from a clean 68.7 mL airway to a 1,126 mL leak at the first threshold. 6-connected labeling is better on these data (Dice 0.806 -> 0.834 before the case 10 fix, false positives 1.6 -> 0.3 mL). The library default was changed to 6-connected after this evaluation (verified: cases 8, 10 and 16 give identical results with no flag), and a regression test covers the corner-contact case.
5. Case 10 (1.25 mm slices): the airway sits near -915 HU and only 1.3 mL is below -950 HU, so the next step (a genuine 36 mL of airway) looked like a leak. Fixed by not applying the ratio rule while the region is a tiny fragment.

## Next

- Going deeper needs more than a global threshold: either branch-wise leak control (grow at a higher threshold and roll back only the leaking front) or the learned model in Phase 2, trained on ATM'22 and AeroPath with a held-out split. The numbers above are the baseline to beat.
- Run `plan` on real cases (AeroPath has no nodules: choose parenchymal targets near a distal airway, or use LUNA16 for real nodules) and judge route and video by eye.
