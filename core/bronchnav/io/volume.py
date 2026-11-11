"""CT volume loading, validation and resampling (Phase 0/1).

Volumes are SimpleITK images in HU, oriented LPS. Spacing is in mm, (x, y, z).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

AIR_HU = -1024


def load_volume(path: str | Path) -> tuple[sitk.Image, list[str]]:
    """Load a DICOM series directory or a single volume file (.mha, .nii.gz, ...).

    Returns (image in HU, LPS, float32) and a list of loader warnings.
    SimpleITK's DICOM reader applies rescale slope/intercept, so values are HU.
    """
    path = Path(path)
    warnings: list[str] = []

    if path.is_dir():
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        scan_dir = path
        if not series_ids:
            subdirs = [p for p in path.rglob("*") if p.is_dir()]
            best_dir = None
            best_id = None
            best_files: list[str] = []
            for sdir in subdirs:
                sids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(sdir))
                for sid in sids:
                    fnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(sdir), sid)
                    if len(fnames) > len(best_files):
                        best_files = list(fnames)
                        best_id = sid
                        best_dir = sdir
            if not best_files or best_dir is None:
                from .image_stack import list_images, load_image_stack

                try:
                    imgs = list_images(path)
                    if len(imgs) >= 3:
                        img, extra_warns = load_image_stack(imgs, pixel_spacing_mm=0.75, slice_spacing_mm=1.0)
                        return img, warnings + extra_warns
                except Exception:
                    pass
                raise ValueError(f"No DICOM series found in {path} or its subdirectories.")
            scan_dir = best_dir
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(scan_dir))

        best_id, best_files = None, []
        for sid in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(scan_dir), sid)
            if len(files) > len(best_files):
                best_id, best_files = sid, list(files)
        if len(series_ids) > 1:
            warnings.append(
                f"{len(series_ids)} series found in folder; using the largest "
                f"({len(best_files)} slices, id {best_id})."
            )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(best_files)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()
    elif path.is_file():
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    else:
        raise FileNotFoundError(path)

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume, got {image.GetDimension()}D")

    image = sitk.DICOMOrient(image, "LPS")
    warnings.extend(validate_volume(image))
    return image, warnings


read_ct = load_volume


def validate_volume(image: sitk.Image) -> list[str]:
    """Return human-readable warnings about suitability for airway planning."""
    warnings: list[str] = []
    sx, sy, sz = image.GetSpacing()
    nx, ny, nz = image.GetSize()

    if sz > 2.5:
        warnings.append(
            f"Slice spacing {sz:.2f} mm is thick (> 2.5 mm); distal airways will not be resolved."
        )
    elif sz > 1.25:
        warnings.append(f"Slice spacing {sz:.2f} mm is above the preferred <= 1.25 mm.")
    if max(sx, sy) > 1.0:
        warnings.append(f"In-plane pixel size {max(sx, sy):.2f} mm is coarse (> 1.0 mm).")
    if nz * sz < 150:
        warnings.append(f"Z coverage {nz * sz:.0f} mm looks too short for a full chest.")

    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    lo, hi = stats.GetMinimum(), stats.GetMaximum()
    if lo > -500:
        warnings.append(
            f"Minimum value {lo:.0f} is above -500: this does not look like an HU chest CT with air."
        )
    if hi < 200:
        warnings.append(f"Maximum value {hi:.0f} is low for a CT in HU (no bone/contrast).")
    return warnings


def resample_isotropic(image: sitk.Image, spacing_mm: float = 0.75) -> sitk.Image:
    """Resample to isotropic voxels with linear interpolation.

    The new grid stays within the original voxel-centre extent. Rounding the size up would add a row of
    voxels past the scan edge filled with the default value (air), and an airway that opens at that edge
    (the trachea at the top of a chest CT) would then connect to the air outside the body.
    """
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(np.floor((n - 1) * s / spacing_mm + 1e-6)) + 1) for n, s in zip(old_size, old_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((spacing_mm,) * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(AIR_HU)
    return resampler.Execute(image)
