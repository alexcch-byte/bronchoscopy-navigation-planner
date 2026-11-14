"""DICOM import: find a CT series in a folder, check it, and store it as a plain de-identified volume.

Real chest CTs arrive as a folder of one file per slice (often with no file extension). `convert_series` picks
the CT series with the most slices, refuses what cannot be planned on (not CT, a localizer, overlapping or
missing slices), and writes an LPS volume in HU as 16-bit integers. The output is built from the voxels and the
geometry only: no DICOM tag (patient name, ID, dates, UIDs, institution, ...) is carried over, so once the
original files are deleted nothing identifying remains on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DICOM_EXTS = (".dcm", ".dicom", ".ima")
MIN_SLICES = 30  # a chest CT has hundreds; this only rejects scouts and single-slice series
MAX_SPACING_DEVIATION = 0.10  # slice gaps or mixed acquisitions show up as spacing that varies by more than this


@dataclass
class Series:
    folder: Path
    series_id: str
    files: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.files)


def is_dicom_file(path: str | Path) -> bool:
    """A DICOM file has a .dcm-like extension or the "DICM" marker after its 128-byte preamble."""
    p = Path(path)
    if p.suffix.lower() in DICOM_EXTS:
        return True
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def find_dicom_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and is_dicom_file(p))


def scan_series(root: str | Path) -> list[Series]:
    """Every series found under `root`, one entry per series id and folder (slices sorted by position)."""
    root = Path(root)
    folders = sorted({p.parent for p in find_dicom_files(root)})
    out: list[Series] = []
    for folder in folders:
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)):
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), sid)
            out.append(Series(folder, sid, tuple(files)))
    return out


def _tag(rd: sitk.ImageFileReader, key: str) -> str:
    return rd.GetMetaData(key).strip() if rd.HasMetaDataKey(key) else ""


def _header(path: str) -> dict:
    rd = sitk.ImageFileReader()
    rd.SetFileName(path)
    rd.LoadPrivateTagsOff()
    rd.ReadImageInformation()
    return {"modality": _tag(rd, "0008|0060").upper(), "image_type": _tag(rd, "0008|0008").upper()}


def pick_series(series: list[Series]) -> tuple[Series, list[int]]:
    """The CT series to use (the one with the most slices, localizers and other modalities excluded) and the slice
    counts of the others. Raises ValueError with a message fit to show the user."""
    if not series:
        raise ValueError("No DICOM series found. Send the folder (or a zip of it) with every slice of the chest CT.")
    usable, modalities = [], set()
    for s in series:
        h = _header(s.files[0])
        if h["modality"] and h["modality"] != "CT":
            modalities.add(h["modality"])
            continue
        if "LOCALIZER" in h["image_type"]:
            continue
        usable.append(s)
    if not usable:
        if modalities:
            raise ValueError(f"The series is {'/'.join(sorted(modalities))}, not CT. Airway planning needs a chest CT.")
        raise ValueError("Only localizer (scout) images were found. Send the axial chest CT series.")
    best = max(usable, key=lambda s: s.n)
    if best.n < MIN_SLICES:
        raise ValueError(f"The largest CT series has only {best.n} slices; a chest CT needs a few hundred.")
    return best, sorted((s.n for s in series if s is not best), reverse=True)


def _check_geometry(reader: sitk.ImageSeriesReader, image: sitk.Image) -> list[str]:
    """Reject slices that overlap or are unevenly spaced (missing slices); warn about gantry tilt."""
    n = image.GetDepth()
    warnings: list[str] = []
    normal = np.array(image.GetDirection()).reshape(3, 3)[:, 2]
    pos = []
    for i in range(n):
        if not reader.HasMetaDataKey(i, "0020|0032"):
            return warnings  # no per-slice positions to compare; the reader's own spacing stands
        pos.append(float(np.dot(normal, [float(v) for v in reader.GetMetaData(i, "0020|0032").split("\\")])))
    d = np.diff(np.array(pos))
    med = float(np.median(d))
    if med < 0.05:
        raise ValueError("The slices overlap (several acquisitions or phases in one series). Send a single series.")
    if float(np.max(np.abs(d - med))) / med > MAX_SPACING_DEVIATION:
        raise ValueError(
            f"The slice spacing is not uniform ({float(d.min()):.2f} to {float(d.max()):.2f} mm): "
            "slices are missing or the series mixes acquisitions. Re-export the complete series."
        )
    if reader.HasMetaDataKey(0, "0018|1120"):
        try:
            if abs(float(reader.GetMetaData(0, "0018|1120"))) > 0.5:
                warnings.append("The scan has gantry tilt, which is not corrected; geometry may be slightly sheared.")
        except ValueError:
            pass
    return warnings


def convert_series(root: str | Path, out_path: str | Path) -> dict:
    """Write the best CT series under `root` to `out_path` (.mha, LPS, HU as int16, no DICOM tags).

    Returns {"n_slices", "size_xyz", "spacing_mm", "other_series": [slice counts], "warnings": [...]}.
    Raises ValueError with a user-facing message when the series cannot be used."""
    best, others = pick_series(scan_series(root))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(best.files))
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    reader.SetOutputPixelType(sitk.sitkFloat32)  # rescale slope and intercept are applied: values are HU
    try:
        image = reader.Execute()
    except RuntimeError as e:
        raise ValueError(f"The DICOM series could not be read ({str(e).splitlines()[0][:120]}).") from None
    if image.GetDimension() != 3 or image.GetDepth() < MIN_SLICES:
        raise ValueError("The DICOM series is not a 3D volume.")
    warnings = _check_geometry(reader, image)
    image = sitk.DICOMOrient(image, "LPS")

    hu = np.clip(np.rint(sitk.GetArrayFromImage(image)), -32768, 32767).astype(np.int16)
    clean = sitk.GetImageFromArray(hu)  # a new image: voxels and geometry only, no metadata dictionary
    clean.CopyInformation(image)
    sitk.WriteImage(clean, str(out_path), useCompression=False)
    if others:
        warnings.append(
            f"The folder holds {len(others) + 1} series; the one with the most slices ({best.n}) was used "
            f"(the others have {', '.join(str(n) for n in others[:5])}{' ...' if len(others) > 5 else ''})."
        )
    return {
        "n_slices": int(clean.GetDepth()),
        "size_xyz": list(clean.GetSize()),
        "spacing_mm": [round(s, 3) for s in clean.GetSpacing()],
        "other_series": others,
        "warnings": warnings,
    }
