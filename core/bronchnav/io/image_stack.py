"""Import a CT given only as a series of ordinary images (JPEG, PNG, BMP, TIFF), for when no DICOM is available.

Such images carry no Hounsfield units and no voxel spacing. This importer rebuilds an APPROXIMATE volume
from three things the user must supply:

  - pixel spacing (mm) and slice spacing (mm), e.g. from the scan report or the viewer that made the images;
  - the display window the images were saved with (default: a typical lung window, centre -600, width 1500),
    to map gray levels back to HU: HU = centre - width/2 + (gray / max) * width;
  - which end of the body the first image is (or let it be detected from where the trachea is).

Anything outside the window was clipped when the images were made and is gone. A lung window keeps air and
lung tissue (about -1350 to +150 HU) and loses bone and contrast. The result is good enough to try the
airway pipeline, not to trust: segmentation quality on such input has not been validated. Not for clinical use.
Image orientation is assumed radiological (patient's left on the right of the picture, anterior at the top),
which maps straight onto LPS.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ARCHIVE_EXTS = (".zip",)
MAX_SLICES = 1500

SCAN_PRESETS: dict[str, dict] = {
    "standard": {"pixel_spacing_mm": 0.75, "slice_spacing_mm": 1.0, "label": "Standard Chest CT (0.75 x 1.0 mm)"},
    "hrct": {"pixel_spacing_mm": 0.50, "slice_spacing_mm": 0.75, "label": "High-Res CT / HRCT (0.50 x 0.75 mm)"},
    "lowdose": {"pixel_spacing_mm": 0.80, "slice_spacing_mm": 1.25, "label": "Low-Dose Chest CT (0.80 x 1.25 mm)"},
    "thick": {"pixel_spacing_mm": 1.00, "slice_spacing_mm": 2.50, "label": "Thick-Slice CT (1.00 x 2.50 mm)"},
}

WINDOW_PRESETS: dict[str, dict] = {
    "lung": {"window_center": -600.0, "window_width": 1500.0, "label": "Lung Window (-600 / 1500 HU)"},
    "mediastinum": {"window_center": 40.0, "window_width": 400.0, "label": "Soft Tissue / Mediastinum (40 / 400 HU)"},
    "bone": {"window_center": 300.0, "window_width": 1500.0, "label": "Bone Window (300 / 1500 HU)"},
    "wide": {"window_center": 0.0, "window_width": 2000.0, "label": "Wide Full Range (0 / 2000 HU)"},
}


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]


def list_images(source: str | Path | list) -> list[Path]:
    """Image files of a folder (or its subfolder) or an explicit list, in natural filename order (img2 < img10)."""
    if isinstance(source, (list, tuple)):
        files = [Path(f) for f in source]
    else:
        src = Path(source)
        if src.is_file() and src.suffix.lower() == ".zip":
            import tempfile
            import zipfile

            cache_dir = Path(tempfile.gettempdir()) / "bronchnav_zip" / src.stem
            cache_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src, "r") as zf:
                for member in zf.infolist():
                    target = (cache_dir / member.filename).resolve()
                    if not str(target).startswith(str(cache_dir.resolve())):
                        raise ValueError(f"unsafe path in zip: {member.filename}")
                zf.extractall(cache_dir)
            src = cache_dir

        if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
            return [src]

        if not src.is_dir():
            raise ValueError(f"{src} is not a folder of images")
        files = [p for p in src.iterdir() if p.is_file()]
        if not any(f.suffix.lower() in IMAGE_EXTS for f in files):
            files = [p for p in src.rglob("*") if p.is_file()]
    files = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise ValueError(f"no image files ({', '.join(IMAGE_EXTS)}) found")
    return sorted(files, key=_natural_key)


def _to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:  # RGB(A) screenshots: luminance
        return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    if arr.ndim == 3 and arr.shape[2] in (1, 2):
        return arr[..., 0]
    raise ValueError(f"unsupported image shape {arr.shape}")


def _scale(arr: np.ndarray) -> float:
    if arr.dtype == np.uint8:
        return 255.0
    if arr.dtype == np.uint16:
        return 65535.0
    return float(max(arr.max(), 1.0))


def _detect_head_first(img_feet_first: sitk.Image) -> tuple[bool | None, str]:
    """Return (flip, note). `flip` True when the first image is the head end. None when undetermined."""
    from ..segmentation import find_trachea_seed, segment_airway
    from .volume import resample_isotropic

    coarse = resample_isotropic(img_feet_first, 1.5)
    hu = sitk.GetArrayFromImage(coarse)
    vol_ml: dict[str, float] = {}
    for name, arr in (("feet_first", hu), ("head_first", hu[::-1].copy())):
        try:
            seed = find_trachea_seed(arr, 1.5, search_mm=80.0, prefer="high", allow_fallback=False)
            seg = segment_airway(arr, 1.5, seed_zyx=seed)
            vol_ml[name] = float(seg.mask.sum()) * 1.5**3 / 1000.0
        except ValueError:
            continue  # no seed at that end, or it leaks straight into the lung: not the trachea
    # A trachea-like blob can also be found at the wrong end (a gas pocket in the stomach or colon), but only
    # the real trachea grows into a large connected airway tree. Take the end whose seed grows the larger one.
    plausible = {k: v for k, v in vol_ml.items() if v >= 8.0}
    if not plausible:
        return None, "could not tell which end the first image is (no trachea-like airway found at either end)"
    if len(plausible) == 2 and max(plausible.values()) < 1.5 * min(plausible.values()):
        return None, "could not tell which end the first image is (both ends grow a similar airway)"
    best = max(plausible, key=plausible.get)
    if best == "feet_first":
        return False, f"detected: first image is the feet end (airway of {plausible[best]:.0f} mL grows from the last images)"
    return True, f"detected: first image is the head end (airway of {plausible[best]:.0f} mL grows from the first images)"


def load_image_stack(
    source: str | Path | list,
    pixel_spacing_mm: float,
    slice_spacing_mm: float,
    window_center: float = -600.0,
    window_width: float = 1500.0,
    first_slice: str = "auto",
    invert: bool = False,
) -> tuple[sitk.Image, list[str]]:
    """Build an approximate HU volume (SimpleITK image, LPS, float32) from a series of images.

    `first_slice` is "head", "feet" or "auto" (detect from where the trachea is; falls back to "feet").
    Returns (image, warnings).
    """
    if pixel_spacing_mm <= 0 or slice_spacing_mm <= 0:
        raise ValueError("pixel spacing and slice spacing must be positive (mm)")
    if window_width <= 0:
        raise ValueError("window width must be positive")
    if first_slice not in ("auto", "head", "feet"):
        raise ValueError("first_slice must be 'auto', 'head' or 'feet'")
    files = list_images(source)
    if len(files) > MAX_SLICES:
        raise ValueError(f"{len(files)} images; the limit is {MAX_SLICES}")
    warnings: list[str] = []

    # Case 1: single 3D multi-page TIFF file
    if len(files) == 1 and files[0].suffix.lower() in (".tif", ".tiff"):
        img = sitk.ReadImage(str(files[0]))
        if img.GetDimension() == 3 and img.GetSize()[2] >= 3:
            arr = sitk.GetArrayFromImage(img)
            scale = _scale(arr)
            if arr.ndim == 3:
                g = arr.astype(np.float32) / scale
            elif arr.ndim == 4 and arr.shape[3] >= 3:
                g = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114).astype(np.float32) / scale
            elif arr.ndim == 4 and arr.shape[3] in (1, 2):
                g = arr[..., 0].astype(np.float32) / scale
            else:
                raise ValueError(f"unsupported multi-slice TIFF shape {arr.shape}")
            frac = np.clip(1.0 - g if invert else g, 0.0, 1.0)
            hu = (window_center - window_width / 2.0 + frac * window_width).astype(np.float32)
            out_img = sitk.GetImageFromArray(hu)
            out_img.SetSpacing((float(pixel_spacing_mm), float(pixel_spacing_mm), float(slice_spacing_mm)))
            if first_slice == "auto":
                flip, note = _detect_head_first(out_img)
                warnings.append(f"Slice order {note}.")
                if flip is None:
                    flip = False
                    warnings.append("Assuming the first image is the feet end. Set it explicitly if the route looks upside down.")
            else:
                flip = first_slice == "head"
            if flip:
                hu = np.ascontiguousarray(hu[::-1])
                out_img = sitk.GetImageFromArray(hu)
                out_img.SetSpacing((float(pixel_spacing_mm), float(pixel_spacing_mm), float(slice_spacing_mm)))
            warnings.insert(
                0,
                f"Imported from multi-slice TIFF ({out_img.GetSize()[2]} slices): reconstructed HU using window centre "
                f"{window_center:g}, width {window_width:g} and spacing {pixel_spacing_mm:g} x {slice_spacing_mm:g} mm.",
            )
            return out_img, warnings

    # Case 2: series of slice images (JPEG, PNG, BMP, TIFF)
    # First pass: determine majority slice dimension and filter out scout images/reports
    shape_counts: dict[tuple[int, int], list[Path]] = {}
    for f in files:
        rd = sitk.ImageFileReader()
        rd.SetFileName(str(f))
        try:
            rd.ReadImageInformation()
            sz = rd.GetSize()
            shape_counts.setdefault((sz[1], sz[0]), []).append(f)
        except Exception:
            continue

    if not shape_counts:
        raise ValueError("could not read any images in the series")

    majority_shape = max(shape_counts, key=lambda s: len(shape_counts[s]))
    valid_files = shape_counts[majority_shape]
    skipped_count = len(files) - len(valid_files)
    if skipped_count > 0:
        skipped_names = [f.name for f in files if f not in valid_files][:3]
        ellipsis_str = "..." if skipped_count > 3 else ""
        warnings.append(
            f"Filtered out {skipped_count} non-matching image(s) ({', '.join(skipped_names)}{ellipsis_str}) "
            f"that did not match the majority slice resolution of {majority_shape[1]}x{majority_shape[0]} pixels."
        )

    if len(valid_files) < 3:
        if skipped_count > 0:
            raise ValueError(
                f"Mismatched image resolutions: expected {majority_shape[1]}x{majority_shape[0]} pixels, "
                f"found only {len(valid_files)} matching images"
            )
        raise ValueError(f"Found only {len(valid_files)} matching slice images; need at least 3")

    files = valid_files

    slices = []
    for f in files:
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)  # 2D: (h, w); RGB: (h, w, c)
        scale = _scale(arr)
        g = _to_gray(arr.astype(np.float32)) / scale
        slices.append(1.0 - g if invert else g)
    frac = np.clip(np.stack(slices, axis=0), 0.0, 1.0)
    hu = (window_center - window_width / 2.0 + frac * window_width).astype(np.float32)

    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((float(pixel_spacing_mm), float(pixel_spacing_mm), float(slice_spacing_mm)))
    # file order = z index. LPS needs z increasing toward the head.
    if first_slice == "auto":
        flip, note = _detect_head_first(img)
        warnings.append(f"Slice order {note}.")
        if flip is None:
            flip = False
            warnings.append("Assuming the first image is the feet end. Set it explicitly if the route looks upside down.")
    else:
        flip = first_slice == "head"
    if flip:
        hu = np.ascontiguousarray(hu[::-1])
        img = sitk.GetImageFromArray(hu)
        img.SetSpacing((float(pixel_spacing_mm), float(pixel_spacing_mm), float(slice_spacing_mm)))

    warnings.insert(
        0,
        f"Imported from {len(files)} ordinary images, not DICOM: HU values are reconstructed from a display window "
        f"(centre {window_center:g}, width {window_width:g}) and spacing is user-supplied. Anything outside the window "
        "was clipped when the images were made. Results are approximate and unvalidated. Not for clinical use.",
    )
    if len(files) < 100:
        warnings.append(f"Only {len(files)} images: too few for a chest CT.")
    z_extent = len(files) * slice_spacing_mm
    if z_extent < 150:
        warnings.append(f"Z coverage {z_extent:.0f} mm looks too short for a full chest.")
    if slice_spacing_mm > 2.5:
        warnings.append(f"Slice spacing {slice_spacing_mm:.2f} mm is thick (> 2.5 mm); distal airways will not be resolved.")
    return img, warnings
