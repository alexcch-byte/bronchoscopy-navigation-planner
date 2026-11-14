"""DICOM series import (bronchnav.io.dicom)."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from bronchnav.io import dicom


def write_series(folder: Path, n=40, spacing=(0.7, 0.7, 1.0), modality="CT", uid="1.2.826.0.1.3680043.8.498.1",
                 skip=(), name_ext=".dcm", patient="DOE^JANE", pid="MRN-99887", rng=0, tilt=None, stem="IM"):
    """Write a synthetic CT as one DICOM file per slice (with identifying tags) and return the volume (HU)."""
    folder.mkdir(parents=True, exist_ok=True)
    r = np.random.default_rng(rng)
    arr = r.integers(-900, 200, size=(n, 24, 24)).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((-10.0, -12.0, 5.0))
    d = img.GetDirection()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    w.SetImageIO("GDCMImageIO")  # extensionless names give no hint, as with files from a scanner
    for i in range(n):
        if i in skip:
            continue
        s = img[:, :, i]
        tags = {
            "0008|0060": modality, "0020|000e": uid, "0010|0010": patient, "0010|0020": pid, "0008|0080": "General Hospital",
            "0008|0020": "20200101", "0020|0013": str(i + 1), "0028|1052": "0", "0028|1053": "1",
            "0020|0032": "\\".join(f"{v:.6f}" for v in img.TransformIndexToPhysicalPoint((0, 0, i))),
            "0020|0037": "\\".join(str(v) for v in (d[0], d[3], d[6], d[1], d[4], d[7])),
            "0008|0018": f"{uid}.{i + 1}",
        }
        if tilt is not None:
            tags["0018|1120"] = str(tilt)
        for k, v in tags.items():
            s.SetMetaData(k, v)
        w.SetFileName(str(folder / f"{stem}{i:04d}{name_ext}"))
        w.Execute(s)
    return arr


def test_finds_files_with_or_without_extension(tmp_path):
    write_series(tmp_path / "a", n=5, name_ext="")           # extensionless, as most scanners write them
    write_series(tmp_path / "b", n=3, name_ext=".dcm")
    (tmp_path / "a" / "notes.txt").write_text("hello")
    found = dicom.find_dicom_files(tmp_path)
    assert len(found) == 8 and all(p.name != "notes.txt" for p in found)


def test_converts_a_series_to_a_clean_volume(tmp_path):
    arr = write_series(tmp_path / "s", n=40, name_ext="")
    out = tmp_path / "ct.mha"
    info = dicom.convert_series(tmp_path / "s", out)
    assert info["n_slices"] == 40 and info["spacing_mm"] == [0.7, 0.7, 1.0]
    img = sitk.ReadImage(str(out))
    assert img.GetSize() == (24, 24, 40) and img.GetPixelID() == sitk.sitkInt16
    # LPS, HU preserved (identity direction here): same voxels back, in slice order
    assert (sitk.GetArrayFromImage(img) == arr).all()
    assert np.allclose(img.GetOrigin(), (-10.0, -12.0, 5.0), atol=1e-3)
    # nothing identifying is carried over: no DICOM tag keys, and none of the tag values anywhere in the file
    import re

    assert not [k for k in img.GetMetaDataKeys() if re.fullmatch(r"[0-9a-f]{4}\|[0-9a-f]{4}", k)]   # no DICOM tags (the reader adds a few of its own keys)
    raw = out.read_bytes()
    for needle in (b"DOE", b"MRN-99887", b"General Hospital", b"1.2.826.0.1.3680043"):
        assert needle not in raw


def test_uses_the_largest_series_and_reports_the_others(tmp_path):
    write_series(tmp_path / "all", n=50, uid="1.2.3.10")
    write_series(tmp_path / "all", n=35, uid="1.2.3.20", stem="B")
    info = dicom.convert_series(tmp_path / "all", tmp_path / "o.mha")
    assert info["n_slices"] == 50 and info["other_series"] == [35]
    assert any("2 series" in w for w in info["warnings"])


@pytest.mark.parametrize("kwargs, text", [
    ({"modality": "MR"}, "not CT"),
    ({"n": 10}, "only 10 slices"),
    ({"skip": (10, 11, 12)}, "not uniform"),
])
def test_refuses_what_cannot_be_planned_on(tmp_path, kwargs, text):
    write_series(tmp_path / "s", **kwargs)
    with pytest.raises(ValueError, match=text):
        dicom.convert_series(tmp_path / "s", tmp_path / "o.mha")
    assert not (tmp_path / "o.mha").exists()


def test_empty_folder_and_tilt_warning(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No DICOM series"):
        dicom.convert_series(tmp_path / "empty", tmp_path / "o.mha")
    write_series(tmp_path / "t", tilt=15)
    info = dicom.convert_series(tmp_path / "t", tmp_path / "o.mha")
    assert any("gantry tilt" in w for w in info["warnings"])
