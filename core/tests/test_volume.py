import numpy as np
import pytest
import SimpleITK as sitk

from bronchnav.cli import main
from bronchnav.io import load_volume, read_ct, resample_isotropic, validate_volume


def _synthetic_ct(spacing=(0.7, 0.7, 2.5), shape_zyx=(40, 32, 32)) -> sitk.Image:
    arr = np.full(shape_zyx, -800, dtype=np.int16)  # lung-ish
    arr[:, 12:20, 12:20] = -1000  # an "airway"
    arr[:, :4, :4] = 400  # something dense
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def test_resample_isotropic_changes_spacing_and_size():
    img = _synthetic_ct()
    out = resample_isotropic(img, 0.75)
    assert np.allclose(out.GetSpacing(), (0.75, 0.75, 0.75))
    # the new grid never extends past the original voxel-centre extent (no padded rows at the scan edge)
    old_extent = (np.array(img.GetSize()) - 1) * np.array(img.GetSpacing())
    new_extent = (np.array(out.GetSize()) - 1) * 0.75
    assert np.all(new_extent <= old_extent + 1e-6) and np.all(new_extent > old_extent - 0.75)


def test_validate_flags_thick_slices_and_short_coverage():
    warnings = validate_volume(_synthetic_ct(spacing=(0.7, 0.7, 3.0)))
    text = " ".join(warnings)
    assert "thick" in text
    assert "too short" in text


def test_validate_flags_non_hu_data():
    arr = np.full((40, 32, 32), 100, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.7, 0.7))
    assert any("does not look like an HU" in w for w in validate_volume(img))


def _write_slices(tmp_path, n=40, h=64, w=64, ext=".jpg", order_padded=True, rgb=False):
    """A tiny 'chest': soft-tissue background, an air column; lung window (-600 / 1500) 8-bit images."""
    import imageio.v2 as imageio

    c, wd = -600.0, 1500.0
    hu = np.full((n, h, w), 40.0, dtype=np.float32)
    hu[:, 20:44, 20:44] = -850.0
    hu[:, 28:36, 28:36] = -1000.0
    gray = np.clip((hu - (c - wd / 2)) / wd, 0, 1) * 255
    for i in range(n):
        im = gray[i].astype(np.uint8)
        if rgb:
            im = np.stack([im, im, im], axis=-1)
        name = f"slice_{i:03d}{ext}" if order_padded else f"slice_{i}{ext}"
        imageio.imwrite(tmp_path / name, im)
    return hu


def test_image_stack_rebuilds_approximate_hu_in_natural_order(tmp_path):
    pytest.importorskip("imageio")
    from bronchnav.io import load_image_stack

    hu = _write_slices(tmp_path, ext=".png", order_padded=False)  # slice_0..slice_39: needs natural sort
    img, warns = load_image_stack(tmp_path, 0.7, 1.5, first_slice="feet")
    arr = sitk.GetArrayFromImage(img)
    assert arr.shape == hu.shape and img.GetSpacing() == (0.7, 0.7, 1.5)
    assert abs(arr[10, 32, 32] - (-1000)) < 12 and abs(arr[10, 5, 5] - 40) < 12  # lung window: ~6 HU per gray level
    assert any("not DICOM" in w and "Not for clinical use" in w for w in warns)
    assert any("too few" in w for w in warns)  # 40 images


def test_image_stack_head_first_flips_z_and_rgb_jpegs_load(tmp_path):
    pytest.importorskip("imageio")
    from bronchnav.io import load_image_stack

    _write_slices(tmp_path, n=30, ext=".jpg", rgb=True)
    # make slice 0 distinguishable
    import imageio.v2 as imageio

    marker = np.full((64, 64, 3), 255, dtype=np.uint8)
    imageio.imwrite(tmp_path / "slice_000.jpg", marker)
    feet = sitk.GetArrayFromImage(load_image_stack(tmp_path, 0.7, 1.0, first_slice="feet")[0])
    head = sitk.GetArrayFromImage(load_image_stack(tmp_path, 0.7, 1.0, first_slice="head")[0])
    assert feet[0].mean() > 100 and head[-1].mean() > 100  # the bright first image ends up at the bottom or top
    assert feet.shape == head.shape


def test_image_stack_rejects_bad_input(tmp_path):
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    from bronchnav.io import load_image_stack

    with pytest.raises(ValueError, match="no image files"):
        load_image_stack(tmp_path, 0.7, 1.0)
    imageio.imwrite(tmp_path / "a_1.png", np.zeros((10, 10), np.uint8))
    imageio.imwrite(tmp_path / "a_2.png", np.zeros((10, 12), np.uint8))
    with pytest.raises(ValueError, match="expected"):
        load_image_stack(tmp_path, 0.7, 1.0, first_slice="feet")
    with pytest.raises(ValueError, match="positive"):
        load_image_stack(tmp_path, 0.0, 1.0)


def test_load_volume_roundtrip_and_cli(tmp_path, capsys):
    path = tmp_path / "ct.mha"
    sitk.WriteImage(_synthetic_ct(), str(path))
    image, _ = load_volume(path)
    assert image.GetSize() == (32, 32, 40)

    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "spacing (mm)" in out


def test_load_volume_recursive_dicom(tmp_path):
    root = tmp_path / "dicom_root"
    nested_dir = root / "sub1" / "study_A" / "series_CT"
    nested_dir.mkdir(parents=True)
    other_dir = root / "sub2" / "scout"
    other_dir.mkdir(parents=True)

    series_uid = "1.2.826.0.1.3680043.2.1125.1.123456789"
    study_uid = "1.2.826.0.1.3680043.2.1125.1.987654321"
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # 4 slices in series_CT
    for i in range(4):
        img2d = sitk.GetImageFromArray(np.full((16, 16), -800, dtype=np.int16))
        img2d.SetSpacing((1.0, 1.0))
        img2d.SetMetaData("0020|000e", series_uid)
        img2d.SetMetaData("0020|000d", study_uid)
        img2d.SetMetaData("0020|0032", f"0\\0\\{i * 2.0}")
        img2d.SetMetaData("0020|0013", str(i + 1))
        writer.SetFileName(str(nested_dir / f"slice_{i:02d}.dcm"))
        writer.Execute(img2d)

    # 1 slice in scout
    scout_uid = "1.2.826.0.1.3680043.2.1125.1.111111111"
    img_scout = sitk.GetImageFromArray(np.full((16, 16), -800, dtype=np.int16))
    img_scout.SetSpacing((1.0, 1.0))
    img_scout.SetMetaData("0020|000e", scout_uid)
    img_scout.SetMetaData("0020|000d", study_uid)
    img_scout.SetMetaData("0020|0032", "0\\0\\0")
    img_scout.SetMetaData("0020|0013", "1")
    writer.SetFileName(str(other_dir / "scout_00.dcm"))
    writer.Execute(img_scout)

    # Loading from root should recursively locate and select the 4-slice series
    vol, warnings = load_volume(root)
    assert vol.GetSize() == (16, 16, 4)
    assert vol.GetPixelID() == sitk.sitkFloat32
    # Verify read_ct alias works identically
    vol2, _ = read_ct(root)
    assert vol2.GetSize() == (16, 16, 4)

    # An empty directory or one without DICOM files raises ValueError
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No DICOM series found"):
        load_volume(empty_dir)


def test_image_stack_filters_outlier_scout_image(tmp_path):
    pytest.importorskip("imageio")
    import imageio.v2 as imageio
    from bronchnav.io import load_image_stack

    # Write 5 normal slices 32x32
    for i in range(5):
        imageio.imwrite(tmp_path / f"slice_{i:02d}.jpg", np.full((32, 32), 128, dtype=np.uint8))
    # Write 1 scout image 32x48
    imageio.imwrite(tmp_path / "scout_view.jpg", np.full((48, 32), 200, dtype=np.uint8))

    img, warns = load_image_stack(tmp_path, pixel_spacing_mm=0.75, slice_spacing_mm=1.0, first_slice="feet")
    assert img.GetSize() == (32, 32, 5)
    assert any("Filtered out 1 non-matching" in w for w in warns)


def test_image_stack_zip_archive_loading(tmp_path):
    pytest.importorskip("imageio")
    import zipfile
    import imageio.v2 as imageio
    from bronchnav.io import load_image_stack

    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    for i in range(6):
        imageio.imwrite(slices_dir / f"axial_{i:02d}.png", np.full((32, 32), 100, dtype=np.uint8))

    zip_file = tmp_path / "patient_ct.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        for p in slices_dir.glob("*.png"):
            zf.write(p, arcname=p.name)

    img, warns = load_image_stack(zip_file, pixel_spacing_mm=0.75, slice_spacing_mm=1.0, first_slice="feet")
    assert img.GetSize() == (32, 32, 6)


def test_image_stack_multi_slice_tiff(tmp_path):
    from bronchnav.io import load_image_stack

    arr = np.full((8, 32, 32), 150, dtype=np.uint8)
    tif_path = tmp_path / "multi_slice.tif"
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(tif_path))

    img, warns = load_image_stack(tif_path, pixel_spacing_mm=0.8, slice_spacing_mm=1.2, first_slice="feet")
    assert img.GetSize() == (32, 32, 8)
    assert any("multi-slice TIFF" in w for w in warns)


def test_load_volume_on_image_folder(tmp_path):
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    img_dir = tmp_path / "jpg_ct"
    img_dir.mkdir()
    for i in range(5):
        imageio.imwrite(img_dir / f"img_{i:02d}.jpg", np.full((32, 32), 120, dtype=np.uint8))

    img, warns = load_volume(img_dir)
    assert img.GetSize() == (32, 32, 5)
    assert any("not DICOM" in w for w in warns)


def test_cli_plan_image_presets(monkeypatch, tmp_path):
    from bronchnav.cli import main

    captured = {}
    def fake_run_plan(ct, target, **kwargs):
        captured["ct"] = ct
        captured["target"] = target
        captured["kwargs"] = kwargs
        from unittest.mock import MagicMock
        res = MagicMock()
        res.route.metrics = {"length_mm": 50.0}
        res.outputs = {}
        return res

    monkeypatch.setattr("bronchnav.pipeline.run_plan", fake_run_plan)
    img_dir = tmp_path / "jpg_series"
    img_dir.mkdir()
    code = main([
        "plan",
        str(img_dir),
        "--target", "10", "20", "30",
        "--preset", "standard",
        "--window-preset", "lung",
        "--no-render",
    ])
    assert code == 0
    assert captured["kwargs"]["image_stack"]["pixel_spacing_mm"] == 0.75
    assert captured["kwargs"]["image_stack"]["slice_spacing_mm"] == 1.0
    assert captured["kwargs"]["image_stack"]["window_center"] == -600.0
    assert captured["kwargs"]["image_stack"]["window_width"] == 1500.0

