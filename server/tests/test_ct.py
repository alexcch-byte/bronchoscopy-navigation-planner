"""Tests for the CT viewer endpoints (open, slice, HU) of the local test console."""

import importlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("multipart")
pytest.importorskip("PIL")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

PASSCODE = "correct-horse-1"
PROXY = {"Tailscale-Funnel-Request": "?1"}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("AEROPATH_ROOT", str(tmp_path / "no-aeropath"))
    monkeypatch.setenv("DEMO_PASSCODE", PASSCODE)
    monkeypatch.setenv("DEMO_INSECURE_COOKIE", "1")
    import app.main as main

    main = importlib.reload(main)
    return TestClient(main.app), tmp_path


@pytest.fixture()
def phantom_ct(api):
    client, tmp = api
    r = client.post("/api/ct/open", json={"source": "phantom"})
    assert r.status_code == 200, r.text
    return client, r.json(), tmp


def _png(resp):
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    return Image.open(io.BytesIO(resp.content))


def test_open_the_phantom_and_reuse_the_cached_volume(phantom_ct):
    client, d, _ = phantom_ct
    assert d["size"] == [200, 200, 200] and d["spacing"] == [0.7, 0.7, 1.0]
    assert d["hint"]["label"] == "the phantom's nodule" and len(d["hint"]["target_mm"]) == 3
    assert d["hu_range"][0] < -900 and d["hu_range"][1] > 400
    again = client.post("/api/ct/open", json={"source": "phantom"}).json()
    assert again["ct_id"] == d["ct_id"]  # not loaded twice


def test_slices_have_the_right_shapes_for_each_plane(phantom_ct):
    client, d, _ = phantom_ct
    cid = d["ct_id"]
    for plane, size in (("axial", (200, 200)), ("coronal", (200, 200)), ("sagittal", (200, 200))):  # (width, height)
        img = _png(client.get(f"/api/ct/{cid}/slice", params={"plane": plane, "i": 100}))
        assert img.size == size and img.mode == "L", plane


def test_window_maps_hu_to_gray_and_the_head_is_at_the_top_of_coronal_slices(phantom_ct):
    client, d, _ = phantom_ct
    cid = d["ct_id"]
    # axial slice through the lungs: air inside the body is dark, soft tissue bright, in a lung window
    ax = np.array(_png(client.get(f"/api/ct/{cid}/slice", params={"plane": "axial", "i": 120, "wc": -600, "ww": 1500})))
    # lung window -1350..150 HU: air (-1000) -> gray 59, lung (-850) -> 85, soft tissue (40) -> 236
    assert 50 < ax.min() < 70 and ax.max() > 200
    # narrower window: the same slice is more contrasty
    soft = np.array(_png(client.get(f"/api/ct/{cid}/slice", params={"plane": "axial", "i": 120, "wc": 40, "ww": 400})))
    assert soft.std() != ax.std()
    # The phantom's trachea opens at the HIGH z (head) end. In a coronal slice through its axis (y = 62 mm -> index
    # 88) the top rows must contain the dark airway column and the bottom rows must not.
    co = np.array(_png(client.get(f"/api/ct/{cid}/slice", params={"plane": "coronal", "i": 88, "wc": -600, "ww": 1500})))
    col = co[:, 100]  # x = 70 mm -> index 100: the trachea's axis
    assert col[:40].mean() < col[-40:].mean() - 50  # dark airway at the top, tissue/lung at the bottom
    assert client.get(f"/api/ct/{cid}/slice", params={"plane": "axial", "i": 120}).headers["cache-control"].startswith("private")


def test_voxel_physical_mapping_is_consistent_with_the_known_nodule(phantom_ct):
    """origin + direction * (index * spacing) must land on the phantom's nodule where the HU is soft tissue."""
    client, d, _ = phantom_ct
    o, s, D = np.array(d["origin"]), np.array(d["spacing"]), np.array(d["direction"]).reshape(3, 3)
    p = np.array(d["hint"]["target_mm"])
    idx = np.rint((D.T @ (p - o)) / s).astype(int)  # (i, j, k)
    assert np.all(idx >= 0) and np.all(idx < np.array(d["size"]))
    hu = client.get(f"/api/ct/{d['ct_id']}/hu", params={"i": idx[0], "j": idx[1], "k": idx[2]}).json()["hu"]
    assert -100 < hu < 150  # the nodule (about 30 HU), not lung (-850) or air
    # and one nodule-radius-plus-margin away it is lung
    far = idx + np.array([0, 0, 12])
    hu_far = client.get(f"/api/ct/{d['ct_id']}/hu", params={"i": far[0], "j": far[1], "k": far[2]}).json()["hu"]
    assert hu_far < -500


def test_bad_requests_are_rejected_cleanly(phantom_ct):
    client, d, _ = phantom_ct
    cid = d["ct_id"]
    assert client.get(f"/api/ct/{cid}/slice", params={"plane": "axial", "i": 200}).status_code == 422
    assert client.get(f"/api/ct/{cid}/slice", params={"plane": "axial", "i": -1}).status_code == 422
    assert client.get(f"/api/ct/{cid}/slice", params={"plane": "oblique", "i": 5}).status_code == 422
    assert client.get(f"/api/ct/{cid}/hu", params={"i": 200, "j": 0, "k": 0}).status_code == 422
    assert client.get("/api/ct/deadbeef0000/slice", params={"plane": "axial", "i": 0}).status_code == 404
    assert client.post("/api/ct/open", json={"source": "aeropath", "case": 3}).status_code == 404  # not on this machine
    assert client.post("/api/ct/open", json={"source": "upload", "upload_id": "0" * 12}).status_code == 404
    assert client.post("/api/ct/open", json={"source": "bogus"}).status_code == 422


def test_open_an_uploaded_volume_and_an_image_series(api):
    client, tmp = api
    arr = np.full((20, 30, 40), -800, dtype=np.int16)
    arr[5, 10, 15] = 123
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.8, 0.8, 2.0))
    vol = tmp / "scan.mha"
    sitk.WriteImage(img, str(vol))
    up = client.post("/api/uploads", files=[("files", ("scan.mha", vol.read_bytes()))]).json()
    d = client.post("/api/ct/open", json={"source": "upload", "upload_id": up["upload_id"]}).json()
    assert d["size"] == [40, 30, 20] and d["spacing"] == [0.8, 0.8, 2.0]
    assert client.get(f"/api/ct/{d['ct_id']}/hu", params={"i": 15, "j": 10, "k": 5}).json()["hu"] == 123

    import imageio.v2 as imageio

    files = []
    for k in range(5):
        b = io.BytesIO()
        imageio.imwrite(b, np.full((24, 32), 30 + k, np.uint8), format="png")
        files.append(("files", (f"IM{k + 1:03d}.png", b.getvalue())))
    ups = client.post("/api/uploads", files=files).json()
    assert client.post("/api/ct/open", json={"source": "upload", "upload_id": ups["upload_id"]}).status_code == 422  # needs spacing
    stack = {"pixel_spacing_mm": 0.7, "slice_spacing_mm": 1.5, "window_center": -600, "window_width": 1500, "first_slice": "feet"}
    ds = client.post("/api/ct/open", json={"source": "upload", "upload_id": ups["upload_id"], "stack": stack})
    assert ds.status_code == 200, ds.text
    assert ds.json()["size"] == [32, 24, 5] and ds.json()["spacing"] == [0.7, 0.7, 1.5]



def test_public_visitors_get_the_phantom_ct_viewer_and_no_other_ct(phantom_ct):
    client, d, tmp = phantom_ct
    # a non-phantom CT held in the cache (opened locally from an uploaded file)
    arr = np.full((10, 12, 14), -500, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    vol = tmp / "other.mha"
    sitk.WriteImage(img, str(vol))
    up = client.post("/api/uploads", files=[("files", ("other.mha", vol.read_bytes()))]).json()["upload_id"]
    other = client.post("/api/ct/open", json={"source": "upload", "upload_id": up}).json()["ct_id"]

    assert client.get(f"/api/ct/{d['ct_id']}/slice", params={"plane": "axial", "i": 1}, headers=PROXY).status_code == 401
    assert client.post("/api/login", json={"passcode": PASSCODE}, headers=PROXY).status_code == 200
    pub = client.post("/api/ct/open", json={"source": "phantom"}, headers=PROXY)
    assert pub.status_code == 200 and pub.json()["ct_id"] == d["ct_id"]  # the shared cached phantom
    assert client.get(f"/api/ct/{d['ct_id']}/slice", params={"plane": "axial", "i": 1}, headers=PROXY).status_code == 200
    assert client.get(f"/api/ct/{d['ct_id']}/hu", params={"i": 1, "j": 1, "k": 1}, headers=PROXY).status_code == 200
    # nothing else: other sources are refused, and another CT's id is not usable even if guessed
    assert client.post("/api/ct/open", json={"source": "aeropath", "case": 3}, headers=PROXY).status_code == 403
    assert client.post("/api/ct/open", json={"source": "upload", "upload_id": up}, headers=PROXY).status_code == 403
    assert client.get(f"/api/ct/{other}/slice", params={"plane": "axial", "i": 1}, headers=PROXY).status_code == 404
    assert client.get(f"/api/ct/{other}/hu", params={"i": 1, "j": 1, "k": 1}, headers=PROXY).status_code == 404
    assert client.get(f"/api/ct/{other}/slice", params={"plane": "axial", "i": 1}).status_code == 200  # local still can


def test_open_phantom_with_custom_nodule_and_validation(api):
    client, _ = api
    nod = {"center": [-34.0, 0.0, 10.0], "radius_mm": 7.0, "hu": -300.0}
    r = client.post("/api/ct/open", json={"source": "phantom", "nodule": nod})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["hint"]["label"] == "the custom nodule"
    assert d["hint"]["target_mm"] == [-34.0, 0.0, 10.0]
    assert d["hint"]["radius_mm"] == 7.0

    # Cached on identical nodule spec
    r2 = client.post("/api/ct/open", json={"source": "phantom", "nodule": nod})
    assert r2.status_code == 200
    assert r2.json()["ct_id"] == d["ct_id"]

    # Invalid placements: outside lung or on airway -> 422
    assert client.post("/api/ct/open", json={"source": "phantom", "nodule": {"center": [0.0, 0.0, 0.0], "radius_mm": 5.0, "hu": 30.0}}).status_code == 422
    assert client.post("/api/ct/open", json={"source": "phantom", "nodule": {"center": [0.0, -8.0, 50.0], "radius_mm": 4.0, "hu": 30.0}}).status_code == 422

    # Public visitor with custom nodule
    assert client.post("/api/login", json={"passcode": PASSCODE}, headers=PROXY).status_code == 200
    r_pub = client.post("/api/ct/open", json={"source": "phantom", "nodule": nod}, headers=PROXY)
    assert r_pub.status_code == 200
    assert r_pub.json()["ct_id"] == d["ct_id"]

    # Public visitor with out-of-bounds nodule -> 422
    out_bounds = {"center": [120.0, 0.0, 0.0], "radius_mm": 5.0, "hu": 30.0}
    assert client.post("/api/ct/open", json={"source": "phantom", "nodule": out_bounds}, headers=PROXY).status_code == 422



