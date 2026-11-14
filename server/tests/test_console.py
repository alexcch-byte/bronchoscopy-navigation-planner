"""Tests for the local test console API (uploads, proxy guard, job validation).

Run from the repo: python -m pytest server/tests   (needs fastapi, httpx, python-multipart, bronchnav)
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("multipart")  # python-multipart, needed for file uploads
from fastapi.testclient import TestClient  # noqa: E402


PASSCODE = "correct-horse-1"
PROXY = {"Tailscale-Funnel-Request": "?1"}  # what a public Funnel request carries


@pytest.fixture()
def no_run(api, monkeypatch):
    """Public runs start for real unless the runner is stubbed: jobs then stay 'queued' until a test changes them."""
    _, main, _ = api
    monkeypatch.setattr(main, "_run_job", lambda job_id, cmd: None)
    return main


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("AEROPATH_ROOT", str(tmp_path / "no-aeropath"))
    monkeypatch.delenv("ALLOW_REMOTE_UPLOADS", raising=False)
    monkeypatch.setenv("DEMO_PASSCODE", PASSCODE)
    monkeypatch.setenv("DEMO_INSECURE_COOKIE", "1")  # the test client speaks http
    import app.main as main

    main = importlib.reload(main)
    return TestClient(main.app), main, tmp_path


def _volume_file(tmp_path, name="ct.mha"):
    img = sitk.GetImageFromArray(np.full((12, 16, 16), -800, dtype=np.int16))
    img.SetSpacing((0.7, 0.7, 1.0))
    p = tmp_path / name
    sitk.WriteImage(img, str(p))
    return p


def _png_bytes(shape=(16, 16)):
    import io

    import imageio.v2 as imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, np.zeros(shape, np.uint8), format="png")
    return buf.getvalue()


def test_health_and_cases(api):
    client, _, _ = api
    assert client.get("/api/health").json()["status"] == "ok"
    c = client.get("/api/cases").json()
    assert c["phantom"] is True and c["imports_allowed"] is True and ".nii.gz" in c["volume_formats"]


def test_imports_are_refused_through_a_proxy(api):
    """Through any proxy (Tailscale serve or Funnel) nothing can be imported or run, even after logging in."""
    client, _, tmp = api
    vol = _volume_file(tmp)
    for hdr in (PROXY, {"Tailscale-User-Login": "someone@example.com"}, {"X-Forwarded-For": "1.2.3.4"}):
        body = [("files", ("ct.mha", vol.read_bytes()))]
        assert client.post("/api/uploads", files=body, headers=hdr).status_code == 401  # not logged in
    assert client.post("/api/login", json={"passcode": PASSCODE}, headers=PROXY).status_code == 200
    assert client.get("/api/cases", headers=PROXY).json()["imports_allowed"] is False
    for hdr in (PROXY, {"Tailscale-User-Login": "someone@example.com"}, {"X-Forwarded-For": "1.2.3.4"}):
        body = [("files", ("ct.mha", vol.read_bytes()))]
        assert client.post("/api/uploads", files=body, headers=hdr).status_code == 403
        assert client.post("/api/uploads/local", json={"path": str(vol)}, headers=hdr).status_code == 403
        assert client.post("/api/jobs", json={"source": "upload", "upload_id": "0" * 12}, headers=hdr).status_code == 403
    assert not any((tmp / "uploads").iterdir())  # nothing was written


def test_upload_a_volume_then_delete_it(api):
    client, main, tmp = api
    vol = _volume_file(tmp)
    r = client.post("/api/uploads", files=[("files", ("my scan.mha", vol.read_bytes()))])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "volume" and d["info"]["size_xyz"] == [16, 16, 12] and d["info"]["spacing_mm"] == [0.7, 0.7, 1.0]
    assert "path" not in d  # the server-side path is not leaked to the page
    saved = list((tmp / "uploads" / d["upload_id"]).glob("*.mha"))
    assert len(saved) == 1
    assert client.delete(f"/api/uploads/{d['upload_id']}").status_code == 200
    assert not (tmp / "uploads" / d["upload_id"]).exists()
    assert client.delete(f"/api/uploads/{d['upload_id']}").status_code == 404


def test_upload_filenames_cannot_escape_the_upload_folder(api):
    client, _, tmp = api
    vol = _volume_file(tmp)
    r = client.post("/api/uploads", files=[("files", ("../../evil.mha", vol.read_bytes()))])
    assert r.status_code == 200
    up = tmp / "uploads" / r.json()["upload_id"]
    assert [p.name for p in up.glob("*.mha")] == ["evil.mha"]
    assert not (tmp / "evil.mha").exists() and not (tmp.parent / "evil.mha").exists()


def test_upload_rejects_bad_input_and_cleans_up(api):
    client, _, tmp = api
    vol = _volume_file(tmp)
    assert client.post("/api/uploads", files=[("files", ("run.exe", b"MZ"))]).status_code == 400
    mixed = [("files", ("ct.mha", vol.read_bytes()))] + [("files", (f"i{k}.png", _png_bytes())) for k in range(3)]
    assert client.post("/api/uploads", files=mixed).status_code == 400
    assert client.post("/api/uploads", files=[("files", (f"i{k}.png", _png_bytes())) for k in range(2)]).status_code == 400  # < 3
    assert client.post("/api/uploads", files=[("files", ("bad.mha", b"not an image"))]).status_code == 400
    assert not any((tmp / "uploads").iterdir())  # failed uploads leave nothing behind


def test_upload_an_image_series_and_job_validation(api):
    pytest.importorskip("imageio")
    client, _, tmp = api
    files = [("files", (f"IM{k:04d}.png", _png_bytes())) for k in range(1, 5)]
    r = client.post("/api/uploads", files=files)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "images" and d["info"]["n_images"] == 4 and d["info"]["size_px"] == [16, 16]
    # a series needs its spacing and window; an unknown upload id is refused
    assert client.post("/api/jobs", json={"source": "upload", "upload_id": d["upload_id"]}).status_code == 422
    assert client.post("/api/jobs", json={"source": "upload", "upload_id": "f" * 12, "stack": {"pixel_spacing_mm": 0.7, "slice_spacing_mm": 1.0}}).status_code == 404
    bad = {"source": "upload", "upload_id": d["upload_id"], "stack": {"pixel_spacing_mm": -1, "slice_spacing_mm": 1.0}}
    assert client.post("/api/jobs", json=bad).status_code == 422


def test_upload_zip_archive_with_images_and_volume(api):
    pytest.importorskip("imageio")
    import zipfile
    client, _, tmp = api

    # 1. Zip with image slices
    zip1 = tmp / "images.zip"
    with zipfile.ZipFile(zip1, "w") as zf:
        for k in range(1, 5):
            zf.writestr(f"slice_{k:03d}.png", _png_bytes())

    r1 = client.post("/api/uploads", files=[("files", ("images.zip", zip1.read_bytes()))])
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["kind"] == "images"
    assert d1["info"]["n_images"] == 4

    # 2. Zip with a 3D volume
    vol = _volume_file(tmp)
    zip2 = tmp / "volume.zip"
    with zipfile.ZipFile(zip2, "w") as zf:
        zf.write(vol, arcname="inside_zip.mha")

    r2 = client.post("/api/uploads", files=[("files", ("volume.zip", zip2.read_bytes()))])
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["kind"] == "volume"
    assert "size_xyz" in d2["info"]


def test_register_a_local_path(api):
    pytest.importorskip("imageio")
    client, _, tmp = api
    vol = _volume_file(tmp)
    r = client.post("/api/uploads/local", json={"path": str(vol)})
    assert r.status_code == 200 and r.json()["kind"] == "volume" and r.json()["copied"] is False
    up_id = r.json()["upload_id"]
    assert client.delete(f"/api/uploads/{up_id}").status_code == 200
    assert vol.exists()  # a registered local file is forgotten, never deleted
    assert client.post("/api/uploads/local", json={"path": str(tmp / "missing.mha")}).status_code == 404
    (tmp / "notes.txt").write_text("x")
    assert client.post("/api/uploads/local", json={"path": str(tmp / "notes.txt")}).status_code == 400
    folder = tmp / "slices"
    folder.mkdir()
    for k in range(4):
        (folder / f"s{k}.png").write_bytes(_png_bytes())
    r = client.post("/api/uploads/local", json={"path": str(folder)})
    assert r.status_code == 200 and r.json()["kind"] == "images" and r.json()["info"]["n_images"] == 4


def test_scans_are_not_reachable_over_http(api):
    client, _, tmp = api
    vol = _volume_file(tmp)
    up = client.post("/api/uploads", files=[("files", ("ct.mha", vol.read_bytes()))]).json()["upload_id"]
    for url in (f"/files/uploads/{up}/ct.mha", f"/files//uploads/{up}/ct.mha", f"/files/../uploads/{up}/ct.mha", f"/files/%2e%2e/uploads/{up}/ct.mha"):
        assert client.get(url).status_code == 404


# ---------------------------------------------------------------- public (proxied) access

def _login(client, code=PASSCODE, headers=PROXY):
    return client.post("/api/login", json={"passcode": code}, headers=headers)


def _make_job(tmp, job_id, source, name=None, ok=True, error=None):
    import json as _json

    d = tmp / "jobs" / job_id
    (d / "stills").mkdir(parents=True)
    result = {"ok": ok, "source": source, "case": 5 if source == "aeropath" else None, "name": name, "seconds": 1.0}
    if error:
        result.update(error=error, traceback="Traceback ... C:/Users/someone/secret")
    (d / "job_result.json").write_text(_json.dumps(result))
    (d / "job.log").write_text("loading C:/Users/someone/datasets/x.nii.gz")
    (d / "route.json").write_text(_json.dumps({"schema": "x", "input": "C:/Users/someone/datasets/x.nii.gz", "path": {"points_mm": []}}))
    (d / "airway_mesh.glb").write_bytes(b"glTF")
    (d / "flythrough.mp4").write_bytes(b"mp4")
    (d / "airway_mask.nii.gz").write_bytes(b"mask")
    (d / "stack.json").write_text("{}")
    (d / "stills" / "bifurcation_01.png").write_bytes(b"png")


def test_public_access_is_closed_when_no_passcode_is_configured(api, monkeypatch):
    client, _, tmp = api
    monkeypatch.delenv("DEMO_PASSCODE", raising=False)
    monkeypatch.setenv("DEMO_PASSCODE_FILE", str(tmp / "no-such-file.txt"))
    assert client.get("/api/jobs", headers=PROXY).status_code == 503
    assert client.get("/", headers=PROXY).status_code == 503
    assert client.post("/api/login", json={"passcode": "anything-at-all"}, headers=PROXY).status_code == 503
    assert client.get("/api/health", headers=PROXY).status_code == 200  # only the health check stays open
    assert client.get("/api/jobs").status_code == 200  # a direct local request is unaffected


def test_a_short_passcode_counts_as_not_configured(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setenv("DEMO_PASSCODE", "short")
    assert client.get("/api/jobs", headers=PROXY).status_code == 503



def test_public_visitors_must_log_in_and_then_get_a_phantom_only_view(api, no_run):
    client, _, tmp = api
    vol = _volume_file(tmp)
    assert client.get("/", headers=PROXY, follow_redirects=False).headers["location"] == "/login"
    assert client.get("/login", headers=PROXY).status_code == 200
    assert client.get("/api/jobs", headers=PROXY).status_code == 401
    assert client.get("/api/cases", headers=PROXY).status_code == 401
    assert client.post("/api/jobs", json={"source": "phantom"}, headers=PROXY).status_code == 401  # not signed in
    assert _login(client, "wrong-passcode").status_code == 401
    assert _login(client).status_code == 200
    assert client.get("/api/jobs", headers=PROXY).status_code == 200
    # it can run the phantom and the public AeroPath cases (404 here: this test machine has no AeroPath), nothing else
    assert client.post("/api/jobs", json={"source": "aeropath", "case": 3}, headers=PROXY).status_code == 404
    assert client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY).status_code == 200
    assert client.post("/api/jobs", json={"source": "upload", "upload_id": "0" * 12}, headers=PROXY).status_code == 403
    # no imports, deletes, or anything else
    assert client.post("/api/uploads", files=[("files", ("ct.mha", vol.read_bytes()))], headers=PROXY).status_code == 403
    assert client.post("/api/uploads/local", json={"path": str(vol)}, headers=PROXY).status_code == 403
    assert client.delete("/api/uploads/abcdef012345", headers=PROXY).status_code == 403
    for url in ("/openapi.json", "/docs", "/api/uploads", "/static/index.html"):
        assert client.get(url, headers=PROXY).status_code == 404
    c = client.get("/api/cases", headers=PROXY).json()
    assert c["mode"] == "public" and c["can_run"] is True and c["aeropath"] == [] and "jobs_dir" not in c and "aeropath_root" not in c
    assert client.post("/api/logout", headers=PROXY).status_code == 200
    assert client.get("/api/jobs", headers=PROXY).status_code == 401  # signed out


def test_login_is_rate_limited_and_cookies_cannot_be_forged_or_reused_after_a_passcode_change(api, monkeypatch):
    client, main, _ = api
    for _ in range(main.LOGIN_ATTEMPTS):
        assert _login(client, "nope-nope-nope", {**PROXY, "X-Forwarded-For": "9.9.9.9"}).status_code == 401
    assert _login(client, PASSCODE, {**PROXY, "X-Forwarded-For": "9.9.9.9"}).status_code == 429  # even the right one
    assert _login(client, PASSCODE, {**PROXY, "X-Forwarded-For": "8.8.8.8"}).status_code == 200  # another client is fine
    client.cookies.clear()
    client.cookies.set("bron_session", "9999999999.deadbeef")
    assert client.get("/api/jobs", headers=PROXY).status_code == 401  # forged
    client.cookies.clear()
    assert _login(client, PASSCODE, {**PROXY, "X-Forwarded-For": "7.7.7.7"}).status_code == 200
    assert client.get("/api/jobs", headers=PROXY).status_code == 200
    monkeypatch.setenv("DEMO_PASSCODE", "a-new-passcode-2")
    assert client.get("/api/jobs", headers=PROXY).status_code == 401  # changing the passcode ends old sessions



def test_public_visitors_see_only_phantom_runs_and_only_safe_files(api):
    client, main, tmp = api
    ph, aero, priv, bad = "20261110-090000-aaaaaa", "20261110-090003-dddddd", "20261110-090001-bbbbbb", "20261110-090002-cccccc"
    _make_job(tmp, ph, "phantom")
    _make_job(tmp, aero, "aeropath")
    _make_job(tmp, priv, "upload", name="secret scan.nii.gz")
    _make_job(tmp, bad, "phantom", ok=False, error="FileNotFoundError: C:/Users/someone/datasets/x")
    main._restore_finished_jobs()
    assert _login(client).status_code == 200

    listed = {j["id"] for j in client.get("/api/jobs", headers=PROXY).json()}
    assert ph in listed and bad in listed
    assert aero in listed and priv not in listed  # public data (phantom, AeroPath) yes; an imported scan never
    assert client.get(f"/api/jobs/{priv}", headers=PROXY).status_code == 404
    assert "log" not in client.get(f"/api/jobs/{aero}", headers=PROXY).json()
    one = client.get(f"/api/jobs/{ph}", headers=PROXY).json()
    assert "log" not in one
    err = client.get(f"/api/jobs/{bad}", headers=PROXY).json()
    assert err["result"]["error"] == "the run failed" and "traceback" not in err["result"]

    ok = ["route.json", "airway_mesh.glb", "flythrough.mp4", "stills/bifurcation_01.png"]
    for f in ok:
        assert client.get(f"/files/{ph}/{f}", headers=PROXY).status_code == 200, f
    for run in (ph, aero):
        for f in ("job.log", "job_result.json", "airway_mask.nii.gz", "stack.json", "owner.txt"):  # not on the allowlist
            assert client.get(f"/files/{run}/{f}", headers=PROXY).status_code == 404, (run, f)
    for f in ok:
        assert client.get(f"/files/{aero}/{f}", headers=PROXY).status_code == 200, f
    assert "input" not in client.get(f"/files/{aero}/route.json", headers=PROXY).json()
    for f in ok:  # an imported scan's outputs are not reachable even by exact name
        assert client.get(f"/files/{priv}/{f}", headers=PROXY).status_code == 404, (priv, f)
    assert client.get(f"/files/{ph}/../{priv}/route.json", headers=PROXY).status_code == 404
    assert client.get(f"/files/{ph}/%2e%2e/{priv}/route.json", headers=PROXY).status_code == 404
    assert client.get(f"/files/{ph}/stills/..%2f..%2fjob.log", headers=PROXY).status_code == 404
    route = client.get(f"/files/{ph}/route.json", headers=PROXY).json()
    assert "input" not in route  # the local path of the CT is stripped


def test_local_requests_still_see_everything(api):
    client, main, tmp = api
    pub, priv = "20261110-090000-aaaaaa", "20261110-090001-bbbbbb"
    _make_job(tmp, pub, "aeropath")
    _make_job(tmp, priv, "upload", name="my scan.nii.gz")
    main._restore_finished_jobs()
    assert {j["id"] for j in client.get("/api/jobs").json()} >= {pub, priv}
    assert "log" in client.get(f"/api/jobs/{priv}").json()
    assert "input" in client.get(f"/files/{priv}/route.json").json()
    assert client.get(f"/files/{pub}/job.log").status_code == 200
    assert client.get("/api/cases").json()["mode"] == "local"


def test_the_owner_is_told_why_public_access_is_closed_but_visitors_are_not(api, monkeypatch):
    client, _, tmp = api
    assert client.get("/api/cases").json()["public_access"] == "open"
    f = tmp / "code.txt"
    monkeypatch.delenv("DEMO_PASSCODE", raising=False)
    monkeypatch.setenv("DEMO_PASSCODE_FILE", str(f))
    assert "no passcode file" in client.get("/api/cases").json()["public_access"]
    f.write_text("short7\n")
    state = client.get("/api/cases").json()["public_access"]
    assert "6 characters" in state and "at least 8" in state and "short7" not in state  # never echoes the passcode
    f.write_text("\n")
    assert "empty" in client.get("/api/cases").json()["public_access"]
    f.write_text("long-enough-1\n")
    assert client.get("/api/cases").json()["public_access"] == "open"
    f.write_text("short7\n")
    r = client.get("/api/jobs", headers=PROXY)  # visitors only get the generic message
    assert r.status_code == 503 and r.json() == {"detail": "public access is not configured"}


# ---------------------------------------------------------------- public limits & isolation tests

def test_public_visitor_per_session_isolation(api, no_run):
    _, main, _ = api
    client_a = TestClient(main.app)
    client_b = TestClient(main.app)

    assert _login(client_a).status_code == 200
    assert _login(client_b).status_code == 200

    # Visitor A creates a phantom run
    r_a = client_a.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r_a.status_code == 200
    job_a_id = r_a.json()["id"]

    # Visitor B cannot see Visitor A's job
    jobs_b = client_b.get("/api/jobs", headers=PROXY).json()
    assert not any(j["id"] == job_a_id for j in jobs_b)
    assert client_b.get(f"/api/jobs/{job_a_id}", headers=PROXY).status_code == 404
    assert client_b.get(f"/files/{job_a_id}/route.json", headers=PROXY).status_code == 404

    # Visitor A can see their own job
    jobs_a = client_a.get("/api/jobs", headers=PROXY).json()
    assert any(j["id"] == job_a_id for j in jobs_a)
    assert client_a.get(f"/api/jobs/{job_a_id}", headers=PROXY).status_code == 200

    # Local request sees it too
    local_client = TestClient(main.app)
    assert any(j["id"] == job_a_id for j in local_client.get("/api/jobs").json())


def test_public_visitor_one_active_run_per_session(api, no_run):
    _, main, _ = api
    client = TestClient(main.app)
    assert _login(client).status_code == 200

    r1 = client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r1.status_code == 200
    # Second job request while first is queued/running is rejected with 429
    r2 = client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r2.status_code == 429
    assert "already have a run in progress" in r2.json()["detail"]


def test_there_is_no_hourly_cap_by_default(api, no_run):
    _, main, _ = api
    assert main.PUBLIC_MAX_PER_HOUR is None
    a, ha = _visitor(main, "6.6.6.6")
    for _ in range(15):  # more than the old 12 per hour, one after another
        r = _run(a, ha)
        assert r.status_code == 200, r.text
        main._jobs[r.json()["id"]]["status"] = "done"


def test_public_visitor_hourly_cap(api, no_run, monkeypatch):
    _, main, _ = api
    monkeypatch.setattr(main, "PUBLIC_MAX_PER_HOUR", 12)  # the cap still works when an owner turns it on
    client = TestClient(main.app)
    assert _login(client).status_code == 200
    # Create first job to establish session and owner
    r1 = client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r1.status_code == 200
    sid = main._jobs[r1.json()["id"]]["owner"]
    main._jobs[r1.json()["id"]]["status"] = "done"

    import time
    now = time.time()
    # Fill up the hourly quota with done jobs
    for i in range(main.PUBLIC_MAX_PER_HOUR - 1):
        jid = f"20261110-1000{i:02d}-ffffff"
        main._jobs[jid] = {"id": jid, "status": "done", "created": now - 60, "owner": sid, "params": {"source": "phantom"}}

    r_capped = client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r_capped.status_code == 429
    assert f"Limit of {main.PUBLIC_MAX_PER_HOUR} runs per hour" in r_capped.json()["detail"]


def test_public_visitor_global_cap(api, no_run):
    _, main, _ = api
    import time
    now = time.time()
    # 3 active public jobs from different sessions
    for i in range(main.PUBLIC_MAX_ACTIVE_TOTAL):
        jid = f"20261110-1000{i:02d}-111111"
        main._jobs[jid] = {"id": jid, "status": "running", "created": now - 30, "owner": f"user{i}", "params": {"source": "phantom"}}

    client = TestClient(main.app)
    assert _login(client).status_code == 200
    r = client.post("/api/jobs", json={"source": "phantom", "video": False}, headers=PROXY)
    assert r.status_code == 429
    assert "busy right now" in r.json()["detail"]


def test_public_visitor_bounds_and_nodule_validation(api, no_run):
    client, _, _ = api
    assert _login(client).status_code == 200

    # Imported scans are refused; AeroPath is allowed but this test machine has none
    assert client.post("/api/jobs", json={"source": "upload", "upload_id": "0" * 12}, headers=PROXY).status_code == 403
    assert client.post("/api/jobs", json={"source": "aeropath", "case": 1}, headers=PROXY).status_code == 404

    # Radius bounds (1-15 mm)
    assert client.post("/api/jobs", json={"source": "phantom", "radius": 0.5}, headers=PROXY).status_code == 422
    assert client.post("/api/jobs", json={"source": "phantom", "radius": 20.0}, headers=PROXY).status_code == 422

    # Scope diameter bounds (1.5-6 mm)
    assert client.post("/api/jobs", json={"source": "phantom", "min_diameter": 1.0}, headers=PROXY).status_code == 422
    assert client.post("/api/jobs", json={"source": "phantom", "min_diameter": 8.0}, headers=PROXY).status_code == 422

    # Target out of volume bounds
    assert client.post("/api/jobs", json={"source": "phantom", "target_mode": "manual", "target": [150.0, 0.0, 0.0]}, headers=PROXY).status_code == 422

    # Phantom nodule out of volume bounds
    bad_bounds_nod = {"source": "phantom", "phantom_nodule": {"center": [200.0, 0.0, 0.0], "radius_mm": 5.0, "hu": 30.0}}
    assert client.post("/api/jobs", json=bad_bounds_nod, headers=PROXY).status_code == 422

    # Phantom nodule placement outside lung / on airway
    bad_lung_nod = {"source": "phantom", "phantom_nodule": {"center": [0.0, 0.0, 0.0], "radius_mm": 5.0, "hu": 30.0}}
    assert client.post("/api/jobs", json=bad_lung_nod, headers=PROXY).status_code == 422

    # Valid phantom nodule accepted
    valid_nod = {"source": "phantom", "phantom_nodule": {"center": [-34.0, 0.0, 10.0], "radius_mm": 7.0, "hu": -300.0}, "video": False}
    r = client.post("/api/jobs", json=valid_nod, headers=PROXY)
    assert r.status_code == 200


def test_job_timeout_handling(api, monkeypatch):
    _, main, tmp = api
    import subprocess

    def mock_wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(cmd=["test"], timeout=timeout)
        return 0

    monkeypatch.setattr(subprocess.Popen, "wait", mock_wait)
    monkeypatch.setattr(subprocess.Popen, "kill", lambda self: None)

    jid = "20261110-100000-timeout"
    out = tmp / "jobs" / jid
    out.mkdir(parents=True)
    main._jobs[jid] = {"id": jid, "status": "queued", "created": 1.0, "owner": "local", "params": {}}
    main._run_job(jid, ["python", "-c", "import time; time.sleep(10)"])

    assert main._jobs[jid]["status"] == "error"
    assert "took too long and was stopped" in main._jobs[jid]["result"]["error"]


def test_pruning_public_jobs(api):
    _, main, tmp = api
    import time
    now = time.time()
    jobs_dir = tmp / "jobs"

    # Create 35 finished public jobs
    for i in range(35):
        jid = f"20261110-1000{i:02d}-prune1"
        d = jobs_dir / jid
        d.mkdir(parents=True, exist_ok=True)
        main._jobs[jid] = {"id": jid, "status": "done", "created": now - i * 10, "owner": "public-sid"}

    # Also create an old public job (> 12 hours old)
    old_jid = "20261109-100000-oldone"
    (jobs_dir / old_jid).mkdir(parents=True, exist_ok=True)
    main._jobs[old_jid] = {"id": old_jid, "status": "done", "created": now - 15 * 3600, "owner": "public-sid"}

    # And a local job (should never be pruned)
    local_jid = "20261109-090000-localone"
    (jobs_dir / local_jid).mkdir(parents=True, exist_ok=True)
    main._jobs[local_jid] = {"id": local_jid, "status": "done", "created": now - 20 * 3600, "owner": "local"}

    main._prune_public_jobs()

    # Old job is gone
    assert old_jid not in main._jobs
    assert not (jobs_dir / old_jid).exists()

    # Local job is kept
    assert local_jid in main._jobs
    assert (jobs_dir / local_jid).exists()

    # Kept public jobs is at most PUBLIC_KEEP_JOBS (30)
    pub_remaining = [j for j in main._jobs.values() if j.get("owner") not in (None, "local")]
    assert len(pub_remaining) <= main.PUBLIC_KEEP_JOBS


def test_end_to_end_edit_run_on_phantom_job(api, monkeypatch):
    client, main, tmp = api
    import json
    from bronchnav.phantom import make_phantom
    from bronchnav.pipeline import run_plan

    # Create real job fixture in tmp/jobs/base_job
    jid = "20261110-100000-baseph"
    out = tmp / "jobs" / jid
    out.mkdir(parents=True)
    ph = make_phantom()
    ct_p = out / "phantom.mha"
    sitk.WriteImage(ph.image, str(ct_p))
    res = run_plan(ct_p, ph.nodule_center_mm, target_radius_mm=ph.nodule_radius_mm, out_dir=out, spacing_mm=0.75, min_diameter_mm=3.0, video=False)
    # Save job_result.json
    job_res = {
        "ok": True, "source": "phantom", "name": "Synthetic phantom",
        "target": {"target_mm": list(ph.nodule_center_mm)}, "target_radius_mm": ph.nodule_radius_mm,
        "metrics": res.route.metrics, "warnings": res.route.warnings,
        "outputs": res.outputs, "seconds": 2.0,
    }
    (out / "job_result.json").write_text(json.dumps(job_res))
    (out / "owner.txt").write_text("local")

    main._restore_finished_jobs()
    assert jid in main._jobs

    # 1. GET /api/jobs/{id}/tree
    tree_resp = client.get(f"/api/jobs/{jid}/tree")
    assert tree_resp.status_code == 200
    tree_data = tree_resp.json()
    assert len(tree_data["nodes"]) > 0
    assert len(tree_data["edges"]) > 0
    assert "origin_node" in tree_data and "exit_node" in tree_data

    # 2. GET /api/jobs/{id}/alternatives
    alts_resp = client.get(f"/api/jobs/{jid}/alternatives")
    assert alts_resp.status_code == 200
    alts = alts_resp.json()
    assert len(alts) > 0
    first_alt = alts[0]
    assert "point_mm" in first_alt and "gap_mm" in first_alt and "route_length_mm" in first_alt

    # 3. POST /api/jobs/{id}/replan
    replan_resp = client.post(f"/api/jobs/{jid}/replan", json={"end": first_alt["point_mm"]})
    assert replan_resp.status_code == 200
    replan_data = replan_resp.json()
    assert "path" in replan_data and "metrics" in replan_data
    assert len(replan_data["path"]["points_mm"]) > 0

    # 4. POST /api/jobs/{id}/edits (using mock runner so test is fast)
    monkeypatch.setattr(main, "_run_job", lambda new_id, cmd: None)
    edit_resp = client.post(f"/api/jobs/{jid}/edits", json={"end": first_alt["point_mm"], "video": False})
    assert edit_resp.status_code == 200
    new_job_id = edit_resp.json()["id"]
    assert new_job_id in main._jobs
    assert main._jobs[new_job_id]["params"]["source"] == "edit"
    assert main._jobs[new_job_id]["params"]["base_job"] == jid



# ---------------------------------------------------------------- public runs: phantom only, with limits

def _visitor(main, ip):
    """A signed-in public visitor from a given client address."""
    c = TestClient(main.app)
    hdr = {**PROXY, "X-Forwarded-For": ip}
    assert c.post("/api/login", json={"passcode": PASSCODE}, headers=hdr).status_code == 200
    return c, hdr


def _run(c, hdr, **body):
    return c.post("/api/jobs", json={"source": "phantom", "video": False, **body}, headers=hdr)


def test_a_public_run_is_a_phantom_run_owned_by_its_session_and_hides_internals(api, no_run):
    client, main, tmp = api
    a, ha = _visitor(main, "1.1.1.1")
    b, hb = _visitor(main, "2.2.2.2")
    r = _run(a, ha)
    assert r.status_code == 200
    job = r.json()
    assert not {"owner", "client", "pid", "cmd"} & set(job)  # internal bookkeeping is not exposed
    assert (tmp / "jobs" / job["id"] / "owner.txt").exists()
    main._jobs[job["id"]]["status"] = "done"
    assert {j["id"] for j in a.get("/api/jobs", headers=ha).json()} == {job["id"]}  # the owner sees it ...
    assert b.get("/api/jobs", headers=hb).json() == []                                # ... other visitors do not
    assert b.get(f"/api/jobs/{job['id']}", headers=hb).status_code == 404
    assert b.get(f"/files/{job['id']}/route.json", headers=hb).status_code == 404
    assert client.get(f"/api/jobs/{job['id']}").status_code == 200  # the owner of the machine (local) sees all
    # a local phantom run is a shared showcase for every visitor
    loc = client.post("/api/jobs", json={"source": "phantom", "video": False}).json()
    main._jobs[loc["id"]]["status"] = "done"
    assert loc["id"] in {j["id"] for j in b.get("/api/jobs", headers=hb).json()}


def test_one_active_run_per_session_and_per_client_address(api, no_run):
    _, main, _ = api
    a, ha = _visitor(main, "1.1.1.1")
    first = _run(a, ha).json()
    assert _run(a, ha).status_code == 429                      # same session, run still queued
    a2, ha2 = _visitor(main, "1.1.1.1")                          # signing in again is a new session, same address
    assert _run(a2, ha2).status_code == 429                    # ... and is still limited
    other, ho = _visitor(main, "3.3.3.3")
    assert _run(other, ho).status_code == 200                  # a different visitor is fine
    main._jobs[first["id"]]["status"] = "done"
    assert _run(a, ha).status_code == 200                      # after the first finished


def test_public_visitors_can_plan_on_aeropath_within_the_same_limits(api, no_run):
    client, main, tmp = api
    (tmp / "aero" / "1").mkdir(parents=True)
    (tmp / "aero" / "1" / "1_CT_HR.nii.gz").write_bytes(b"x")
    main.AEROPATH_ROOT = tmp / "aero"
    c, h = _visitor(main, "4.4.4.4")
    cases = c.get("/api/cases", headers=h).json()
    assert cases["aeropath"] == [1] and cases["mode"] == "public" and "aeropath_root" not in cases and "jobs_dir" not in cases
    assert _run(c, h, source="aeropath", case=1).status_code == 200
    d = [j for j in main._jobs.values() if j["status"] == "queued"][0]
    assert d["params"]["source"] == "aeropath" and d["owner"] != "local"
    main._jobs[d["id"]]["status"] = "done"
    # the same bounds apply; a real scan has no fixed frame, but absurd coordinates are refused
    assert _run(c, h, source="aeropath", case=1, target_mode="manual", target=[10, 20, 3000]).status_code == 422
    assert _run(c, h, source="aeropath", case=1, target_mode="manual", target=[10, 20, 30]).status_code == 200
    main._jobs[[j for j in main._jobs.values() if j["status"] == "queued"][0]["id"]]["status"] = "done"
    assert _run(c, h, source="aeropath", case=1, radius=40).status_code == 422
    assert _run(c, h, source="aeropath", case=1, phantom_nodule={"center": [-34, 0, 10]}).status_code == 422  # phantom only
    assert _run(c, h, source="aeropath", case=2).status_code == 404  # no such case
    # still no imports, and a visitor's AeroPath run is invisible to other visitors
    assert c.post("/api/jobs", json={"source": "upload", "upload_id": "0" * 12}, headers=h).status_code == 403
    other, ho = _visitor(main, "5.5.5.5")
    assert other.get("/api/jobs", headers=ho).json() == []


def test_known_failures_get_a_fixed_explanation_and_other_errors_stay_generic(api):
    client, main, tmp = api
    emph, other = "20261110-090010-eeeeee", "20261110-090011-ffffff"
    _make_job(tmp, emph, "aeropath", ok=False,
              error="ValueError: Region growing already exceeds 400 mL (1246 mL) at C:/Users/someone/secret/9.nii.gz")
    _make_job(tmp, other, "phantom", ok=False, error="OSError: C:/Users/someone/secret/x")
    main._restore_finished_jobs()
    assert _login(client).status_code == 200
    e = client.get(f"/api/jobs/{emph}", headers=PROXY).json()["result"]["error"]
    assert "emphysema" in e and "someone" not in e and "secret" not in e and "1246" not in e  # fixed text, nothing copied
    assert client.get(f"/api/jobs/{other}", headers=PROXY).json()["result"]["error"] == "the run failed"
    cases = client.get("/api/cases", headers=PROXY).json()
    assert "9" in cases["aeropath_unsupported"]


def test_a_forged_forwarded_for_header_does_not_dodge_the_login_limit(api):
    """The proxy appends the real address LAST; a client can put anything before it."""
    client, main, _ = api
    for i in range(main.LOGIN_ATTEMPTS):
        hdr = {**PROXY, "X-Forwarded-For": f"10.0.0.{i}, 9.9.9.9"}
        assert _login(client, "nope-nope-nope", hdr).status_code == 401
    hdr = {**PROXY, "X-Forwarded-For": "10.0.0.99, 9.9.9.9"}
    assert _login(client, PASSCODE, hdr).status_code == 429


def test_a_malformed_session_cookie_is_a_401_not_a_server_error(api):
    client, _, _ = api
    for cookie in (b"bron_session=9999999999.abcdef12.\xe9\xe9", b"bron_session=\xb2.abcdef12.zz", b"bron_session=a.b"):
        r = client.get("/api/jobs", headers={**PROXY, "cookie": cookie})
        assert r.status_code == 401, cookie


def test_parallel_requests_from_one_visitor_start_only_one_run(api, no_run):
    from concurrent.futures import ThreadPoolExecutor

    _, main, _ = api
    a, ha = _visitor(main, "1.1.1.1")
    cookie = a.cookies.get("bron_session")

    def go(_):
        c = TestClient(main.app)
        c.cookies.set("bron_session", cookie)
        return _run(c, ha).status_code

    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(go, range(8)))
    assert codes.count(200) == 1, codes
    assert sum(1 for j in main._jobs.values() if j["status"] == "queued") == 1


def test_hourly_cap_and_global_cap(api, no_run, monkeypatch):
    _, main, _ = api
    monkeypatch.setattr(main, "PUBLIC_MAX_PER_HOUR", 2)
    a, ha = _visitor(main, "1.1.1.1")
    for _ in range(2):
        r = _run(a, ha)
        assert r.status_code == 200
        main._jobs[r.json()["id"]]["status"] = "done"
    assert _run(a, ha).status_code == 429                      # third in the hour
    monkeypatch.setattr(main, "PUBLIC_MAX_PER_HOUR", None)
    monkeypatch.setattr(main, "PUBLIC_MAX_ACTIVE_TOTAL", 2)
    vs = [_visitor(main, f"9.9.9.{i}") for i in range(3)]
    assert [_run(c, h).status_code for c, h in vs] == [200, 200, 429]  # the third finds the demo busy


def test_public_parameters_are_bounded(api, no_run):
    _, main, _ = api
    a, ha = _visitor(main, "1.1.1.1")
    for bad in (
        {"radius": 50.0},                       # model limit is 40; the public limit is 15
        {"radius": 20.0},
        {"min_diameter": 1.2},
        {"min_diameter": 8.0},
        {"target_mode": "manual", "target": [0.0, 0.0, 500.0]},   # outside the phantom's volume
        {"target_mode": "manual", "target": [999.0, 0.0, 0.0]},
    ):
        assert _run(a, ha, **bad).status_code == 422, bad
    assert _run(a, ha, target_mode="manual", target=[10.0, -5.0, -37.0], radius=6.0).status_code == 200


def test_a_run_that_takes_too_long_is_stopped(api, monkeypatch):
    import sys
    import time

    _, main, tmp = api
    monkeypatch.setattr(main, "JOB_TIMEOUT_S", 1)
    jid = "20261110-090000-abcdef"
    (tmp / "jobs" / jid).mkdir(parents=True)
    main._jobs[jid] = {"id": jid, "status": "queued", "created": time.time(), "params": {"source": "phantom"}, "result": None, "owner": "local"}
    t0 = time.time()
    main._run_job(jid, [sys.executable, "-c", "import time; time.sleep(60)"])
    assert time.time() - t0 < 20
    assert main._jobs[jid]["status"] == "error" and "too long" in main._jobs[jid]["result"]["error"]


def test_old_and_surplus_public_runs_are_deleted_but_local_runs_are_kept(api, monkeypatch):
    import time

    _, main, tmp = api
    monkeypatch.setattr(main, "PUBLIC_KEEP_JOBS", 2)
    now = time.time()
    for i, (owner, age) in enumerate([("aaaaaaaa", 10), ("aaaaaaaa", 20), ("aaaaaaaa", 30), ("local", 10**6), ("aaaaaaaa", 13 * 3600)]):
        jid = f"20261110-09000{i}-00000{i}"
        (tmp / "jobs" / jid).mkdir(parents=True)
        main._jobs[jid] = {"id": jid, "status": "done", "created": now - age, "params": {"source": "phantom"}, "result": {}, "owner": owner}
    main._prune_public_jobs()
    left = set(main._jobs)
    assert left == {"20261110-090000-000000", "20261110-090001-000001", "20261110-090003-000003"}  # 2 newest public + the local run
    assert not (tmp / "jobs" / "20261110-090002-000002").exists() and not (tmp / "jobs" / "20261110-090004-000004").exists()
    assert (tmp / "jobs" / "20261110-090003-000003").exists()  # the owner's own run is never pruned


def test_job_and_edit_camera_roll_passed(api, monkeypatch):
    import json
    client, main, tmp = api
    captured_cmd = []
    monkeypatch.setattr(main, "_run_job", lambda jid, cmd: captured_cmd.extend(cmd))

    # 1. JobRequest with camera_roll
    r = client.post("/api/jobs", json={"source": "phantom", "camera_roll": 45.0, "video": True})
    assert r.status_code == 200
    jid = r.json()["id"]
    assert "--camera-roll" in captured_cmd
    idx = captured_cmd.index("--camera-roll")
    assert captured_cmd[idx + 1] == "45.0"

    # 2. EditRun with camera_roll
    # Set up dummy finished base job
    base_jid = "20261110-120000-baseedit"
    d = tmp / "jobs" / base_jid
    d.mkdir(parents=True)
    (d / "airway_mask.nii.gz").write_bytes(b"dummy")
    (d / "route.json").write_text(json.dumps({"target": {"center_mm": [0,0,0], "radius_mm": 5}}))
    (d / "owner.txt").write_text("local")
    main._jobs[base_jid] = {"id": base_jid, "status": "done", "created": 1000, "params": {"source": "phantom"}, "owner": "local"}

    captured_cmd.clear()
    edit_r = client.post(f"/api/jobs/{base_jid}/edits", json={"end": [10.0, 20.0, -30.0], "camera_roll": -90.0, "video": True})
    assert edit_r.status_code == 200
    assert "--camera-roll" in captured_cmd
    idx2 = captured_cmd.index("--camera-roll")
    assert captured_cmd[idx2 + 1] == "-90.0"


# ---------------------------------------------------------------- adding an airway the segmentation missed

def _phantom_base_job(main, tmp, jid="20261114-100000-addaws"):
    import json
    from bronchnav.phantom import make_phantom
    from bronchnav.pipeline import run_plan

    out = tmp / "jobs" / jid
    out.mkdir(parents=True)
    ph = make_phantom()
    sitk.WriteImage(ph.image, str(out / "phantom.mha"))
    res = run_plan(out / "phantom.mha", ph.nodule_center_mm, target_radius_mm=ph.nodule_radius_mm, out_dir=out,
                   spacing_mm=0.75, min_diameter_mm=3.0, video=False)
    (out / "job_result.json").write_text(json.dumps({
        "ok": True, "source": "phantom", "name": "Synthetic phantom", "target": {"target_mm": list(ph.nodule_center_mm)},
        "target_radius_mm": ph.nodule_radius_mm, "metrics": res.route.metrics, "warnings": res.route.warnings,
        "outputs": res.outputs, "seconds": 2.0}))
    (out / "owner.txt").write_text("local")
    main._restore_finished_jobs()
    assert jid in main._jobs
    return jid


def _stub_branch(client, jid, length_mm=16.0):
    """Two or three points that start on a tip of the existing tree and run straight on past it (away from the
    tree's centre), tried tip by tip until one stays inside the volume."""
    t = client.get(f"/api/jobs/{jid}/tree").json()
    nodes = np.array(t["nodes"])
    nbrs = {}
    for a, b in t["edges"]:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    centre = nodes.mean(axis=0)
    for n, nb in sorted(nbrs.items(), key=lambda kv: -np.linalg.norm(nodes[kv[0]] - centre)):
        if len(nb) != 1:
            continue
        d = nodes[n] - nodes[nb[0]]
        d /= np.linalg.norm(d)
        pts = [nodes[n] + d * s for s in (0.0, length_mm * 0.4, length_mm)]
        yield [[float(v) for v in p] for p in pts]


def test_add_airway_preview_and_save(api, monkeypatch):
    import json
    import subprocess

    client, main, tmp = api
    jid = _phantom_base_job(main, tmp)
    seen = None
    for pts in _stub_branch(client, jid):
        r = client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": pts, "radius_mm": 1.2})
        if r.status_code == 200:
            seen = (pts, r.json())
            break
        assert r.status_code == 422, r.text
    assert seen, "no tip of the phantom tree admitted a 16 mm continuation"
    pts, data = seen
    assert data["added_airway"]["length_mm"] >= 15 and data["added_airway"]["added_ml"] > 0
    assert any("added by the operator" in w for w in data["warnings"])
    end = np.array(data["path"]["points_mm"][-1])
    assert np.linalg.norm(end - np.array(pts[-1])) < 4.0   # the route really ends in the new branch

    # bad input is refused with a reason, and nothing is saved
    far = [[900.0, 900.0, 900.0], [905.0, 900.0, 900.0]]
    assert client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": far, "radius_mm": 1.0}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": pts[:1], "radius_mm": 1.0}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": pts, "radius_mm": 9.0}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": [[1, 2], [3, 4]], "radius_mm": 1.0}).status_code == 422
    short = [pts[0], [v + 1.0 for v in pts[0]]]
    assert client.post(f"/api/jobs/{jid}/addairway", json={"points_mm": short, "radius_mm": 1.0}).status_code == 422

    # saving: the runner gets the drawn airway and the route end is its far end
    monkeypatch.setattr(main, "_run_job", lambda new_id, cmd: None)
    saved = client.post(f"/api/jobs/{jid}/edits", json={"end": [0, 0, 0], "video": False,
                        "add_airway": {"points_mm": pts, "radius_mm": 1.2}})
    assert saved.status_code == 200, saved.text
    new_id = saved.json()["id"]
    params = main._jobs[new_id]["params"]
    assert params["new_airway"]["radius_mm"] == 1.2 and params["end"] == pts[-1]
    spec = tmp / "jobs" / new_id / "added_airway.json"
    assert json.loads(spec.read_text())["points_mm"] == pts

    # and the real job script builds the run from it
    out = tmp / "jobs" / "20261114-100001-addrun"
    cmd = [sys.executable, "-u", "-I", str(main.JOB_SCRIPT), "--source", "edit", "--out", str(out),
           "--base-job-dir", str(tmp / "jobs" / jid), "--end", *map(str, pts[-1]), "--min-diameter", "3.0",
           "--add-airway-json", str(spec), "--no-video"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((out / "job_result.json").read_text())
    assert result["ok"] and any("added by the operator" in w for w in result["warnings"])
    route = json.loads((out / "route.json").read_text())
    assert route["edit"]["added_airway"]["radius_mm"] == 1.2
    old = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp / "jobs" / jid / "airway_mask.nii.gz"))).sum()
    new = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "airway_mask.nii.gz"))).sum()
    assert new > old


# ---------------------------------------------------------------- DICOM import

def _dicom_series(folder, n=40, modality="CT", ext="", uid="1.2.826.0.1.3680043.8.498.7"):
    """A synthetic CT written as one DICOM file per slice, with identifying tags (like a scanner export)."""
    folder.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.random.default_rng(1).integers(-900, 200, size=(n, 24, 24)).astype(np.int16))
    img.SetSpacing((0.7, 0.7, 1.0))
    d = img.GetDirection()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    w.SetImageIO("GDCMImageIO")
    for i in range(n):
        s = img[:, :, i]
        for k, v in {"0008|0060": modality, "0020|000e": uid, "0010|0010": "DOE^JANE", "0010|0020": "MRN-99887",
                     "0020|0013": str(i + 1), "0028|1052": "0", "0028|1053": "1",
                     "0020|0032": "\\".join(f"{x:.6f}" for x in img.TransformIndexToPhysicalPoint((0, 0, i))),
                     "0020|0037": "\\".join(str(x) for x in (d[0], d[3], d[6], d[1], d[4], d[7])),
                     "0008|0018": f"{uid}.{i + 1}"}.items():
            s.SetMetaData(k, v)
        w.SetFileName(str(folder / f"IM{i:04d}{ext}"))
        w.Execute(s)
    return sorted(folder.iterdir())


def _upload_files(client, paths):
    return client.post("/api/uploads", files=[("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in paths])


def _assert_clean_upload(main, up_id):
    d = main.UPLOAD_DIR / up_id
    assert sorted(p.name for p in d.iterdir()) == ["ct.mha", "meta.json"]   # no DICOM original is left behind
    for p in d.iterdir():
        assert b"DOE" not in p.read_bytes() and b"MRN-99887" not in p.read_bytes()


def test_dicom_folder_upload_becomes_a_clean_volume(api):
    client, main, tmp = api
    files = _dicom_series(tmp / "src", n=40)                        # extensionless, as scanners write them
    (tmp / "src" / "README.txt").write_text("viewer notes")        # strays that travel with a folder are ignored
    r = _upload_files(client, sorted((tmp / "src").iterdir()))
    assert r.status_code == 200, r.text
    u = r.json()
    assert u["kind"] == "volume" and u["name"] == "DICOM CT, 40 slices"
    assert u["info"]["size_xyz"] == [24, 24, 40] and u["info"]["dicom"]["n_slices"] == 40
    _assert_clean_upload(main, u["upload_id"])
    assert len(files) == 40 and all(p.exists() for p in files)    # the client's own files are untouched


def test_dicom_zip_upload_and_local_folder_and_errors(api):
    import zipfile

    client, main, tmp = api
    _dicom_series(tmp / "z", n=35, ext=".dcm")
    (tmp / "z" / "preview1.png").write_bytes(_png_bytes())
    zp = tmp / "series.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for p in (tmp / "z").iterdir():
            zf.write(p, p.name)
    r = _upload_files(client, [zp])
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "DICOM CT, 35 slices"
    _assert_clean_upload(main, r.json()["upload_id"])

    _dicom_series(tmp / "folder", n=32)
    r = client.post("/api/uploads/local", json={"path": str(tmp / "folder")})
    assert r.status_code == 200, r.text
    assert len(list((tmp / "folder").iterdir())) == 32             # a local folder is read, never modified
    _assert_clean_upload(main, r.json()["upload_id"])

    r = client.post("/api/uploads/local", json={"path": str(tmp / "folder" / "IM0000")})
    assert r.status_code == 400 and "one DICOM slice" in r.json()["detail"]

    _dicom_series(tmp / "mr", n=40, modality="MR")
    r = _upload_files(client, sorted((tmp / "mr").iterdir()))
    assert r.status_code == 400 and "not CT" in r.json()["detail"]
    assert not any(main.UPLOAD_DIR.glob("*/dicom_raw"))             # a refused upload leaves nothing behind

    r = _upload_files(client, [tmp / "z" / "preview1.png"])
    assert r.status_code == 400                                     # one image is not a series
    junk = tmp / "junk.bin"
    junk.write_bytes(b"not a scan")
    r = _upload_files(client, [junk])
    assert r.status_code == 400 and "unsupported type" in r.json()["detail"]


def test_public_visitors_cannot_import_dicom(api, monkeypatch):
    client, main, tmp = api
    files = _dicom_series(tmp / "p", n=35)
    hdr = {**PROXY, "X-Forwarded-For": "203.0.113.9"}
    assert client.post("/api/login", json={"passcode": PASSCODE}, headers=hdr).status_code == 200
    r = client.post("/api/uploads", files=[("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in files], headers=hdr)
    assert r.status_code in (401, 403)


def test_stray_files_beside_an_image_series_are_ignored(api):
    client, main, tmp = api
    imgs = []
    for i in range(4):
        p = tmp / f"s{i}.png"
        p.write_bytes(_png_bytes())
        imgs.append(p)
    junk = tmp / "Thumbs.db"
    junk.write_bytes(b"x")
    r = _upload_files(client, [*imgs, junk])
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "images"
