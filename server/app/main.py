"""FastAPI server: health endpoint plus a LOCAL TEST CONSOLE.

The test console runs one planning job at a time as a subprocess (core/scripts/plan_job.py) and serves
the page in app/static plus each job's output folder. It is meant for a developer's own machine:
bind to 127.0.0.1 only, test data only, never expose it. The production job queue arrives in Phase 4.

Importing your own CT (a DICOM folder, a volume file, or a series of JPEG/PNG images) copies it under UPLOADS_DIR or
registers a path on this machine. Both are REFUSED when the request arrives through a proxy (for example
`tailscale serve`, which adds identity/forwarding headers): the console has no login, so scans must not
travel through a shared URL. Set ALLOW_REMOTE_UPLOADS=1 to override (not recommended).

Environment:
  JOBS_DIR        where job outputs go (default ~/bronchnav-jobs, outside the Google Drive folder)
  UPLOADS_DIR     where imported scans are copied (default ~/bronchnav-uploads; never served over HTTP)
  AEROPATH_ROOT   folder holding the numbered AeroPath case folders
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from bronchnav import __version__ as core_version
from bronchnav.segmentation.manual import MAX_POINTS as MAX_AIRWAY_POINTS, RADIUS_RANGE_MM as AIRWAY_RADIUS_RANGE
from bronchnav.io import IMAGE_EXTS

from . import edit as _edit

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
JOB_SCRIPT = REPO_ROOT / "core" / "scripts" / "plan_job.py"
JOBS_DIR = Path(os.environ.get("JOBS_DIR", str(Path.home() / "bronchnav-jobs")))
AEROPATH_ROOT = Path(os.environ.get("AEROPATH_ROOT", str(Path.home() / "datasets" / "aeropath" / "extracted" / "AeroPath")))
# Uploads live OUTSIDE the served jobs tree (the /files mount below can never reach the scans).
UPLOAD_DIR = Path(os.environ.get("UPLOADS_DIR", str(Path.home() / "bronchnav-uploads")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VOLUME_EXTS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".nhdr", ".tif", ".tiff")
RAW_EXTS = (".raw", ".zraw", ".img")
ARCHIVE_EXTS = (".zip",)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MAX_UPLOAD_BYTES = 3 * 1024**3
PROXY_HEADERS = (
    "tailscale-funnel-request", "tailscale-user-login", "tailscale-user-name",
    "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded", "x-real-ip",
)

app = FastAPI(title="Bronchoscopy Navigation Planner", version="0.0.1")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "core_version": core_version,
        "data_dir": os.environ.get("DATA_DIR", "/data"),
        "intended_use": "Planning aid only. Not for clinical use.",
    }


# ---------------------------------------------------------------- access guard

def _proxied(request: Request) -> bool:
    if any(h in request.headers for h in PROXY_HEADERS):
        return True
    host = request.client.host if request.client else ""
    return host not in ("127.0.0.1", "::1", "localhost", "testclient")


def _local_only(request: Request) -> bool:
    return os.environ.get("ALLOW_REMOTE_UPLOADS") == "1" or not _proxied(request)


def _require_local(request: Request) -> None:
    if not _local_only(request):
        raise HTTPException(
            403,
            "Importing scans is disabled over a shared/proxied URL (Tailscale etc.): this console has no login. "
            "Open http://127.0.0.1:8000 on the machine that runs it.",
        )


# ---------------------------------------------------------------- public (proxied) access
#
# Two tiers, decided only by whether the request came through a proxy (Tailscale serve/funnel adds
# forwarding headers; a direct request from this PC has none):
#   local   - no proxy headers: everything works (run jobs, import scans, logs, file paths).
#   public  - any proxied request: passcode login required, then FUNCTIONAL on PUBLIC DATA ONLY: the synthetic
#             phantom and the AeroPath dataset (CC BY 4.0). Use the CT viewer, pick a nodule, edit the path, render.
#             No imports, no local paths, no logs, and the raw volumes/masks are never downloadable. Each session
#             sees only its own runs plus the owner's local phantom/AeroPath runs; one run at a time per session,
#             an optional hourly cap (off), a global cap, a job timeout, bounded parameters, and old public runs are deleted.
# The tier is NOT taken from Tailscale identity headers (a client could forge them on a public Funnel), so a
# phone on the owner's own tailnet also logs in with the passcode. With no passcode configured, public access
# is closed (503), never open.

SESSION_SECRET = os.urandom(32)  # sessions end when the server restarts
SESSION_HOURS = 12
MIN_PASSCODE = 8
LOGIN_ATTEMPTS = 8  # per client, per LOGIN_WINDOW seconds
LOGIN_WINDOW = 600
PUBLIC_MAX_ACTIVE_PER_SESSION = 1  # a visitor can have one run queued or running at a time
PUBLIC_MAX_PER_HOUR: int | None = None  # runs per visitor per hour; None = no hourly cap (released 2026-11-13)
PUBLIC_MAX_ACTIVE_TOTAL = 3        # public runs queued or running across all visitors
PUBLIC_KEEP_JOBS = 30              # finished public runs kept on disk
PUBLIC_JOB_MAX_AGE_S = 12 * 3600
JOB_TIMEOUT_S = 420                # a run is stopped after this long
PHANTOM_BOUNDS = ((-80.0, 80.0), (-80.0, 80.0), (-110.0, 110.0))  # physical LPS mm around the phantom volume
# CTs a public visitor may open and plan on: the synthetic phantom and the public AeroPath dataset (CC BY 4.0,
# anonymised research data). Imported scans and local paths are never available. The raw volumes and masks are
# not downloadable either: only route.json, the mesh, stills, video and windowed CT slices are served.
PUBLIC_SOURCES = ("phantom", "aeropath")
AEROPATH_OPENS_PER_WINDOW = 20  # CT-viewer loads of an AeroPath case per client per LOGIN_WINDOW (each reads ~200 MB)
_aeropath_opens: dict[str, list[float]] = {}
SID_RE = re.compile(r"^[a-f0-9]{8}$")
JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-f0-9]{6}$")
PUBLIC_POST_RE = re.compile(r"^/api/jobs/[^/]+/(replan|addairway|edits)$")
PUBLIC_FILE_RE = re.compile(r"^(route\.json|airway_mesh\.glb|airway_mesh\.ply|flythrough\.mp4|stills/[A-Za-z0-9_.-]+\.png)$")
_login_attempts: dict[str, list[float]] = {}
_closed_logged = False


def _passcode() -> str | None:
    """The shared passcode: env DEMO_PASSCODE, else the first line of DEMO_PASSCODE_FILE (default
    ~/bronchnav-demo-passcode.txt). Read on every call so it can be changed without a restart. None = not set."""
    code = os.environ.get("DEMO_PASSCODE")
    if not code:
        f = Path(os.environ.get("DEMO_PASSCODE_FILE", str(Path.home() / "bronchnav-demo-passcode.txt")))
        try:
            code = f.read_text(encoding="utf-8").splitlines()[0] if f.exists() else ""
        except (OSError, IndexError):
            code = ""
    code = (code or "").strip()
    return code if len(code) >= MIN_PASSCODE else None


def _passcode_state() -> str:
    """Why public access is open or closed, for the owner (local requests and the server log only)."""
    f = Path(os.environ.get("DEMO_PASSCODE_FILE", str(Path.home() / "bronchnav-demo-passcode.txt")))
    if os.environ.get("DEMO_PASSCODE"):
        return "open" if _passcode() else f"closed: DEMO_PASSCODE is shorter than {MIN_PASSCODE} characters"
    if not f.exists():
        return f"closed: no passcode file at {f}"
    try:
        first = f.read_text(encoding="utf-8").splitlines()[0].strip() if f.stat().st_size else ""
    except (OSError, IndexError):
        first = ""
    if not first:
        return f"closed: the passcode file {f} is empty"
    if len(first) < MIN_PASSCODE:
        return f"closed: the passcode in {f} is {len(first)} characters; it needs at least {MIN_PASSCODE}"
    return "open"


def _token(passcode: str, exp: int, sid: str) -> str:
    key = hashlib.sha256(SESSION_SECRET + passcode.encode()).digest()  # a new passcode invalidates old sessions
    return f"{exp}.{sid}.{hmac.new(key, f'{exp}.{sid}'.encode(), hashlib.sha256).hexdigest()}"


def _session(request: Request) -> tuple[int, str] | None:
    """(expiry, session id) of a valid signed session cookie, else None."""
    code = _passcode()
    parts = request.cookies.get("bron_session", "").split(".")
    if not code or len(parts) != 3:
        return None
    exp_s, sid, _ = parts
    if not exp_s.isdigit() or int(exp_s) < time.time() or not SID_RE.match(sid):
        return None
    cookie = ".".join(parts)
    # compare bytes: str comparison raises TypeError on a non-ASCII cookie (a 500 for any anonymous visitor)
    return (int(exp_s), sid) if hmac.compare_digest(cookie.encode(), _token(code, int(exp_s), sid).encode()) else None


def _authed(request: Request) -> bool:
    return _session(request) is not None


def _sid(request: Request) -> str | None:
    s = _session(request)
    return s[1] if s else None


def _client_key(request: Request) -> str:
    # The LAST forwarded address is the one our own proxy appended; earlier entries are whatever the client sent,
    # so keying on the first would let a visitor dodge the login limit and quotas by rotating a forged header.
    return (request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
            or (request.client.host if request.client else "unknown"))


def _is_public(request: Request) -> bool:
    return _proxied(request)


def _data_source(job: dict | None) -> str | None:
    """Which CT a run was made on: an edited path has the source of the run it edits."""
    p = (job or {}).get("params") or {}
    return p.get("base_source") or p.get("source")


def _can_see(request: Request, job: dict | None) -> bool:
    """Local sees everything. A public visitor sees phantom runs that are their own session's, or the owner's
    (local) phantom runs, and nothing else."""
    if not job:
        return False
    if not _is_public(request):
        return True
    return _data_source(job) in PUBLIC_SOURCES and job.get("owner") in ("local", _sid(request))


class Login(BaseModel):
    passcode: str = Field(max_length=200)


@app.post("/api/login")
def login(body: Login, request: Request):
    code = _passcode()
    if code is None:
        return JSONResponse({"detail": "public access is not configured"}, status_code=503)
    key, now = _client_key(request), time.time()
    recent = [t for t in _login_attempts.get(key, []) if now - t < LOGIN_WINDOW]
    if len(recent) >= LOGIN_ATTEMPTS:
        return JSONResponse({"detail": "too many attempts; try again in a few minutes"}, status_code=429)
    if not hmac.compare_digest(body.passcode.encode(), code.encode()):
        _login_attempts[key] = recent + [now]
        return JSONResponse({"detail": "wrong passcode"}, status_code=401)
    _login_attempts.pop(key, None)
    exp = int(now + SESSION_HOURS * 3600)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        "bron_session", _token(code, exp, uuid.uuid4().hex[:8]), max_age=SESSION_HOURS * 3600, httponly=True, samesite="lax",
        secure=os.environ.get("DEMO_INSECURE_COOKIE") != "1", path="/",
    )
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("bron_session", path="/")
    return resp


@app.get("/login")
def login_page() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "login.html")


@app.middleware("http")
async def public_gate(request: Request, call_next):
    if not _is_public(request):
        return await call_next(request)  # local: unrestricted
    path = posixpath.normpath("/" + request.url.path.lstrip("/"))
    if path == "/api/health":
        return await call_next(request)
    if _passcode() is None:
        global _closed_logged
        if not _closed_logged:  # tell the owner in the server log, once; the visitor only sees a generic message
            _closed_logged = True
            print(f"WARNING public access {_passcode_state()}", flush=True)
        return JSONResponse({"detail": "public access is not configured"}, status_code=503)
    _closed_logged = False
    if path in ("/login", "/api/login"):
        return await call_next(request)
    if not _authed(request):
        if request.method in ("GET", "HEAD") and path == "/":
            return RedirectResponse("/login")
        return JSONResponse({"detail": "login required"}, status_code=401)
    if request.method in ("GET", "HEAD"):
        if path in ("/", "/api/cases", "/api/jobs") or path.startswith(("/api/jobs/", "/files/", "/api/ct/")):
            return await call_next(request)
        return JSONResponse({"detail": "not found"}, status_code=404)
    if request.method == "POST" and (
        path in ("/api/jobs", "/api/ct/open", "/api/logout") or PUBLIC_POST_RE.match(path)
    ):
        return await call_next(request)  # the handlers still enforce public-data-only and the limits
    return JSONResponse({"detail": "not available in the public demo"}, status_code=403)


# ---------------------------------------------------------------- imports

class StackParams(BaseModel):
    pixel_spacing_mm: float = Field(gt=0, le=5)
    slice_spacing_mm: float = Field(gt=0, le=10)
    window_center: float = -600.0
    window_width: float = Field(1500.0, gt=0, le=6000)
    first_slice: Literal["auto", "head", "feet"] = "auto"
    invert: bool = False


def _safe_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(". ")
    if not base:
        raise HTTPException(400, "empty file name")
    return base


def _is_volume(name: str) -> bool:
    return name.lower().endswith(VOLUME_EXTS) and not name.lower().endswith((".tif", ".tiff"))


def _is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))


def _is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_EXTS)


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def _safe_extract_zip(zip_file: Path, target_dir: Path) -> list[Path]:
    import zipfile

    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    extracted = []
    with zipfile.ZipFile(zip_file, "r") as zf:
        for member in zf.infolist():
            target_path = (target_dir / member.filename).resolve()
            if not str(target_path).startswith(str(target_dir)):
                raise HTTPException(400, f"unsafe path in zip: {member.filename}")
            zf.extract(member, target_dir)
            if target_path.is_file():
                extracted.append(target_path)
    return extracted


def _describe(kind: str, primary: Path) -> dict:
    """Header-level facts about an import, without reading voxels."""
    import SimpleITK as sitk

    if kind == "volume":
        rd = sitk.ImageFileReader()
        rd.SetFileName(str(primary))
        try:
            rd.ReadImageInformation()
        except Exception as e:  # unreadable or unsupported
            raise HTTPException(400, f"cannot read {primary.name} as a 3D image: {e}")
        if rd.GetDimension() != 3:
            raise HTTPException(400, f"{primary.name} is {rd.GetDimension()}D; a 3D CT volume is needed")
        return {
            "size_xyz": list(rd.GetSize()),
            "spacing_mm": [round(s, 3) for s in rd.GetSpacing()],
            "pixel_type": sitk.GetPixelIDValueAsString(rd.GetPixelID()),
        }
    from bronchnav.io import list_images

    imgs = list_images(primary)
    rd = sitk.ImageFileReader()
    rd.SetFileName(str(imgs[0]))
    rd.ReadImageInformation()
    formats = sorted({f.suffix.lower().lstrip(".") for f in imgs})
    return {
        "n_images": len(imgs),
        "size_px": [rd.GetSize()[0], rd.GetSize()[1]],
        "first": imgs[0].name,
        "last": imgs[-1].name,
        "formats": formats,
        "pixel_type": sitk.GetPixelIDValueAsString(rd.GetPixelID()),
    }


def _save_meta(up_id: str, meta: dict) -> dict:
    (UPLOAD_DIR / up_id).mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / up_id / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"upload_id": up_id, **{k: v for k, v in meta.items() if k != "path"}}


def _load_meta(up_id: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{12}", up_id or ""):
        raise HTTPException(404, "unknown upload")
    p = UPLOAD_DIR / up_id / "meta.json"
    if not p.exists():
        raise HTTPException(404, "unknown upload")
    return json.loads(p.read_text(encoding="utf-8"))


def _finish_dicom(up_id: str, root: Path, bytes_in: int | None, remove: list[Path]) -> dict:
    """Convert the CT series under `root` to a plain volume in this upload's folder, then delete the DICOM originals
    that were copied for it (`remove`). Only voxels and geometry are kept, so no patient identifiers stay on disk."""
    from bronchnav.io import dicom

    dest = UPLOAD_DIR / up_id
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "ct.mha"
    try:
        dinfo = dicom.convert_series(root, out)
    except ValueError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(400, str(e))
    for p in remove:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    info = _describe("volume", out)
    info["dicom"] = {"n_slices": dinfo["n_slices"], "other_series": dinfo["other_series"], "warnings": dinfo["warnings"]}
    name = f"DICOM CT, {dinfo['n_slices']} slices"
    return _save_meta(up_id, {"kind": "volume", "name": name, "path": str(out), "bytes": bytes_in, "info": info, "copied": True})


@app.post("/api/uploads")
async def upload(request: Request, files: list[UploadFile] = File(...)) -> dict:
    """Copy a CT volume file, ZIP archive, DICOM series or a series of images (JPEG, PNG, TIFF) into the uploads
    folder. A DICOM series is converted to a plain volume and the originals are deleted."""
    _require_local(request)
    up_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / up_id
    dest.mkdir(parents=True)
    total, saved, unknown = 0, [], []
    try:
        for f in files:
            name = _safe_name(f.filename or "")
            if _is_volume(name) or _is_image(name) or _is_archive(name) or name.lower().endswith(RAW_EXTS):
                target = dest / name
                n = 2
                while target.exists():
                    target = dest / f"{target.stem}_{n}{target.suffix}"
                    n += 1
            else:  # most DICOM slices have no (or an odd) extension: keep them apart and check what they are below
                (dest / "dicom_raw").mkdir(exist_ok=True)
                target = dest / "dicom_raw" / f"{len(unknown):05d}_{name}"
                unknown.append(target)
            with open(target, "wb") as out:
                while chunk := await f.read(8 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "upload larger than 3 GB")
                    out.write(chunk)
            saved.append(target)

        # DICOM slices (a folder of files, usually without extensions)
        if unknown:
            from bronchnav.io import dicom

            dcm = [p for p in unknown if dicom.is_dicom_file(p)]
            others = [p for p in unknown if p not in dcm]
            known = [p for p in saved if p not in unknown]   # volumes, images and archives that came with them
            if len(dcm) >= 3:
                # other files that came along with the folder (previews, viewers, notes) are ignored and deleted
                return await asyncio.to_thread(_finish_dicom, up_id, dest / "dicom_raw", total, [dest / "dicom_raw", *known])
            if dcm or not known:
                what = "a DICOM series needs at least 3 slices; send the folder with every slice, or a .zip of it" if dcm else                     re.sub(r"^\d{5}_", "", others[0].name) + ": unsupported type. Use a DICOM folder or .zip, .nii/.nii.gz/.mha/.mhd(+.raw)/.nrrd, or JPEG/PNG/BMP/TIFF images"
                raise HTTPException(400, what)
            shutil.rmtree(dest / "dicom_raw", ignore_errors=True)  # stray files beside a volume or an image series

        # Check for zip archives
        zips = [p for p in saved if _is_archive(p.name)]
        if zips:
            extracted_dir = dest / "extracted"
            for z in zips:
                _safe_extract_zip(z, extracted_dir)
            # Scan extracted contents
            ex_vols = [p for p in extracted_dir.rglob("*") if p.is_file() and _is_volume(p.name)]
            ex_imgs = [p for p in extracted_dir.rglob("*") if p.is_file() and _is_image(p.name) and not _is_volume(p.name)]
            if not ex_vols:
                from bronchnav.io import dicom

                if len(dicom.find_dicom_files(extracted_dir)) >= 3:
                    return await asyncio.to_thread(_finish_dicom, up_id, extracted_dir, total, [extracted_dir, *saved])
            if ex_vols:
                kind, primary, display = "volume", ex_vols[0], f"{ex_vols[0].name} (from {zips[0].name})"
            elif len(ex_imgs) >= 3:
                folder = dest / "images"
                folder.mkdir(parents=True, exist_ok=True)
                for p in ex_imgs:
                    target_p = folder / p.name
                    if not target_p.exists():
                        shutil.copy(str(p), str(target_p))
                kind, primary, display = "images", folder, f"{len(ex_imgs)} images from {zips[0].name}"
            else:
                raise HTTPException(400, f"no 3D volume or series of >=3 images found inside {zips[0].name}")
            info = _describe(kind, primary)
            return _save_meta(up_id, {"kind": kind, "name": display, "path": str(primary), "bytes": total, "info": info, "copied": True})

        # Single 3D TIFF file
        if len(saved) == 1 and _is_tiff(saved[0].name):
            import SimpleITK as sitk

            rd = sitk.ImageFileReader()
            rd.SetFileName(str(saved[0]))
            try:
                rd.ReadImageInformation()
                if rd.GetDimension() == 3 and rd.GetSize()[2] >= 3:
                    kind, primary, display = "volume", saved[0], saved[0].name
                    info = _describe(kind, primary)
                    return _save_meta(up_id, {"kind": kind, "name": display, "path": str(primary), "bytes": total, "info": info, "copied": True})
            except Exception:
                pass

        vols = [p for p in saved if _is_volume(p.name)]
        imgs = [p for p in saved if _is_image(p.name)]
        if vols and imgs:
            raise HTTPException(400, "send either one volume file or a series of images, not both")
        if len(vols) > 1:
            raise HTTPException(400, "send one volume file at a time")
        if vols:
            kind, primary, display = "volume", vols[0], vols[0].name
        elif len(imgs) >= 3:
            folder = dest / "images"
            folder.mkdir()
            for p in imgs:
                shutil.move(str(p), str(folder / p.name))
            kind, primary, display = "images", folder, f"{len(imgs)} images"
        else:
            raise HTTPException(400, "need a volume file, a ZIP archive, or at least 3 images in slice order")
        info = _describe(kind, primary)
        return _save_meta(up_id, {"kind": kind, "name": display, "path": str(primary), "bytes": total, "info": info, "copied": True})
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise


class LocalPath(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


@app.post("/api/uploads/local")
def register_local(req: LocalPath, request: Request) -> dict:
    """Use a volume file, ZIP archive, DICOM folder or image folder that already sits on this machine. Nothing is
    copied, except that a DICOM series is converted to a de-identified volume in the uploads folder."""
    _require_local(request)
    p = Path(req.path.strip().strip('"')).expanduser()
    if not p.exists():
        raise HTTPException(404, f"not found: {p}")
    p = p.resolve()
    if p.is_file():
        if _is_archive(p.name):
            up_id = uuid.uuid4().hex[:12]
            ex_dir = UPLOAD_DIR / up_id / "extracted"
            _safe_extract_zip(p, ex_dir)
            ex_vols = [f for f in ex_dir.rglob("*") if f.is_file() and _is_volume(f.name)]
            ex_imgs = [f for f in ex_dir.rglob("*") if f.is_file() and _is_image(f.name) and not _is_volume(f.name)]
            if not ex_vols:
                from bronchnav.io import dicom

                if len(dicom.find_dicom_files(ex_dir)) >= 3:
                    return _finish_dicom(up_id, ex_dir, p.stat().st_size, [ex_dir])
            if ex_vols:
                kind, primary, display = "volume", ex_vols[0], f"{ex_vols[0].name} (from {p.name})"
            elif len(ex_imgs) >= 3:
                kind, primary, display = "images", ex_imgs[0].parent, f"{len(ex_imgs)} images from {p.name}"
            else:
                raise HTTPException(400, f"no volume or image series found in {p.name}")
            info = _describe(kind, primary)
            return _save_meta(up_id, {"kind": kind, "name": display, "path": str(primary), "bytes": p.stat().st_size, "info": info, "copied": True})

        if _is_tiff(p.name):
            import SimpleITK as sitk

            rd = sitk.ImageFileReader()
            rd.SetFileName(str(p))
            try:
                rd.ReadImageInformation()
                if rd.GetDimension() == 3 and rd.GetSize()[2] >= 3:
                    kind, display = "volume", p.name
                    info = _describe(kind, p)
                    return _save_meta(uuid.uuid4().hex[:12], {"kind": kind, "name": display, "path": str(p), "bytes": None, "info": info, "copied": False})
            except Exception:
                pass

        if not _is_volume(p.name):
            from bronchnav.io import dicom

            if dicom.is_dicom_file(p):
                raise HTTPException(400, f"{p.name} is one DICOM slice; give the folder that holds all the slices")
            raise HTTPException(400, f"{p.name}: need a volume file ({', '.join(VOLUME_EXTS)}), a ZIP archive (.zip), a DICOM folder, or a folder of images")
        kind, display = "volume", p.name
    else:
        from bronchnav.io import dicom, list_images

        if len(dicom.find_dicom_files(p)) >= 3:
            return _finish_dicom(uuid.uuid4().hex[:12], p, None, [])

        try:
            imgs = list_images(p)
            if len(imgs) >= 3:
                kind, display = "images", f"{len(imgs)} images in {p.name}"
                info = _describe(kind, p)
                return _save_meta(uuid.uuid4().hex[:12], {"kind": kind, "name": display, "path": str(p), "bytes": None, "info": info, "copied": False})
        except Exception:
            pass
        kind, display = "volume", f"volume directory {p.name}"
    info = _describe(kind, p)
    return _save_meta(uuid.uuid4().hex[:12], {"kind": kind, "name": display, "path": str(p), "bytes": None, "info": info, "copied": False})


@app.delete("/api/uploads/{up_id}")
def delete_upload(up_id: str, request: Request) -> dict:
    """Remove an upload (and the copy of the scan). A registered local path is only forgotten, never deleted."""
    _require_local(request)
    _load_meta(up_id)
    shutil.rmtree(UPLOAD_DIR / up_id, ignore_errors=True)
    return {"deleted": up_id}


# ---------------------------------------------------------------- CT viewer
def _ct_guard_open(request: Request, req) -> None:
    if _is_public(request) and req.source not in PUBLIC_SOURCES:
        raise HTTPException(403, "the public demo only has the synthetic phantom and the public AeroPath CTs")
    if _is_public(request) and req.source == "aeropath":
        now, key = time.time(), _client_key(request)
        recent = [t for t in _aeropath_opens.get(key, []) if now - t < LOGIN_WINDOW]
        if len(recent) >= AEROPATH_OPENS_PER_WINDOW:
            raise HTTPException(429, "Too many CT loads. Try again in a few minutes.")
        _aeropath_opens[key] = recent + [now]
    if _is_public(request) and getattr(req, "nodule", None):
        nod = req.nodule
        for v, (lo, hi), name in zip(nod.center, PHANTOM_BOUNDS, "xyz"):
            if not lo <= v <= hi:
                raise HTTPException(422, f"nodule {name} must be within {lo:g}..{hi:g} mm (the phantom's volume)")


def _ct_guard_use(request: Request, ct) -> None:
    if _is_public(request) and not ct.key.startswith(("phantom", "aeropath:")):
        raise HTTPException(404, "unknown or expired CT: open it again")


from . import ct as _ct  # noqa: E402

_ct.register(
    app,
    aeropath_root=lambda: AEROPATH_ROOT,
    load_upload_meta=_load_meta,
    guard_open=_ct_guard_open,
    guard_use=_ct_guard_use,
)


# ---------------------------------------------------------------- test console jobs

class PhantomNodule(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)
    radius_mm: float = Field(5.0, ge=1.0, le=15.0)
    hu: float = Field(30.0, ge=-900.0, le=500.0)


class JobRequest(BaseModel):
    source: Literal["phantom", "aeropath", "upload"] = "phantom"
    case: int = Field(1, ge=1, le=27)
    upload_id: str | None = None
    stack: StackParams | None = None
    target_mode: Literal["auto", "manual"] = "auto"
    percentile: float = Field(0.7, ge=0.0, le=1.0)
    ahead_mm: float = Field(15.0, ge=2.0, le=40.0)
    target: list[float] | None = Field(None, min_length=3, max_length=3)
    radius: float = Field(6.0, ge=1.0, le=40.0)
    min_diameter: float = Field(3.0, ge=1.0, le=10.0)
    camera_roll: float = Field(0.0, ge=-360.0, le=360.0)
    video: bool = True
    phantom_nodule: PhantomNodule | None = None


_jobs: dict[str, dict] = {}
_run_lock = threading.Semaphore(1)  # one pipeline at a time: each uses several GB of RAM


def _aeropath_cases() -> list[int]:
    if not AEROPATH_ROOT.is_dir():
        return []
    return sorted(
        int(p.name) for p in AEROPATH_ROOT.iterdir() if p.name.isdigit() and (p / f"{p.name}_CT_HR.nii.gz").exists()
    )


def _command(job_id: str, req: JobRequest) -> list[str]:
    out = JOBS_DIR / job_id
    cmd = [sys.executable, "-u", "-I", str(JOB_SCRIPT), "--source", req.source, "--out", str(out)]
    cmd += ["--aeropath-root", str(AEROPATH_ROOT), "--case", str(req.case)]
    cmd += ["--target-mode", req.target_mode, "--percentile", str(req.percentile), "--ahead-mm", str(req.ahead_mm)]
    cmd += ["--radius", str(req.radius), "--min-diameter", str(req.min_diameter)]
    if req.target_mode == "manual":
        if not req.target:
            raise HTTPException(422, "manual target needs target = [x, y, z] in physical LPS mm")
        cmd += ["--target", *[str(v) for v in req.target]]
    if req.source == "phantom" and req.phantom_nodule:
        cmd += [
            "--nodule", *[str(v) for v in req.phantom_nodule.center],
            "--nodule-radius", str(req.phantom_nodule.radius_mm),
            "--nodule-hu", str(req.phantom_nodule.hu),
        ]
    if req.source == "upload":
        meta = _load_meta(req.upload_id or "")
        out.mkdir(parents=True, exist_ok=True)
        cmd += ["--ct-path", meta["path"], "--display-name", meta["name"]]
        if meta["kind"] == "images":
            if req.stack is None:
                raise HTTPException(422, "an image series needs stack = pixel/slice spacing and window")
            (out / "stack.json").write_text(json.dumps(req.stack.model_dump()), encoding="utf-8")
            cmd += ["--stack-json", str(out / "stack.json")]
    if req.camera_roll != 0.0:
        cmd += ["--camera-roll", str(req.camera_roll)]
    if not req.video:
        cmd.append("--no-video")
    return cmd


def _run_job(job_id: str, cmd: list[str]) -> None:
    job = _jobs[job_id]
    with _run_lock:
        job["status"] = "running"
        job["started"] = time.time()
        out = JOBS_DIR / job_id
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "job.log", "w", encoding="utf-8", errors="replace") as logf:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(out))
            job["pid"] = proc.pid
            timed_out = False
            try:
                proc.wait(timeout=JOB_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()
        job["finished"] = time.time()
        res = out / "job_result.json"
        if res.exists() and not timed_out:
            job["result"] = json.loads(res.read_text(encoding="utf-8"))
            job["status"] = "done" if job["result"].get("ok") else "error"
        else:
            job["status"] = "error"
            why = "the run took too long and was stopped" if timed_out else f"job exited with code {proc.returncode} without a result"
            job["result"] = {"ok": False, "error": why}


def _public(job: dict, with_log: bool = False) -> dict:
    d = {k: v for k, v in job.items() if k not in ("cmd", "client", "owner", "pid")}  # internal bookkeeping
    if with_log:
        p = JOBS_DIR / job["id"] / "job.log"
        d["log"] = p.read_text(encoding="utf-8", errors="replace")[-6000:] if p.exists() else ""
    return d


@app.get("/api/cases")
def cases(request: Request) -> dict:
    if _is_public(request):  # phantom + public AeroPath cases; no local paths; nothing to import
        return {"phantom": True, "aeropath": _aeropath_cases(), "mode": "public", "can_run": True, "imports_allowed": False,
                "aeropath_unsupported": {str(k): v for k, v in AEROPATH_UNSUPPORTED.items()}}
    return {
        "phantom": True,
        "aeropath": _aeropath_cases(),
        "aeropath_root": str(AEROPATH_ROOT),
        "jobs_dir": str(JOBS_DIR),
        "imports_allowed": _local_only(request),
        "volume_formats": list(VOLUME_EXTS),
        "image_formats": list(IMAGE_EXTS),
        "mode": "local",
        "can_run": True,
        "aeropath_unsupported": {str(k): v for k, v in AEROPATH_UNSUPPORTED.items()},
        "public_access": _passcode_state(),  # owner-only: whether visitors can sign in, and why not
    }


def _enforce_public_quota(request: Request) -> str:
    """Limits for a public visitor starting a run. Returns the session id."""
    sid = _sid(request)
    if not sid:
        raise HTTPException(401, "login required")
    now = time.time()
    # A visitor is a session AND a client address: signing in again makes a new session but not a new address,
    # so the per-visitor limits count both.
    client = _client_key(request)
    mine = [j for j in _jobs.values() if j.get("owner") == sid or (j.get("client") == client and j.get("owner") not in (None, "local"))]
    if any(j["status"] in ("queued", "running") for j in mine):
        raise HTTPException(429, "You already have a run in progress. Wait for it to finish.")
    if PUBLIC_MAX_PER_HOUR is not None and sum(1 for j in mine if now - j["created"] < 3600) >= PUBLIC_MAX_PER_HOUR:
        raise HTTPException(429, f"Limit of {PUBLIC_MAX_PER_HOUR} runs per hour reached. Try again later.")
    active = sum(1 for j in _jobs.values() if j.get("owner") not in (None, "local") and j["status"] in ("queued", "running"))
    if active >= PUBLIC_MAX_ACTIVE_TOTAL:
        raise HTTPException(429, "The demo is busy right now. Try again in a minute.")
    return sid


def _check_public_params(
    source: str,
    target: list[float] | None,
    radius: float,
    min_diameter: float,
    phantom_nodule: PhantomNodule | None = None,
) -> None:
    if source not in PUBLIC_SOURCES:
        raise HTTPException(403, "the public demo only has the synthetic phantom and the public AeroPath CTs")
    if not (1.0 <= radius <= 15.0):
        raise HTTPException(422, "nodule radius must be 1-15 mm in the public demo")
    if not (1.5 <= min_diameter <= 6.0):
        raise HTTPException(422, "scope diameter must be 1.5-6 mm in the public demo")
    if target is not None:
        if not all(math.isfinite(v) for v in target):
            raise HTTPException(422, "target must be finite numbers")
        if source == "phantom":
            for v, (lo, hi), name in zip(target, PHANTOM_BOUNDS, "xyz"):
                if not lo <= v <= hi:
                    raise HTTPException(422, f"target {name} must be within {lo:g}..{hi:g} mm (the phantom's volume)")
        elif any(abs(v) > 2000.0 for v in target):  # a real scan's frame is not fixed; the planner still checks the target is inside the lung
            raise HTTPException(422, "target is outside any plausible scan volume")
    if phantom_nodule is not None and source != "phantom":
        raise HTTPException(422, "a custom nodule only applies to the phantom")
    if phantom_nodule is not None:
        if not (1.0 <= phantom_nodule.radius_mm <= 15.0):
            raise HTTPException(422, "phantom nodule radius must be 1-15 mm in the public demo")
        if not (-900.0 <= phantom_nodule.hu <= 500.0):
            raise HTTPException(422, "phantom nodule hu must be between -900 and 500 in the public demo")
        if not all(math.isfinite(v) for v in phantom_nodule.center):
            raise HTTPException(422, "phantom nodule center must be finite numbers")
        for v, (lo, hi), name in zip(phantom_nodule.center, PHANTOM_BOUNDS, "xyz"):
            if not lo <= v <= hi:
                raise HTTPException(422, f"phantom nodule {name} must be within {lo:g}..{hi:g} mm (the phantom's volume)")


def _prune_public_jobs() -> None:
    """Delete finished public runs beyond the newest PUBLIC_KEEP_JOBS or older than PUBLIC_JOB_MAX_AGE_S."""
    now = time.time()
    pub = sorted(
        (j for j in _jobs.values() if j.get("owner") not in (None, "local") and j["status"] in ("done", "error")),
        key=lambda j: j["created"], reverse=True,
    )
    for n, j in enumerate(pub):
        if n >= PUBLIC_KEEP_JOBS or now - j["created"] > PUBLIC_JOB_MAX_AGE_S:
            shutil.rmtree(JOBS_DIR / j["id"], ignore_errors=True)
            _jobs.pop(j["id"], None)
            _edit.forget(j["id"])


_admit_lock = threading.Lock()


def _admit(request: Request, params: dict, cmd_for) -> dict:
    """Check the public limits and register the run as one step. Without the lock, parallel requests from one
    visitor all pass the "one run at a time" check before any of them is registered."""
    with _admit_lock:
        if _is_public(request):
            _enforce_public_quota(request)
        return _new_job(request, params, cmd_for)


def _new_job(request: Request, params: dict, cmd_for) -> dict:
    """Register and start a run (callers go through _admit). The owner is recorded on disk so the run keeps its
    owner across a server restart."""
    public = _is_public(request)
    owner = _sid(request) if public else "local"
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    cmd = cmd_for(job_id)
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / job_id / "owner.txt").write_text(owner or "orphan", encoding="utf-8")
    job = {"id": job_id, "status": "queued", "created": time.time(), "params": params, "result": None, "owner": owner,
           "client": _client_key(request) if public else None}
    _jobs[job_id] = job
    _prune_public_jobs()
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return job


@app.post("/api/jobs")
def create_job(req: JobRequest, request: Request) -> dict:
    if _is_public(request):
        _check_public_params(
            req.source,
            req.target if req.target_mode == "manual" else None,
            req.radius,
            req.min_diameter,
            phantom_nodule=req.phantom_nodule,
        )
        _enforce_public_quota(request)
    if req.source == "phantom" and req.phantom_nodule:
        from bronchnav.phantom import NoduleSpec, make_phantom

        try:
            spec = NoduleSpec(
                center_mm=tuple(float(v) for v in req.phantom_nodule.center),
                radius_mm=float(req.phantom_nodule.radius_mm),
                hu=float(req.phantom_nodule.hu),
            )
            make_phantom(nodule=spec)
        except ValueError as e:
            raise HTTPException(422, str(e))
    if req.source == "aeropath" and req.case not in _aeropath_cases():
        raise HTTPException(404, f"AeroPath case {req.case} not found under {AEROPATH_ROOT}")
    if req.source == "upload":
        _require_local(request)
    params = req.model_dump()
    if req.source == "upload":
        params["name"] = _load_meta(req.upload_id)["name"]
    return _public(_admit(request, params, lambda job_id: _command(job_id, req)))


# ---------------------------------------------------------------- editing the path (the "breadcrumbs")

class EndPoint(BaseModel):
    end: list[float] = Field(min_length=3, max_length=3)


class NewAirway(BaseModel):
    """An airway the operator drew on the CT: clicked points (physical LPS mm, the first on or beside an airway
    that was already found) and the radius they state for it."""

    points_mm: list[list[float]] = Field(min_length=2, max_length=MAX_AIRWAY_POINTS)
    radius_mm: float = Field(1.0, ge=AIRWAY_RADIUS_RANGE[0], le=AIRWAY_RADIUS_RANGE[1])

    @field_validator("points_mm")
    @classmethod
    def _points(cls, pts: list[list[float]]) -> list[list[float]]:
        for p in pts:
            if len(p) != 3 or not all(math.isfinite(v) for v in p):
                raise ValueError("every point needs three finite coordinates")
        return pts


class EditRun(EndPoint):
    camera_roll: float = Field(0.0, ge=-360.0, le=360.0)
    video: bool = True
    add_airway: NewAirway | None = None


def _edit_base(request: Request, job_id: str) -> tuple[dict, Path]:
    job = _jobs.get(job_id)
    if not _can_see(request, job):
        raise HTTPException(404, "unknown job")
    if job["status"] != "done":
        raise HTTPException(409, "the run is not finished")
    d = JOBS_DIR / job_id
    if not (d / "airway_mask.nii.gz").exists() or not (d / "route.json").exists():
        raise HTTPException(404, "this run has no airway to edit")
    return job, d


def _finite(end: list[float]) -> list[float]:
    if not all(math.isfinite(v) for v in end):
        raise HTTPException(422, "the end point must be finite numbers")
    return end


def _bundle(job_id: str, job: dict, d: Path):
    try:
        return _edit.load_bundle(job_id, d, float((job.get("params") or {}).get("min_diameter", 3.0)))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(422, str(e))


@app.get("/api/jobs/{job_id}/tree")
def job_tree(job_id: str, request: Request) -> dict:
    """The whole airway centerline (nodes, radii, edges) so the operator can pick another end point."""
    job, d = _edit_base(request, job_id)
    return _bundle(job_id, job, d).tree_payload()


@app.get("/api/jobs/{job_id}/alternatives")
def job_alternatives(job_id: str, request: Request, n: int = 6, search_mm: float = 45.0) -> list[dict]:
    job, d = _edit_base(request, job_id)
    return _bundle(job_id, job, d).alternatives(max(1, min(n, 10)), max(10.0, min(search_mm, 150.0)))


@app.post("/api/jobs/{job_id}/replan")
def job_replan(job_id: str, body: EndPoint, request: Request) -> dict:
    """Preview a route that ends at the airway point nearest `end` (physical LPS mm). Nothing is saved."""
    job, d = _edit_base(request, job_id)
    try:
        return _bundle(job_id, job, d).replan(_finite(body.end))
    except ValueError as e:
        raise HTTPException(422, str(e))


_airway_slot = threading.Semaphore(1)  # rebuilding a tree costs seconds of CPU: one preview at a time


@app.post("/api/jobs/{job_id}/addairway")
def job_add_airway(job_id: str, body: NewAirway, request: Request) -> dict:
    """Preview a route into an airway the operator drew on the CT. Nothing is saved."""
    job, d = _edit_base(request, job_id)
    min_d = float((job.get("params") or {}).get("min_diameter", 3.0))
    if not _airway_slot.acquire(blocking=False):
        raise HTTPException(429, "Another new-airway preview is running; try again in a moment.")
    try:
        return _edit.preview_added_airway(d, min_d, body.points_mm, body.radius_mm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        _airway_slot.release()


@app.post("/api/jobs/{job_id}/edits")
def create_edit(job_id: str, body: EditRun, request: Request) -> dict:
    """Save an edited path as its own run (new route, mesh, stills and video). With `add_airway`, the run is built
    on a mask that also holds the airway the operator drew, and the route ends at its far end."""
    job, d = _edit_base(request, job_id)
    if body.add_airway:
        body.end = list(body.add_airway.points_mm[-1])
    _finite(body.end)
    if _is_public(request):
        _enforce_public_quota(request)
    min_d = float((job.get("params") or {}).get("min_diameter", 3.0))
    params = {"source": "edit", "base_job": job_id, "base_source": _data_source(job), "end": body.end,
              "min_diameter": min_d, "camera_roll": body.camera_roll, "video": body.video}
    if body.add_airway:
        params["new_airway"] = body.add_airway.model_dump()

    def cmd_for(new_id: str) -> list[str]:
        cmd = [sys.executable, "-u", "-I", str(JOB_SCRIPT), "--source", "edit", "--out", str(JOBS_DIR / new_id),
               "--base-job-dir", str(d), "--end", *[str(v) for v in body.end], "--min-diameter", str(min_d)]
        if body.add_airway:
            spec = JOBS_DIR / new_id / "added_airway.json"
            spec.parent.mkdir(parents=True, exist_ok=True)
            spec.write_text(json.dumps(body.add_airway.model_dump()), encoding="utf-8")
            cmd += ["--add-airway-json", str(spec)]
        if body.camera_roll != 0.0:
            cmd += ["--camera-roll", str(body.camera_roll)]
        return cmd if body.video else cmd + ["--no-video"]

    return _public(_admit(request, params, cmd_for))


# Failures a visitor is told the reason for. The text shown is fixed (matched on the raw error, never copied
# from it), so a path or any other detail of the scan cannot leak through it. Anything else is "the run failed".
KNOWN_FAILURES = (
    ("Region growing already exceeds",
     "This scan cannot be segmented automatically: the lung tissue itself is as dark as air (typical of emphysema), "
     "so threshold growing floods the lung. A learned model is planned for cases like this. Try another case."),
    ("No trachea seed found",
     "The trachea could not be found in this scan. Check that the top of the chest is included."),
    ("seed is not in air at any tested threshold",
     "The airway could not be located in this scan."),
)
# AeroPath cases the classical segmenter is known to reject (shown in the case list so visitors are not surprised).
AEROPATH_UNSUPPORTED = {9: "emphysema: the lung is as dark as air"}


def _public_error(raw: object) -> str:
    text = str(raw or "")
    for key, message in KNOWN_FAILURES:
        if key in text:
            return message
    return "the run failed"


def _redacted(job: dict) -> dict:
    """What a public visitor may see of a run: no log, no tracebacks or raw error text (they can hold local paths)."""
    d = _public(job)
    r = d.get("result")
    if isinstance(r, dict):
        r = {k: v for k, v in r.items() if k not in ("traceback", "image_stack")}
        if "error" in r:
            r["error"] = _public_error(r["error"])
        d["result"] = r
    return d


@app.get("/api/jobs")
def list_jobs(request: Request) -> list[dict]:
    jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    if _is_public(request):
        return [_redacted(j) for j in jobs if _can_see(request, j)]
    return [_public(j) for j in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    if _is_public(request):
        if not _can_see(request, _jobs[job_id]):
            raise HTTPException(404, "unknown job")
        return _redacted(_jobs[job_id])
    return _public(_jobs[job_id], with_log=True)


def _restore_finished_jobs() -> None:
    """Re-list jobs from earlier server runs (their output folders persist)."""
    for d in JOBS_DIR.iterdir():
        res = d / "job_result.json"
        if d.is_dir() and res.exists() and d.name not in _jobs:
            try:
                r = json.loads(res.read_text(encoding="utf-8"))
            except Exception:
                continue
            owner_f = d / "owner.txt"
            owner = owner_f.read_text(encoding="utf-8").strip() if owner_f.exists() else "local"
            _jobs[d.name] = {
                "id": d.name,
                "status": "done" if r.get("ok") else "error",
                "created": d.stat().st_mtime,
                "params": {"source": r.get("source"), "case": r.get("case"), "name": r.get("name"),
                           "base_source": r.get("base_source"), "base_job": (r.get("edit") or {}).get("base_job")},
                "result": r,
                "owner": owner or "orphan",
            }


_restore_finished_jobs()
_prune_public_jobs()  # drop public runs left from earlier server sessions that are old or beyond the cap


@app.get("/files/{job_id}/{rest:path}")
def job_file(job_id: str, rest: str, request: Request):
    """Job outputs. Resolved inside JOBS_DIR/<job id> only; the uploads folder is a separate directory.
    A public visitor gets only an allowlist of files of public-data runs, and route.json without the input path."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(404, "not found")
    clean = posixpath.normpath(rest.replace("\\", "/"))
    if clean.startswith(("/", "..")) or clean == ".":
        raise HTTPException(404, "not found")
    public = _is_public(request)
    if public and (not _can_see(request, _jobs.get(job_id)) or not PUBLIC_FILE_RE.match(clean)):
        raise HTTPException(404, "not found")
    base = (JOBS_DIR / job_id).resolve()
    target = (base / clean).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "not found")
    if public and clean == "route.json":
        data = json.loads(target.read_text(encoding="utf-8"))
        data.pop("input", None)  # the local path of the CT
        return JSONResponse(data)
    return FileResponse(target)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")
