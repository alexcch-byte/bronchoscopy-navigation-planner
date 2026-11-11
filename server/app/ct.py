"""CT viewer endpoints for the local test console: open a CT, serve windowed slices, report HU.

Local requests may open any CT. A proxied (public) request may only open and read the synthetic phantom: main.py's
gate lets /api/ct/ through after the passcode login, and `guard_open` / `guard_use` refuse every other source.
One or two volumes stay in memory as int16, shared by all callers.

Orientation: volumes are LPS (the loader applies DICOMOrient), so the array is (z, y, x) with x toward the
patient's left, y toward posterior and z toward the head. An axial slice arr[k] already shows in radiological
convention (patient's left on the right of the picture, anterior at the top); coronal and sagittal slices are
flipped vertically so the head is at the top.
"""

from __future__ import annotations

import io
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

MAX_CACHED = 2
PLANES = ("axial", "coronal", "sagittal")


@dataclass
class CT:
    ct_id: str
    key: str
    name: str
    arr: np.ndarray  # (z, y, x) int16 HU
    spacing: tuple[float, float, float]  # (x, y, z) mm
    origin: tuple[float, float, float]
    direction: tuple[float, ...]  # 3x3 row-major, ITK
    hint: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def size_xyz(self) -> tuple[int, int, int]:
        return (self.arr.shape[2], self.arr.shape[1], self.arr.shape[0])

    def public(self) -> dict:
        lo, hi = float(self.arr.min()), float(self.arr.max())
        return {
            "ct_id": self.ct_id,
            "name": self.name,
            "size": list(self.size_xyz),
            "spacing": [round(s, 4) for s in self.spacing],
            "origin": [round(o, 4) for o in self.origin],
            "direction": [round(d, 6) for d in self.direction],
            "hu_range": [lo, hi],
            "hint": self.hint,
        }


class NoduleIn(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)
    radius_mm: float = Field(5.0, ge=1.0, le=15.0)
    hu: float = Field(30.0, ge=-900.0, le=500.0)


class CTOpen(BaseModel):
    source: str = Field("phantom", pattern="^(phantom|aeropath|upload)$")
    case: int = Field(1, ge=1, le=27)
    upload_id: str | None = None
    stack: dict | None = None
    nodule: NoduleIn | None = None


def _to_ct(image: sitk.Image, ct_id: str, key: str, name: str, hint: dict | None = None) -> CT:
    image = sitk.DICOMOrient(image, "LPS")
    arr = np.clip(np.rint(sitk.GetArrayFromImage(image)), -32768, 32767).astype(np.int16)
    return CT(
        ct_id=ct_id,
        key=key,
        name=name,
        arr=arr,
        spacing=tuple(float(s) for s in image.GetSpacing()),
        origin=tuple(float(o) for o in image.GetOrigin()),
        direction=tuple(float(d) for d in image.GetDirection()),
        hint=hint,
    )


def register(
    app: FastAPI,
    *,
    aeropath_root: Callable[[], object],
    load_upload_meta: Callable[[str], dict],
    guard_open: Callable[[Request, object], None],
    guard_use: Callable[[Request, CT], None],
) -> None:
    cache: "OrderedDict[str, CT]" = OrderedDict()
    lock = threading.Lock()

    def get(ct_id: str) -> CT:
        ct = cache.get(ct_id)
        if ct is None:
            raise HTTPException(404, "unknown or expired CT: open it again")
        cache.move_to_end(ct_id)
        return ct

    @app.post("/api/ct/open")
    def open_ct(req: CTOpen, request: Request) -> dict:
        guard_open(request, req)
        if req.source == "upload":
            meta = load_upload_meta(req.upload_id or "")
            if meta["kind"] == "images" and not req.stack:
                raise HTTPException(422, "an image series needs stack = pixel/slice spacing and window")
            key = f"upload:{req.upload_id}:{sorted((req.stack or {}).items())}"
        elif req.source == "phantom" and req.nodule:
            c = [round(float(v), 2) for v in req.nodule.center]
            key = f"phantom:{c[0]},{c[1]},{c[2]}:{round(float(req.nodule.radius_mm), 1)}:{round(float(req.nodule.hu), 1)}"
        else:
            key = f"{req.source}:{req.case if req.source == 'aeropath' else ''}"
        with lock:  # one load at a time; a repeat request reuses the cached volume
            for ct in cache.values():
                if ct.key == key:
                    cache.move_to_end(ct.ct_id)
                    return ct.public()
            ct_id = uuid.uuid4().hex[:12]
            if req.source == "phantom":
                from bronchnav.phantom import NoduleSpec, make_phantom

                spec = None
                if req.nodule:
                    try:
                        spec = NoduleSpec(
                            center_mm=tuple(float(v) for v in req.nodule.center),
                            radius_mm=float(req.nodule.radius_mm),
                            hu=float(req.nodule.hu),
                        )
                        ph = make_phantom(nodule=spec)
                    except ValueError as e:
                        raise HTTPException(422, str(e))
                else:
                    ph = make_phantom()
                hint = {
                    "target_mm": [round(float(v), 2) for v in ph.nodule_center_mm],
                    "radius_mm": float(ph.nodule_radius_mm),
                    "label": "the custom nodule" if spec else "the phantom's nodule",
                    "hu": float(ph.nodule_hu),
                }
                name = "Synthetic phantom (custom nodule)" if spec else "Synthetic phantom"
                ct = _to_ct(ph.image, ct_id, key, name, hint)
            elif req.source == "aeropath":
                from pathlib import Path

                p = Path(str(aeropath_root())) / str(req.case) / f"{req.case}_CT_HR.nii.gz"
                if not p.exists():
                    raise HTTPException(404, f"AeroPath case {req.case} not found")
                from bronchnav.io import load_volume

                img, _ = load_volume(p)
                ct = _to_ct(img, ct_id, key, f"AeroPath case {req.case}")
            else:
                meta = load_upload_meta(req.upload_id or "")
                try:
                    if meta["kind"] == "images":
                        from bronchnav.io import load_image_stack

                        img, _ = load_image_stack(meta["path"], **req.stack)
                    else:
                        from bronchnav.io import load_volume

                        img, _ = load_volume(meta["path"])
                except (ValueError, TypeError) as e:
                    raise HTTPException(400, f"cannot load this CT: {e}")
                ct = _to_ct(img, ct_id, key, meta["name"])
            cache[ct_id] = ct
            while len(cache) > MAX_CACHED:
                cache.popitem(last=False)
            return ct.public()

    @app.get("/api/ct/{ct_id}/slice")
    def slice_png(ct_id: str, request: Request, plane: str = "axial", i: int = 0, wc: float = -600.0, ww: float = 1500.0):
        if plane not in PLANES:
            raise HTTPException(422, f"plane must be one of {PLANES}")
        ct = get(ct_id)
        guard_use(request, ct)
        nx, ny, nz = ct.size_xyz
        n = {"axial": nz, "coronal": ny, "sagittal": nx}[plane]
        if not 0 <= i < n:
            raise HTTPException(422, f"index {i} outside 0..{n - 1} for the {plane} plane")
        ww = max(1.0, min(ww, 20000.0))
        a = ct.arr[i] if plane == "axial" else (ct.arr[:, i, :][::-1] if plane == "coronal" else ct.arr[:, :, i][::-1])
        g = np.clip((a.astype(np.float32) - (wc - ww / 2.0)) / ww, 0.0, 1.0)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray((g * 255.0 + 0.5).astype(np.uint8), mode="L").save(buf, format="PNG", compress_level=1)
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/ct/{ct_id}/hu")
    def hu(ct_id: str, request: Request, i: int, j: int, k: int) -> dict:
        ct = get(ct_id)
        guard_use(request, ct)
        nx, ny, nz = ct.size_xyz
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise HTTPException(422, "voxel outside the volume")
        return {"hu": int(ct.arr[k, j, i])}
