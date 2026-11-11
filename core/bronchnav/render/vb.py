"""Surface-based virtual bronchoscopy renderer (VTK, offscreen).

Camera rides the smoothed route (grid-frame mm). Airway surface = marching cubes of the smoothed mask.
Overlays: green route line, distance readout, and a translucent "ghost" nodule that shows through the wall
once the camera is within `ghost_range_mm` of the target.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import struct

import numpy as np
import scipy.ndimage as ndi
import vtk
from vtk.util import numpy_support as ns

from ..geometry import Grid
from ..planning import Route


def mask_to_polydata(
    mask: np.ndarray,
    spacing_mm: float,
    smooth_iters: int = 20,
    target_reduction: float = 0.5,
    max_triangles: int = 300_000,
) -> vtk.vtkPolyData:
    """Airway mask (z, y, x) -> smoothed triangle mesh in grid-frame mm."""
    objs = ndi.find_objects(mask.astype(np.uint8))
    if not objs:
        raise ValueError("Cannot generate mesh from an empty airway mask")
    box = objs[0]
    m = 3
    sl = tuple(slice(max(0, s.start - m), min(d, s.stop + m)) for s, d in zip(box, mask.shape))
    # Zero-pad so a lumen touching the volume border (trachea at the scan edge) still gets a closed cap.
    pad = 2
    # sigma 0.5 keeps 1-voxel-wide distal tubes above the 0.5 iso-level (sigma 0.7 erased them).
    sub = ndi.gaussian_filter(np.pad(mask[sl].astype(np.float32), pad), sigma=0.5)
    nz, ny, nx = sub.shape

    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
    img.SetOrigin(
        (sl[2].start - pad) * spacing_mm, (sl[1].start - pad) * spacing_mm, (sl[0].start - pad) * spacing_mm
    )
    img.GetPointData().SetScalars(ns.numpy_to_vtk(sub.ravel(), deep=True, array_type=vtk.VTK_FLOAT))

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, 0.5)
    mc.ComputeNormalsOff()
    mc.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(mc.GetOutputPort())
    smooth.SetNumberOfIterations(smooth_iters)
    smooth.SetPassBand(0.1)
    smooth.BoundarySmoothingOff()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    cur_triangles = smooth.GetOutput().GetNumberOfPolys()
    reduction = target_reduction
    if max_triangles is not None and cur_triangles > max_triangles:
        needed = 1.0 - float(max_triangles) / float(cur_triangles)
        reduction = max(reduction, needed)

    if 0.0 < reduction < 1.0:
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(smooth.GetOutputPort())
        decimate.SetTargetReduction(reduction)
        decimate.Update()
        last_port = decimate.GetOutputPort()
    else:
        last_port = smooth.GetOutputPort()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(last_port)
    normals.SetFeatureAngle(120.0)
    normals.ConsistencyOn()
    # Marching cubes orients faces down the scalar gradient (out of the lumen); we view from inside, and the
    # exported PLY should face the lumen too.
    normals.FlipNormalsOn()
    normals.Update()
    return normals.GetOutput()


def _polyline(points: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points))
    for i in range(len(points)):
        lines.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _grid_to_physical_polydata(polydata: vtk.vtkPolyData, grid: Grid) -> vtk.vtkPolyData:
    """Transform polydata from grid-frame mm to physical LPS mm using Grid origin and direction."""
    mat = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            mat.SetElement(r, c, float(grid.direction[r, c]))
        mat.SetElement(r, 3, float(grid.origin[r]))
    mat.SetElement(3, 0, 0.0)
    mat.SetElement(3, 1, 0.0)
    mat.SetElement(3, 2, 0.0)
    mat.SetElement(3, 3, 1.0)

    transform = vtk.vtkTransform()
    transform.SetMatrix(mat)

    tf_filter = vtk.vtkTransformFilter()
    tf_filter.SetInputData(polydata)
    tf_filter.SetTransform(transform)
    tf_filter.Update()
    return tf_filter.GetPolyDataOutput()


def _gltf_to_glb(gltf_dict: dict) -> bytes:
    """Pack a glTF JSON dict with inline data URIs into binary GLB format."""
    buffers = gltf_dict.get("buffers", [])
    bin_data = bytearray()
    buffer_views = gltf_dict.get("bufferViews", [])
    buffer_offsets = []
    for b in buffers:
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            raw = base64.b64decode(uri.split(",", 1)[1])
        else:
            raw = b""
        pad = (4 - (len(bin_data) % 4)) % 4
        bin_data.extend(b"\x00" * pad)
        buffer_offsets.append(len(bin_data))
        bin_data.extend(raw)

    for bv in buffer_views:
        old_buf_idx = bv.get("buffer", 0)
        old_offset = bv.get("byteOffset", 0)
        bv["buffer"] = 0
        bv["byteOffset"] = buffer_offsets[old_buf_idx] + old_offset

    gltf_dict["buffers"] = [{"byteLength": len(bin_data)}]
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad
    bin_pad = (4 - (len(bin_data) % 4)) % 4
    bin_data.extend(b"\x00" * bin_pad)

    total_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_data))
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    chunk0_hdr = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # JSON
    chunk1_hdr = struct.pack("<II", len(bin_data), 0x004E4942)   # BIN
    return header + chunk0_hdr + json_bytes + chunk1_hdr + bytes(bin_data)


class VBRenderer:
    def __init__(
        self,
        mask: np.ndarray,
        spacing_mm: float,
        route: Route,
        size: tuple[int, int] = (640, 480),
        fov_deg: float = 100.0,
        ghost_range_mm: float = 50.0,
        approach_blend_mm: float = 0.0,
        camera_roll_deg: float = 0.0,
        smooth_iters: int = 20,
        target_reduction: float = 0.5,
        max_triangles: int = 300_000,
    ):
        self.route = route
        # yuv420p H.264 needs even dimensions
        self.size = (max(2, size[0] - size[0] % 2), max(2, size[1] - size[1] % 2))
        size = self.size
        self.ghost_range = ghost_range_mm
        self.approach_blend = approach_blend_mm
        self.camera_roll_deg = float(camera_roll_deg)
        self.mesh = mask_to_polydata(
            mask,
            spacing_mm,
            smooth_iters=smooth_iters,
            target_reduction=target_reduction,
            max_triangles=max_triangles,
        )

        ren = vtk.vtkRenderer()
        ren.SetBackground(0.02, 0.0, 0.0)
        ren.AutomaticLightCreationOff()
        ren.RemoveAllLights()
        self.ren = ren

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        mapper.ScalarVisibilityOff()
        wall = vtk.vtkActor()
        wall.SetMapper(mapper)
        p = wall.GetProperty()
        p.SetColor(0.86, 0.52, 0.46)
        p.SetAmbient(0.12)
        p.SetDiffuse(0.85)
        p.SetSpecular(0.45)
        p.SetSpecularPower(35)
        ren.AddActor(wall)

        # route guide line (thin emissive tube), rebuilt per frame to cover only the path ahead
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetInputData(_polyline(route.points_mm[:2]))
        self.tube.SetRadius(0.1)
        self.tube.SetNumberOfSides(6)
        self.tube.Update()
        tmap = vtk.vtkPolyDataMapper()
        tmap.SetInputConnection(self.tube.GetOutputPort())
        line = vtk.vtkActor()
        line.SetMapper(tmap)
        lp = line.GetProperty()
        lp.SetColor(0.1, 0.95, 0.35)
        lp.SetAmbient(1.0)
        lp.SetDiffuse(0.0)
        lp.LightingOff()
        ren.AddActor(line)
        self.line = line

        # headlight with distance falloff
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        light.SetPositional(True)
        light.SetPosition(0, 0, 0)
        light.SetFocalPoint(0, 0, -1)
        light.SetConeAngle(90)
        light.SetAttenuationValues(1.0, 0.0, 0.004)
        light.SetIntensity(1.3)
        ren.AddLight(light)

        cam = ren.GetActiveCamera()
        cam.SetViewAngle(fov_deg)
        cam.SetClippingRange(0.1, 500.0)

        # nodule ghost on an overlay layer sharing the camera (no depth test against the wall)
        ghost_ren = vtk.vtkRenderer()
        ghost_ren.SetLayer(1)
        ghost_ren.SetActiveCamera(cam)
        ghost_ren.InteractiveOff()
        ghost_ren.AutomaticLightCreationOn()
        sph = vtk.vtkSphereSource()
        sph.SetCenter(*route.target_mm)
        sph.SetRadius(max(route.target_radius_mm, 0.5))
        sph.SetThetaResolution(32)
        sph.SetPhiResolution(32)
        smap = vtk.vtkPolyDataMapper()
        smap.SetInputConnection(sph.GetOutputPort())
        self.ghost = vtk.vtkActor()
        self.ghost.SetMapper(smap)
        gp = self.ghost.GetProperty()
        gp.SetColor(1.0, 0.85, 0.1)
        gp.SetOpacity(0.4)
        gp.SetAmbient(0.6)
        gp.BackfaceCullingOn()  # camera inside a large nodule sphere should not be tinted yellow
        ghost_ren.AddActor(self.ghost)

        # end-of-path blue ball on the overlay layer
        end_sph = vtk.vtkSphereSource()
        end_sph.SetCenter(*route.points_mm[-1])
        end_sph.SetRadius(1.2)
        end_sph.SetThetaResolution(24)
        end_sph.SetPhiResolution(24)
        emap = vtk.vtkPolyDataMapper()
        emap.SetInputConnection(end_sph.GetOutputPort())
        self.end_ball = vtk.vtkActor()
        self.end_ball.SetMapper(emap)
        ep = self.end_ball.GetProperty()
        ep.SetColor(0.17, 0.51, 1.0)
        ep.SetAmbient(0.8)
        ep.SetDiffuse(0.2)
        ep.LightingOff()
        ghost_ren.AddActor(self.end_ball)

        self.ghost_ren = ghost_ren

        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetFontSize(16)
        self.text.GetTextProperty().SetColor(1, 1, 1)
        self.text.GetTextProperty().SetShadow(True)
        self.text.SetPosition(10, 10)
        ghost_ren.AddViewProp(self.text)

        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetMultiSamples(4)
        win.SetNumberOfLayers(2)
        win.SetSize(*size)
        win.AddRenderer(ren)
        win.AddRenderer(ghost_ren)
        self.win = win

        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(win)
        self.w2i.SetInputBufferTypeToRGB()
        self.w2i.ReadFrontBufferOff()

    # --- path sampling -------------------------------------------------
    def _at(self, s: float):
        r = self.route
        s = float(np.clip(s, 0.0, r.s_mm[-1]))
        p = np.array([np.interp(s, r.s_mm, r.points_mm[:, k]) for k in range(3)])
        T = np.array([np.interp(s, r.s_mm, r.tangents[:, k]) for k in range(3)])
        N = np.array([np.interp(s, r.s_mm, r.normals[:, k]) for k in range(3)])
        T /= np.linalg.norm(T)
        N = N - (N @ T) * T
        N /= np.linalg.norm(N)
        return p, T, N

    def frame(self, s_mm: float) -> np.ndarray:
        r = self.route
        s_mm = float(np.clip(s_mm, 0.0, r.s_mm[-1]))  # stills "lead_mm" before an early bifurcation can go < 0
        p, T, N = self._at(s_mm)
        to_target = r.target_mm - p
        dist_t = float(np.linalg.norm(to_target))
        L = float(r.s_mm[-1])
        # optional turn toward the target over the last `approach_blend` mm (0 = always follow the lumen)
        w = 0.0 if self.approach_blend <= 0 else 1.0 - _smoothstep((L - s_mm) / self.approach_blend)
        u_t = to_target / dist_t if dist_t > 1e-6 else T
        d = (1 - w) * T + w * u_t
        dn = float(np.linalg.norm(d))
        d = d / dn if dn > 1e-6 else T  # T and the target direction can cancel at w = 0.5
        up = N - (N @ d) * d
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(d, np.eye(3)[int(np.argmin(np.abs(d)))])
        up /= np.linalg.norm(up)

        if abs(self.camera_roll_deg) > 1e-3:
            rad = float(np.deg2rad(self.camera_roll_deg))
            up = up * np.cos(rad) + np.cross(d, up) * np.sin(rad)
            up /= np.linalg.norm(up)

        cam = self.ren.GetActiveCamera()
        cam.SetPosition(*p)
        cam.SetFocalPoint(*(p + d * 10.0))
        cam.SetViewUp(*up)
        cam.SetClippingRange(0.1, 500.0)

        # guide line: from 2.5 mm ahead of the camera up to 80 mm ahead
        ahead = (r.s_mm >= s_mm + 2.5) & (r.s_mm <= s_mm + 80.0)
        if ahead.sum() >= 2:
            self.tube.SetInputData(_polyline(r.points_mm[ahead]))
            self.line.SetVisibility(1)
        else:
            self.line.SetVisibility(0)

        self.ghost.SetVisibility(1 if dist_t < self.ghost_range else 0)
        dist_end = float(np.linalg.norm(r.points_mm[-1] - p))
        self.end_ball.SetVisibility(1 if dist_end < self.ghost_range else 0)
        gen = sum(1 for b in r.bifurcations if b["s_mm"] <= s_mm + 1e-6)
        self.text.SetInput(
            f"Target: {max(0.0, dist_t - r.target_radius_mm):5.1f} mm   "
            f"Path: {s_mm:5.1f}/{L:.0f} mm   Bifurcations passed: {gen}/{len(r.bifurcations)}   "
            f"Lumen: {2 * float(np.interp(s_mm, r.s_mm, r.radius_mm)):.1f} mm"
        )

        self.win.Render()
        self.w2i.Modified()
        self.w2i.Update()
        im = self.w2i.GetOutput()
        w_, h_, _ = im.GetDimensions()
        arr = ns.vtk_to_numpy(im.GetPointData().GetScalars()).reshape(h_, w_, -1)
        return np.ascontiguousarray(arr[::-1, :, :3])

    # --- outputs -------------------------------------------------------
    def render_stills(self, outdir: str | Path, lead_mm: float = 4.0) -> list[Path]:
        import imageio.v2 as imageio

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        files = []
        for i, b in enumerate(self.route.bifurcations, start=1):
            f = outdir / f"bifurcation_{i:02d}.png"
            imageio.imwrite(f, self.frame(b["s_mm"] - lead_mm))
            files.append(f)
        f = outdir / "end_of_route.png"
        imageio.imwrite(f, self.frame(float(self.route.s_mm[-1])))
        files.append(f)
        return files

    def schedule(self, fps: int, speed_mm_s: float, pause_s: float, end_hold_s: float) -> list[float]:
        L = float(self.route.s_mm[-1])
        step = speed_mm_s / fps
        pause_at = [max(0.0, b["s_mm"] - 4.0) for b in self.route.bifurcations]
        s_list: list[float] = []
        s = 0.0
        pending = sorted(pause_at)
        while s < L:
            s_list.append(s)
            if pending and s >= pending[0]:
                s_list.extend([s] * int(pause_s * fps))
                pending.pop(0)
            s += step
        s_list.extend([L] * int(end_hold_s * fps))
        return s_list

    def render_video(
        self,
        path: str | Path,
        fps: int = 30,
        speed_mm_s: float = 8.0,
        pause_s: float = 0.5,
        end_hold_s: float = 1.5,
        progress=None,
    ) -> Path:
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s_list = self.schedule(fps, speed_mm_s, pause_s, end_hold_s)
        w, h = self.size
        with imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p"
        ) as writer:
            for k, s in enumerate(s_list):
                writer.append_data(self.frame(s))
                if progress and k % 30 == 0:
                    progress(k, len(s_list))
        return path

    def export_mesh(self, path: str | Path, grid: Grid | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd = self.mesh if grid is None else _grid_to_physical_polydata(self.mesh, grid)
        wr = vtk.vtkPLYWriter()
        wr.SetFileName(str(path))
        wr.SetInputData(pd)
        wr.SetFileTypeToBinary()
        wr.Write()
        return path

    def export_glb(self, path: str | Path, grid: Grid | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd = self.mesh if grid is None else _grid_to_physical_polydata(self.mesh, grid)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(0.86, 0.52, 0.46)
        p.SetAmbient(0.12)
        p.SetDiffuse(0.85)
        p.SetSpecular(0.45)
        p.SetSpecularPower(35)

        ren = vtk.vtkRenderer()
        ren.AddActor(actor)
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.AddRenderer(ren)

        tmp_gltf = path.with_suffix(".tmp.gltf")
        exp = vtk.vtkGLTFExporter()
        exp.SetRenderWindow(win)
        exp.InlineDataOn()
        exp.SetFileName(str(tmp_gltf))
        exp.Write()

        try:
            with open(tmp_gltf, "r", encoding="utf-8") as f:
                gltf_json = json.load(f)
            glb_bytes = _gltf_to_glb(gltf_json)
            path.write_bytes(glb_bytes)
        finally:
            if tmp_gltf.exists():
                tmp_gltf.unlink(missing_ok=True)
        return path
