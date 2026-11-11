"""bronchnav command line.

  bronchnav info CT                  geometry + suitability warnings
  bronchnav phantom OUT.mha          write a synthetic CT with a known nodule (for testing)
  bronchnav plan CT --target X Y Z   segment, route to the target, render stills/video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import load_volume


def _parse_coords(val: str | list | tuple) -> list[float]:
    """Parse 3D coordinates from various formats.

    Accepts:
      - 3 floats/strings (e.g. ['-60.2', '-1.4', '-37.0'] or [-60.2, -1.4, -37.0])
      - A single comma-delimited or space-delimited string (e.g. '-60.2, -1.4, -37.0' or '-60.2 -1.4 -37.0')
    Returns list[float] of length 3.
    """
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            val = val[0]
        elif len(val) == 3:
            try:
                return [float(x) for x in val]
            except (ValueError, TypeError):
                pass
        if isinstance(val, (list, tuple)):
            val = " ".join(str(x) for x in val)

    if isinstance(val, str):
        parts = val.replace(",", " ").split()
        coords = [float(p) for p in parts]
        if len(coords) != 3:
            raise ValueError(f"Expected 3 coordinates (X, Y, Z), got {len(coords)} from '{val}'")
        return coords

    raise ValueError(f"Cannot parse coordinates from {val}")


def _cmd_info(args: argparse.Namespace) -> int:
    image, warnings = load_volume(args.path)
    print(f"size (x,y,z):    {image.GetSize()}")
    print(f"spacing (mm):    {tuple(round(s, 3) for s in image.GetSpacing())}")
    print(f"origin (mm):     {tuple(round(o, 2) for o in image.GetOrigin())}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("warnings:        none")
    return 0


def _cmd_phantom(args: argparse.Namespace) -> int:
    import SimpleITK as sitk

    from .phantom import make_phantom

    p = make_phantom(seed=args.seed, leak=args.leak)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(p.image, str(out))
    info = {"target_mm": [round(v, 2) for v in p.nodule_center_mm], "target_radius_mm": p.nodule_radius_mm}
    out.with_suffix(".target.json").write_text(json.dumps(info), encoding="utf-8")
    print(f"wrote {out}")
    print(f"nodule center (physical mm): {info['target_mm']}  radius: {info['target_radius_mm']}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .io.image_stack import SCAN_PRESETS, WINDOW_PRESETS
    from .pipeline import run_plan

    target = _parse_coords(args.target)
    origin = None if args.origin is None else _parse_coords(args.origin)
    stack = None
    if args.image_stack or args.preset or args.window_preset:
        px = args.pixel_spacing
        sl = args.slice_spacing
        wc = args.window_center
        ww = args.window_width
        if args.preset and args.preset in SCAN_PRESETS:
            p = SCAN_PRESETS[args.preset]
            if px is None:
                px = p["pixel_spacing_mm"]
            if sl is None:
                sl = p["slice_spacing_mm"]
        if args.window_preset and args.window_preset in WINDOW_PRESETS:
            w = WINDOW_PRESETS[args.window_preset]
            if args.window_center is None:
                wc = w["window_center"]
            if args.window_width is None:
                ww = w["window_width"]
        if wc is None:
            wc = -600.0
        if ww is None:
            ww = 1500.0
        if px is None or sl is None:
            raise SystemExit("--image-stack needs --pixel-spacing and --slice-spacing (mm), or a --preset (standard, hrct, lowdose, thick)")
        stack = dict(
            pixel_spacing_mm=px,
            slice_spacing_mm=sl,
            window_center=wc,
            window_width=ww,
            first_slice=args.first_slice,
            invert=args.invert,
        )
    res = run_plan(
        args.ct,
        np.array(target, dtype=float),
        target_radius_mm=args.target_radius,
        origin_mm_phys=None if origin is None else np.array(origin, dtype=float),
        out_dir=args.out,
        spacing_mm=args.spacing,
        min_diameter_mm=args.min_diameter,
        min_bend_radius_limit_mm=args.min_bend_radius,
        leak_factor=args.leak_factor,
        min_leak_jump_ml=args.min_leak_jump,
        fov_deg=args.fov,
        render=not args.no_render,
        video=not args.no_video,
        size=(args.width, args.height),
        speed_mm_s=args.speed,
        image_stack=stack,
    )
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in res.route.metrics.items()}, indent=1))
    for k, v in res.outputs.items():
        print(f"{k}: {Path(args.out) / v if args.out else v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bronchnav",
        description="Virtual bronchoscopy planning core (planning aid, not for clinical use).",
    )
    parser.add_argument("--version", action="version", version=f"bronchnav {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Load a CT and print geometry and suitability warnings.")
    info.add_argument("path", help="DICOM series folder or volume file (.mha/.nii.gz)")
    info.set_defaults(func=_cmd_info)

    ph = sub.add_parser("phantom", help="Write a synthetic airway phantom CT (testing only).")
    ph.add_argument("out", help="output .mha path")
    ph.add_argument("--seed", type=int, default=0)
    ph.add_argument("--leak", action="store_true", help="open one branch tip so region growing leaks")
    ph.set_defaults(func=_cmd_phantom)

    pl = sub.add_parser("plan", help="Plan a route to a target and render virtual bronchoscopy.")
    pl.add_argument("ct", help="DICOM series folder or volume file")
    pl.add_argument("--target", nargs="+", required=True, metavar="COORD",
                    help="nodule center in physical LPS mm (3 numbers, space- or comma-separated)")
    pl.add_argument("--target-radius", type=float, default=5.0, help="nodule radius in mm (default 5)")
    pl.add_argument("--origin", nargs="+", metavar="COORD",
                    help="route origin in physical mm (default: top of trachea)")
    pl.add_argument("--out", default="out", help="output directory")
    pl.add_argument("--spacing", type=float, default=0.75, help="isotropic working spacing in mm")
    pl.add_argument("--min-diameter", type=float, default=3.0, help="smallest airway the scope can enter (mm)")
    pl.add_argument("--min-bend-radius", type=float, default=12.0,
                    help="Minimum allowed bend radius in mm before warning")
    pl.add_argument("--fov", type=float, default=100.0,
                    help="Virtual bronchoscopy camera FOV in degrees")
    pl.add_argument("--leak-factor", type=float, default=2.0,
                    help="Relative volume jump threshold for leak detection")
    pl.add_argument("--min-leak-jump", type=float, default=30.0,
                    help="Minimum absolute volume jump in mL for leak detection")
    pl.add_argument("--no-render", action="store_true", help="skip stills/video/mesh")
    pl.add_argument("--no-video", action="store_true", help="skip the MP4 only")
    pl.add_argument("--width", type=int, default=640)
    pl.add_argument("--height", type=int, default=480)
    pl.add_argument("--speed", type=float, default=8.0, help="fly-through speed in mm/s")
    pl.add_argument("--image-stack", action="store_true",
                    help="treat CT as a folder of JPEG/PNG/BMP/TIFF slices (approximate; no DICOM needed)")
    pl.add_argument("--preset", choices=("standard", "hrct", "lowdose", "thick"),
                    help="resolution preset: standard (0.75x1.0mm), hrct (0.5x0.75mm), lowdose (0.8x1.25mm), thick (1.0x2.5mm)")
    pl.add_argument("--window-preset", choices=("lung", "mediastinum", "bone", "wide"),
                    help="display window preset: lung (-600/1500 HU), mediastinum (40/400 HU), bone (300/1500 HU), wide (0/2000 HU)")
    pl.add_argument("--pixel-spacing", type=float, help="image stack: in-plane pixel size in mm")
    pl.add_argument("--slice-spacing", type=float, help="image stack: distance between slices in mm")
    pl.add_argument("--window-center", type=float, default=None, help="image stack: display window centre (HU)")
    pl.add_argument("--window-width", type=float, default=None, help="image stack: display window width (HU)")
    pl.add_argument("--first-slice", choices=("auto", "head", "feet"), default="auto",
                    help="image stack: which end of the body the first image is")
    pl.add_argument("--invert", action="store_true", help="image stack: images are negatives (air is white)")
    pl.set_defaults(func=_cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
