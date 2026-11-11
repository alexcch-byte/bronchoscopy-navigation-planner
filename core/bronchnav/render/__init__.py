"""Virtual bronchoscopy rendering. Requires the optional `render` extra (vtk, imageio, imageio-ffmpeg)."""

from .vb import VBRenderer, mask_to_polydata

__all__ = ["VBRenderer", "mask_to_polydata"]
