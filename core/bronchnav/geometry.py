"""Coordinate frames.

grid frame:     x, y, z in mm = (index_x, index_y, index_z) * spacing   (used internally and for rendering)
physical frame: DICOM/ITK physical mm (LPS) = origin + direction @ grid  (used at the user/JSON boundary)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class Grid:
    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3, 3)
    spacing_mm: float

    @classmethod
    def from_image(cls, image: sitk.Image) -> "Grid":
        sp = image.GetSpacing()
        if not np.allclose(sp, sp[0], atol=1e-6):
            raise ValueError(f"Image is not isotropic: spacing={sp}")
        return cls(
            origin=np.array(image.GetOrigin(), dtype=np.float64),
            direction=np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3),
            spacing_mm=float(sp[0]),
        )

    def points_to_physical(self, pts_grid: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(pts_grid) @ self.direction.T

    def point_to_physical(self, pt_grid: np.ndarray) -> np.ndarray:
        return self.points_to_physical(pt_grid)

    def points_to_grid(self, pts_phys: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_phys) - self.origin) @ self.direction  # direction is orthonormal

    def vectors_to_physical(self, v_grid: np.ndarray) -> np.ndarray:
        return np.asarray(v_grid) @ self.direction.T
