"""Operator path editing for the test console.

A finished job leaves its airway mask and route.json on disk. This module rebuilds the centerline tree from
the mask (about a second, then cached) so the operator can

  - see the whole tree and click any airway to end the route there (in an airway tree there is exactly one
    path to any point, so choosing a different end airway is choosing a different route),
  - get ranked alternative end points with their metrics,
  - preview any choice instantly (`replan`) and compare it with the automatic route.

All points in and out are physical LPS mm. Saving an edited path as its own run is done by the job runner
(core/scripts/plan_job.py --source edit), which also renders its stills and video.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.sparse import triu

from bronchnav.airway_tree import AirwayTree
from bronchnav.geometry import Grid
from bronchnav.pipeline import rebuild_from_job, route_payload
from bronchnav.planning import exit_candidates, plan_route

MAX_BUNDLES = 2
_cache: "OrderedDict[str, Bundle]" = OrderedDict()
_lock = threading.Lock()


@dataclass
class Bundle:
    grid: Grid
    tree: AirwayTree
    base_route: dict  # route.json of the job
    target_grid: np.ndarray
    radius_mm: float
    origin_grid: np.ndarray
    min_diameter_mm: float

    def tree_payload(self) -> dict:
        nodes = self.grid.points_to_physical(self.tree.pts_mm)
        up = triu(self.tree.adj, k=1).tocoo()
        end_grid = self.grid.points_to_grid(np.array(self.base_route["path"]["points_mm"][-1]))
        exit_node = int(np.argmin(np.linalg.norm(self.tree.pts_mm - end_grid, axis=1)))
        origin_node = int(np.argmin(np.linalg.norm(self.tree.pts_mm - self.origin_grid, axis=1)))
        return {
            "nodes": np.round(nodes, 1).tolist(),
            "radius_mm": np.round(self.tree.radius_mm, 2).tolist(),
            "edges": [[int(a), int(b)] for a, b in zip(up.row, up.col)],
            "origin_node": origin_node,
            "exit_node": exit_node,
        }

    def alternatives(self, n: int = 6, search_mm: float = 45.0) -> list[dict]:
        cands = exit_candidates(
            self.tree, self.target_grid, self.radius_mm, origin_mm=self.origin_grid,
            min_diameter_mm=self.min_diameter_mm, n=n, search_mm=search_mm,
        )
        for c in cands:
            c["point_mm"] = [round(float(v), 1) for v in self.grid.points_to_physical(np.array(c["point_mm"]))]
        return cands

    def replan(self, end_phys: list[float]) -> dict:
        force = self.grid.points_to_grid(np.array(end_phys, dtype=float))
        route = plan_route(
            self.tree, self.target_grid, self.radius_mm, origin_mm=self.origin_grid,
            min_diameter_mm=self.min_diameter_mm, force_exit_mm=force,
        )
        return route_payload(route, self.grid, self.tree)


def load_bundle(job_id: str, job_dir: Path, min_diameter_mm: float = 3.0) -> Bundle:
    with _lock:
        if job_id in _cache:
            _cache.move_to_end(job_id)
            return _cache[job_id]
        mask_path, route_path = job_dir / "airway_mask.nii.gz", job_dir / "route.json"
        if not mask_path.exists() or not route_path.exists():
            raise FileNotFoundError("this run has no airway mask or route to edit")
        route = json.loads(route_path.read_text(encoding="utf-8"))
        grid, _mask, tree, origin_grid, target_grid, radius = rebuild_from_job(sitk.ReadImage(str(mask_path)), route)
        b = Bundle(
            grid=grid,
            tree=tree,
            base_route=route,
            target_grid=target_grid,
            radius_mm=radius,
            origin_grid=origin_grid,
            min_diameter_mm=float(min_diameter_mm),
        )
        _cache[job_id] = b
        while len(_cache) > MAX_BUNDLES:
            _cache.popitem(last=False)
        return b


def forget(job_id: str) -> None:
    with _lock:
        _cache.pop(job_id, None)
