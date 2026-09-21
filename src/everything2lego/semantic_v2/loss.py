"""LossFunction — measures how well an Assembly matches the MeshTarget.

Operates on a list[Placement] against a MeshTarget. Returns a scalar loss
plus a term-by-term breakdown for debugging.

Terms (all normalised to ~[0, 1] scale so weights are meaningful):

    shape_surface  — mean chamfer distance from LEGO surface points to the
                     nearest point on the *original* mesh surface (normalised
                     by stud size)
    shape_volume   — 1 − IoU between LEGO occupancy and mesh voxel occupancy
    color          — mean LAB error at surface placements against the
                     mesh's true (pre-quantisation) colour at each point
    count          — placement count normalised against mesh voxel volume
    diversity      — negative: fraction of catalog part-types actually used
                     (reward variety)
    coherence      — fraction of face-adjacent placements sharing colour +
                     shape family (reward contiguity)

Every term has an explicit knob in `LossWeights`; the composite is a
weighted sum, which is what SA minimises.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .. import bricks, colors
from ..pack import Placement
from .mesh_target import MeshTarget


@dataclass
class LossWeights:
    shape_surface: float = 1.0
    shape_volume: float = 0.5
    color:         float = 0.5
    count:         float = 0.001
    diversity:     float = 0.1
    coherence:     float = 0.2


@dataclass
class LossBreakdown:
    shape_surface: float = 0.0
    shape_volume:  float = 0.0
    color:         float = 0.0
    count:         float = 0.0
    diversity:     float = 0.0
    coherence:     float = 0.0
    total:         float = 0.0

    def as_line(self) -> str:
        return (f"L={self.total:+.4f}  sS={self.shape_surface:.3f}  "
                f"sV={self.shape_volume:.3f}  col={self.color:.3f}  "
                f"div={self.diversity:+.3f}  coh={self.coherence:.3f}  "
                f"cnt={self.count:.3f}")


class LossFunction:
    """Callable — score an Assembly relative to a MeshTarget.

    Caches KDTree of mesh surface samples on the target for cheap chamfer.
    """

    def __init__(self, target: MeshTarget, weights: Optional[LossWeights] = None) -> None:
        self.target = target
        self.weights = weights or LossWeights()
        from scipy.spatial import cKDTree
        self._mesh_kd = cKDTree(target.surface_points)
        # Palette RGB as int32 for L2 in RGB (Lab is stricter but this is a proxy).
        pal = colors.palette()
        self._palette_rgb = np.array([c.rgb for c in pal], dtype=np.int32)
        # Grid-space bounds for fast IoU.
        occ = target.voxel_grid.occupied()
        self._target_occ = occ

    # ------------- evaluate -------------------------------------------------

    def evaluate(self, placements: Iterable[Placement]
                  ) -> tuple[LossBreakdown, LossBreakdown]:
        """Returns (raw, weighted)."""
        placements = list(placements)
        w = self.weights
        raw = LossBreakdown()
        if not placements:
            raw.total = 0.0
            return raw, raw

        raw.shape_surface = self._shape_surface(placements)
        raw.shape_volume  = self._shape_volume(placements)
        raw.color         = self._color(placements)
        raw.count         = self._count(placements)
        raw.diversity     = self._diversity(placements)
        raw.coherence     = self._coherence(placements)

        weighted = LossBreakdown(
            shape_surface=w.shape_surface * raw.shape_surface,
            shape_volume=w.shape_volume  * raw.shape_volume,
            color=w.color        * raw.color,
            count=w.count        * raw.count,
            diversity=w.diversity    * raw.diversity,
            coherence=w.coherence    * raw.coherence,
        )
        weighted.total = (weighted.shape_surface + weighted.shape_volume
                           + weighted.color + weighted.count
                           + weighted.diversity + weighted.coherence)
        return raw, weighted

    # ------------- term implementations ------------------------------------

    def _shape_surface(self, placements: list[Placement]) -> float:
        """Mean mm distance from placements' surface centres to nearest mesh
        surface point, in stud-sized units."""
        S = bricks.MM_PER_STUD
        P = bricks.MM_PER_PLATE
        Tinv = np.linalg.inv(self.target.mesh_to_voxel)
        # Sample the top-centre of each placement in voxel-index space,
        # then map to mesh-mm via the inverse transform.
        homos = []
        for p in placements:
            cx = p.x_stud + p.w_world / 2
            cy = p.y_plate + p.part.height_units
            cz = p.z_stud + p.d_world / 2
            homos.append([cx, cy, cz, 1.0])
        H = np.asarray(homos, dtype=np.float32)
        world = (Tinv @ H.T).T[:, :3]
        d, _ = self._mesh_kd.query(world)
        return float(d.mean() / S)

    def _shape_volume(self, placements: list[Placement]) -> float:
        """1 − IoU of placements' occupancy vs target voxel_grid."""
        occ = np.zeros_like(self._target_occ, dtype=bool)
        Y, X, Z = occ.shape
        for p in placements:
            for dy in range(p.part.height_units):
                yi = p.y_plate + dy
                if not (0 <= yi < Y):
                    continue
                for dx in range(p.w_world):
                    xi = p.x_stud + dx
                    if not (0 <= xi < X):
                        continue
                    for dz in range(p.d_world):
                        zi = p.z_stud + dz
                        if 0 <= zi < Z:
                            occ[yi, xi, zi] = True
        inter = int((occ & self._target_occ).sum())
        union = int((occ | self._target_occ).sum())
        return 1.0 - (inter / union if union > 0 else 1.0)

    def _color(self, placements: list[Placement]) -> float:
        """Mean per-placement RGB L2 error vs the mesh's actual colour at
        the nearest surface sample. Normalised to [0, 1] by dividing by
        the diagonal of RGB cube (441.6)."""
        # Nearest mesh sample index for each placement centre.
        Tinv = np.linalg.inv(self.target.mesh_to_voxel)
        centres_vox = np.array([[p.x_stud + p.w_world / 2,
                                    p.y_plate + p.part.height_units / 2,
                                    p.z_stud + p.d_world / 2]
                                    for p in placements], dtype=np.float32)
        homos = np.concatenate([centres_vox, np.ones((len(centres_vox), 1))], axis=1)
        world = (Tinv @ homos.T).T[:, :3]
        _, idx = self._mesh_kd.query(world)
        target_rgb = self.target.surface_colors[idx].astype(np.int32)
        placement_rgb = self._palette_rgb[[p.color_idx for p in placements]]
        d = np.linalg.norm(placement_rgb - target_rgb, axis=1)
        return float(d.mean() / 441.6)

    def _count(self, placements: list[Placement]) -> float:
        n_cells = max(1, int(self._target_occ.sum()))
        return float(len(placements)) / n_cells

    def _diversity(self, placements: list[Placement]) -> float:
        """Negative fraction of the catalog actually used — reward variety."""
        used = {p.part.ldraw for p in placements}
        catalog_size = max(1, len(bricks.catalog()))
        return -float(len(used)) / catalog_size

    def _coherence(self, placements: list[Placement]) -> float:
        """Fraction of face-adjacent placement pairs that DON'T share
        (colour, shape). Lower = more contiguous same-material regions.

        Approximates adjacency via bucket lookup on cell overlap — cheap
        for the placement counts we care about (< 5000)."""
        if len(placements) < 2:
            return 0.0
        # Bucket placements by (color, shape) then count boundary pairs.
        # For simplicity, count adjacent placements sharing top plate.
        pairs = 0
        diffs = 0
        by_pos: dict[tuple[int, int, int], int] = {}
        for i, p in enumerate(placements):
            for dx in range(p.w_world):
                for dz in range(p.d_world):
                    by_pos[(p.y_plate, p.x_stud + dx, p.z_stud + dz)] = i
        seen: set[tuple[int, int]] = set()
        for (y, x, z), i in by_pos.items():
            for dy, dx, dz in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nb = by_pos.get((y, x + dx, z + dz))
                if nb is None or nb == i:
                    continue
                key = (min(i, nb), max(i, nb))
                if key in seen:
                    continue
                seen.add(key)
                pairs += 1
                pi, pn = placements[i], placements[nb]
                if pi.color_idx != pn.color_idx or pi.part.shape != pn.part.shape:
                    diffs += 1
        return (diffs / pairs) if pairs else 0.0
