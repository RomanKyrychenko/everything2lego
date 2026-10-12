"""MILPPacker — rectangle exact-cover on a 2D mask via scipy's HiGHS.

Given a boolean mask M (cells that must be covered), a list of candidate
part footprints, and a per-cell colour map, find the placement set that:
    - covers every cell in M exactly once
    - covers no cell outside M
    - minimises total part count + λ × number of distinct SKUs

Falls back to a size-first greedy when the ILP is infeasible or times out.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csr_matrix
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


@dataclass
class Placement2D:
    """One 2D placement result: which part, where, what orientation."""
    part_index: int          # index into the part list passed to MILPPacker
    x: int                   # origin x
    z: int                   # origin z
    w: int                   # world footprint width  (after rotation)
    d: int                   # world footprint depth  (after rotation)
    rotation_deg: int


@dataclass
class PackResult:
    placements: list[Placement2D]
    method: str              # "milp" or "greedy"
    n_parts: int
    n_distinct_parts: int
    covered: bool = True
    solve_time_s: float = 0.0


class MILPPacker:
    """Solves rectangle exact-cover on a 2D mask using an MILP formulation.

    Usage
    -----
        packer = MILPPacker(parts=[(1,1), (2,1), (1,2), (2,2), (4,2)],
                            diversity_weight=0.01)
        result = packer.pack(mask)
        for p in result.placements:
            emit part `parts[p.part_index]` at (p.x, p.z) rotation p.rotation_deg
    """

    def __init__(self, parts: list[tuple[int, int]], *,
                 diversity_weight: float = 0.05,
                 time_limit_s: float = 10.0,
                 max_mask_area: int = 1000) -> None:
        # Deduplicate and store; we also generate rotated variants inside pack().
        self.parts = list(parts)
        self.diversity_weight = float(diversity_weight)
        self.time_limit_s = float(time_limit_s)
        # Above this many cells, skip MILP and go straight to greedy — the
        # variable count explodes and HiGHS is too slow.
        self.max_mask_area = int(max_mask_area)

    def pack(self, mask: np.ndarray) -> PackResult:
        """Pack the boolean mask (X, Z). Returns a PackResult."""
        import time
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        if area == 0:
            return PackResult(placements=[], method="empty",
                              n_parts=0, n_distinct_parts=0)

        # Enumerate all legal placements.
        cands = self._enumerate_placements(mask)
        if not cands:
            return PackResult(placements=[], method="empty",
                              n_parts=0, n_distinct_parts=0)

        t0 = time.time()
        # Skip MILP for very small or very large masks.
        if _HAS_MILP and area <= self.max_mask_area:
            try:
                result = self._solve_milp(mask, cands)
                if result is not None:
                    result.solve_time_s = time.time() - t0
                    return result
            except Exception as exc:
                warnings.warn(f"MILP packing failed ({exc}); using the greedy packer",
                              RuntimeWarning, stacklevel=2)
        result = self._greedy(mask, cands)
        result.solve_time_s = time.time() - t0
        return result

    # ------------------------------------------------------------------

    def _enumerate_placements(self, mask: np.ndarray
                                ) -> list[tuple[int, int, int, int, int, int]]:
        """Return legal placements as
        (part_index, x, z, w, d, rotation_deg).
        A placement is legal when its footprint lies entirely inside the mask.
        """
        X, Z = mask.shape
        out: list[tuple[int, int, int, int, int, int]] = []
        for i, (w0, d0) in enumerate(self.parts):
            orientations = [(w0, d0, 0)]
            if w0 != d0:
                orientations.append((d0, w0, 90))
            for pw, pd, rot in orientations:
                if pw > X or pd > Z:
                    continue
                # Precompute where a footprint of (pw, pd) fits.
                for x in range(X - pw + 1):
                    for z in range(Z - pd + 1):
                        if mask[x:x + pw, z:z + pd].all():
                            out.append((i, x, z, pw, pd, rot))
        return out

    def _solve_milp(self, mask: np.ndarray,
                    cands: list[tuple[int, int, int, int, int, int]]
                    ) -> Optional[PackResult]:
        """Build and solve the exact-cover ILP."""
        X, Z = mask.shape
        # Only cells in the mask are constrained. Map (x, z) -> row index.
        cell_ids = -np.ones((X, Z), dtype=np.int32)
        cells = list(zip(*np.where(mask)))
        for r, (x, z) in enumerate(cells):
            cell_ids[x, z] = r
        n_cells = len(cells)
        n_placements = len(cands)
        n_parts = len(self.parts)

        # Placement variables x[j] + auxiliary "part-in-use" variables y[i].
        n_vars = n_placements + n_parts
        # Objective: sum(x) + λ · sum(y)
        c = np.zeros(n_vars, dtype=np.float64)
        c[:n_placements] = 1.0
        c[n_placements:] = self.diversity_weight

        # Coverage rows: n_cells × n_vars
        cov = lil_matrix((n_cells, n_vars), dtype=np.float64)
        for j, (i, x, z, pw, pd, rot) in enumerate(cands):
            for dx in range(pw):
                for dz in range(pd):
                    cov[cell_ids[x + dx, z + dz], j] = 1.0
        # Link rows: y[i] × M >= sum_{j uses part i} x[j]
        # Rewrite as: sum_{j uses part i} x[j] − M · y[i] ≤ 0
        # M can be the maximum count of placements per part = n_cells (loose bound).
        big_m = float(n_cells + 1)
        link = lil_matrix((n_parts, n_vars), dtype=np.float64)
        for j, (i, *_rest) in enumerate(cands):
            link[i, j] = 1.0
        for i in range(n_parts):
            link[i, n_placements + i] = -big_m

        # Constraints: coverage == 1, link ≤ 0.
        constraints = [
            LinearConstraint(csr_matrix(cov), lb=1.0, ub=1.0),
            LinearConstraint(csr_matrix(link), lb=-np.inf, ub=0.0),
        ]
        integrality = np.ones(n_vars, dtype=np.int32)
        bounds = Bounds(lb=0, ub=1)

        opts = {"time_limit": self.time_limit_s, "disp": False,
                 "presolve": True, "mip_rel_gap": 0.01}
        res = milp(c, constraints=constraints, integrality=integrality,
                     bounds=bounds, options=opts)
        if not res.success or res.x is None:
            return None

        chosen = np.where(res.x[:n_placements] > 0.5)[0]
        placements = [
            Placement2D(part_index=cands[int(j)][0],
                          x=cands[int(j)][1], z=cands[int(j)][2],
                          w=cands[int(j)][3], d=cands[int(j)][4],
                          rotation_deg=cands[int(j)][5])
            for j in chosen
        ]
        distinct = len({p.part_index for p in placements})
        return PackResult(placements=placements, method="milp",
                          n_parts=len(placements),
                          n_distinct_parts=distinct)

    def _greedy(self, mask: np.ndarray,
                cands: list[tuple[int, int, int, int, int, int]]
                ) -> PackResult:
        """Size-first-across-positions greedy — the same approach as the
        v1 export packer, kept here as a certain-fit fallback for when the
        MILP times out or is infeasible."""
        filled = ~mask.copy()
        # Sort candidates by footprint area descending.
        cands_sorted = sorted(cands, key=lambda t: -(t[3] * t[4]))
        placements: list[Placement2D] = []
        for i, x, z, pw, pd, rot in cands_sorted:
            if filled[x:x + pw, z:z + pd].any():
                continue
            placements.append(Placement2D(part_index=i, x=x, z=z,
                                            w=pw, d=pd, rotation_deg=rot))
            filled[x:x + pw, z:z + pd] = True
        distinct = len({p.part_index for p in placements})
        return PackResult(placements=placements, method="greedy",
                          n_parts=len(placements),
                          n_distinct_parts=distinct)
