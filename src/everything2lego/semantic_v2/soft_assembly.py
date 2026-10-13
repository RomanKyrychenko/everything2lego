"""SoftAssembly — differentiable relaxation of a LEGO cell-grid.

Each cell is a soft categorical distribution over cell types and colors.
Gumbel-Softmax makes sampling differentiable so we can backprop the loss
into the per-cell logits. Anneal the temperature during training so the
distributions sharpen toward one-hot; discretise at the end.

Cell type vocabulary (11 types, deliberately compact for tensor economy):

    0: EMPTY
    1: PLATE
    2: TILE
    3–6: SLOPE_XP / SLOPE_XM / SLOPE_ZP / SLOPE_ZM (2×1×3 45° slopes)
    7–10: CHEESE_XP / CHEESE_XM / CHEESE_ZP / CHEESE_ZM (1×1×2 cheese slopes)

Each type maps to (LDraw part, footprint, height_plates) at discretisation
time.  Larger footprints (SLOPE = 2×1×3, CHEESE = 1×1×2) are represented
per-cell in the tensor — the discretiser handles the block claim by
finding contiguous same-type regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Cell-type integer constants (kept independent of v1's loss module).
EMPTY = 0
PLATE = 1
TILE = 2
SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM = 3, 4, 5, 6
CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM = 7, 8, 9, 10

N_TYPES = 11
TYPE_NAMES = ["empty", "plate", "tile",
               "slope_xp", "slope_xm", "slope_zp", "slope_zm",
               "cheese_xp", "cheese_xm", "cheese_zp", "cheese_zm"]


def _placement_to_type(pl) -> int:
    """Map a v1 Placement -> our compact cell-type enum.

    Slopes and cheese slopes get an orientation from the placement's
    rotation_deg and the part's slope_dir attribute (defaulting to +x).
    """
    part = pl.part
    shape = getattr(part, "shape", "plate")
    if shape == "tile":
        return TILE
    if shape in ("plate", "brick"):
        return PLATE
    if shape in ("slope", "slope_inv"):
        # Slope directions are defined relative to the base part's
        # slope_dir ("+x" by default).  Rotation modifies which world
        # axis the slope points down toward.
        base = getattr(part, "slope_dir", "+x") or "+x"
        rot = int(getattr(pl, "rotation_deg", 0)) % 360
        rot_map = {
            ("+x", 0):   SLOPE_XP,
            ("+x", 90):  SLOPE_ZP,
            ("+x", 180): SLOPE_XM,
            ("+x", 270): SLOPE_ZM,
            ("-x", 0):   SLOPE_XM,
            ("-x", 90):  SLOPE_ZM,
            ("-x", 180): SLOPE_XP,
            ("-x", 270): SLOPE_ZP,
            ("+z", 0):   SLOPE_ZP,
            ("+z", 90):  SLOPE_XM,
            ("+z", 180): SLOPE_ZM,
            ("+z", 270): SLOPE_XP,
            ("-z", 0):   SLOPE_ZM,
            ("-z", 90):  SLOPE_XP,
            ("-z", 180): SLOPE_ZP,
            ("-z", 270): SLOPE_XM,
        }
        if part.ldraw == "54200.dat":
            # Cheese: same direction encoding but the CHEESE_* variants.
            slope_to_cheese = {SLOPE_XP: CHEESE_XP, SLOPE_XM: CHEESE_XM,
                                 SLOPE_ZP: CHEESE_ZP, SLOPE_ZM: CHEESE_ZM}
            key = rot_map.get((base, rot), SLOPE_XP)
            return slope_to_cheese.get(key, CHEESE_XP)
        return rot_map.get((base, rot), SLOPE_XP)
    return PLATE


@dataclass
class SoftAssembly:
    """Learnable soft cell-grid.

    Shapes:
        type_logits : (Y, X, Z, N_TYPES)  float32   — per-cell type logits
        color_logits: (Y, X, Z, N_COLORS) float32   — per-cell colour logits

    Attributes are torch.nn.Parameter tensors on `device`.
    """
    type_logits: "torch.Tensor"
    color_logits: "torch.Tensor"
    device: str = "cpu"

    @property
    def Y(self) -> int: return int(self.type_logits.shape[0])
    @property
    def X(self) -> int: return int(self.type_logits.shape[1])
    @property
    def Z(self) -> int: return int(self.type_logits.shape[2])
    @property
    def N_COLORS(self) -> int: return int(self.color_logits.shape[-1])

    @staticmethod
    def from_target_occupancy(occ: np.ndarray, palette_idx: np.ndarray,
                                n_colors: int, *, device: str = "cpu",
                                geometric_prior: Optional["torch.Tensor"] = None,
                                allowed_palette: Optional[np.ndarray] = None,
                                ) -> "SoftAssembly":
        """Initial state:
        - If `geometric_prior` given: initialise type logits as
          `log(prior + eps) * 5` so the initial argmax matches the
          per-cell "ideal" type — surface cells start as TILE/SLOPE,
          interior as PLATE, empty as EMPTY.
        - Otherwise: fallback to the simple PLATE-everywhere-occupied
          scheme.
        Colour comes from the voxeliser's per-cell palette pick."""
        assert _HAS_TORCH, "PyTorch required for SoftAssembly"
        Y, X, Z = occ.shape
        occ_t = torch.from_numpy(occ.astype(np.bool_))
        if geometric_prior is not None:
            # Use log(prior) scaled so the argmax is sharp but not saturated.
            # Larger scale + smaller epsilon widens the initial logit gap
            # between the "right" type and the alternatives — reduces the
            # chance that Adam / coherence gradients can flip pitch-tip
            # cells from SLOPE_* back to PLATE.
            type_logits = 8.0 * torch.log(geometric_prior + 1e-4)
        else:
            # Fallback: PLATE-only initialisation.
            type_logits = torch.zeros(Y, X, Z, N_TYPES, dtype=torch.float32)
            type_logits[..., EMPTY] = torch.where(occ_t, torch.tensor(0.0),
                                                       torch.tensor(5.0))
            type_logits[..., PLATE] = torch.where(occ_t, torch.tensor(5.0),
                                                       torch.tensor(0.0))
        # Color logits: use voxeliser's palette pick per cell where occupied.
        color_logits = torch.zeros(Y, X, Z, n_colors, dtype=torch.float32)
        # scatter one-hot along last dim only where occupied
        pi_clip = np.clip(np.where(occ, palette_idx, 0),
                            0, n_colors - 1).astype(np.int64)
        pi_t = torch.from_numpy(pi_clip)
        color_logits.scatter_(-1, pi_t.unsqueeze(-1),
                                 torch.full_like(pi_t.unsqueeze(-1).float(), 5.0))
        # Optional palette lock: mask out palette indices that never
        # appear in the mesh's voxel target.  Prevents the trainer from
        # drifting into off-tone LDraw colours (e.g., picking Rust because
        # a slate roof cell is one LAB step closer to it than to Dark
        # Bluish Gray).  Very-negative initial logits stay negative under
        # Adam's finite step budget.
        if allowed_palette is not None:
            mask = np.ones(n_colors, dtype=bool)
            allowed = np.asarray(allowed_palette, dtype=np.int64)
            allowed = allowed[(allowed >= 0) & (allowed < n_colors)]
            if allowed.size:
                mask[allowed] = False
            color_logits[..., mask] = -1e10
        # Non-occupied cells still get some prior (won't matter — masked at loss).
        return SoftAssembly(
            type_logits=torch.nn.Parameter(type_logits.to(device)),
            color_logits=torch.nn.Parameter(color_logits.to(device)),
            device=device,
        )

    @staticmethod
    def from_placements(shape_yxz: tuple[int, int, int],
                          placements: list,
                          n_colors: int,
                          *,
                          device: str = "cpu",
                          geometric_prior: Optional["torch.Tensor"] = None,
                          allowed_palette: Optional[np.ndarray] = None,
                          per_region_palette: Optional[list] = None,
                          ) -> "SoftAssembly":
        """Bridge from the semantic-MILP layer to the soft trainer.

        Rasterises each Placement into a per-cell (type, color) grid and
        seeds the SoftAssembly's logits from that.  Where MILP produced
        a placement the corresponding cells inherit its type + color
        with a large positive logit; unclaimed cells fall back to
        `geometric_prior` (or PLATE-if-occupied).

        `per_region_palette` (optional) maps each cell (as a boolean
        mask) to an allowed set of palette indices, so the trainer can
        only choose from that region's LDraw shortlist — used for
        region-specific palette lock.
        """
        assert _HAS_TORCH, "PyTorch required for SoftAssembly"
        Y, X, Z = shape_yxz
        # Rasterise placements into (Y, X, Z) grids of type + color.
        rast_type = np.full((Y, X, Z), -1, dtype=np.int64)
        rast_col = np.zeros((Y, X, Z), dtype=np.int64)
        for pl in placements:
            t = _placement_to_type(pl)
            for dy in range(int(getattr(pl.part, "height_units", 1))):
                for dx in range(int(pl.w_world)):
                    for dz in range(int(pl.d_world)):
                        y = int(pl.y_plate) + dy
                        x = int(pl.x_stud) + dx
                        z = int(pl.z_stud) + dz
                        if 0 <= y < Y and 0 <= x < X and 0 <= z < Z:
                            rast_type[y, x, z] = t
                            rast_col[y, x, z] = int(pl.color_idx)
        claimed = rast_type >= 0
        occ_effective = claimed | (
            geometric_prior is not None
            and (geometric_prior.argmax(dim=-1).cpu().numpy() != EMPTY))

        # Type logits.  Base = geometric prior (log-scaled).  Where a
        # placement claimed the cell, add a strong bump for THAT type.
        if geometric_prior is not None:
            type_logits = 8.0 * torch.log(geometric_prior + 1e-4)
        else:
            type_logits = torch.zeros(Y, X, Z, N_TYPES, dtype=torch.float32)
            claim_t = torch.from_numpy(claimed)
            type_logits[..., EMPTY] = torch.where(
                claim_t, torch.tensor(-5.0), torch.tensor(5.0))
            type_logits[..., PLATE] = torch.where(
                claim_t, torch.tensor(3.0), torch.tensor(0.0))
        # Bump the placed type by +10 so it dominates over the prior.
        rast_t_torch = torch.from_numpy(np.where(claimed, rast_type, 0)
                                             .astype(np.int64))
        claim_mask = torch.from_numpy(claimed)
        # Scatter-add 10.0 at (y, x, z, rast_type).
        add = torch.zeros_like(type_logits)
        add.scatter_(-1, rast_t_torch.unsqueeze(-1),
                       torch.full_like(rast_t_torch.unsqueeze(-1).float(),
                                          10.0))
        type_logits = type_logits + add * claim_mask.unsqueeze(-1).float()

        # Color logits: one-hot the placement color where claimed,
        # otherwise leave zero (trainer will fit).
        color_logits = torch.zeros(Y, X, Z, n_colors, dtype=torch.float32)
        col_clip = np.clip(rast_col, 0, n_colors - 1).astype(np.int64)
        col_t = torch.from_numpy(col_clip)
        color_logits.scatter_(-1, col_t.unsqueeze(-1),
                                 torch.full_like(col_t.unsqueeze(-1).float(),
                                                    5.0))

        # Global palette lock.
        if allowed_palette is not None:
            mask = np.ones(n_colors, dtype=bool)
            allowed = np.asarray(allowed_palette, dtype=np.int64)
            allowed = allowed[(allowed >= 0) & (allowed < n_colors)]
            if allowed.size:
                mask[allowed] = False
            color_logits[..., mask] = -1e10

        # Per-region palette lock.  Region masks are bounding boxes and
        # overlap (a roof pitch's box spans most of a house), so the locks
        # are resolved as a UNION per cell: a cell inside one region is
        # locked to that region's colours; a cell inside several keeps
        # every candidate.  Applying them cumulatively would intersect the
        # sets and leave overlap cells with no allowed colour at all —
        # a saturated colour loss the optimiser can only relieve by
        # emptying the cell.  Cells in no region are unrestricted here
        # (the global lock above still applies).
        if per_region_palette:
            cell_allowed = np.zeros((Y, X, Z, n_colors), dtype=bool)
            covered = np.zeros((Y, X, Z), dtype=bool)
            for region_mask, allowed_ids in per_region_palette:
                rmask = np.asarray(region_mask, dtype=bool)
                if rmask.shape != (Y, X, Z):
                    continue
                aids = np.asarray(list(allowed_ids), dtype=np.int64)
                aids = aids[(aids >= 0) & (aids < n_colors)]
                if not aids.size:
                    continue
                for c in aids:
                    cell_allowed[..., int(c)] |= rmask
                covered |= rmask
            cell_allowed[~covered] = True
            # torch.where, not boolean-mask assignment: `t[mask] = v` runs
            # nonzero() and materialises int64 index tuples for every True
            # entry — with a tight per-cell lock that's ~Y·X·Z·n_colors
            # entries, hundreds of MB in one spike on a house-sized grid.
            color_logits = torch.where(torch.from_numpy(cell_allowed),
                                       color_logits,
                                       torch.tensor(-1e10, dtype=color_logits.dtype))

        return SoftAssembly(
            type_logits=torch.nn.Parameter(type_logits.to(device)),
            color_logits=torch.nn.Parameter(color_logits.to(device)),
            device=device,
        )

    def soft_type_probs(self, tau: float) -> "torch.Tensor":
        """Soft-argmax over types with temperature tau."""
        return F.softmax(self.type_logits / tau, dim=-1)

    def soft_color_probs(self, tau: float) -> "torch.Tensor":
        return F.softmax(self.color_logits / tau, dim=-1)

    def hard_types(self) -> np.ndarray:
        """Argmax types (used at discretisation)."""
        return self.type_logits.detach().argmax(dim=-1).cpu().numpy()

    def hard_colors(self) -> np.ndarray:
        return self.color_logits.detach().argmax(dim=-1).cpu().numpy()

    def parameters(self):
        return [self.type_logits, self.color_logits]
