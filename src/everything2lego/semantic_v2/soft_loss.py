"""SoftLoss — differentiable loss over a SoftAssembly.

Terms (all end-to-end differentiable in PyTorch):

    shape_volume — mean squared error between soft occupancy and the
                   mesh voxel-grid occupancy.  Occupancy of cell c is the
                   probability that c's type ≠ EMPTY, i.e., 1 − p(EMPTY).

    color        — RGB-space L2 between soft-rendered colour at each cell
                   and the mesh's true colour at that cell.  Only counted
                   where the mesh IS occupied, weighted by soft occupancy
                   so we don't punish "empty cells have wrong colour".

    coherence    — Total variation on soft type probabilities across
                   face-adjacent cells.  Encourages contiguous same-type
                   regions.

    diversity    — Negative Shannon entropy of the type-usage histogram
                   summed over the grid.  Reward for spreading probability
                   mass across many types.

Every term is a plain torch operation; gradients flow through
SoftAssembly.type_logits and .color_logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .soft_assembly import SoftAssembly, N_TYPES, EMPTY


@dataclass
class SoftLossWeights:
    """Similarity dominates; the rest are tie-breakers.

    Rule: `shape_volume` + `color` (+ photo losses when enabled) drive
    the model toward the mesh.  `coherence`, `type_prior`, and `diversity`
    are weak enough that they can only pick a winner when the similarity
    terms are indifferent — they never overpower a similarity gradient.
    """
    # ---- similarity terms (dominant) ---------------------------------
    shape_volume: float = 2.0
    color:        float = 1.0
    photo_pixel:  float = 0.0    # weight of L2 pixel loss per rendered view
    photo_percep: float = 0.0    # weight of VGG perceptual loss per view
    # ---- tie-breakers (must stay << similarity terms) ----------------
    coherence:    float = 0.05   # smooth same-type neighbours when shape/
                                   # color are indifferent.  Larger same-type
                                   # runs → bigger MILP footprints → parts
                                   # that span course seams and stay attached.
    type_prior:   float = 0.05   # tiny nudge toward the geometric prior
                                   # (SLOPE at pitched-roof tips, TILE at
                                   # flat top, PLATE elsewhere).  Init
                                   # logits are pre-loaded from the prior,
                                   # so this weight only holds them in
                                   # place against small perturbations.
    diversity:    float = 0.0    # OFF — non-zero lets the trainer
                                   # reward-hack by randomising types.
    support:      float = 0.25   # penalise cells whose 'below' neighbour is
                                   # EMPTY.  Every non-ground occupied cell
                                   # should have something below to stud
                                   # onto.  The term is a mean over cells of
                                   # "floating" (typically 0.01-0.05), so
                                   # 0.25 puts it on par with the shape
                                   # term's residual rather than dominating.


@dataclass
class SoftLossBreakdown:
    shape_volume: float = 0.0
    color:        float = 0.0
    coherence:    float = 0.0
    diversity:    float = 0.0
    type_prior:   float = 0.0
    support:      float = 0.0
    photo_pixel:  float = 0.0
    photo_percep: float = 0.0
    total:        float = 0.0

    def as_line(self) -> str:
        s = (f"L={self.total:+.4f}  sV={self.shape_volume:.3f}  "
              f"col={self.color:.3f}  coh={self.coherence:.3f}  "
              f"div={self.diversity:+.3f}  prior={self.type_prior:.3f}  "
              f"sup={self.support:.3f}")
        if self.photo_pixel > 0 or self.photo_percep > 0:
            s += f"  photo_px={self.photo_pixel:.3f}  photo_p={self.photo_percep:.3f}"
        return s


def _build_geometric_type_prior(target_occ: np.ndarray,
                                  *,
                                  surface_points_vox: Optional[np.ndarray] = None,
                                  surface_normals: Optional[np.ndarray] = None,
                                  ) -> "torch.Tensor":
    """Geometry-aware type prior. Returns (Y, X, Z, N_TYPES) tensor.

    When both `surface_points_vox` (N,3 in voxel coords) and
    `surface_normals` (N,3 unit vectors in mesh frame) are supplied, the
    slope-direction assignment is refined by looking up each surface
    cell's nearest mesh sample and reading its outward normal, instead
    of inferring from empty-neighbour heuristics.  This produces cleaner
    pitched-roof edges (fewer stair-step artefacts) because the LEGO
    slope direction follows the actual mesh pitch.
    """
    from .soft_assembly import (EMPTY, PLATE, TILE,
                                    SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
                                    CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM,
                                    N_TYPES)
    Y, X, Z = target_occ.shape
    # Face-neighbour occupancy tests. neighbour_empty[direction] == True
    # means the face in that direction is exposed.
    ne = {}
    ne["+y"] = np.zeros_like(target_occ)
    ne["-y"] = np.zeros_like(target_occ)
    ne["+x"] = np.zeros_like(target_occ)
    ne["-x"] = np.zeros_like(target_occ)
    ne["+z"] = np.zeros_like(target_occ)
    ne["-z"] = np.zeros_like(target_occ)
    ne["+y"][:-1, :, :] = ~target_occ[1:,  :, :]
    ne["+y"][-1, :, :] = True   # boundary counts as empty
    ne["-y"][1:,  :, :] = ~target_occ[:-1, :, :]
    ne["-y"][0,  :, :] = True
    ne["+x"][:, :-1, :] = ~target_occ[:, 1:, :]
    ne["+x"][:, -1, :] = True
    ne["-x"][:, 1:,  :] = ~target_occ[:, :-1, :]
    ne["-x"][:, 0,  :] = True
    ne["+z"][:, :, :-1] = ~target_occ[:, :, 1:]
    ne["+z"][:, :, -1] = True
    ne["-z"][:, :, 1:]  = ~target_occ[:, :, :-1]
    ne["-z"][:, :, 0]  = True

    prior = np.full((Y, X, Z, N_TYPES), 0.005, dtype=np.float32)

    # Non-occupied cells: EMPTY.
    prior[~target_occ, EMPTY] = 1.0

    # Occupied interior cells (no face exposed): PLATE.
    exposed_any = (ne["+y"] | ne["-y"] | ne["+x"]
                    | ne["-x"] | ne["+z"] | ne["-z"])
    interior = target_occ & ~exposed_any
    prior[interior, PLATE] = 1.0

    # Surface cells.  Priority scheme:
    #   - Interior of a flat wall (only ONE side face exposed, no top/bottom)
    #       → PLATE  (physically it's just a vertical brick)
    #   - Top face exposed → TILE (mesh's upper surface)
    #   - Top face exposed AND a side face exposed → SLOPE_* pointing
    #     outward on that side (an eave / pitched-roof edge)
    #   - Bottom face exposed only → PLATE (invisible under model)
    surface = target_occ & exposed_any
    # Count how many faces of each cell are exposed.
    exposed_count = (ne["+y"].astype(np.int8) + ne["-y"].astype(np.int8)
                       + ne["+x"].astype(np.int8) + ne["-x"].astype(np.int8)
                       + ne["+z"].astype(np.int8) + ne["-z"].astype(np.int8))

    # Flat top of a solid slab: TILE.
    top_flat = surface & ne["+y"] & ~ne["+x"] & ~ne["-x"] & ~ne["+z"] & ~ne["-z"]
    prior[top_flat, TILE] = 1.0

    # A cell whose top face AND a side face are exposed = pitched-roof
    # tip.  Assign the SLOPE that faces the exposed side (LOW toward the
    # exposed direction).  The mesh normal there is up-and-out, which
    # matches how 3040 default (low = +x) is oriented.
    for open_key, slope_t, cheese_t in (
            ("+x", SLOPE_XP, CHEESE_XP),
            ("-x", SLOPE_XM, CHEESE_XM),
            ("+z", SLOPE_ZP, CHEESE_ZP),
            ("-z", SLOPE_ZM, CHEESE_ZM)):
        pitch_tip = surface & ne["+y"] & ne[open_key]
        prior[pitch_tip, slope_t] = 0.6
        prior[pitch_tip, cheese_t] = 0.2
        prior[pitch_tip, TILE] = 0.15
    # Wall cells (only side face(s) exposed, no top): PLATE (or PANEL).
    wall = surface & ~ne["+y"] & ~ne["-y"]
    prior[wall, PLATE] = 1.0
    # Bottom-only exposed cells: PLATE (hidden).
    bottom_only = surface & ne["-y"] & ~ne["+y"] & ~ne["+x"] \
                    & ~ne["-x"] & ~ne["+z"] & ~ne["-z"]
    prior[bottom_only, PLATE] = 1.0

    # Slope-from-normal refinement.  When mesh surface samples are
    # provided, for each surface voxel look up the nearest sample and
    # read its outward normal.  If it's tilted upward (ny in a band
    # away from 0 and 1) — a genuine roof pitch — pick the SLOPE_*
    # facing the horizontal projection of the normal; otherwise leave
    # the prior alone (walls/tiles stay as heuristic assigned).
    if surface_points_vox is not None and surface_normals is not None \
            and len(surface_points_vox) > 0:
        try:
            from scipy.spatial import cKDTree
            surf_cells = np.argwhere(surface)  # (M, 3) in (y, x, z)
            if len(surf_cells) > 0:
                # KDTree over (x, y, z) sample coords -> lookup at cell
                # centres (offset by 0.5 to hit the voxel middle).
                tree = cKDTree(surface_points_vox[:, [1, 0, 2]])
                cell_centres = surf_cells[:, [1, 0, 2]] + 0.5
                _, nn = tree.query(cell_centres, k=1)
                nvec = surface_normals[nn]  # (M, 3) in mesh frame
                ny = nvec[:, 1]
                # A cell is a "roof pitch" when the surface it belongs
                # to is tilted noticeably upward but not flat.  The band
                # 0.25 < ny < 0.85 catches ~15-75 degree pitches.
                pitch_mask = (ny > 0.25) & (ny < 0.85)
                if pitch_mask.any():
                    # Pick horizontal component with larger magnitude.
                    nh_x = nvec[:, 0]
                    nh_z = nvec[:, 2]
                    slope_choice = np.where(
                        np.abs(nh_x) >= np.abs(nh_z),
                        np.where(nh_x >= 0, SLOPE_XP, SLOPE_XM),
                        np.where(nh_z >= 0, SLOPE_ZP, SLOPE_ZM),
                    )
                    sel = surf_cells[pitch_mask]
                    slope_choice = slope_choice[pitch_mask]
                    # Reset that cell's prior and set the chosen slope
                    # to 0.75, its adjacent CHEESE to 0.15, TILE 0.05.
                    ys, xs, zs = sel[:, 0], sel[:, 1], sel[:, 2]
                    prior[ys, xs, zs, :] = 0.005
                    prior[ys, xs, zs, slope_choice] = 0.75
                    # Corresponding CHEESE_*:
                    cheese_of = {SLOPE_XP: CHEESE_XP, SLOPE_XM: CHEESE_XM,
                                   SLOPE_ZP: CHEESE_ZP, SLOPE_ZM: CHEESE_ZM}
                    for slope_t, cheese_t in cheese_of.items():
                        m = slope_choice == slope_t
                        if m.any():
                            prior[ys[m], xs[m], zs[m], cheese_t] = 0.15
                    prior[ys, xs, zs, TILE] = 0.05
        except ImportError:
            pass  # scipy unavailable; heuristic prior remains

    prior /= prior.sum(axis=-1, keepdims=True) + 1e-9
    return torch.from_numpy(prior)


class SoftLoss:
    """Differentiable loss for a SoftAssembly against a MeshTarget."""

    def __init__(self, target_occ: np.ndarray, target_color_idx: np.ndarray,
                  n_colors: int, weights: SoftLossWeights | None = None,
                  device: str = "cpu",
                  photo_views: dict | None = None,
                  renderer=None,
                  perceptual=None) -> None:
        assert _HAS_TORCH, "PyTorch required for SoftLoss"
        self.weights = weights or SoftLossWeights()
        self.device = device
        self.target_occ = torch.from_numpy(
            np.asarray(target_occ, dtype=np.float32)).to(device)
        # One-hot target colour (n_cells, n_colors).
        tci = np.clip(target_color_idx, 0, n_colors - 1)
        oh = np.zeros((*target_occ.shape, n_colors), dtype=np.float32)
        y, x, z = np.mgrid[0:target_occ.shape[0], 0:target_occ.shape[1],
                              0:target_occ.shape[2]]
        oh[y, x, z, tci] = 1.0
        self.target_color_oh = torch.from_numpy(oh).to(device)
        # photo_views: {axis: (H, W, 3) torch.Tensor} — target image per view.
        self.photo_views = photo_views or {}
        self.renderer = renderer
        self.perceptual = perceptual
        # Type prior: geometry-aware distribution.  For each cell, look at
        # which of its 6 face-neighbours are EMPTY — that tells us which
        # face of the LEGO cell is EXPOSED and hence which type is most
        # appropriate:
        #    Interior (all 6 neighbours occupied)     → PLATE
        #    +Y face empty (top exposed)              → TILE
        #    -Y face empty (bottom exposed only)      → PLATE (invisible)
        #    +X face empty (right side exposed)       → SLOPE_XP
        #    -X face empty                            → SLOPE_XM
        #    +Z face empty                            → SLOPE_ZP
        #    -Z face empty                            → SLOPE_ZM
        #    Two side faces exposed (corner cell)     → CHEESE_* (compact)
        # When several faces are exposed, we prefer the SLOPE direction
        # that matches the majority-exposed side; ties go to CHEESE.
        self.type_prior = _build_geometric_type_prior(target_occ).to(device)

    def evaluate(self, a: SoftAssembly, tau: float
                  ) -> tuple[SoftLossBreakdown, torch.Tensor]:
        w = self.weights
        # Soft occupancy = 1 - P(EMPTY).
        soft_types = a.soft_type_probs(tau)
        soft_occ = 1.0 - soft_types[..., EMPTY]

        # shape_volume: MSE against target occupancy.
        sv = ((soft_occ - self.target_occ) ** 2).mean()

        # color: soft-color vs target one-hot, weighted by soft occupancy.
        soft_col = a.soft_color_probs(tau)
        # Cross-entropy-ish: negative dot product (higher when colours match).
        col_dot = (soft_col * self.target_color_oh).sum(dim=-1)  # (Y,X,Z)
        # Loss = 1 - correlation, weighted by soft occupancy.
        col_err = ((1.0 - col_dot) * soft_occ).sum() / (soft_occ.sum() + 1e-6)

        # coherence: TV on soft types across 3 axes.
        d_y = (soft_types[1:, :, :, :] - soft_types[:-1, :, :, :]).abs().mean()
        d_x = (soft_types[:, 1:, :, :] - soft_types[:, :-1, :, :]).abs().mean()
        d_z = (soft_types[:, :, 1:, :] - soft_types[:, :, :-1, :]).abs().mean()
        coh = (d_y + d_x + d_z) / 3.0

        # diversity: entropy of TYPE-usage histogram.  Higher entropy = more
        # varied use.  Loss = -normalized entropy so lower loss ⇒ higher diversity.
        type_hist = soft_types.sum(dim=(0, 1, 2))
        type_hist = type_hist / (type_hist.sum() + 1e-9)
        entropy = -(type_hist * torch.log(type_hist + 1e-9)).sum()
        max_ent = float(np.log(N_TYPES))
        div = -entropy / max_ent

        # type_prior: KL(soft_types || target_prior). Small pull toward the
        # initial-state type distribution unless another term overrides it.
        prior_kl = (soft_types
                     * (torch.log(soft_types + 1e-9)
                          - torch.log(self.type_prior + 1e-9))
                     ).sum(dim=-1).mean()

        # support: penalise "floating" cells — occupied cells whose
        # BELOW-neighbour is EMPTY.  Ground layer (y = 0) is exempt (it
        # sits on the baseplate).  Slopes/cheeses whose flat face
        # points down are penalised the same way — the trainer picks
        # attachment-friendly types where the shape term is indifferent.
        below_occ = torch.zeros_like(soft_occ)
        below_occ[1:, :, :] = soft_occ[:-1, :, :]
        # Fully-occupied ground row is free from support penalty.
        ground_mask = torch.zeros_like(soft_occ)
        ground_mask[0, :, :] = 1.0
        floating = soft_occ * (1.0 - below_occ) * (1.0 - ground_mask)
        sup = floating.mean()

        total = (w.shape_volume * sv + w.color * col_err
                  + w.coherence * coh + w.diversity * div
                  + w.type_prior * prior_kl
                  + w.support * sup)

        # Photo losses — only compute when weights + views are set.
        photo_px = torch.zeros((), device=self.device)
        photo_perc = torch.zeros((), device=self.device)
        if self.renderer is not None and self.photo_views and (
                w.photo_pixel > 0 or w.photo_percep > 0):
            from .soft_renderer import photo_loss as _photo_loss
            for axis, tgt in self.photo_views.items():
                rendered = self.renderer.render(a, tau=tau, axis=axis)
                if w.photo_pixel > 0:
                    photo_px = photo_px + _photo_loss(rendered, tgt)
                if w.photo_percep > 0 and self.perceptual is not None:
                    photo_perc = photo_perc + self.perceptual.compute(
                        rendered, tgt)
            total = total + w.photo_pixel * photo_px + w.photo_percep * photo_perc

        breakdown = SoftLossBreakdown(
            shape_volume=float(sv.detach()),
            color=float(col_err.detach()),
            coherence=float(coh.detach()),
            diversity=float(div.detach()),
            type_prior=float(prior_kl.detach()),
            support=float(sup.detach()),
            photo_pixel=float(photo_px.detach()),
            photo_percep=float(photo_perc.detach()),
        )
        breakdown.total = float(total.detach())
        return breakdown, total
