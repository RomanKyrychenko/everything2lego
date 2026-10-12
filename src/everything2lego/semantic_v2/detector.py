"""RegionDetector — uses trimesh.facets as the atomic region unit.

`trimesh.Trimesh.facets` returns groups of face indices whose face normals
are within a small angular tolerance AND that are physically adjacent.
That's the RIGHT primitive to use for region detection: it's the mesh
telling us what its own planar surfaces are, not a clustering algorithm
guessing.

We augment that with:
  - single-face "facets" for the leftover triangles that didn't group
    (typical for organic curved meshes)
  - a min-Y "foundation" special-case (horizontal top-face group at ground
    level → FoundationRegion instead of PlanarRegion)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import trimesh

from .mesh_target import MeshTarget
from .primitive_fit import fit_cylinder, fit_sphere
from .regions import (Region, RegionLabel, PlanarRegion, CylindricalRegion,
                        SphericalRegion, CornerRegion, FoundationRegion,
                        DetailRegion)


@dataclass
class DetectorConfig:
    """Parameters for RegionDetector."""
    # Absolute minimum area (in the mesh's own coordinate system) for a
    # facet to become its own Region.  Kept for backwards-compat but
    # made small enough that it never dominates: the *relative* threshold
    # (`min_facet_area_frac`) is the one that matters in practice — it
    # scales with the mesh, so it works for a 200 mm gable and a
    # 1.0-unit Hunyuan mesh alike.
    min_facet_area_mm2: float = 0.0
    # Minimum facet area as a fraction of TOTAL mesh area.  A facet
    # smaller than this is considered surface detail and skipped from
    # region-typed detection; the leftover triangles feed the RANSAC
    # primitive fit or the DetailRegion fallback.  Default 0.5 % catches
    # true walls / roof pitches while ignoring the sub-pixel facets a
    # marching-cubes / Hunyuan mesh emits.
    min_facet_area_frac: float = 0.005
    # A horizontal facet whose max Y is within this fraction of the model's
    # Y-span from the bottom becomes the FoundationRegion.
    foundation_y_frac: float = 0.15
    # A facet's plane_normal.y >= this ⇒ horizontal top; <= -this ⇒ bottom.
    horizontal_ny_thresh: float = 0.85
    # A facet's plane_normal.y in [-thresh, +thresh] ⇒ vertical.
    vertical_ny_thresh: float = 0.15
    # RANSAC-lite: after facet-based classification, singleton triangles
    # (typically the tessellated curved surfaces) are grouped and fit for
    # sphere / cylinder primitives.  If the residual is below this fraction
    # of the primitive's radius, we accept the fit.
    primitive_fit_max_residual_frac: float = 0.06
    # Minimum face count to attempt a sphere/cylinder fit.
    min_primitive_faces: int = 20


class RegionDetector:
    """Classifies a MeshTarget's mesh into typed Regions using facets.

    A facet is a group of coplanar+adjacent faces trimesh identifies
    intrinsically. It's the physically-right unit for a "planar region" of
    the mesh — a real house wall is exactly one facet, a real roof pitch is
    exactly one facet, etc.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def detect(self, target: MeshTarget) -> List[Region]:
        mesh = target.mesh
        cfg = self.config
        regions: List[Region] = []

        facets = list(mesh.facets)         # each is a list of face indices
        # Faces NOT in any facet — trimesh only groups faces that agree
        # within its default angular tolerance. Curved meshes (spheres,
        # cylinders) leave nearly every face as a singleton; we set those
        # aside and try a primitive fit at the end.
        in_facet = set()
        for face_group in facets:
            for f in face_group:
                in_facet.add(int(f))
        curved_face_ids = [f for f in range(len(mesh.faces)) if f not in in_facet]

        y_min = float(mesh.bounds[0, 1])
        y_max = float(mesh.bounds[1, 1])
        y_span = max(1e-6, y_max - y_min)
        foundation_cutoff = y_min + cfg.foundation_y_frac * y_span

        # Sample-index KD-tree so each facet can grab its own point subset.
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(target.surface_points)
        except Exception as exc:
            warnings.warn(f"no KD-tree for facet sampling ({exc}); facets get no point subsets",
                          RuntimeWarning, stacklevel=2)
            tree = None

        # Relative area threshold — robust across mesh scales (mm-scale
        # gable meshes vs unit-normalised Hunyuan meshes).
        total_area = float(mesh.area_faces.sum())
        min_area_rel = cfg.min_facet_area_frac * total_area

        for face_ids in facets:
            face_ids = np.asarray(face_ids)
            area = float(mesh.area_faces[face_ids].sum())
            if area < max(cfg.min_facet_area_mm2, min_area_rel):
                continue

            # Facet normal (area-weighted mean).
            fn = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
            areas = np.asarray(mesh.area_faces[face_ids], dtype=np.float32)
            plane_normal = (fn * areas[:, None]).sum(axis=0)
            n_norm = float(np.linalg.norm(plane_normal))
            if n_norm < 1e-6:
                continue
            plane_normal /= n_norm

            # Facet centroid → plane_origin.
            face_verts = mesh.vertices[mesh.faces[face_ids]]  # (F, 3, 3)
            plane_origin = face_verts.reshape(-1, 3).mean(axis=0).astype(np.float32)

            # Grab points from target.surface_points that lie ON this facet.
            # Approximation: use faces' vertices as the point set (much
            # cheaper than mesh proximity queries and good enough for
            # per-region colour sampling).
            pts = face_verts.reshape(-1, 3).astype(np.float32)
            normals = np.broadcast_to(plane_normal, (len(pts), 3)).astype(np.float32)
            cols = _facet_colors(mesh, face_ids, len(pts))
            mn = pts.min(axis=0).astype(np.float32)
            mx = pts.max(axis=0).astype(np.float32)

            ny = float(plane_normal[1])
            # Classify the facet.
            if (ny > cfg.horizontal_ny_thresh
                    and pts[:, 1].max() <= foundation_cutoff):
                label = RegionLabel.FOUNDATION
                cls = FoundationRegion
            elif ny > cfg.horizontal_ny_thresh:
                label = RegionLabel.PLANAR_ROOF  # flat top of building
                cls = PlanarRegion
            elif ny < -cfg.horizontal_ny_thresh:
                # Bottom face of a solid — invisible, skip.
                continue
            elif abs(ny) < cfg.vertical_ny_thresh:
                label = RegionLabel.PLANAR_WALL
                cls = PlanarRegion
            else:
                # Tilted plane = pitched roof.
                label = RegionLabel.PLANAR_ROOF
                cls = PlanarRegion

            # Local (u, v) axes in the plane so builders can index 2D.
            tmp = (np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    if abs(plane_normal[1]) < 0.9
                    else np.array([1.0, 0.0, 0.0], dtype=np.float32))
            u_axis = np.cross(plane_normal, tmp)
            u_axis /= (np.linalg.norm(u_axis) + 1e-9)
            v_axis = np.cross(plane_normal, u_axis)
            v_axis /= (np.linalg.norm(v_axis) + 1e-9)

            regions.append(cls(
                label=label,
                points=pts,
                normals=normals,
                colors=cols,
                bounds=(mn, mx),
                metadata={
                    "facet_area_mm2": area,
                    "face_ids": face_ids.tolist(),
                    "facet_size": int(len(face_ids)),
                },
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                u_axis=u_axis.astype(np.float32),
                v_axis=v_axis.astype(np.float32),
            ))

        # Plane-fitting fallback for marching-cubes meshes.  Faces that
        # trimesh.facets left ungrouped often form NOISY-but-coplanar
        # pitched-roof surfaces: the true normal varies within ~5-15° of
        # the pitch normal, so strict co-planarity fails.  Cluster those
        # leftover faces by normal direction and emit each cluster as a
        # PLANAR_ROOF region when it covers a meaningful fraction of the
        # mesh.  This is what recovers the two roof pitches on the
        # Hunyuan gable cabin.
        if len(curved_face_ids) >= cfg.min_primitive_faces:
            planar_extra = self._cluster_leftover_planes(
                target, curved_face_ids, min_area_rel)
            regions.extend(planar_extra)
            # If we consumed a planar-cluster majority, don't try
            # sphere/cylinder afterwards — those would spuriously fit to
            # the residual noise.
            consumed = {f for r in planar_extra
                          for f in r.metadata.get("face_ids", [])}
            curved_face_ids = [f for f in curved_face_ids if f not in consumed]

        # Curved geometry: fit primitives to what's left.  Guard: only
        # attempt when the residual singleton set is a MAJORITY of the
        # mesh (a curved mesh leaves nearly all faces ungrouped), not a
        # handful of edge-cases from a planar mesh.
        residual_frac = (len(curved_face_ids) / max(1, len(mesh.faces)))
        if (len(curved_face_ids) >= cfg.min_primitive_faces
                and residual_frac >= 0.5):
            regions.extend(self._fit_curved_primitives(target, curved_face_ids))
        return regions

    def _cluster_leftover_planes(self, target: MeshTarget,
                                    face_ids: list[int],
                                    min_area_rel: float) -> List[Region]:
        """Group leftover faces by shared normal direction and emit a
        PLANAR_ROOF Region for each cluster that meets `min_area_rel`.
        Clustering uses a normal-space threshold of ~10 degrees.
        """
        cfg = self.config
        mesh = target.mesh
        face_ids = np.asarray(face_ids, dtype=np.int64)
        normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
        areas = np.asarray(mesh.area_faces[face_ids], dtype=np.float32)
        # Sort faces by area descending — largest faces seed clusters.
        order = np.argsort(-areas)
        cluster_id = np.full(len(face_ids), -1, dtype=np.int64)
        centroids: list[np.ndarray] = []
        cos_thresh = float(np.cos(np.deg2rad(10.0)))
        for i in order:
            n = normals[i]
            # Compare to each existing centroid (angular).
            best = -1
            best_cos = cos_thresh
            for k, c in enumerate(centroids):
                dot = float(np.dot(n, c))
                if dot > best_cos:
                    best = k
                    best_cos = dot
            if best < 0:
                centroids.append(n.copy())
                cluster_id[i] = len(centroids) - 1
            else:
                cluster_id[i] = best
                # Update centroid: area-weighted mean.
                pass  # keep centroid as first-seen normal; simpler.
        out: List[Region] = []
        for k in range(len(centroids)):
            mask = cluster_id == k
            if not mask.any():
                continue
            cluster_face_ids = face_ids[mask]
            cluster_area = float(mesh.area_faces[cluster_face_ids].sum())
            if cluster_area < min_area_rel:
                continue
            # Recompute a stable area-weighted normal for the cluster.
            fn = np.asarray(mesh.face_normals[cluster_face_ids],
                              dtype=np.float32)
            fa = np.asarray(mesh.area_faces[cluster_face_ids],
                              dtype=np.float32)
            plane_normal = (fn * fa[:, None]).sum(axis=0)
            nn = float(np.linalg.norm(plane_normal))
            if nn < 1e-6:
                continue
            plane_normal /= nn
            ny = float(plane_normal[1])
            # Only take UPWARD-tilted surfaces — roof pitches, not the
            # eave undersides (which have negative ny and would produce
            # a phantom lower roof overlapping the upper one).  Skip
            # pure-vertical (already a wall facet) and pure-flat (top).
            if ny < 0.25 or ny > 0.85:
                continue
            # Skip clusters whose normal is near-identical to one we've
            # already accepted (within 15°) — the same pitch sometimes
            # shows up twice from noise.
            dup = False
            for r in out:
                if float(np.dot(plane_normal, r.plane_normal)) > 0.966:  # cos(15°)
                    dup = True
                    break
            if dup:
                continue
            face_verts = mesh.vertices[mesh.faces[cluster_face_ids]] \
                            .reshape(-1, 3).astype(np.float32)
            plane_origin = face_verts.mean(axis=0).astype(np.float32)
            pts = face_verts
            normals_bc = np.broadcast_to(plane_normal, (len(pts), 3)) \
                              .astype(np.float32)
            cols = _facet_colors(mesh, cluster_face_ids, len(pts))
            mn = pts.min(axis=0).astype(np.float32)
            mx = pts.max(axis=0).astype(np.float32)
            tmp = (np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    if abs(plane_normal[1]) < 0.9
                    else np.array([1.0, 0.0, 0.0], dtype=np.float32))
            u_axis = np.cross(plane_normal, tmp)
            u_axis /= (np.linalg.norm(u_axis) + 1e-9)
            v_axis = np.cross(plane_normal, u_axis)
            v_axis /= (np.linalg.norm(v_axis) + 1e-9)
            out.append(PlanarRegion(
                label=RegionLabel.PLANAR_ROOF,
                points=pts, normals=normals_bc, colors=cols,
                bounds=(mn, mx),
                metadata={
                    "facet_area_mm2": cluster_area,
                    "face_ids": cluster_face_ids.tolist(),
                    "facet_size": int(len(cluster_face_ids)),
                    "recovered_from": "leftover-plane-cluster",
                },
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                u_axis=u_axis.astype(np.float32),
                v_axis=v_axis.astype(np.float32),
            ))
        return out

    # ---- curved-geometry primitives ---------------------------------------

    def _fit_curved_primitives(self, target: MeshTarget,
                                 face_ids: list[int]) -> List[Region]:
        """Attempt sphere then cylinder RANSAC on the leftover faces.

        v1 approach: fit ONE primitive to the whole leftover set. That's
        correct for a pure sphere or pure cylinder mesh (like sample.glb).
        For meshes mixing multiple curved primitives, a proper approach
        would iteratively fit-and-subtract; we treat that as Phase 4.5.
        """
        mesh = target.mesh
        face_ids = np.asarray(face_ids, dtype=np.int64)
        centroids = np.asarray(mesh.triangles_center[face_ids], dtype=np.float32)
        normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)

        cfg = self.config
        out: List[Region] = []

        sphere = fit_sphere(centroids, normals)
        cylinder = fit_cylinder(centroids, normals)

        def _residual_ok(fit, radius: float) -> bool:
            if radius <= 0:
                return False
            return fit.residual_mm / radius < cfg.primitive_fit_max_residual_frac

        # Prefer whichever primitive fits better (lower relative residual).
        s_score = (sphere.residual_mm / sphere.radius) if (sphere and sphere.radius) else 1e9
        c_score = (cylinder.residual_mm / cylinder.radius) if (cylinder and cylinder.radius) else 1e9
        picked = None
        if sphere and _residual_ok(sphere, sphere.radius) and s_score <= c_score:
            picked = ("sphere", sphere)
        elif cylinder and _residual_ok(cylinder, cylinder.radius):
            picked = ("cylinder", cylinder)

        # Build the corresponding Region.
        face_verts = mesh.vertices[mesh.faces[face_ids]].reshape(-1, 3).astype(np.float32)
        cols = _facet_colors(mesh, face_ids, len(face_verts))
        mn = face_verts.min(axis=0).astype(np.float32)
        mx = face_verts.max(axis=0).astype(np.float32)

        if picked and picked[0] == "sphere":
            fit = picked[1]
            out.append(SphericalRegion(
                label=RegionLabel.SPHERICAL,
                points=face_verts,
                normals=np.broadcast_to(np.array([0., 1., 0.], dtype=np.float32),
                                          (len(face_verts), 3)).astype(np.float32),
                colors=cols,
                bounds=(mn, mx),
                metadata={"n_faces": int(len(face_ids)),
                            "residual_mm": fit.residual_mm},
                center=fit.center, radius=float(fit.radius),
            ))
        elif picked and picked[0] == "cylinder":
            fit = picked[1]
            axis = fit.axis
            # Estimate height as the projection extent along the axis.
            proj = ((face_verts - fit.center) @ axis)
            height = float(proj.max() - proj.min())
            out.append(CylindricalRegion(
                label=RegionLabel.CYLINDRICAL,
                points=face_verts,
                normals=np.broadcast_to(axis, (len(face_verts), 3)).astype(np.float32),
                colors=cols,
                bounds=(mn, mx),
                metadata={"n_faces": int(len(face_ids)),
                            "residual_mm": fit.residual_mm},
                axis=axis, center=fit.center,
                radius=float(fit.radius), height=height,
            ))
        else:
            # No primitive fit: hand the whole thing to DetailRegion for the
            # GenericVoxelBuilder to handle.
            out.append(DetailRegion(
                label=RegionLabel.DETAIL,
                points=face_verts,
                normals=np.asarray(mesh.face_normals[face_ids], dtype=np.float32).repeat(3, axis=0)[:len(face_verts)],
                colors=cols,
                bounds=(mn, mx),
                metadata={"n_faces": int(len(face_ids)),
                            "reason": "no_primitive_fit"},
            ))
        return out


def _facet_colors(mesh: trimesh.Trimesh, face_ids: np.ndarray,
                    n_points: int) -> np.ndarray:
    """RGB per sample point on this facet. Uses vertex_colors if present,
    otherwise a mid-gray default."""
    try:
        vc = mesh.visual.vertex_colors  # (V, 4)
        # face_verts index — mesh.faces[face_ids] gives (F, 3) vertex indices
        vidx = mesh.faces[face_ids].reshape(-1)
        rgb = np.asarray(vc[vidx, :3], dtype=np.uint8)
        # Ensure length matches n_points.
        if len(rgb) >= n_points:
            return rgb[:n_points]
        pad = np.full((n_points - len(rgb), 3), 200, dtype=np.uint8)
        return np.concatenate([rgb, pad], axis=0)
    except Exception as exc:
        warnings.warn(f"colour sampling failed ({exc}); the region will be uniform grey",
                      RuntimeWarning, stacklevel=2)
        return np.full((n_points, 3), 200, dtype=np.uint8)
