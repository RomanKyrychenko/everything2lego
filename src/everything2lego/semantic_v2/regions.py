"""Region abstractions.

A Region is a semantic subset of the mesh surface — a wall, a roof plane,
a cylinder wall, a dome cap, etc. RegionDetector emits typed Region
subclasses; RegionBuilder subclasses dispatch on them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RegionLabel(str, Enum):
    PLANAR_WALL = "planar_wall"
    PLANAR_ROOF = "planar_roof"
    FOUNDATION = "foundation"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    CORNER = "corner"
    DETAIL = "detail"
    UNCLASSIFIED = "unclassified"


@dataclass
class Region(ABC):
    """Abstract base class for a semantic region of the mesh."""

    label: RegionLabel
    # Sample points inside this region (mesh mm)
    points: np.ndarray                   # (N, 3) float32
    normals: np.ndarray                  # (N, 3) float32
    colors: np.ndarray                   # (N, 3) uint8
    # Bounding box in mesh mm, (mins, maxs) each shape (3,)
    bounds: tuple[np.ndarray, np.ndarray]
    metadata: dict = field(default_factory=dict)

    def size_mm(self) -> np.ndarray:
        mn, mx = self.bounds
        return mx - mn

    def n_points(self) -> int:
        return int(len(self.points))

    @abstractmethod
    def characteristic_normal(self) -> np.ndarray:
        """The single normal that best describes this region (for dispatch)."""


@dataclass
class PlanarRegion(Region):
    """A flat surface — wall, roof plane, floor."""

    plane_normal: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    plane_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Local 2-D axes for the plane, so builders can index cells in (u, v).
    u_axis: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0], np.float32))
    v_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1], np.float32))

    def characteristic_normal(self) -> np.ndarray:
        return self.plane_normal

    @property
    def is_horizontal(self) -> bool:
        return abs(float(self.plane_normal[1])) > 0.85

    @property
    def is_vertical(self) -> bool:
        return abs(float(self.plane_normal[1])) < 0.15

    @property
    def is_tilted_up(self) -> bool:
        """True for any pitched (non-vertical, non-horizontal) plane. Uses
        |ny| so both normal-out (roof) and normal-in (mis-wound mesh face)
        pitches are recognised — the builder recovers the pitch direction
        from the raw normal sign."""
        ny = float(self.plane_normal[1])
        return 0.15 <= abs(ny) <= 0.9


@dataclass
class CylindricalRegion(Region):
    """A cylinder — round tower, pillar, drum."""

    axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], np.float32))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    radius: float = 0.0
    height: float = 0.0

    def characteristic_normal(self) -> np.ndarray:
        # Radial "outward" from center — approximate as x direction.
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class SphericalRegion(Region):
    """A sphere or dome."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    radius: float = 0.0

    def characteristic_normal(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class CornerRegion(Region):
    """A junction between two planes (an outer or inner corner)."""

    incident_normals: list[np.ndarray] = field(default_factory=list)

    def characteristic_normal(self) -> np.ndarray:
        if not self.incident_normals:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n = np.mean(self.incident_normals, axis=0)
        norm = float(np.linalg.norm(n))
        return (n / norm) if norm > 1e-6 else n


@dataclass
class FoundationRegion(PlanarRegion):
    """The horizontal slab at the model's ground level (special-case wall)."""

    def characteristic_normal(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class DetailRegion(Region):
    """Fallback: mesh geometry we couldn't fit into a primitive family."""

    def characteristic_normal(self) -> np.ndarray:
        if len(self.normals) == 0:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n = np.mean(self.normals, axis=0)
        norm = float(np.linalg.norm(n))
        return (n / norm) if norm > 1e-6 else n
