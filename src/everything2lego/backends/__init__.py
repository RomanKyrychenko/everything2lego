"""Pluggable image -> voxel-grid backends.

Every backend implements a single function:

    def build(image_paths: Sequence[str | Path], *, max_studs: int, **kwargs) -> VoxelGrid

Backends currently available:
- `carve`  - classical multi-view silhouette space carving. Deterministic, CPU only.
- `triposr` - single-image neural reconstruction (Stability AI TripoSR).
  Runs on MPS in an arm64 Python venv; falls back to CPU.
- `hybrid` - TripoSR mesh intersected with the multi-view silhouette hull:
  neural fills in concavities, silhouettes tighten the outline.
- `dust3r` - unstructured multi-view photos (any angles, real-world subjects
  like houses). Naver DUSt3R estimates camera poses + dense point cloud;
  the point cloud is voxelized directly. Weights are CC-BY-NC.
- `hunyuan3d` - stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ..carve import carve as _carve
from ..voxelize import VoxelGrid


def _stub(name: str):
    def _f(*_a, **_kw):
        raise NotImplementedError(
            f"backend '{name}' is not implemented yet."
        )
    return _f


def _lazy(module: str, attr: str = "build"):
    """Import-on-first-call so backends with heavy deps (torch, TripoSR) don't
    break users who only want the carve/mesh path."""
    def _f(*args, **kwargs):
        from importlib import import_module
        mod = import_module(module)
        return getattr(mod, attr)(*args, **kwargs)
    return _f


Backend = Callable[[Sequence[Path]], VoxelGrid]

BACKENDS: dict[str, Callable] = {
    "carve": _carve,
    "triposr": _lazy("everything2lego.backends.triposr"),
    "hybrid": _lazy("everything2lego.backends.hybrid"),
    "dust3r": _lazy("everything2lego.backends.dust3r"),
    "hunyuan3d": _lazy("everything2lego.backends.hunyuan3d"),
    "hunyuan3d_mv": _lazy("everything2lego.backends.hunyuan3d_mv"),
    "hunyuan3d_v21": _lazy("everything2lego.backends.hunyuan3d_v21"),
}


def build(name: str, image_paths: Sequence[str | Path], **kwargs) -> VoxelGrid:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](image_paths, **kwargs)
