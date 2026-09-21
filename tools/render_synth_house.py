"""Build a simple textured house mesh and render 8 turntable views of it.

Not required by the runtime — this exists only so we have realistic-looking
house photos to test the DUSt3R pipeline against on machines with no camera.
The house has enough distinct colored regions (windows, door, chimney, roof)
for DUSt3R's feature matching to find correspondences.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image


def _colored_box(size: tuple[float, float, float], color: tuple[int, int, int],
                  center: tuple[float, float, float],
                  noise: int = 0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    fc = np.tile(list(color) + [255], (len(m.faces), 1))
    if noise:
        rng = np.random.default_rng(abs(hash(center)) % 2**32)
        jitter = rng.integers(-noise, noise + 1, size=(len(m.faces), 3))
        fc[:, :3] = np.clip(fc[:, :3].astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    m.visual.face_colors = fc
    return m


def _textured_wall_grid(size: tuple[float, float, float], base: tuple[int, int, int],
                         center: tuple[float, float, float],
                         cells: tuple[int, int] = (8, 6),
                         noise: int = 22) -> list[trimesh.Trimesh]:
    """Subdivide a wall into a grid of small colored panels so DUSt3R has
    per-pixel features to match across views. Flat-shaded synthetic renders
    otherwise give it nothing to correspond."""
    sx, sy, sz = size
    cx, cy, cz = center
    nx, ny = cells
    tiles = []
    rng = np.random.default_rng(abs(hash(center)) % 2**32)
    for i in range(nx):
        for j in range(ny):
            px = cx - sx / 2 + (i + 0.5) * sx / nx
            py = cy - sy / 2 + (j + 0.5) * sy / ny
            jitter = rng.integers(-noise, noise + 1, size=3)
            c = tuple(int(np.clip(base[k] + jitter[k], 0, 255)) for k in range(3))
            tiles.append(_colored_box((sx / nx * 0.98, sy / ny * 0.98, 0.02),
                                       c, (px, py, cz)))
    return tiles


def _gable_roof(width: float, depth: float, base_y: float, peak_h: float,
                color: tuple[int, int, int]) -> trimesh.Trimesh:
    """Triangular-prism roof sitting on y = base_y, peak at y = base_y + peak_h."""
    half_w = width / 2
    half_d = depth / 2
    verts = np.array([
        [-half_w, base_y, -half_d],
        [ half_w, base_y, -half_d],
        [ half_w, base_y,  half_d],
        [-half_w, base_y,  half_d],
        [ 0, base_y + peak_h, -half_d],
        [ 0, base_y + peak_h,  half_d],
    ])
    faces = np.array([
        [0, 1, 4],           # front gable triangle
        [3, 5, 2],           # back gable triangle
        [0, 4, 5], [0, 5, 3],  # left slope
        [1, 2, 5], [1, 5, 4],  # right slope
        [0, 3, 2], [0, 2, 1],  # underside
    ])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.visual.face_colors = np.tile(list(color) + [255], (len(m.faces), 1))
    return m


def build_house() -> trimesh.Scene:
    """Two-story cottage with textured walls. The wall texture (an 8x6 grid of
    slightly-jittered stone-colored tiles per face) is what gives DUSt3R
    per-pixel features to correspond between views. Without it the shaded
    renders are too smooth for depth estimation."""
    scene_meshes = []
    wall_base = (232, 220, 190)
    # Interior box (so we can see through if the outer skin were transparent —
    # here just to occupy the volume).
    scene_meshes.append(_colored_box((11.6, 7.6, 9.6), wall_base, (0, 4, 0), noise=8))
    # Textured outer wall panels on all four sides.
    # Fine-grained tile grid: 24x16 tiles per wall so DUSt3R has abundant
    # distinct features to match at 512 downscale.
    fine = (24, 16)
    scene_meshes.extend(_textured_wall_grid((11.6, 7.6, 0.05), wall_base, (0, 4, 5.02),
                                             cells=fine, noise=14))
    scene_meshes.extend(_textured_wall_grid((11.6, 7.6, 0.05), wall_base, (0, 4, -5.02),
                                             cells=fine, noise=14))
    scene_meshes.extend(_textured_wall_grid((0.05, 7.6, 9.6), wall_base, (6.02, 4, 0),
                                             cells=(20, 16), noise=14))
    scene_meshes.extend(_textured_wall_grid((0.05, 7.6, 9.6), wall_base, (-6.02, 4, 0),
                                             cells=(20, 16), noise=14))
    # Roof: rust red, gable along +z axis.
    scene_meshes.append(_gable_roof(width=12, depth=10, base_y=8, peak_h=5,
                                     color=(160, 60, 40)))
    # Chimney.
    scene_meshes.append(_colored_box((1.6, 4, 1.6), (110, 60, 50),
                                      (3.5, 10.5, 0), noise=20))
    # Front door.
    scene_meshes.append(_colored_box((1.4, 3.2, 0.15), (100, 55, 25),
                                      (0, 1.6, 5.06)))
    # Windows on all four walls.
    for x in (-3.5, 3.5):
        for z in (-5.06, 5.06):
            scene_meshes.append(_colored_box((1.6, 1.6, 0.06),
                                              (155, 175, 200), (x, 3.5, z)))
        for zside in (-2, 2):
            scene_meshes.append(_colored_box((0.06, 1.6, 1.6),
                                              (155, 175, 200),
                                              (6.06 if x > 0 else -6.06,
                                               3.5, zside)))
    # Textured ground so DUSt3R has features to anchor scale/pose.
    scene_meshes.extend(_textured_wall_grid((30, 0.02, 30), (120, 155, 90),
                                             (0, -0.05, 0), cells=(20, 20), noise=25))
    return trimesh.util.concatenate(scene_meshes)


def render_views(mesh: trimesh.Trimesh, out_dir: Path, *,
                 n_views: int = 8, size: int = 640, radius: float = 26.0,
                 height: float = 8.0) -> list[Path]:
    scene = pyrender.Scene(bg_color=(0.75, 0.82, 0.9, 1.0),
                            ambient_light=(0.35, 0.35, 0.35))
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pr_mesh)

    # Directional light at a fixed sun angle (not turntable — so the light
    # baked into pixels varies view-to-view, which is how real photos look).
    sun = pyrender.DirectionalLight(color=(1.0, 0.95, 0.85), intensity=4.0)
    sun_pose = np.eye(4)
    sun_pose[:3, :3] = trimesh.transformations.euler_matrix(-math.pi / 4,
                                                             math.pi / 6, 0)[:3, :3]
    scene.add(sun, pose=sun_pose)

    camera = pyrender.PerspectiveCamera(yfov=math.radians(45.0), aspectRatio=1.0)
    cam_node = scene.add(camera)

    renderer = pyrender.OffscreenRenderer(size, size)
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        cx = radius * math.cos(theta)
        cz = radius * math.sin(theta)
        # Camera looks at (0, height, 0) from (cx, height, cz).
        eye = np.array([cx, height, cz])
        target = np.array([0.0, 5.5, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        pose = _look_at(eye, target, up)
        scene.set_pose(cam_node, pose)
        color, _ = renderer.render(scene)
        img = Image.fromarray(color)
        p = out_dir / f"view_{i:02d}.png"
        img.save(p)
        paths.append(p)
    renderer.delete()
    return paths


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL-style look-at camera pose (world_from_camera)."""
    forward = eye - target                       # camera looks along -forward
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/synth_house", type=Path)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    mesh = build_house()
    print(f"house mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"bounds {mesh.bounds.tolist()}")
    paths = render_views(mesh, args.out, n_views=args.views, size=args.size)
    print(f"wrote {len(paths)} views to {args.out}")
