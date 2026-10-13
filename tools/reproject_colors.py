"""Re-run photo→vertex colour projection on an existing mesh.

Lets you fix colours without redoing the (slow, memory-hungry) shape
reconstruction: load the mesh hunyuan saved, project the photo again with
the current `project_photo_colors` (back-face + background-hit detection,
nearest-known inpainting), and save.

Usage:
  reproject_colors.py <mesh.glb|stl> <photo> <dst.glb> [forward_axis=-z] [--flip-v]
"""
import sys
import trimesh
from everything2lego.voxelize import project_photo_colors

args = [a for a in sys.argv[1:] if not a.startswith("--")]
src, photo, dst = args[0], args[1], args[2]
forward = args[3] if len(args) > 3 else "-z"
flip_v = "--flip-v" in sys.argv

m = trimesh.load(src, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
out = project_photo_colors(m, photo, forward_axis=forward, flip_v=flip_v)
md = out.metadata
print(f"faces={len(out.faces)}  unknown={md.get('projection_unknown_frac', 0):.1%} "
      f"(back-facing {md.get('projection_back_frac', 0):.1%}, "
      f"background-hit {md.get('projection_bg_frac', 0):.1%}) → inpainted from nearest known")
out.export(dst)
print("saved:", dst)
