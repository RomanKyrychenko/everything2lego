"""Quantise a mesh's vertex colours to K clusters (kmeans in RGB space)
so shading gradients collapse to a few flat regions.  Removes the
scattered per-voxel colour noise the LEGO discretiser inherits from the
Hunyuan mesh's per-vertex shaded texture projection.

Usage:
  quantise_colors.py <src.glb> <dst.glb> <K>
"""
import sys
import numpy as np
import trimesh
from sklearn.cluster import KMeans

src, dst = sys.argv[1], sys.argv[2]
K = int(sys.argv[3])

m = trimesh.load(src, force="mesh")
if hasattr(m, "to_geometry"):
    m = m.to_geometry()
elif hasattr(m, "dump"):
    m = m.dump(concatenate=True)

print("verts:", len(m.vertices), "kind:", m.visual.kind)

vc = np.asarray(m.visual.vertex_colors)  # (V, 4) uint8 RGBA
rgb = vc[:, :3].astype(np.float32)

km = KMeans(n_clusters=K, n_init=8, random_state=0)
labels = km.fit_predict(rgb)
centroids = km.cluster_centers_.astype(np.uint8)  # (K, 3)

print("cluster sizes (top 8):",
      sorted(np.bincount(labels).tolist(), reverse=True)[:8])
print("centroids:")
for i, c in enumerate(centroids):
    n = int((labels == i).sum())
    print(f"  {i}: RGB={tuple(c.tolist())}  count={n}")

# Replace per-vertex RGB with centroid RGB, keep alpha.
new_vc = np.zeros_like(vc)
new_vc[:, :3] = centroids[labels]
new_vc[:, 3] = vc[:, 3]
m.visual.vertex_colors = new_vc

m.export(dst)
print("saved:", dst)
