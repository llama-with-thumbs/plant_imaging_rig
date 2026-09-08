"""Final carve: solved perspective camera, then a printable mesh.

Every parameter of the camera is now fitted rather than assumed -- where the
rotation axis falls in the image, how far the camera tilts down, and how strong
the perspective is -- each scored by how well the resulting hull predicts the
eighteen views it was not built from.
"""

import json
import math
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import measure

import persp

HERE = os.path.dirname(os.path.abspath(__file__))
OBJECT_HEIGHT_MM = 270.0


def largest_body(vol):
    lbl, n = ndimage.label(vol)
    if n <= 1:
        return vol, n
    sizes = ndimage.sum(vol, lbl, range(1, n + 1))
    return lbl == (1 + int(np.argmax(sizes))), n


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_v3"
    NXZ = int(os.environ.get("NXZ", 280))
    NY = int(os.environ.get("NY", 400))

    g = json.load(open(os.path.join(HERE, "solved_persp.json")))
    axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
    top, bottom, radius = g["top"], g["bottom"], g["radius"]
    height = bottom - top

    masks, angles = persp.load()
    print("carving %dx%dx%d from %d views" % (NXZ, NY, NXZ, len(masks)))
    print("   axis %.2f  pitch %+.2f deg  k %.6f (camera %.0f px away)"
          % (axis, pitch, k, 1 / k), flush=True)
    vol, xs, ys, zs = persp.carve(masks, angles, axis, top, height, radius,
                                  pitch, k, NXZ, NY)
    print("   kept %d of %d (%.2f%%)" % (vol.sum(), vol.size,
                                         100 * vol.sum() / vol.size), flush=True)
    vol, ncomp = largest_body(vol)
    print("   largest of %d components: %d voxels" % (ncomp, vol.sum()))
    np.save(os.path.join(HERE, "hull%s.npy" % tag), vol)

    px_xz = (2 * radius) / (NXZ - 1)
    px_y = height / (NY - 1)
    mm = OBJECT_HEIGHT_MM / height

    smooth = ndimage.gaussian_filter(vol.astype(np.float32), 1.2)
    verts, faces, _, _ = measure.marching_cubes(np.pad(smooth, 2), level=0.5)
    print("   marching cubes: %d verts, %d tris" % (len(verts), len(faces)))

    v = verts
    pts = np.column_stack([
        (v[:, 0] - v[:, 0].mean()) * px_xz * mm,
        (np.ptp(v[:, 1]) - (v[:, 1] - v[:, 1].min())) * px_y * mm,
        (v[:, 2] - v[:, 2].mean()) * px_xz * mm,
    ])
    pts[:, 1] -= pts[:, 1].min()
    np.save(os.path.join(HERE, "verts%s.npy" % tag), pts.astype(np.float32))
    np.save(os.path.join(HERE, "faces%s.npy" % tag), faces.astype(np.int32))

    bb = pts.max(axis=0) - pts.min(axis=0)
    print("   bounding box %.1f x %.1f x %.1f mm" % tuple(bb))
    json.dump({"axis": axis, "pitch_deg": pitch, "k": k, "top": top,
               "bottom": bottom, "radius": radius, "grid_xz": NXZ, "grid_y": NY,
               "px_per_voxel_xz": px_xz, "px_per_voxel_y": px_y, "mm_per_px": mm,
               "voxels": int(vol.sum()), "bbox_mm": [float(x) for x in bb]},
              open(os.path.join(HERE, "geometry%s.json" % tag), "w"), indent=2)


if __name__ == "__main__":
    main()
