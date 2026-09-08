"""Turn the carved voxel hull into a printable STL.

Three things happen between voxels and a usable mesh:

* the volume is blurred slightly before the surface is extracted, which turns
  the voxel staircase into a smooth skin without moving the surface;
* marching cubes runs at step 2, roughly quartering the triangle count so the
  file loads in a browser;
* only the largest connected body is kept, since imperfect masks leave a few
  crumbs floating nearby.

Scale.  The mesh comes out in image pixels, which mean nothing to a printer, so
everything is scaled by OBJECT_HEIGHT_MM / (height in pixels).  That constant is
the one number here that cannot be derived from the images -- set it from a
tape measure.
"""

import os
import struct

import numpy as np
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the one measurement that has to come from the real world --------------
OBJECT_HEIGHT_MM = 270.0        # measured with a tape
# ---------------------------------------------------------------------------

SMOOTH_SIGMA = 1.4
STEP = 2


def largest_body(vol):
    lbl, n = ndimage.label(vol)
    if n <= 1:
        return vol
    sizes = ndimage.sum(vol, lbl, range(1, n + 1))
    return lbl == (1 + int(np.argmax(sizes)))


def write_binary_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"Plant Imaging Rig - visual hull from 36 silhouettes".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln else np.zeros(3)
            fh.write(struct.pack("<3f", *n))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(b"\x00\x00")


if __name__ == "__main__":
    TAG = os.environ.get("TAG", "")
    src = os.path.join(HERE, "hull_voxels_clean%s.npy" % TAG)
    if not os.path.exists(src):
        src = os.path.join(HERE, "hull_voxels%s.npy" % TAG)
    vol = np.load(src)
    vol = largest_body(vol.astype(bool))
    print("voxels: %d" % vol.sum())

    smooth = ndimage.gaussian_filter(vol.astype(np.float32), SMOOTH_SIGMA)
    padded = np.pad(smooth, 2)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, step_size=STEP)
    print("mesh: %d verts, %d tris" % (len(verts), len(faces)))

    # Voxel indices -> a right-handed, upright, millimetre model.
    # Axis 0 and 2 are the turntable plane, axis 1 is image rows (downwards).
    #
    # Voxels are not cubes.  Converting indices straight to millimetres treats
    # them as though they were and stretches the model across by the ratio of
    # the two voxel sizes -- which on this grid is about 1.5x.
    import json
    g = json.load(open(os.path.join(HERE, "geometry%s.json" % TAG)))
    px_xz = g["px_per_voxel_xz"]
    px_y = g["px_per_voxel_y"]
    mm_per_px = OBJECT_HEIGHT_MM / g["object_height_px"]
    print("voxel %.2f x %.2f px;  %.4f mm/px" % (px_xz, px_y, mm_per_px))

    v = verts.copy()
    xs = (v[:, 0] - v[:, 0].mean()) * px_xz * mm_per_px
    ys = (np.ptp(v[:, 1]) - (v[:, 1] - v[:, 1].min())) * px_y * mm_per_px  # row 0 is the top
    zs = (v[:, 2] - v[:, 2].mean()) * px_xz * mm_per_px
    pts = np.column_stack([xs, ys, zs])
    pts[:, 1] -= pts[:, 1].min()                              # sit on the Z=0 plane

    tris = pts[faces]
    out = os.path.join(HERE, "bottle%s.stl" % TAG)
    write_binary_stl(out, tris)
    size = os.path.getsize(out)
    print("wrote %s  (%d triangles, %.1f MB)" % (out, len(faces), size / 1e6))

    bb = pts.max(axis=0) - pts.min(axis=0)
    print("bounding box: %.1f x %.1f x %.1f mm  (assuming height = %.0f mm)"
          % (bb[0], bb[1], bb[2], OBJECT_HEIGHT_MM))
    voxel_mm3 = (px_xz * mm_per_px) ** 2 * (px_y * mm_per_px)
    print("hull volume: ~%.0f cm3" % (vol.sum() * voxel_mm3 / 1000.0))

    np.save(os.path.join(HERE, "mesh_pts.npy"), pts.astype(np.float32))
    np.save(os.path.join(HERE, "mesh_faces.npy"), faces.astype(np.int32))
