"""Carve the 8 mm capture from measured geometry rather than fitted geometry.

Every previous model inferred the camera by asking which parameters best
explained the silhouettes they were meant to explain, and that process quietly
absorbed model error into whichever parameter was loosest -- it put the
principal point 370 px from where the capture chain said it was and pulled the
pitch to -15 deg when a tape says roughly zero.

This one starts from measurements:

    lens to rotation axis        340 mm
    lens height above the base   140 mm
    object                       270 mm tall, 115 mm at its widest

which fix three of the four things the old solver had to guess.

  * Pitch. The lens sits at 140 mm and the bottle's mid-height is 135 mm, so the
    optical axis is atan(5/340) = 0.8 deg below level. Not fitted, measured.
  * Perspective. k is one over the camera distance in pixels, and both terms are
    now known.
  * Principal point. There is no ROI crop any more, so it is the image centre.
    That is the whole reason for dropping the crop.

Only the rotation axis is still estimated, and that one is safe: on a turntable
it is the mean of the silhouette centres.

The 115 mm width is new, and it is the first independent check this project has
had. Every earlier model was scaled by height alone, so width was free to come
out wrong without anything noticing.
"""

import glob
import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

import persp
import robust

HERE = os.path.dirname(os.path.abspath(__file__))
# The object's own size, not a constant. It feeds the scale AND the camera
# pitch: the lens sits 140 mm above the platter, so with a 270 mm object it was
# level with the mid-height and the tilt was 0.8 degrees, while a 150 mm object
# puts the mid-height at 75 mm and the camera looks down 10.8. Swapping the
# subject changes the camera geometry even though nothing on the rig moved.
_OBJ = json.load(open(os.path.join(HERE, "object.json")))
OBJ_H_MM = float(_OBJ["height_mm"])
OBJ_W_MM = float(_OBJ["width_mm"]) if _OBJ.get("width_mm") else None
MASK_DIR = _OBJ.get("masks", "masks8")
DIST_MM, LENS_H_MM = 340.0, 140.0


def load_masks(d=None):
    d = d or MASK_DIR
    fs = sorted(glob.glob(os.path.join(HERE, d, "v*.png")))
    ms = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127 for f in fs])
    # Negated on purpose. Standing the bottle up means rotating every frame 90
    # degrees, and that also mirrors the horizontal image axis relative to the
    # platter's rotation -- so the frames read as turning the other way. Left
    # uncorrected, each silhouette is matched against the object as it looked
    # from the far side of the platter: reprojection IoU 0.80 against 0.91, and
    # a strict intersection that collapses to a third of the volume.
    ang = -np.deg2rad(np.arange(len(fs)) * 360.0 / len(fs))
    return ms, ang


def geometry(masks):
    n, h, w = masks.shape
    tops, bots, cxs, halfw = [], [], [], []
    for m in masks:
        ys, xs = np.where(m)
        tops.append(ys.min()); bots.append(ys.max())
        cxs.append(0.5 * (xs.min() + xs.max())); halfw.append(0.5 * (xs.max() - xs.min()))
    top = int(np.percentile(tops, 10))       # an off-axis tip sweeps; take a low quantile
    bottom = int(np.median(bots))
    axis = float(np.mean(cxs))               # on a turntable, the axis is the mean centre
    height_px = bottom - top
    mm_per_px = OBJ_H_MM / height_px
    # Without a measured width, bound the grid by the widest silhouette instead.
    # That is not a free lunch -- it removes the one independent check on scale,
    # since height alone can be satisfied by a model of any width.
    widest = max(halfw) * 1.06
    radius_px = max((OBJ_W_MM / 2.0) / mm_per_px, widest) if OBJ_W_MM else widest
    pitch = -math.degrees(math.atan((LENS_H_MM - OBJ_H_MM / 2.0) / DIST_MM))
    k = mm_per_px / DIST_MM                  # 1/k is the camera distance in pixels
    return dict(top=top, bottom=bottom, height_px=height_px, axis=axis,
                radius=radius_px, mm_per_px=mm_per_px, pitch=pitch, k=k,
                cu=w / 2.0, cv=h / 2.0, shape=(h, w))


def main():
    masks, ang = load_masks()
    g = geometry(masks)
    persp.CU, persp.CV = g["cu"], g["cv"]
    print("masks %d x %d x %d" % masks.shape)
    print("measured geometry")
    print("  object      : %d px tall -> %.4f mm/px" % (g["height_px"], g["mm_per_px"]))
    print("  axis        : %.1f px   (image centre %.1f)" % (g["axis"], g["cu"]))
    print("  radius      : %.0f px  = %.1f mm" % (g["radius"], g["radius"] * g["mm_per_px"]))
    print("  pitch       : %+.2f deg  (measured, not fitted)" % g["pitch"])
    print("  k           : %.6f  -> camera %.0f mm" % (g["k"], g["mm_per_px"] / g["k"]))
    print("  princ. point: %.1f, %.1f  = the image centre, no crop" % (g["cu"], g["cv"]))

    NXZ = int(os.environ.get("NXZ", 260))
    NY = int(os.environ.get("NY", 380))
    VOTES = int(os.environ.get("VOTES", 32))
    print("\ncarving %dx%dx%d, keeping voxels %d of %d views accept"
          % (NXZ, NY, NXZ, VOTES, len(masks)), flush=True)
    votes, xs, ys, zs = robust.vote_volume(masks, ang, g["axis"], g["top"],
                                           g["height_px"], g["radius"],
                                           g["pitch"], g["k"], NXZ, NY)
    for thr in (36, 34, 32, 30):
        vol = votes >= thr
        if vol.sum() < 1000:
            print("  >=%2d of 36 : empty" % thr); continue
        s = robust.score(vol, xs, ys, zs, masks, ang, g["axis"], g["top"],
                         g["pitch"], g["k"], g["radius"], g["height_px"])
        print("  >=%2d of 36 : %8d voxels   reprojection IoU %.4f" % (thr, vol.sum(), s),
              flush=True)

    vol = votes >= VOTES
    lbl, ncomp = ndimage.label(vol)
    if ncomp > 1:
        sizes = ndimage.sum(vol, lbl, range(1, ncomp + 1))
        vol = lbl == (1 + int(np.argmax(sizes)))
    print("\nlargest of %d components: %d voxels" % (ncomp, vol.sum()))
    np.save(os.path.join(HERE, "hull_g8.npy"), vol)

    px_xz = (2 * g["radius"]) / (NXZ - 1)
    px_y = g["height_px"] / (NY - 1)
    mm = g["mm_per_px"]
    smooth = ndimage.gaussian_filter(vol.astype(np.float32), 1.2)
    verts, faces, _, _ = measure.marching_cubes(np.pad(smooth, 2), level=0.5)
    v = verts
    pts = np.column_stack([
        (v[:, 0] - v[:, 0].mean()) * px_xz * mm,
        (np.ptp(v[:, 1]) - (v[:, 1] - v[:, 1].min())) * px_y * mm,
        (v[:, 2] - v[:, 2].mean()) * px_xz * mm,
    ])
    pts[:, 1] -= pts[:, 1].min()
    np.save(os.path.join(HERE, "verts_g8.npy"), pts.astype(np.float32))
    np.save(os.path.join(HERE, "faces_g8.npy"), faces.astype(np.int32))

    bb = pts.max(axis=0) - pts.min(axis=0)
    print("marching cubes: %d verts, %d tris" % (len(verts), len(faces)))
    print("\nbounding box %.1f x %.1f x %.1f mm" % tuple(bb))
    print("  height : %.1f mm vs %.1f measured  (%+.1f%%)"
          % (bb[1], OBJ_H_MM, 100 * (bb[1] / OBJ_H_MM - 1)))
    if OBJ_W_MM:
        print("  width  : %.1f mm vs %.1f measured  (%+.1f%%)   <- independent check"
              % (max(bb[0], bb[2]), OBJ_W_MM, 100 * (max(bb[0], bb[2]) / OBJ_W_MM - 1)))
    else:
        print("  width  : %.1f mm   (no measured width, so nothing checks the scale"
              % max(bb[0], bb[2]))
        print("           independently -- height alone is satisfied by any width)")
    json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in g.items() if k != "shape"},
              open(os.path.join(HERE, "geometry_g8.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
