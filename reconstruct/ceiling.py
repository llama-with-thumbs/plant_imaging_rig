"""How good could the score possibly get?

An IoU of 0.83 against the photographs means little without knowing what a
perfect reconstruction would score. Two things cap it below 1.0 regardless of
how good the model is:

* a visual hull is not the object -- it is the intersection of silhouette
  cones, which for a non-convex shape is strictly larger, and with only 36
  views strictly larger again;
* the score itself is discrete. Voxels are projected and dilated by their own
  footprint, which never lines up exactly with a silhouette edge.

So this runs the whole pipeline on a synthetic object whose silhouettes are
generated exactly -- no camera error, no segmentation error, no lens. Whatever
it scores is the ceiling, and the gap between that and 0.83 is the part
actually worth chasing.
"""

import math
import os

import cv2
import numpy as np

import persp

HERE = os.path.dirname(os.path.abspath(__file__))

H, W = 1520, 980
AXIS, TOP = 522.82, 47.0
HEIGHT = 1392.0
RADIUS = 330.0


def synth_masks(n=36, kind="bottle"):
    """Silhouettes of a known solid of revolution, rendered analytically."""
    angles = np.deg2rad(np.arange(n) * 360.0 / n)
    masks = np.zeros((n, H, W), dtype=bool)
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    t = (yy - TOP) / HEIGHT                       # 0 at the top, 1 at the base
    for i in range(n):
        if kind == "cylinder":
            rad = np.where((t >= 0) & (t <= 1), 190.0, -1.0)
        else:
            # a bottle-ish profile: narrow neck widening to a rounded body
            prof = 0.22 + 0.78 * np.clip((t - 0.30) / 0.45, 0, 1) ** 0.6
            prof = np.where(t > 0.92, prof * (1 - (t - 0.92) / 0.35), prof)
            rad = np.where((t >= 0) & (t <= 1), 235.0 * prof, -1.0)
        masks[i] = np.abs(xx - AXIS) <= rad
    return masks, angles


def score(vol, xs, ys, zs, masks, angles, axis, top, pitch, k):
    nxz, ny, _ = vol.shape
    fx = max(1, int(round((2 * RADIUS) / nxz)) | 1)
    fy = max(1, int(round(HEIGHT / ny)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    p = math.radians(pitch)
    ious = []
    for m in range(len(masks)):
        img = np.zeros((H, W), dtype=np.bool_)
        persp.project_kernel(vol, xs, ys, zs, np.float32(axis), np.float32(top),
                             np.float32(math.cos(angles[m])), np.float32(math.sin(angles[m])),
                             np.float32(math.cos(p)), np.float32(math.sin(p)),
                             np.float32(k), np.float32(persp.CU), np.float32(persp.CV), img)
        solid = cv2.dilate(img.astype(np.uint8), ker)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, ker).astype(bool)
        inter = np.logical_and(solid, masks[m]).sum()
        union = np.logical_or(solid, masks[m]).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious))


if __name__ == "__main__":
    print("ceiling of the scoring pipeline, on synthetic silhouettes")
    print("(exact geometry, no segmentation error, no lens)\n")
    print(" object     | grid      | IoU")
    for kind in ("cylinder", "bottle"):
        masks, angles = synth_masks(36, kind)
        for nxz, ny in ((140, 190), (180, 260), (280, 400)):
            vol, xs, ys, zs = persp.carve(masks, angles, AXIS, TOP, HEIGHT,
                                          RADIUS, 0.0, 0.0, nxz, ny)
            s = score(vol, xs, ys, zs, masks, angles, AXIS, TOP, 0.0, 0.0)
            print(" %-10s | %3dx%3d   | %.4f" % (kind, nxz, ny, s), flush=True)
    print("\nfor comparison, the real bottle scores 0.8257 on all 36 views")
