"""Robust carving: let a voxel survive if most views agree, not all of them.

Strict visual hull carving is an intersection -- a voxel must be inside every
silhouette. That makes it maximally sensitive to the worst mask in the set: one
view that clips the object by a few pixels removes that material from the model
permanently, and no amount of agreement from the other 35 views brings it back.

The silhouettes here are measurably inconsistent. Adding views keeps shrinking
the hull long after it should have plateaued (-4.4% from 18 to 36 views, where
synthetic consistent silhouettes lose 0.7%), and every view has 13-20% of its
mask unexplained by the hull built from the others.

So instead of requiring unanimity, count the votes and keep voxels that enough
views accept. A threshold of 100% is exactly the classic algorithm; lower
values trade a little bloat for tolerance of bad masks.
"""

import json
import math
import os

import cv2
import numpy as np
from numba import njit, prange

import persp

HERE = os.path.dirname(os.path.abspath(__file__))


@njit(parallel=True, cache=True, fastmath=True)
def vote_kernel(masks, cos_t, sin_t, xs, ys, zs, axis, top,
                cos_p, sin_p, k, cu, cv, votes):
    nx, ny, nz = xs.size, ys.size, zs.size
    nviews, h, w = masks.shape
    for ix in prange(nx):
        X = xs[ix]
        for iy in range(ny):
            Y = ys[iy]
            for iz in range(nz):
                Z = zs[iz]
                c = 0
                for m in range(nviews):
                    lat = X * cos_t[m] + Z * sin_t[m]
                    d = -X * sin_t[m] + Z * cos_t[m]
                    s = 1.0 + k * d
                    if s < 0.2:
                        continue
                    u = cu + (axis + lat - cu) / s
                    v = cv + (top + Y * cos_p + d * sin_p - cv) / s
                    ui = int(u + 0.5); vi = int(v + 0.5)
                    if 0 <= ui < w and 0 <= vi < h and masks[m, vi, ui]:
                        c += 1
                votes[ix, iy, iz] = c


def vote_volume(masks, angles, axis, top, height, radius, pitch, k, nxz, ny):
    xs = np.linspace(-radius, radius, nxz).astype(np.float32)
    zs = xs.copy()
    ys = np.linspace(0.0, height, ny).astype(np.float32)
    votes = np.zeros((nxz, ny, nxz), dtype=np.uint8)
    p = math.radians(pitch)
    vote_kernel(masks, np.cos(angles).astype(np.float32),
                np.sin(angles).astype(np.float32), xs, ys, zs,
                np.float32(axis), np.float32(top),
                np.float32(math.cos(p)), np.float32(math.sin(p)),
                np.float32(k), np.float32(persp.CU), np.float32(persp.CV), votes)
    return votes, xs, ys, zs


def score(vol, xs, ys, zs, masks, angles, axis, top, pitch, k, radius, height):
    nxz, ny, _ = vol.shape
    fx = max(1, int(round((2 * radius) / nxz)) | 1)
    fy = max(1, int(round(height / ny)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    p = math.radians(pitch)
    ious = []
    for m in range(len(masks)):
        img = np.zeros(masks[m].shape, dtype=np.bool_)
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
    masks, angles = persp.load()
    g = json.load(open(os.path.join(HERE, "solved_persp.json")))
    axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
    top, bottom, radius = g["top"], g["bottom"], g["radius"]
    height = bottom - top
    n = len(masks)

    train = np.arange(0, n, 2)
    test = np.arange(1, n, 2)
    print("votes counted once, then thresholded -- carving is not repeated\n")

    vt, xs, ys, zs = vote_volume(masks[train], angles[train], axis, top, height,
                                 radius, pitch, k, 170, 230)
    ntr = len(train)
    print(" threshold      | voxels  | held-out IoU")
    best = None
    for miss in range(0, 5):
        thr = ntr - miss
        vol = vt >= thr
        s = score(vol, xs, ys, zs, masks[test], angles[test], axis, top, pitch, k,
                  radius, height)
        label = "all %d views" % ntr if miss == 0 else "%d of %d (-%d)" % (thr, ntr, miss)
        print("  %-13s | %7d | %.4f" % (label, vol.sum(), s), flush=True)
        if best is None or s > best[0]:
            best = (s, thr, miss)
    print("\nbest: allow %d dissenting view(s), held-out IoU %.4f" % (best[2], best[0]))
