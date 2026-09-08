"""Find the camera pitch by cross-validation rather than by maximising volume.

Maximising the carved volume is the obvious objective and the wrong one: a
wrong geometry can inflate volume as easily as shrink it, and the landscape it
produces here is broad, shallow, and peaks at an implausible 30 degrees.

Held-out validation asks a question that cannot be gamed.  Carve using only
half the views, then project that hull into the views it never saw and compare
against their real silhouettes.  A correct geometry predicts unseen views well.
A wrong one produces a hull that happens to satisfy the views it was built from
and misses the rest.

The score is intersection-over-union against the held-out masks.
"""

import glob
import math
import os

import cv2
import numpy as np
from PIL import Image
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))


def load(mask_dir):
    files = sorted(glob.glob(os.path.join(mask_dir, "v*.png")))
    masks = np.stack([np.asarray(Image.open(f).convert("L")) > 127 for f in files])
    angles = np.deg2rad(np.arange(len(files)) * 360.0 / len(files))
    return masks, angles


def bounds(masks):
    tops, bots, half, cxs = [], [], [], []
    for m in masks:
        ys, xs = np.where(m)
        tops.append(ys.min()); bots.append(ys.max())
        cxs.append(xs.mean())
        half.append(max(xs.max() - xs.mean(), xs.mean() - xs.min()))
    return int(min(tops)), int(max(bots)), float(np.mean(cxs)), float(max(half))


@njit(parallel=True, cache=True, fastmath=True)
def carve_kernel(masks, cos_t, sin_t, xs, ys, zs, axis, top, cos_p, sin_p, out):
    nx, ny, nz = xs.size, ys.size, zs.size
    nviews, h, w = masks.shape
    for ix in prange(nx):
        X = xs[ix]
        for iy in range(ny):
            Y = ys[iy]
            for iz in range(nz):
                Z = zs[iz]
                keep = True
                for k in range(nviews):
                    u = axis + X * cos_t[k] + Z * sin_t[k]
                    d = -X * sin_t[k] + Z * cos_t[k]
                    v = top + Y * cos_p + d * sin_p
                    ui = int(u + 0.5); vi = int(v + 0.5)
                    if ui < 0 or ui >= w or vi < 0 or vi >= h or not masks[k, vi, ui]:
                        keep = False
                        break
                out[ix, iy, iz] = keep


@njit(parallel=True, cache=True, fastmath=True)
def project_kernel(vol, xs, ys, zs, axis, top, cos_t, sin_t, cos_p, sin_p, img):
    nx, ny, nz = xs.size, ys.size, zs.size
    h, w = img.shape
    for ix in prange(nx):
        X = xs[ix]
        for iy in range(ny):
            Y = ys[iy]
            for iz in range(nz):
                if not vol[ix, iy, iz]:
                    continue
                Z = zs[iz]
                u = axis + X * cos_t + Z * sin_t
                d = -X * sin_t + Z * cos_t
                v = top + Y * cos_p + d * sin_p
                ui = int(u + 0.5); vi = int(v + 0.5)
                if 0 <= ui < w and 0 <= vi < h:
                    img[vi, ui] = True


def carve(masks, angles, axis, top, height, radius, pitch_deg, nxz, ny):
    xs = np.linspace(-radius, radius, nxz).astype(np.float32)
    zs = xs.copy()
    ys = np.linspace(0.0, height, ny).astype(np.float32)
    out = np.zeros((nxz, ny, nxz), dtype=np.bool_)
    p = math.radians(pitch_deg)
    carve_kernel(masks, np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32),
                 xs, ys, zs, np.float32(axis), np.float32(top),
                 np.float32(math.cos(p)), np.float32(math.sin(p)), out)
    return out, xs, ys, zs


def heldout_iou(mask_dir="masks_g", pitches=range(-4, 33, 4), nxz=110, ny=150):
    masks, angles = load(mask_dir)
    top, bottom, axis0, half = bounds(masks)
    radius = half * 1.12
    height = bottom - top
    train = np.arange(0, len(masks), 2)          # even views build the hull
    test = np.arange(1, len(masks), 2)           # odd views judge it
    print("training on %d views, testing on %d" % (len(train), len(test)))
    print(" pitch |  held-out IoU | carved volume")
    scores = []
    for p in pitches:
        vol, xs, ys, zs = carve(masks[train], angles[train], axis0, top, height,
                                radius, float(p), nxz, ny)
        if vol.sum() == 0:
            print("  %+4d  |     -        | 0" % p)
            continue
        ious = []
        pr = math.radians(p)
        for k in test:
            img = np.zeros(masks[k].shape, dtype=np.bool_)
            project_kernel(vol, xs, ys, zs, np.float32(axis0), np.float32(top),
                           np.float32(math.cos(angles[k])), np.float32(math.sin(angles[k])),
                           np.float32(math.cos(pr)), np.float32(math.sin(pr)), img)
            # A voxel is a box, not a point. Projecting centres leaves the
            # image striped -- at zero pitch every voxel of a given height
            # lands on one row, so a 150-row grid fills 150 of 1392 image
            # rows and the score measures grid resolution, not geometry.
            # Dilating by the voxel's own footprint restores a solid area.
            fx = max(1, int(round((2 * radius) / nxz)) | 1)
            fy = max(1, int(round(height / ny)) | 1)
            solid = cv2.dilate(img.astype(np.uint8), np.ones((fy, fx), np.uint8))
            solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((fy, fx), np.uint8))
            solid = solid.astype(bool)
            inter = np.logical_and(solid, masks[k]).sum()
            union = np.logical_or(solid, masks[k]).sum()
            ious.append(inter / union if union else 0.0)
        mean_iou = float(np.mean(ious))
        scores.append((mean_iou, float(p), int(vol.sum())))
        print("  %+4d  |    %.4f    | %d" % (p, mean_iou, vol.sum()), flush=True)
    scores.sort(reverse=True)
    return scores


if __name__ == "__main__":
    scores = heldout_iou()
    print("\nbest pitch by held-out IoU: %+.0f deg (IoU %.4f)" % (scores[0][1], scores[0][0]))
    print("runners-up:", ", ".join("%+.0f (%.4f)" % (s[1], s[0]) for s in scores[1:4]))
