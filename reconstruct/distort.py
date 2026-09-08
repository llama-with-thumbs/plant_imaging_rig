"""Add radial lens distortion to the camera model and solve for it.

What remains after fitting the axis, the tilt and the perspective is the lens.
It is visibly a fisheye -- straight walls bow in every frame -- and a bowed lens
maps a straight silhouette edge onto a curved one, so a hull carved with a
pinhole model has to shrink to fit inside every curve.

The standard single-term radial model is enough here:

    r2 = ((u-cu)^2 + (v-cv)^2) / f^2
    u_distorted = cu + (u-cu) * (1 + k1*r2)

with r normalised by a focal length in pixels so k1 is dimensionless. Negative
k1 is barrel distortion, which is what a wide lens has.

The distortion is applied *after* projection, so the model describes the whole
path from object to sensor: rotate, tilt, perspective-divide, then bend.
"""

import json
import math
import os

import cv2
import numpy as np
from numba import njit, prange

import persp

HERE = os.path.dirname(os.path.abspath(__file__))

# Normalising radius. The crop is ~980 px across, taken from a 4056 px frame,
# so the true focal length in pixels is far larger than the crop; 2000 keeps k1
# in a comfortable numerical range without claiming to be the real focal length.
FNORM = 2000.0


@njit(parallel=True, cache=True, fastmath=True)
def carve_kernel(masks, cos_t, sin_t, xs, ys, zs, axis, top,
                 cos_p, sin_p, k, k1, cu, cv, out):
    nx, ny, nz = xs.size, ys.size, zs.size
    nviews, h, w = masks.shape
    inv_f2 = 1.0 / (FNORM * FNORM)
    for ix in prange(nx):
        X = xs[ix]
        for iy in range(ny):
            Y = ys[iy]
            for iz in range(nz):
                Z = zs[iz]
                keep = True
                for m in range(nviews):
                    lat = X * cos_t[m] + Z * sin_t[m]
                    d = -X * sin_t[m] + Z * cos_t[m]
                    s = 1.0 + k * d
                    if s < 0.2:
                        keep = False
                        break
                    du = (axis + lat - cu) / s
                    dv = (top + Y * cos_p + d * sin_p - cv) / s
                    r2 = (du * du + dv * dv) * inv_f2
                    g = 1.0 + k1 * r2
                    ui = int(cu + du * g + 0.5)
                    vi = int(cv + dv * g + 0.5)
                    if ui < 0 or ui >= w or vi < 0 or vi >= h or not masks[m, vi, ui]:
                        keep = False
                        break
                out[ix, iy, iz] = keep


@njit(parallel=True, cache=True, fastmath=True)
def project_kernel(vol, xs, ys, zs, axis, top, cos_t, sin_t,
                   cos_p, sin_p, k, k1, cu, cv, img):
    nx, ny, nz = xs.size, ys.size, zs.size
    h, w = img.shape
    inv_f2 = 1.0 / (FNORM * FNORM)
    for ix in prange(nx):
        X = xs[ix]
        for iy in range(ny):
            Y = ys[iy]
            for iz in range(nz):
                if not vol[ix, iy, iz]:
                    continue
                Z = zs[iz]
                lat = X * cos_t + Z * sin_t
                d = -X * sin_t + Z * cos_t
                s = 1.0 + k * d
                if s < 0.2:
                    continue
                du = (axis + lat - cu) / s
                dv = (top + Y * cos_p + d * sin_p - cv) / s
                r2 = (du * du + dv * dv) * inv_f2
                g = 1.0 + k1 * r2
                ui = int(cu + du * g + 0.5)
                vi = int(cv + dv * g + 0.5)
                if 0 <= ui < w and 0 <= vi < h:
                    img[vi, ui] = True


def carve(masks, angles, axis, top, height, radius, pitch, k, k1, nxz, ny):
    xs = np.linspace(-radius, radius, nxz).astype(np.float32)
    zs = xs.copy()
    ys = np.linspace(0.0, height, ny).astype(np.float32)
    out = np.zeros((nxz, ny, nxz), dtype=np.bool_)
    p = math.radians(pitch)
    carve_kernel(masks, np.cos(angles).astype(np.float32),
                 np.sin(angles).astype(np.float32), xs, ys, zs,
                 np.float32(axis), np.float32(top),
                 np.float32(math.cos(p)), np.float32(math.sin(p)),
                 np.float32(k), np.float32(k1),
                 np.float32(persp.CU), np.float32(persp.CV), out)
    return out, xs, ys, zs


def heldout(masks, angles, axis, top, height, radius, pitch, k, k1,
            nxz, ny, train, test):
    vol, xs, ys, zs = carve(masks[train], angles[train], axis, top, height,
                            radius, pitch, k, k1, nxz, ny)
    if vol.sum() < 1000:
        return 0.0, int(vol.sum())
    fx = max(1, int(round((2 * radius) / nxz)) | 1)
    fy = max(1, int(round(height / ny)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    p = math.radians(pitch)
    ious = []
    for m in test:
        img = np.zeros(masks[m].shape, dtype=np.bool_)
        project_kernel(vol, xs, ys, zs, np.float32(axis), np.float32(top),
                       np.float32(math.cos(angles[m])), np.float32(math.sin(angles[m])),
                       np.float32(math.cos(p)), np.float32(math.sin(p)),
                       np.float32(k), np.float32(k1),
                       np.float32(persp.CU), np.float32(persp.CV), img)
        solid = cv2.dilate(img.astype(np.uint8), ker)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, ker).astype(bool)
        inter = np.logical_and(solid, masks[m]).sum()
        union = np.logical_or(solid, masks[m]).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)), int(vol.sum())


if __name__ == "__main__":
    masks, angles = persp.load()
    g = json.load(open(os.path.join(HERE, "solved_persp.json")))
    axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
    top, bottom, radius = g["top"], g["bottom"], g["radius"]
    height = bottom - top
    train = np.arange(0, len(masks), 2)
    test = np.arange(1, len(masks), 2)

    print("held-out IoU against radial distortion k1")
    print("(k1 = 0 is the undistorted pinhole used so far; negative is barrel)")
    print("     k1    |  IoU   | volume")
    best = None
    for k1 in [-0.30, -0.20, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12, 0.20]:
        iou, vol = heldout(masks, angles, axis, top, height, radius, pitch, k,
                           k1, 140, 190, train, test)
        print("  %+.3f   | %.4f | %d" % (k1, iou, vol), flush=True)
        if best is None or iou > best[0]:
            best = (iou, k1)
    print("\nbest k1 = %+.3f (IoU %.4f)" % (best[1], best[0]))
