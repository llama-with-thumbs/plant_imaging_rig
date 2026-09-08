"""Carve with a perspective camera, and solve for how strong the perspective is.

Everything so far assumed orthographic projection -- that an object's size in
the image does not depend on its distance.  The rig plainly violates this: a
sheet of paper behind the bottle measured out at a scale implying a 190 mm
bottle where a tape says 270 mm, a 35% error that only depth can explain.

The model here perturbs the orthographic projection radially about the
principal point, by an amount proportional to depth:

    u = cu + (u_ortho - cu) / (1 + k*d)
    v = cv + (v_ortho - cv) / (1 + k*d)

k is inverse camera distance in the same pixel units as the object, so k = 0 is
exactly the old orthographic model and larger k means a closer camera.  Solving
for k rather than assuming it avoids needing a checkerboard calibration.

The principal point is the original frame's centre carried through the crop:
the sensor centre was (2028, 1520) and the crop began at (1500, 1080), so it
lands at (528, 440) in these images -- reassuringly close to the rotation axis
found independently at 524.8.
"""

import glob
import json
import math
import os

import cv2
import numpy as np
from PIL import Image
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))

CU, CV = 528.0, 440.0          # principal point in crop coordinates


@njit(parallel=True, cache=True, fastmath=True)
def carve_kernel(masks, cos_t, sin_t, xs, ys, zs, axis, top,
                 cos_p, sin_p, k, cu, cv, out):
    nx, ny, nz = xs.size, ys.size, zs.size
    nviews, h, w = masks.shape
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
                    uo = axis + lat
                    vo = top + Y * cos_p + d * sin_p
                    s = 1.0 + k * d
                    if s < 0.2:
                        keep = False
                        break
                    u = cu + (uo - cu) / s
                    v = cv + (vo - cv) / s
                    ui = int(u + 0.5); vi = int(v + 0.5)
                    if ui < 0 or ui >= w or vi < 0 or vi >= h or not masks[m, vi, ui]:
                        keep = False
                        break
                out[ix, iy, iz] = keep


@njit(parallel=True, cache=True, fastmath=True)
def project_kernel(vol, xs, ys, zs, axis, top, cos_t, sin_t,
                   cos_p, sin_p, k, cu, cv, img):
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
                lat = X * cos_t + Z * sin_t
                d = -X * sin_t + Z * cos_t
                s = 1.0 + k * d
                if s < 0.2:
                    continue
                u = cu + (axis + lat - cu) / s
                v = cv + (top + Y * cos_p + d * sin_p - cv) / s
                ui = int(u + 0.5); vi = int(v + 0.5)
                if 0 <= ui < w and 0 <= vi < h:
                    img[vi, ui] = True


def load(mask_dir="masks_g"):
    files = sorted(glob.glob(os.path.join(HERE, mask_dir, "v*.png")))
    masks = np.stack([np.asarray(Image.open(f).convert("L")) > 127 for f in files])
    angles = np.deg2rad(np.arange(len(files)) * 360.0 / len(files))
    return masks, angles


def carve(masks, angles, axis, top, height, radius, pitch, k, nxz, ny):
    xs = np.linspace(-radius, radius, nxz).astype(np.float32)
    zs = xs.copy()
    ys = np.linspace(0.0, height, ny).astype(np.float32)
    out = np.zeros((nxz, ny, nxz), dtype=np.bool_)
    p = math.radians(pitch)
    carve_kernel(masks, np.cos(angles).astype(np.float32),
                 np.sin(angles).astype(np.float32), xs, ys, zs,
                 np.float32(axis), np.float32(top),
                 np.float32(math.cos(p)), np.float32(math.sin(p)),
                 np.float32(k), np.float32(CU), np.float32(CV), out)
    return out, xs, ys, zs


def heldout(masks, angles, axis, top, height, radius, pitch, k, nxz, ny,
            train, test):
    vol, xs, ys, zs = carve(masks[train], angles[train], axis, top, height,
                            radius, pitch, k, nxz, ny)
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
                       np.float32(k), np.float32(CU), np.float32(CV), img)
        solid = cv2.dilate(img.astype(np.uint8), ker)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, ker).astype(bool)
        inter = np.logical_and(solid, masks[m]).sum()
        union = np.logical_or(solid, masks[m]).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)), int(vol.sum())


if __name__ == "__main__":
    import solve_pitch as sp
    masks, angles = load()
    top, bottom, axis0, half = sp.bounds(masks)
    radius = half * 1.12
    height = bottom - top
    g = json.load(open(os.path.join(HERE, "solved_geometry.json")))
    axis, pitch = g["axis"], g["pitch_deg"]
    train = np.arange(0, len(masks), 2)
    test = np.arange(1, len(masks), 2)

    print("held-out IoU against inverse camera distance k")
    print("(k = 0 is the orthographic model used until now)")
    print("     k      | distance      | IoU    | volume")
    best = None
    for k in [0.0, 2e-4, 4e-4, 6e-4, 8e-4, 1.0e-3, 1.3e-3, 1.6e-3, 2.0e-3, 2.5e-3]:
        iou, vol = heldout(masks, angles, axis, top, height, radius, pitch, k,
                           140, 190, train, test)
        dist = ("%8.0f px" % (1.0 / k)) if k else "  infinite"
        print("  %.5f  | %s | %.4f | %d" % (k, dist, iou, vol), flush=True)
        if best is None or iou > best[0]:
            best = (iou, k)
    print("\nbest k = %.5f  (IoU %.4f)" % (best[1], best[0]))
