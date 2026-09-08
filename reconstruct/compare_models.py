"""Score two hulls by how well they reproduce the photographs they came from.

Reprojection IoU against all 36 silhouettes is the closest thing to ground
truth available without a reference scan: a hull that predicts every view it
was built from is, at minimum, consistent with the evidence.  It cannot catch
errors shared by every view -- the lens distortion is invisible to it -- but it
does catch geometry that is simply wrong.
"""

import glob
import json
import math
import os

import cv2
import numpy as np
from PIL import Image

import solve_pitch as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def masks_and_angles(mask_dir="masks_g"):
    files = sorted(glob.glob(os.path.join(HERE, mask_dir, "v*.png")))
    masks = np.stack([np.asarray(Image.open(f).convert("L")) > 127 for f in files])
    angles = np.deg2rad(np.arange(len(files)) * 360.0 / len(files))
    return masks, angles


def evaluate(hull_path, axis, top, height, radius, pitch, masks, angles, label):
    vol = np.load(os.path.join(HERE, hull_path))
    nxz, ny, _ = vol.shape
    xs = np.linspace(-radius, radius, nxz).astype(np.float32)
    zs = xs.copy()
    ys = np.linspace(0.0, height, ny).astype(np.float32)
    fx = max(1, int(round((2 * radius) / nxz)) | 1)
    fy = max(1, int(round(height / ny)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    pr = math.radians(pitch)
    ious, covers = [], []
    for k in range(len(masks)):
        img = np.zeros(masks[k].shape, dtype=np.bool_)
        sp.project_kernel(vol, xs, ys, zs, np.float32(axis), np.float32(top),
                          np.float32(math.cos(angles[k])), np.float32(math.sin(angles[k])),
                          np.float32(math.cos(pr)), np.float32(math.sin(pr)), img)
        solid = cv2.dilate(img.astype(np.uint8), ker)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, ker).astype(bool)
        inter = np.logical_and(solid, masks[k]).sum()
        union = np.logical_or(solid, masks[k]).sum()
        ious.append(inter / union if union else 0.0)
        covers.append(inter / masks[k].sum())
    print("  %-28s IoU %.4f  (min %.4f)   silhouette coverage %.4f   %d voxels, grid %dx%dx%d"
          % (label, np.mean(ious), np.min(ious), np.mean(covers), vol.sum(), nxz, ny, nxz))
    return float(np.mean(ious))


if __name__ == "__main__":
    masks, angles = masks_and_angles()
    print("reprojection against all %d views:" % len(masks))

    g1 = json.load(open(os.path.join(HERE, "geometry_g.json")))
    evaluate("hull_voxels_g.npy", g1["axis"], g1["top"], g1["bottom"] - g1["top"],
             g1["radius"], 0.0, masks, angles, "v1  level camera, 180 grid")

    g2 = json.load(open(os.path.join(HERE, "geometry_v2.json")))
    evaluate("hull_v2.npy", g2["axis"], g2["top"], g2["bottom"] - g2["top"],
             g2["radius"], g2["pitch_deg"], masks, angles, "v2  solved pitch, 280 grid")

    # what does the solved geometry buy on its own, at the old resolution?
    vol, xs, ys, zs = sp.carve(masks, angles, g2["axis"], g2["top"],
                               g2["bottom"] - g2["top"], g2["radius"],
                               g2["pitch_deg"], 180, 260)
    np.save(os.path.join(HERE, "hull_ablate.npy"), vol)
    evaluate("hull_ablate.npy", g2["axis"], g2["top"], g2["bottom"] - g2["top"],
             g2["radius"], g2["pitch_deg"], masks, angles, "    solved pitch, 180 grid")
