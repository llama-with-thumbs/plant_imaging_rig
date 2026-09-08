"""Jointly solve for the rotation axis and camera pitch by held-out validation.

Carve from the even-numbered views, predict the odd ones, and score by IoU.
Volume is not the objective: a wrong geometry inflates it as readily as it
shrinks it, and optimising volume alone put the pitch at an implausible 30
degrees with no interior peak at all.

Both parameters are searched together because they trade off -- shifting the
assumed axis can partly mimic a tilt, so optimising them one at a time lands
in the wrong place.
"""

import math
import os

import cv2
import numpy as np

import solve_pitch as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def score(masks, angles, train, test, axis, top, height, radius, pitch,
          nxz, ny):
    vol, xs, ys, zs = sp.carve(masks[train], angles[train], axis, top, height,
                               radius, pitch, nxz, ny)
    if vol.sum() == 0:
        return 0.0, 0
    fx = max(1, int(round((2 * radius) / nxz)) | 1)
    fy = max(1, int(round(height / ny)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    pr = math.radians(pitch)
    ious = []
    for k in test:
        img = np.zeros(masks[k].shape, dtype=np.bool_)
        sp.project_kernel(vol, xs, ys, zs, np.float32(axis), np.float32(top),
                          np.float32(math.cos(angles[k])), np.float32(math.sin(angles[k])),
                          np.float32(math.cos(pr)), np.float32(math.sin(pr)), img)
        solid = cv2.dilate(img.astype(np.uint8), ker)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, ker).astype(bool)
        inter = np.logical_and(solid, masks[k]).sum()
        union = np.logical_or(solid, masks[k]).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)), int(vol.sum())


def solve(mask_dir="masks_g", nxz=140, ny=190):
    masks, angles = sp.load(mask_dir)
    top, bottom, axis0, half = sp.bounds(masks)
    radius = half * 1.12
    height = bottom - top
    train = np.arange(0, len(masks), 2)
    test = np.arange(1, len(masks), 2)

    best = (0.0, axis0, 0.0, 0)
    for stage, (da_list, dp_list) in enumerate([
        (range(-16, 17, 8), range(-25, -12, 3)),
        (range(-6, 7, 3), np.arange(-3.0, 3.1, 1.5)),
        (range(-2, 3, 1), np.arange(-1.0, 1.1, 0.5)),
    ]):
        base_a, base_p = best[1], best[2]
        if stage == 0:
            base_a, base_p = axis0, 0.0
        results = []
        for da in da_list:
            for dp in dp_list:
                a = base_a + da
                p = float(base_p + dp) if stage else float(dp)
                iou, vol = score(masks, angles, train, test, a, top, height,
                                 radius, p, nxz, ny)
                results.append((iou, a, p, vol))
        results.sort(reverse=True)
        best = results[0]
        print("   stage %d: IoU %.4f  axis %.1f  pitch %+.2f  volume %d"
              % (stage + 1, best[0], best[1], best[2], best[3]), flush=True)
    return best, (top, bottom, radius, axis0)


if __name__ == "__main__":
    print("joint search over axis and pitch, scored on 18 held-out views")
    best, (top, bottom, radius, axis0) = solve()
    iou, axis, pitch, vol = best
    print()
    print("axis  %.1f px  (initial guess %.1f, moved %+.1f)" % (axis, axis0, axis - axis0))
    print("pitch %+.2f deg" % pitch)
    print("held-out IoU %.4f" % iou)
    import json
    json.dump({"axis": axis, "pitch_deg": pitch, "iou": iou,
               "top": top, "bottom": bottom, "radius": radius},
              open(os.path.join(HERE, "solved_geometry.json"), "w"), indent=2)
    print("wrote solved_geometry.json")
