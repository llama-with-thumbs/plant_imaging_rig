"""Is the platter actually where the software thinks it is?

The carve assumes view m was taken at exactly m * 360/36 degrees. That comes
from counting motor steps through a belt, and belts have backlash. If the real
stops drift from the commanded ones, every silhouette is consistent with the
object and inconsistent with the *model*, which is exactly the symptom measured:
each mask has 13-20% unexplained by the hull built from the others.

So solve for a small correction to each view's angle, leave-one-out: build the
hull from the other 35 views, then ask which angle makes it best explain the
held-out mask. That question never uses view m's own assumed angle, so it is a
measurement rather than a fit to itself.

The result is falsifiable, which is the point. A genuine mechanical error is
*structured* -- eccentricity and belt error repeat once or twice per revolution,
so the corrections should trace a smooth curve around the platter. Overfitting
to mask noise would scatter. Fitting a 1- and 2-per-rev sinusoid to the
corrections and reporting how much variance it explains separates the two.
"""

import json
import math
import os

import cv2
import numpy as np

import persp
import robust

HERE = os.path.dirname(os.path.abspath(__file__))

NXZ, NY = 140, 190
SWEEP = np.arange(-3.0, 3.01, 0.25)     # degrees of angle correction to test
SHIFTS = np.arange(-8, 9)               # pixels of sideways correction to test


def project(vol, xs, ys, zs, ang, axis, top, pitch, k, shape):
    img = np.zeros(shape, dtype=np.bool_)
    p = math.radians(pitch)
    persp.project_kernel(vol, xs, ys, zs, np.float32(axis), np.float32(top),
                         np.float32(math.cos(ang)), np.float32(math.sin(ang)),
                         np.float32(math.cos(p)), np.float32(math.sin(p)),
                         np.float32(k), np.float32(persp.CU), np.float32(persp.CV), img)
    return img


def solidify(img, radius, height):
    fx = max(1, int(round((2 * radius) / NXZ)) | 1)
    fy = max(1, int(round(height / NY)) | 1)
    ker = np.ones((fy, fx), np.uint8)
    s = cv2.dilate(img.astype(np.uint8), ker)
    return cv2.morphologyEx(s, cv2.MORPH_CLOSE, ker).astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def fourier_fit(corr):
    """How much of the correction is 1- and 2-per-revolution structure?"""
    n = len(corr)
    t = np.arange(n) * 2 * np.pi / n
    A = np.column_stack([np.ones(n), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)])
    coef, *_ = np.linalg.lstsq(A, corr, rcond=None)
    fit = A @ coef
    resid = corr - fit
    var = np.var(corr)
    frac = 1.0 - np.var(resid) / var if var > 0 else 0.0
    amp1 = math.hypot(coef[1], coef[2])
    amp2 = math.hypot(coef[3], coef[4])
    return frac, coef[0], amp1, amp2, fit


def main():
    masks, angles = persp.load()
    g = json.load(open(os.path.join(HERE, "solved_persp.json")))
    axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
    top, bottom, radius = g["top"], g["bottom"], g["radius"]
    height = bottom - top
    n = len(masks)
    thr_frac = 32.0 / 36.0

    print("leave-one-out angle refinement, %d views, grid %dx%dx%d\n" % (n, NXZ, NY, NXZ))
    print(" view | assumed | best fit | shift |  IoU before -> after")

    d_ang = np.zeros(n)
    d_off = np.zeros(n)
    before, after = [], []

    for m in range(n):
        keep = np.array([i for i in range(n) if i != m])
        votes, xs, ys, zs = robust.vote_volume(masks[keep], angles[keep], axis, top,
                                               height, radius, pitch, k, NXZ, NY)
        vol = votes >= max(1, int(round(thr_frac * len(keep))))
        if vol.sum() < 1000:
            print(" %4d | (empty hull, skipped)" % m)
            continue

        best = None
        base = None
        for dd in SWEEP:
            img = project(vol, xs, ys, zs, angles[m] + math.radians(dd),
                          axis, top, pitch, k, masks[m].shape)
            solid = solidify(img, radius, height)
            for sh in SHIFTS:
                cand = np.roll(solid, int(sh), axis=1)
                s = iou(cand, masks[m])
                if dd == 0.0 and sh == 0:
                    base = s
                if best is None or s > best[0]:
                    best = (s, float(dd), int(sh))

        d_ang[m], d_off[m] = best[1], best[2]
        before.append(base)
        after.append(best[0])
        print(" %4d | %6.1f  | %+7.2f  | %+4d  |  %.4f -> %.4f"
              % (m, math.degrees(angles[m]), best[1], best[2], base, best[0]), flush=True)

    print("\nmean explanation of each mask: %.4f -> %.4f" % (np.mean(before), np.mean(after)))
    print("angle corrections: mean %+.2f deg, spread %.2f deg, max |%.2f|"
          % (d_ang.mean(), d_ang.std(), np.abs(d_ang).max()))

    frac, bias, a1, a2, fit = fourier_fit(d_ang)
    print("\nis the correction structured, or is it noise?")
    print("  1- and 2-per-rev sinusoid explains %.1f%% of the variance" % (100 * frac))
    print("  constant bias %+.2f deg | once-per-rev %.2f deg | twice-per-rev %.2f deg"
          % (bias, a1, a2))
    if frac > 0.5:
        print("  -> structured. This is mechanical: the platter is not stopping where")
        print("     it is told, in a pattern that repeats every revolution.")
    else:
        print("  -> unstructured. The corrections are not a consistent mechanical")
        print("     error, so applying them per-view would be fitting mask noise.")

    json.dump({"d_angle_deg": d_ang.tolist(), "d_offset_px": d_off.tolist(),
               "smooth_d_angle_deg": fit.tolist(), "structured_fraction": frac,
               "iou_before": float(np.mean(before)), "iou_after": float(np.mean(after))},
              open(os.path.join(HERE, "view_corrections.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
