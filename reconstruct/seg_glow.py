"""Silhouettes from a backlit screen, which needs no colour at all.

Against a green screen the test is hue: find the backdrop's colour and remove
it. That works, but it is a proxy -- it asks "is this pixel backdrop-coloured"
when the question is "is this pixel backdrop". With a lit panel behind the
object the two coincide: the backdrop emits light and the object blocks it, so
the silhouette is simply the shadow.

The method is almost embarrassingly direct:

  1. the screen is the large bright region -- take the biggest bright blob
  2. fill it in, and the filled area is screen plus whatever sits in front
  3. inside that area, anything not bright is the object

No hue thresholds, no saturation floor, no assumption about the subject's
colour. A bare aluminium pot and a dark blue bottle are the same problem here,
which is the point: the earlier pipeline needed a different key for every
backdrop and nearly lost the pot entirely to a rule written for the bottle.

Two things still need care. The panel does not fill the frame, so the bezel and
the room sit outside it and must be excluded rather than keyed -- hence working
inside the screen's own region. And the object is lit from behind, so its front
is in shadow but its edges can still catch a highlight; the bright test is
deliberately generous so a glinting facet does not punch a hole in the mask.
"""

import glob
import os

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.environ.get("SRC_DIR", "glow"))
OUT = os.path.join(HERE, os.environ.get("OUT_DIR", "masks_glow"))
WORK_H = 1900


def fill_small_holes(mask, max_frac=0.02):
    """Close specks, keep genuine openings.

    A blanket binary_fill_holes was closing the gap inside the pot's handle.
    That gap is real -- you can see through it -- and the backlight makes it
    unmistakable for the first time, so filling it turned the handle into a
    solid wedge. Speckle from a glinting facet is a few hundred pixels; the
    handle's opening is tens of thousands. Fill by size and the distinction
    makes itself.
    """
    filled = ndimage.binary_fill_holes(mask.astype(bool))
    holes = filled & ~mask.astype(bool)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    out = mask.astype(bool).copy()
    limit = max_frac * float(mask.sum())
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= limit:
            out |= (lbl == i)
    return out.astype(np.uint8)


def silhouette(path):
    bgr = cv2.rotate(cv2.imread(path), cv2.ROTATE_90_COUNTERCLOCKWISE)
    h0, w0 = bgr.shape[:2]
    bgr = cv2.resize(bgr, (int(round(w0 * WORK_H / float(h0))), WORK_H),
                     interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Otsu splits the frame into lit and unlit without a hand-picked number,
    # which matters because the panel's brightness is not ours to choose.
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (g >= thr).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, lbl, st, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        return np.zeros((h, w), np.uint8), bgr, 0.0

    # The panel is the hull of EVERY lit patch, not of the largest one.
    #
    # The object divides the panel: a wide subject leaves lit strips down either
    # side, and with a coffee bag the largest single strip is a fraction of the
    # screen, so its hull is not the panel at all -- silhouette heights varied
    # 44% and areas by a factor of twelve. Taking all the lit patches together
    # recovers the panel's outline however the object cuts it up.
    #
    # This is only safe with the lamps off. Lit the pot's polished facets were
    # bright too, so "every lit patch" included reflections on the object and
    # punched holes through the body. An unlit object against a lit panel has no
    # such patches -- which is why turning the lamps off simplified the problem
    # rather than merely improving it.
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, ar in enumerate(areas) if ar > 0.01 * float(areas.max())]
    screen = np.isin(lbl, keep)

    pts = cv2.findNonZero(screen.astype(np.uint8))
    hull = cv2.convexHull(pts)
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, hull.reshape(-1, 2), 1)
    region = cv2.erode(region, np.ones((25, 25), np.uint8)).astype(bool)

    m = (region & ~screen).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

    n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
    if n2 > 1:
        m = (lbl2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    m = fill_small_holes(m)
    return m, bgr, float(screen.mean())


def waist_cut(mask, drop=0.55, run=12):
    """Find where the object ends and its stand begins, scanning downwards.

    Looking for the narrowest row does not work. Below the object the profile is
    legs (narrow), then the pedestal disc -- which is WIDER than the object --
    and then a taper to a few pixels at the very tip. The global minimum is that
    tip, not the legs, and a threshold derived from it puts the cut at the
    bottom of everything.

    Scanning down from the body avoids the question entirely. Take the body's
    typical width from the upper half, then walk down and cut at the first place
    the silhouette narrows past `drop` of it and stays narrow for `run` rows.
    Nothing below that point is consulted, so a wide pedestal cannot confuse it.
    """
    widths = mask.sum(axis=1).astype(float)
    rows = np.where(widths > 0)[0]
    if len(rows) < 20:
        return mask.shape[0]
    lo, hi = int(rows.min()), int(rows.max())
    upper = widths[lo:lo + max(10, (hi - lo) // 2)]
    body = float(np.median(upper[upper > 0])) if (upper > 0).any() else 0.0
    if body <= 0:
        return mask.shape[0]
    limit = drop * body
    for r in range(lo + (hi - lo) // 3, hi - run):
        if widths[r] < limit and np.all(widths[r:r + run] < body):
            return r
    return hi + 1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    cuts = []
    masks = []
    for f in files:
        m, bgr, frac = silhouette(f)
        cuts.append(waist_cut(m))
        masks.append((f, m, frac))
    cut = int(np.median(cuts)) if cuts else WORK_H
    print("pedestal cut agreed at row %d (spread %d px)"
          % (cut, max(cuts) - min(cuts) if cuts else 0))

    rows = []
    for f, m, frac in masks:
        m = m.copy(); m[cut:, :] = 0
        n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
        if n2 > 1:
            m = (lbl2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")), m * 255)
        ys, xs = np.where(m)
        rows.append((frac, int(m.sum()), int(xs.max() - xs.min()),
                     int(ys.min()), int(ys.max())))

    A = np.array([r[1] for r in rows], float)
    W = np.array([r[2] for r in rows]); T = np.array([r[3] for r in rows])
    B = np.array([r[4] for r in rows]); Hh = B - T
    print("masks written to %s" % OUT)
    print("screen coverage : %.0f%%..%.0f%%" % (100 * min(r[0] for r in rows),
                                                100 * max(r[0] for r in rows)))
    print("area            : %d..%d  (max/min %.2f)" % (A.min(), A.max(), A.max() / A.min()))
    print("width           : %d..%d px  (max/min %.2f)" % (W.min(), W.max(), W.max() / W.min()))
    print("height          : %d..%d px  (spread %d px = %.1f%%)"
          % (Hh.min(), Hh.max(), Hh.max() - Hh.min(), 100 * (Hh.max() - Hh.min()) / Hh.mean()))
    print("top edge spread : %d px      <- should be ~0 on a turntable" % (T.max() - T.min()))
    print("base edge spread: %d px" % (B.max() - B.min()))
