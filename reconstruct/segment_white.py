"""Cut the object out of each view, against a white paper backdrop.

The tests invert relative to the black screen, and the awkward part changes.
Against black, the white trigger head was the easy bit and the dark flanks of
the bottle were hard.  Against white it is the reverse -- but the head is still
separable, because it is *brighter* than the paper rather than similar to it:

    paper         V ~ 129-148, almost unsaturated, very uniform (sd 3-6)
    trigger head  V ~ 184 median -- brighter than the paper
    bottle body   S ~ 168 -- hugely more saturated than paper's S ~ 9
    cork stand    tan, saturated at hue 5-35 -> excluded
    room below    dark, unsaturated -> excluded by needing brightness OR colour

So: anything strongly coloured, or brighter than the paper, is object.  The
paper's uniformity is what makes this work; a textured backdrop would not.
"""

import glob
import os

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "orbit36w")
OUT = os.path.join(HERE, "masks_w")
os.makedirs(OUT, exist_ok=True)

STAND_CUT = 0.948


def silhouette(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Estimate the paper level per ROW from the left and right margins, not
    # from a couple of corners. The sheet is lit unevenly across its width, so
    # a single global level makes brightly-lit paper read as object.
    margin = np.concatenate([V[:, 8:68], V[:, -68:-8]], axis=1)
    paper_row = np.median(margin, axis=1).astype(np.float32)
    paper_row = cv2.GaussianBlur(paper_row.reshape(-1, 1), (1, 61), 0).ravel()
    # Below the sheet's bottom edge the margins are dark table, which would
    # drag the threshold down until shadows counted as object. Clamp it.
    paper_row = np.maximum(paper_row, 120.0)
    paper_col = paper_row[:, None]

    # A brightness floor before trusting saturation: HSV saturation is
    # meaningless in near-black pixels, and the unlit room below the paper
    # reads as S=76 at V=27.
    # Two colour tests, not one. The bottle's shadowed edge sits at S~60,
    # V~75 -- just under a single threshold, so it was being eroded away and
    # taking a bite out of the silhouette. Since a visual hull intersects every
    # view, one bite in one view removes that material from the model
    # permanently, which is why a few bad edges cost so much.
    coloured = (S > 60) & (V > 55)
    dim_blue = (H > 98) & (H < 142) & (S > 35) & (V > 45)
    coloured = coloured | dim_blue
    # +12 rather than more: the trigger head is only ~24 levels above the
    # paper, so a tighter margin loses it in half the views (top-edge spread
    # 199 px at +16, 50 px at +12).
    brighter = V > paper_col + 12
    cork = (H > 4) & (H < 48) & (S > 55)

    m = ((coloured | brighter) & ~cork).astype(np.uint8)

    m[int(h * STAND_CUT):, :] = 0
    m[:, :8] = 0
    m[:, -8:] = 0
    m[:8, :] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lbl == keep).astype(np.uint8)

    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    return m, float(np.median(paper_row))


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    rows = []
    for f in files:
        m, pv = silhouette(f)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")),
                    m * 255)
        ys, xs = np.where(m)
        rows.append((os.path.basename(f), pv, int(m.sum()),
                     xs.min(), xs.max(), ys.min(), ys.max()))

    A = np.array([r[2] for r in rows], float)
    Wd = np.array([r[3] - r[2] for r in rows])
    Wd = np.array([r[4] - r[3] for r in rows])
    tops = np.array([r[5] for r in rows])
    bots = np.array([r[6] for r in rows])
    pv = np.array([r[1] for r in rows])
    print("paper level    : %.0f .. %.0f  (drift %.0f)" % (pv.min(), pv.max(), pv.max()-pv.min()))
    print("area           : min %d  max %d  (max/min %.2f)" % (A.min(), A.max(), A.max()/A.min()))
    print("silhouette width: min %d  max %d" % (Wd.min(), Wd.max()))
    print("top edge spread : %d px" % (tops.max()-tops.min()))
    print("bottom spread   : %d px" % (bots.max()-bots.min()))
    edge = [r[0] for r in rows if r[3] <= 10 or r[4] >= 969]
    print("touching a side edge: %s" % (edge if edge else "none"))
