"""Cut the object out of each view.

Brightness alone does not work here.  The camera auto-exposes every frame, so
when the bottle turns edge-on and presents little bright surface the exposure
lifts, the threshold drops, and patches of the grey screen get classified as
object.  Those patches then carve real material out of the model.

Colour separates the three things in frame cleanly, and none of them are grey:

    bottle body   strong blue     hue 100-140, saturated
    trigger head  near-white      unsaturated but very bright
    backdrop      grey            unsaturated and mid-dark   -> excluded
    cork stand    tan/brown       saturated at hue 5-35      -> excluded

The stand is excluded twice over -- by hue and by a cut line -- because it turns
with the object, so unlike the backdrop it would otherwise be carved into the
model as a plinth rather than averaged away.
"""

import glob
import os

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "orbit36")
OUT = os.path.join(HERE, "masks")
os.makedirs(OUT, exist_ok=True)

# The blue body reaches row ~1438 of 1520 (0.946). Sit just below that: a sliver
# of cork is far less harmful to the model than a truncated base.
STAND_CUT = 0.948


def silhouette(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blue = (H > 98) & (H < 142) & (S > 80) & (V > 40)
    # The head is unsaturated, so it can only be told from the grey screen by
    # being much brighter -- hence a high floor rather than an adaptive one.
    white = (S < 75) & (V > 165)
    cork = (H > 5) & (H < 35) & (S > 70)

    m = ((blue | white) & ~cork).astype(np.uint8)

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
    return m


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    rows = []
    for f in files:
        m = silhouette(f)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")),
                    m * 255)
        ys, xs = np.where(m)
        rows.append((os.path.basename(f), int(m.sum()),
                     xs.min(), xs.max(), ys.min(), ys.max()))

    A = np.array([r[1] for r in rows], float)
    Wd = np.array([r[3] - r[2] for r in rows])
    tops = np.array([r[4] for r in rows])
    bots = np.array([r[5] for r in rows])
    print("area            : min %d  max %d  (max/min %.2f)" % (A.min(), A.max(), A.max()/A.min()))
    print("silhouette width: min %d  max %d" % (Wd.min(), Wd.max()))
    print("top edge spread : %d px   (object does not move vertically)" % (tops.max()-tops.min()))
    print("bottom spread   : %d px" % (bots.max()-bots.min()))
    edge = [r[0] for r in rows if r[2] <= 10 or r[3] >= 969]
    print("touching a side edge: %s" % (edge if edge else "none"))
