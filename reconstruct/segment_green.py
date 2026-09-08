"""Cut the object out of each view by chroma key.

This is the backdrop the method wants.  Both earlier attempts failed the same
way: they had to separate object from background using *brightness*, and the
subject contains both a dark blue body and a near-white head, so no single
threshold could keep both.  Against black the dark flanks were lost; against
white paper the head was.

Green solves it because the test is hue, not brightness:

    screen        H 71-72, and remarkably uniform -- standard deviation 0.6-0.8
    bottle blue   H 96
    label green   H 112 -- teal, comfortably clear of the screen
    trigger head  saturation 8.5 against the screen's 150

So the head is separated by being unsaturated and the body by being a different
hue, and neither depends on exposure.  Nothing here needs a per-image threshold
the way the previous two backdrops did.

The mask is built by keying out the screen and keeping the rest, which is the
opposite way round from before and is why it holds up: the screen is the
uniform thing in frame, so it is what should be modelled.
"""

import glob
import os

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "orbit36g")
OUT = os.path.join(HERE, "masks_g")
os.makedirs(OUT, exist_ok=True)

STAND_CUT = 0.948

HUE_LO, HUE_HI = 60, 86       # screen hue 71-72, generous either side
SAT_MIN = 70                  # the screen is saturated; the head is not
VAL_MIN = 25                  # ignore near-black, where hue is meaningless


def silhouette(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    screen = (H >= HUE_LO) & (H <= HUE_HI) & (S >= SAT_MIN) & (V >= VAL_MIN)

    # The cork stand is not green, so a plain key keeps it -- and because the
    # bottle stands on it the two are one connected region, so "largest
    # component" cannot separate them. It has to go by hue as well. It matters
    # more than ordinary background because it turns *with* the object, so it
    # would be carved into the model rather than averaged away.
    cork = (H > 4) & (H < 40) & (S > 50) & (V > 40)

    m = (~screen & ~cork).astype(np.uint8)

    m[int(h * STAND_CUT):, :] = 0
    m[:, :10] = 0
    m[:, -10:] = 0
    m[:10, :] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lbl == keep).astype(np.uint8)

    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    return m, float(screen.mean())


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    rows = []
    for f in files:
        m, frac = silhouette(f)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")),
                    m * 255)
        ys, xs = np.where(m)
        rows.append((os.path.basename(f), frac, int(m.sum()),
                     xs.min(), xs.max(), ys.min(), ys.max()))

    A = np.array([r[2] for r in rows], float)
    Wd = np.array([r[4] - r[3] for r in rows])
    tops = np.array([r[5] for r in rows])
    bots = np.array([r[6] for r in rows])
    fr = np.array([r[1] for r in rows])
    print("screen coverage : %.0f%%..%.0f%% of frame" % (fr.min()*100, fr.max()*100))
    print("area            : min %d  max %d  (max/min %.2f)" % (A.min(), A.max(), A.max()/A.min()))
    print("silhouette width: min %d  max %d" % (Wd.min(), Wd.max()))
    print("top edge spread : %d px" % (tops.max()-tops.min()))
    print("bottom spread   : %d px" % (bots.max()-bots.min()))
    edge = [r[0] for r in rows if r[3] <= 12 or r[4] >= 967]
    print("touching a side edge: %s" % (edge if edge else "none"))
