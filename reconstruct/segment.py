"""Cut the object out of each view.

A fixed brightness threshold cannot work here: the camera auto-exposes every
frame, so the same black screen reads V=42 in one view and V=78 in another as
the bottle turns more or less of its bright face to the lens.  The threshold is
therefore computed per image with Otsu, which adapts to whatever exposure that
frame happened to get.

Three refinements matter as much as the threshold:

* largest connected component only -- a highlight on the stand or a sliver of
  room at the frame edge would otherwise join the object;
* holes filled, since dark label text and the shadowed flank of the bottle read
  as background but are plainly inside the silhouette;
* everything below STAND_CUT discarded, because the stand turns with the object
  and would be carved into the model as a plinth.
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

# Row-mean brightness climbs from y=0.78 as the cork stand enters the frame.
STAND_CUT = 0.80


def silhouette(path):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Otsu over the region that can contain the object, so the stand's
    # brightness does not drag the threshold up.
    region = V[: int(h * STAND_CUT), :]
    thr, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bright = V > thr
    blue = (H > 100) & (H < 140) & (S > 110) & (V > thr * 0.55)
    m = (bright | blue).astype(np.uint8)

    m[int(h * STAND_CUT):, :] = 0
    m[:, :8] = 0
    m[:, -8:] = 0
    m[:8, :] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lbl == keep).astype(np.uint8)

    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    return m, thr


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    rows = []
    for f in files:
        m, thr = silhouette(f)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")),
                    m * 255)
        ys, xs = np.where(m)
        rows.append((os.path.basename(f), thr, int(m.sum()),
                     xs.min(), xs.max(), ys.min(), ys.max()))

    A = np.array([r[2] for r in rows], float)
    Wd = np.array([r[4] - r[3] for r in rows])
    tops = np.array([r[5] for r in rows])
    touch = [r[0] for r in rows if r[3] <= 10 or r[4] >= 969]
    print("otsu threshold: min %d max %d" % (min(r[1] for r in rows), max(r[1] for r in rows)))
    print("area           : min %d  max %d  mean %d  (max/min %.2f)"
          % (A.min(), A.max(), A.mean(), A.max() / A.min()))
    print("silhouette width: min %d  max %d" % (Wd.min(), Wd.max()))
    print("top edge spread : %d px" % (tops.max() - tops.min()))
    print("views touching a side edge: %s" % (touch if touch else "none"))
