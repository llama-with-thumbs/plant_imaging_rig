"""Chroma key for the 8 mm portrait capture.

Same principle as before -- key out the screen and keep what is left, because
the screen is the uniform thing in frame and so it is what should be modelled --
but three things differ from the fisheye pipeline.

The frames arrive on their side, because the camera is mounted rotated, so they
are turned upright first. There is no ROI crop any more, so the object sits in a
full 3040x4056 frame rather than a hand-picked window. And the platter disc is
now the same green as the screen, which is a gift: the chroma key removes it for
free, and only the cork stand below it needs an explicit cut.

Downscales the masks. Carving cost goes as the voxel count, not the mask size,
and a silhouette edge is not sharper for being sampled at 4056 px than at 1520 --
but memory and cache pressure very much are.
"""

import glob
import os

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.environ.get("SRC_DIR", "green8mm"))
OUT = os.path.join(HERE, os.environ.get("OUT_DIR", "masks8"))
WORK_H = 1900                 # mask height to work at

HUE_LO, HUE_HI = 40, 95       # screen measured at 68.4, sd 5.2
SAT_MIN = 55
VAL_MIN = 30


STAND_CUT = [None]          # filled in by stand_cut(), shared by every frame


def lo_ok(cut, h):
    """A waist cut is only believable in the lower part of the frame."""
    return 0.30 * h < cut < 0.98 * h


def waist_cut(mask, h):
    """Cut where the silhouette pinches between the stand and the object.

    The object now sits on a pedestal: a wide white disc carrying thin wire
    legs. Neither is green and neither is cork, so the chroma key keeps both and
    they would be carved into the model as part of the subject -- and they turn
    with it, so no other view averages them away.

    Looking for the stand by colour no longer works, because the disc is the
    same pale metal as the pot. But the shape gives it away: read the silhouette
    width from the bottom and it goes wide (disc), narrow (legs), wide (object).
    Find the narrowest row in the lower half, then walk up out of the legs to
    where the object begins, and cut there.
    """
    widths = mask.sum(axis=1).astype(float)
    rows = np.where(widths > 0)[0]
    if len(rows) < 10:
        return h
    lo, hi = rows.min(), rows.max()
    band = slice(int(lo + 0.45 * (hi - lo)), hi)
    seg = widths[band]
    if not seg.size or seg.max() <= 0:
        return h
    waist = int(np.argmin(np.where(seg > 0, seg, seg.max()))) + band.start
    thresh = max(2.0 * widths[waist], 0.25 * widths.max())
    r = waist
    while r > lo and widths[r] < thresh:
        r -= 1
    return int(r + 1)


def stand_cut(paths, probe=12):
    """Decide one cut row for the whole set, from where the cork actually is.

    The platter is fixed, so its top edge is the same row in every frame, and a
    single agreed value is far steadier than a per-frame detection that can be
    fooled by whatever the label happens to show at that angle. Taking the
    median across frames means a few confused views cannot move it.
    """
    rows = []
    for p in paths[::max(1, len(paths) // probe)]:
        bgr = cv2.rotate(cv2.imread(p), cv2.ROTATE_90_COUNTERCLOCKWISE)
        h0, w0 = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(round(w0 * WORK_H / float(h0))), WORK_H),
                         interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cork = ((H > 4) & (H < 34) & (S > 45) & (V > 35))
        floor = int(h * 0.86)                 # below anything the label reaches
        hit = np.where(cork[floor:].sum(axis=1) > w * 0.12)[0]
        if len(hit):
            rows.append(floor + int(hit.min()))
    STAND_CUT[0] = int(np.median(rows)) if rows else None
    return STAND_CUT[0], rows


def silhouette(path):
    bgr = cv2.rotate(cv2.imread(path), cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = bgr.shape[:2]
    scale = WORK_H / float(h)
    bgr = cv2.resize(bgr, (int(round(w * scale)), WORK_H), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    screen = (H >= HUE_LO) & (H <= HUE_HI) & (S >= SAT_MIN) & (V >= VAL_MIN)

    # the cork stand and its ruler sticker turn with the object, so other views
    # cannot average them away -- they have to go by colour, not by position
    cork = ((H > 4) & (H < 34) & (S > 45) & (V > 35))
    m = (~screen & ~cork).astype(np.uint8)

    # Find where the stand starts, from the cork rather than from the screen.
    # Keying off "a wide band of green" does not work: green covers most of the
    # row on both sides of the bottle for almost the whole frame, so the test
    # fires half way up the object. The cork only exists below the platter, so
    # its topmost row is an unambiguous floor.
    # The platter does not move, so the cut is decided once for the whole set
    # (see stand_cut) rather than guessed per frame. Guessing per frame fails:
    # the bottle's back label carries a beige panel in the same hue band as
    # cork, and at 150-210 deg that panel was taken for the stand and sliced
    # 350 px off the object.
    cut = STAND_CUT[0] if STAND_CUT[0] is not None else int(h * 0.95)
    m[cut:, :] = 0
    # then, if the object stands on a pedestal, cut again at the pinch between
    # the legs and the object itself
    if os.environ.get("PEDESTAL"):
        n0, l0, st0, _ = cv2.connectedComponentsWithStats(
            cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)), 8)
        if n0 > 1:
            big = (l0 == 1 + int(np.argmax(st0[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
            wc = waist_cut(big, h)
            if lo_ok(wc, h):
                m[wc:, :] = 0
                cut = wc
    # There was a rule here removing pale, bright pixels in the lower half of
    # the frame, to kill the white band on the stand. It was redundant -- the
    # stand cut above already zeroes everything below the platter -- and it was
    # actively destructive on anything short: a 150 mm moka pot sits low in the
    # frame and IS pale bright metal, so the rule ate most of the object and
    # left silhouettes varying 61% in height. A rule aimed at the background by
    # position rather than by colour only works while the subject stays out of
    # that position.

    m[:, :6] = 0
    m[:, -6:] = 0
    m[:6, :] = 0

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lbl == keep).astype(np.uint8)
    m = ndimage.binary_fill_holes(m.astype(bool)).astype(np.uint8)
    return m, bgr, float(screen.mean()), cut


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "v*.jpg")))
    cut, seen = stand_cut(files)
    print("stand cut agreed at row %s from %d probes (%s)"
          % (cut, len(seen), ", ".join(str(v) for v in seen)))
    rows = []
    for f in files:
        m, bgr, frac, cut = silhouette(f)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f).replace(".jpg", ".png")), m * 255)
        ys, xs = np.where(m)
        rows.append((os.path.basename(f), frac, int(m.sum()),
                     int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), cut))

    A = np.array([r[2] for r in rows], float)
    W = np.array([r[4] - r[3] for r in rows])
    T = np.array([r[5] for r in rows])
    B = np.array([r[6] for r in rows])
    Hh = B - T
    print("masks written to %s at %d px tall" % (OUT, WORK_H))
    print("screen coverage : %.0f%%..%.0f%%" % (100 * min(r[1] for r in rows),
                                                100 * max(r[1] for r in rows)))
    print("area            : %d..%d  (max/min %.2f)" % (A.min(), A.max(), A.max() / A.min()))
    print("width           : %d..%d px  (max/min %.2f)" % (W.min(), W.max(), W.max() / W.min()))
    print("height          : %d..%d px  (spread %d px = %.1f%%)"
          % (Hh.min(), Hh.max(), Hh.max() - Hh.min(), 100 * (Hh.max() - Hh.min()) / Hh.mean()))
    print("top edge spread : %d px      <- should be ~0 on a turntable" % (T.max() - T.min()))
    print("base edge spread: %d px" % (B.max() - B.min()))
    edge = [r[0] for r in rows if r[3] <= 8 or r[4] >= 0]
    print("median top %d  median base %d" % (int(np.median(T)), int(np.median(B))))
