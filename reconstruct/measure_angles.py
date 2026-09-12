"""Recover each view's real angle from the witness frames.

The step counter says every stop is 360/N degrees. It is not: the belt has
about a degree of backlash and the error accumulates per move, which is why 72
views reconstruct worse than 36 and why 36 views reached in 72 moves reconstruct
worse than the same 36 reached in 36.

The witness frames make the angle measurable. Each is matched against the one
before it, which gives the rotation that move actually achieved; the increments
accumulate into a real angle for every view. Matching against a fixed reference
would not work -- the rim leaves the secondary camera's field after a few stops
-- but consecutive frames always overlap heavily, because one stop moves the rim
only a fraction of the window.

Two things are deliberately not assumed:

  * Absolute scale. Pixels per degree varies across the field (the same 400
    steps read 13.5 px at one platter angle and 20.6 at another), so instead of
    trusting a constant, the increments are normalised so that a full
    revolution sums to exactly 360 degrees. The carve only cares about relative
    angles, and one full turn is the one thing known exactly.
  * That every match is good. A pair whose peak correlation is poor gets the
    median increment instead of its own bad measurement.
"""

import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The rotating rim ONLY. A window spanning the frame width puts the drive belt,
# the tensioner and the table edge back inside it, and template matching weights
# by contrast rather than by relevance -- the stationary furniture wins and the
# increments come out anywhere between -5 and +34 degrees. Same window the zero
# check uses, widened just enough to hold a 7 degree move.
ROI = (0.52, 0.64, 0.16, 0.60)
MIN_PEAK = 0.30


def band(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    y0, y1, x0, x1 = ROI
    b = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].astype(np.float32)
    b -= cv2.GaussianBlur(b, (0, 0), 15)          # kill the lighting gradient
    return b


def shift(a, b, margin=90):
    tpl = a[:, margin:-margin]
    res = cv2.matchTemplate(b, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    row = res[loc[1]]
    x = loc[0]
    dx = float(x - margin)
    if 0 < x < len(row) - 1:
        y0, y1, y2 = float(row[x - 1]), float(row[x]), float(row[x + 1])
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            dx += 0.5 * (y0 - y2) / den
    return dx, float(peak)


def measure(witness_dir):
    files = sorted(glob.glob(os.path.join(witness_dir, "v*.jpg")))
    n = len(files)
    bands = [band(f) for f in files]
    inc_px, peaks = [], []
    for i in range(n):
        a, b = bands[i], bands[(i + 1) % n]       # wrap: the last closes the turn
        if a is None or b is None:
            inc_px.append(np.nan); peaks.append(0.0); continue
        dx, pk = shift(a, b)
        inc_px.append(dx); peaks.append(pk)
    inc = np.array(inc_px, dtype=float)
    pk = np.array(peaks, dtype=float)

    good = (pk >= MIN_PEAK) & np.isfinite(inc)
    if good.sum() < n * 0.5:
        raise SystemExit("only %d of %d witness pairs matched; check the ROI" % (good.sum(), n))
    med = float(np.median(inc[good]))
    # a pair that matched badly, or that jumped, is replaced by the typical move
    bad = ~good | (np.abs(inc - med) > 4 * np.std(inc[good]) + 1e-6)
    inc[bad] = med

    # one revolution is the only angle known exactly, so make the turn close
    scale = 360.0 / inc.sum()
    deg = inc * scale
    ang = np.concatenate([[0.0], np.cumsum(deg)[:-1]])
    nominal = np.arange(n) * 360.0 / n
    return dict(n=n, angles=ang.tolist(), nominal=nominal.tolist(),
                increments_deg=deg.tolist(), peaks=pk.tolist(),
                replaced=int(bad.sum()), px_per_deg=float(1.0 / scale),
                residual=(ang - nominal).tolist())


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "g50", "witness")
    r = measure(d)
    res = np.array(r["residual"])
    inc = np.array(r["increments_deg"])
    print("%d witness frames, %d pair(s) replaced as unreliable" % (r["n"], r["replaced"]))
    print("scale: %.2f px per degree" % r["px_per_deg"])
    print("per-move rotation: %.3f deg nominal, measured %.3f .. %.3f (sd %.3f)"
          % (360.0 / r["n"], inc.min(), inc.max(), inc.std()))
    print("\nview angle vs the step counter:")
    print("  worst error %+.2f deg, rms %.2f deg" % (res[np.argmax(np.abs(res))], np.sqrt((res**2).mean())))
    for i in range(0, r["n"], max(1, r["n"] // 12)):
        print("   v%02d  nominal %6.2f   measured %6.2f   %+5.2f  (peak %.2f)"
              % (i, r["nominal"][i], r["angles"][i], res[i], r["peaks"][i]))
    json.dump(r, open(os.path.join(HERE, "angles_%s.json"
                                   % os.path.basename(os.path.dirname(d))), "w"), indent=2)
    print("\nwrote angles json")
