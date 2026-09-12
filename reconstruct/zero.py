"""Re-zero the platter against the ruler sticker, driven from this machine.

The platter's error accumulates per move rather than per degree: 36 views at 10
degrees reached in 72 moves drift as badly as 72 views, and worse than the same
36 views reached in 36 moves. A finer step does not help; going back to a known
mark and cancelling what has crept in does.

The secondary camera looks straight at a ruler sticker on the platter rim, and
phase correlation against a stored reference reads that mark to a fraction of a
pixel. Over the small angles drift ever reaches, the rim moves across this
camera almost exactly linearly in motor steps.

Nothing assumes the platter radius, the camera distance or the lens. The
pixels-per-step figure is measured by stepping a known amount and watching how
far the ruler goes, which absorbs all of that at once.

    python zero.py --calibrate      once, to learn pixels per step
    python zero.py --set-ref        once, to define zero
    python zero.py --check          report drift, change nothing
    python zero.py --correct        report drift and null it
    python zero.py --correct --every 2      act only on every 2nd call
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PI = os.environ.get("RIG_HOST", "pi@10.0.1.238")
STATE = os.path.join(HERE, "zero_state.json")
REF = os.path.join(HERE, "zero_ref.npy")
COUNTER = os.path.join(HERE, "zero_counter.json")
LOG = os.path.join(HERE, "zero_log.json")

# A stable path, not a number. /dev/videoN is assigned in probe order, so the
# USB camera was video2 before a reboot and video0 after -- and video2 became
# the CSI sensor, which does not speak MJPEG and fails with a format error that
# looks nothing like "you are talking to the wrong camera".
USB = ("/dev/v4l/by-id/"
       "usb-HD_USB_Camera_HD_USB_Camera_HD_USB_Camera-video-index0")

# The upper, centimetre ruler band ONLY.
#
# A generous window is worse than useless here. Most of what the secondary
# camera sees never moves -- the drive belt, the tensioner, the table -- and
# phase correlation weights by contrast, not by relevance, so the stationary
# furniture outvotes the ruler and the answer comes back near zero. With the
# whole lower frame, 400 steps read as 0.56 px; cropped to the ruler alone, the
# same pair reads 13.5 px, and an independently tighter crop agrees at 13.6.
#
# The lower, inch ruler is excluded on purpose. It moves only 3.7 px over the
# same 400 steps, far too little for a radius difference, which says it is on
# the fixed pedestal rather than the turning platter -- there is what looks like
# a bearing surface between the two cork discs.
ROI = (0.52, 0.63, 0.20, 0.52)          # y0, y1, x0, x1 as fractions


def ssh(cmd, timeout=180):
    return subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", PI, cmd],
                          capture_output=True, text=True, timeout=timeout)


def grab(tag="now", lights=True):
    """One frame from the secondary camera, past the auto-exposure ramp."""
    remote = "/tmp/zero_%s.jpg" % tag
    on = "pinctrl set 17 op dh; sleep 4; " if lights else ""
    cmd = (on +
           "ffmpeg -loglevel error -y -f v4l2 -input_format mjpeg "
           "-video_size 1280x960 -i %s " % USB +
           "-vf 'select=gte(n\\,8)' -frames:v 1 %s" % remote)
    # The USB camera is not always released the instant its last reader exits,
    # and v4l2 reports that as "Device or resource busy" -- a transient, not a
    # fault. Clear any stale grabber and try again rather than failing the whole
    # check on a race.
    r = ssh(cmd)
    for attempt in range(3):
        if not r.returncode:
            break
        if "busy" not in (r.stderr or "").lower():
            break
        ssh("pkill -f '[f]fmpeg' 2>/dev/null; sleep 2")
        r = ssh(cmd)
    if r.returncode:
        raise RuntimeError("capture failed: %s" % (r.stderr.strip()[:200]))
    local = os.path.join(HERE, "zero_%s.jpg" % tag)
    subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no",
                    "%s:%s" % (PI, remote), local], check=True, timeout=120)
    img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("could not read the pulled frame")
    h, w = img.shape
    y0, y1, x0, x1 = ROI
    band = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].astype(np.float32)
    # flatten the lighting so the correlation keys on ruler marks, not shading
    band -= cv2.GaussianBlur(band, (0, 0), 25)
    return band


def ambiguity(res_row, best_i, guard=12):
    """How close the runner-up peak is to the best one.

    A ruler is a periodic target, so template matching finds a strong peak at
    every mark, not just the right one. Returns second/best: near 1.0 means the
    match is a coin toss between periods and the number it returns is arbitrary.
    """
    mask = np.ones(len(res_row), bool)
    lo, hi = max(0, best_i - guard), min(len(res_row), best_i + guard + 1)
    mask[lo:hi] = False
    if not mask.any():
        return 0.0
    return float(res_row[mask].max() / max(res_row[best_i], 1e-6))


def shift_px(a, b, margin=60):
    """Horizontal shift of b relative to a, by template matching.

    Phase correlation is the obvious choice and it fails on this picture. The
    window is a narrow strip of near-periodic ruler marks, and the Hanning
    window phase correlation needs throws away the ends where most of the
    distinguishing detail is -- on a pair where template matching finds +19 px
    at 0.61 peak correlation, phase correlation returned -0.01 confidence and a
    meaningless number. Matching a cut-down template against the full strip has
    no windowing and gives a peak height that is directly interpretable.

    Sub-pixel by fitting a parabola through the peak and its two neighbours.
    """
    tpl = a[:, margin:-margin]
    res = cv2.matchTemplate(b, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    x = loc[0]
    r = res[0] if res.ndim > 1 else res
    r = r.ravel() if r.ndim == 1 else res[loc[1]]
    dx = float(x - margin)
    if 0 < x < len(r) - 1:                       # parabolic sub-pixel refinement
        y0, y1, y2 = float(r[x - 1]), float(r[x]), float(r[x + 1])
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            dx += 0.5 * (y0 - y2) / denom
    return dx, ambiguity(r, x), float(peak)


def step(n):
    r = ssh("cd /home/pi && python3 rig_step.py %d --light" % int(n), timeout=300)
    if r.returncode:
        raise RuntimeError("motor move failed: %s" % (r.stderr.strip()[:200]))
    return r.stdout.strip()


def lights_off():
    ssh("pinctrl set 17 op dl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--set-ref", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--deadband", type=float, default=1.5)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    if a.every > 1:
        n = (json.load(open(COUNTER))["n"] + 1) if os.path.exists(COUNTER) else 1
        json.dump({"n": n}, open(COUNTER, "w"))
        if n % a.every:
            print("zero: call %d of every %d -- skipping this time" % (n, a.every))
            return

    try:
        if a.calibrate:
            before = grab("a")
            step(a.steps)
            after = grab("b", lights=True)
            dx, amb, resp = shift_px(before, after)
            if abs(dx) < 2 or resp < 0.25 or amb > 0.80:
                print("calibration failed: %d steps moved the ruler %.2f px "
                      "(confidence %.3f). Check the ruler is lit and in view."
                      % (a.steps, dx, resp))
                return
            pps = dx / a.steps
            json.dump({"px_per_step": pps, "probe_steps": a.steps,
                       "probe_dx": dx, "confidence": resp}, open(STATE, "w"), indent=2)
            step(-a.steps)
            print("calibrated: %+d steps moved the ruler %+.2f px  (confidence %.3f)"
                  % (a.steps, dx, resp))
            print("  %.5f px per step; 1 px of drift = %.0f steps = %.4f deg"
                  % (pps, 1 / abs(pps), 360.0 / 38851.0 / abs(pps)))
            return

        if a.set_ref:
            np.save(REF, grab("ref"))
            print("reference saved; this position is now zero")
            return

        if not os.path.exists(REF):
            print("no reference yet -- run --set-ref first")
            return
        ref = np.load(REF)
        now = grab("now")
        dx, amb, resp = shift_px(ref, now)
        st = json.load(open(STATE)) if os.path.exists(STATE) else None
        line = "drift %+.2f px  (peak %.3f, runner-up %.0f%% of it)" % (dx, resp, 100 * amb)
        steps = None
        if st:
            steps = -dx / st["px_per_step"]
            line += "  =  %+.0f steps  =  %+.3f deg" % (steps, steps * 360.0 / 38851.0)
        print(line)
        rec = {"when": time.strftime("%Y-%m-%d %H:%M"), "dx": dx, "conf": resp,
               "steps": steps, "corrected": False}
        if resp < 0.25:
            print("  match too weak to trust (peak %.2f) -- doing nothing" % resp)
        elif amb > 0.80:
            print("  AMBIGUOUS: the runner-up peak is %.0f%% of the best, so the"
                  % (100 * amb))
            print("  ruler is aliasing and this offset is arbitrary. Refusing to move.")
            print("  A non-repeating mark on the rim would fix this for good.")
        elif a.correct and st:
            if abs(dx) < a.deadband:
                print("  inside the %.1f px deadband -- left alone" % a.deadband)
            else:
                # Closed loop rather than one calculated move. The rim's
                # apparent speed is not constant across the field -- the same
                # 400 steps read 13.5 px at one platter angle and 19 px at
                # another -- so a single px/step constant cannot null the drift
                # in one go. Measuring after each move and re-estimating from
                # what actually happened absorbs that.
                # Re-estimating the scale from each move sounds right and is a
                # trap. The first move after a reversal is swallowed by about a
                # degree of backlash, so it looks like the platter barely
                # responds -- 1070 steps moved the ruler 7 px, implying 0.0065
                # px/step against a calibrated 0.051. The loop then asked for
                # 7354 steps, swung 68 degrees past, and left the platter worse
                # than it found it.
                #
                # So: never stray far from the calibrated scale, never command a
                # move larger than the calibrated one needs, and stop the moment
                # a move makes things worse rather than trying to recover.
                cal = st["px_per_step"]
                pps = cal
                moved_total = 0
                best_dx = dx
                for it in range(4):
                    n = int(round(-dx / pps))
                    limit = int(abs(dx / cal) * 1.5) + 50
                    n = max(-limit, min(limit, n))
                    if n == 0:
                        break
                    step(n); moved_total += n
                    after = grab("after%d" % it)
                    dx_new, amb_new, r_new = shift_px(ref, after)
                    if amb_new > 0.80:
                        print("  match became ambiguous -- stopping")
                        break
                    print("  step %+d -> drift %+.2f px (was %+.2f, peak %.2f)"
                          % (n, dx_new, dx, r_new))
                    if r_new < 0.25:
                        print("  match lost -- stopping here")
                        break
                    if abs(dx_new) > abs(best_dx) + 0.5:
                        print("  that made it worse -- stopping, not chasing it")
                        dx = dx_new
                        break
                    best_dx = min(best_dx, abs(dx_new), key=abs) if False else dx_new
                    # only trust a re-estimate that stays within 2x of calibration
                    if abs(dx_new - dx) > 1.0:
                        est = (dx - dx_new) / float(-n)
                        if 0.5 * abs(cal) < abs(est) < 2.0 * abs(cal):
                            pps = est
                    dx = dx_new
                    if abs(dx) < a.deadband:
                        break
                rec["corrected"] = True
                rec["residual"] = dx
                rec["moved_steps"] = moved_total
        log = json.load(open(LOG)) if os.path.exists(LOG) else []
        log.append(rec)
        json.dump(log, open(LOG, "w"), indent=2)
    finally:
        lights_off()


if __name__ == "__main__":
    main()
