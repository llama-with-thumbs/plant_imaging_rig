"""Capture the object and, at every stop, a witness frame of the platter rim.

More photographs have made the model worse on this rig, and the reason is not
the spacing: 36 views at 10 degrees reached through 72 moves drift as badly as
72 views do. The error accumulates per move, and with about a degree of
backlash in the belt there is no step size small enough to escape it.

So stop trusting the step counter. The secondary camera looks straight at the
platter rim, and the rim is covered in cork grain and ruler marks -- plenty to
match on. A frame of it at every stop turns each view's angle from something
assumed into something measured, and a view whose angle is known is worth
having however many there already are.

The witness frames are only useful consecutively: matching each against the one
before gives the rotation actually achieved for that move, and those add up to
a real angle for every view. Matching against a fixed reference would not work,
because the rim leaves the secondary camera's field after a few stops.
"""

import argparse
import json
import os
import subprocess
import sys
import time

RIG = "/home/pi/Documents/plant_imaging_rig"
sys.path.insert(0, RIG)

import config
from rig.capture import capture_still
from rig.lights import Lights
from rig.motor import Platter

USB = ("/dev/v4l/by-id/"
       "usb-HD_USB_Camera_HD_USB_Camera_HD_USB_Camera-video-index0")

# Exposure and white balance are pinned, not left to auto.
#
# The chroma key decides object from backdrop by hue and saturation, so it
# assumes the backdrop looks the same in every frame. Auto exposure does not
# promise that: it restarts on every still, and a frame that lands a third of a
# stop bright turns the green screen yellow-green and the key keeps most of the
# frame instead of removing it. One frame in a 50-stop run did exactly that,
# and its mask was four times wider than its neighbours'.
#
# The values are what the camera settles on by itself under the rig's LEDs, so
# pinning them changes nothing except that they stop moving.
LOCK = ["--shutter", "29999", "--gain", "2.11",
        "--awbgains", "2.16,2.71", "--denoise", "cdn_off"]

# For a backlit screen the exposure has to be set for the SCREEN, not the
# object. At the green-screen setting the panel blows out over a third of the
# frame, and an over-bright backdrop blooms into the object's edge -- exactly
# the outline the whole method depends on. Measured across four shutters, 12000
# keeps essentially all the contrast (169 against 178) and edge sharpness (126
# against 132) while dropping clipping from 32.7% to 1.8%.
BACKLIT = ["--shutter", "12000", "--gain", "1.0",
           "--awbgains", "2.16,2.71", "--denoise", "cdn_off"]


def grab_usb(path, skip=8):
    """One frame from the secondary camera, past the auto-exposure ramp."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "v4l2",
           "-input_format", "mjpeg", "-video_size", "1280x960", "-i", USB,
           "-vf", "select=gte(n\\,%d)" % skip, "-frames:v", "1", path]
    try:
        subprocess.run(cmd, check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stops", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle-ms", type=int, default=2500)
    ap.add_argument("--lock", action="store_true",
                    help="pin exposure and white balance (recommended)")
    ap.add_argument("--backlit", action="store_true",
                    help="exposure set for a glowing backdrop rather than a lit object")
    ap.add_argument("--no-lights", action="store_true",
                    help="leave the rig lamps off -- for a backlit silhouette, the "
                         "front lighting only adds reflections to segment around")
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    wit = os.path.join(out, "witness")
    os.makedirs(wit, exist_ok=True)

    lights = Lights(pin=config.LIGHT_PIN, schedule=(),
                    settle_seconds=config.LIGHT_SETTLE_SECONDS,
                    active_high=config.LIGHT_ACTIVE_HIGH)
    platter = Platter(pins=config.MOTOR_PINS, steps_per_rev=config.STEPS_PER_REV,
                      stops=a.stops,
                      state_path=os.path.join(out, "platter_state.json"),
                      step_delay=config.STEP_DELAY, ramp_steps=config.RAMP_STEPS)

    meta = {"stops": a.stops, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_mode": config.SENSOR_MODE, "locked": bool(a.lock),
            "steps_per_rev": config.STEPS_PER_REV, "frames": []}

    t0 = time.time()
    if a.no_lights:
        lights.off()
        print("lamps OFF -- silhouette comes from the backlight alone", flush=True)
        time.sleep(2)
    else:
        lights.on()
        print("lights on, settling %d s" % config.LIGHT_SETTLE_SECONDS, flush=True)
        time.sleep(config.LIGHT_SETTLE_SECONDS)
    try:
        with platter:
            for i in range(a.stops):
                name = "v%02d.jpg" % i
                dest = os.path.join(out, name)
                ts = time.time()
                capture_still(dest, mode=config.SENSOR_MODE,
                              settle_ms=a.settle_ms, timeout=90,
                              extra_args=(BACKLIT if a.backlit
                                          else LOCK if a.lock else None))
                ok = os.path.exists(dest) and os.path.getsize(dest) > 0
                # the witness frame goes with the object frame, same stop
                wok = grab_usb(os.path.join(wit, name))
                meta["frames"].append(
                    {"file": name, "nominal_deg": i * 360.0 / a.stops,
                     "ok": ok, "witness": wok,
                     "bytes": os.path.getsize(dest) if ok else 0})
                print("  %s  %6.1f deg  %s  witness %s  %.1fs"
                      % (name, i * 360.0 / a.stops, "ok" if ok else "FAILED",
                         "ok" if wok else "--", time.time() - ts), flush=True)
                if i < a.stops - 1:
                    platter.index()
                    time.sleep(0.6)
    finally:
        lights.off()

    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta["ok_count"] = sum(1 for f in meta["frames"] if f["ok"])
    meta["witness_count"] = sum(1 for f in meta["frames"] if f["witness"])
    json.dump(meta, open(os.path.join(out, "run.json"), "w"), indent=2)
    print("\n%d of %d frames (%d witnesses) in %.0f s -> %s"
          % (meta["ok_count"], a.stops, meta["witness_count"], meta["elapsed_s"], out))


if __name__ == "__main__":
    main()
