"""Turntable capture for the rebuilt camera: 8 mm lens, portrait mount, no crop.

The old capture path cropped inside the camera pipeline, because the fisheye put
the bottle in a small part of a big frame and the rest was wasted. That crop is
now wrong twice over: the lens has changed, and the camera is mounted on its
side, so the object already fills the frame. The crop was also the source of the
principal-point error -- sensor centre minus a crop origin, with a rescale that
was never applied. Shooting the full sensor makes the principal point simply the
image centre, so that whole class of mistake stops being possible.

Full frame, full resolution, one revolution, lights held on throughout rather
than toggled per shot so the exposure does not hunt between frames.

The platter keeps its position in a state file, and that file records a stop
counter interpreted against a stop count. This run uses 36 stops where the rig's
own config says 24, so it writes to its own state file -- reusing the shared one
would reinterpret a 24-stop counter as a 36-stop one and throw the platter a
long way on the first move.
"""

import argparse
import json
import os
import sys
import time

RIG = "/home/pi/Documents/plant_imaging_rig"
sys.path.insert(0, RIG)

import config
from rig.capture import capture_still
from rig.lights import Lights
from rig.motor import Platter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stops", type=int, default=36)
    ap.add_argument("--out", default=None)
    ap.add_argument("--settle-ms", type=int, default=2500)
    ap.add_argument("--test", action="store_true",
                    help="one frame, no rotation, no state file")
    a = ap.parse_args()

    out = a.out or os.path.expanduser(
        "~/captures/run_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    lights = Lights(pin=config.LIGHT_PIN, schedule=(),
                    settle_seconds=config.LIGHT_SETTLE_SECONDS,
                    active_high=config.LIGHT_ACTIVE_HIGH)

    if a.test:
        lights.on()
        time.sleep(config.LIGHT_SETTLE_SECONDS)
        try:
            p = capture_still(os.path.join(out, "test.jpg"),
                              mode=config.SENSOR_MODE,
                              settle_ms=a.settle_ms, timeout=90)
            print("test frame: %s" % p, flush=True)
            if p and os.path.exists(p):
                print("bytes: %d" % os.path.getsize(p))
        finally:
            lights.off()
        return

    platter = Platter(pins=config.MOTOR_PINS,
                      steps_per_rev=config.STEPS_PER_REV,
                      stops=a.stops,
                      state_path=os.path.join(out, "platter_state.json"),
                      step_delay=config.STEP_DELAY,
                      ramp_steps=config.RAMP_STEPS)

    meta = {"stops": a.stops, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_mode": config.SENSOR_MODE, "roi": None,
            "steps_per_rev": config.STEPS_PER_REV,
            "lens": "GJ-M12-8IR 8mm, portrait mount, full frame",
            "frames": []}

    t0 = time.time()
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
                              settle_ms=a.settle_ms, timeout=90)
                ok = os.path.exists(dest) and os.path.getsize(dest) > 0
                size = os.path.getsize(dest) if ok else 0
                meta["frames"].append({"file": name,
                                       "angle_deg": i * 360.0 / a.stops,
                                       "ok": ok, "bytes": size})
                print("  %s  %6.1f deg  %s  %.1fs"
                      % (name, i * 360.0 / a.stops,
                         "%8d bytes" % size if ok else "FAILED",
                         time.time() - ts), flush=True)
                if i < a.stops - 1:
                    platter.index()
                    time.sleep(0.6)       # let the platter stop ringing
    finally:
        lights.off()

    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta["ok_count"] = sum(1 for f in meta["frames"] if f["ok"])
    json.dump(meta, open(os.path.join(out, "run.json"), "w"), indent=2)
    print("\n%d of %d frames in %.0f s -> %s"
          % (meta["ok_count"], a.stops, meta["elapsed_s"], out))


if __name__ == "__main__":
    main()
