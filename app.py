"""Main capture loop: photograph the plant, index the platter, repeat.

Each tick captures at the position the platter is already resting at, then
moves to the next stop.  Shooting first means the move happens in the slack
between ticks, so the platter has minutes to settle rather than seconds.
"""

import os
import time
from datetime import datetime, timezone

from contextlib import nullcontext

import config
from rig.capture import cameras_available, capture_still
from rig.motor import Platter


def frame_path(cycle, angle_index, timestamp):
    """One directory per revolution, so a 24-frame turntable set stays together."""
    return os.path.join(
        config.OUTPUT_DIR,
        f"cycle_{cycle:05d}",
        f"angle_{angle_index:02d}_{timestamp}.jpg",
    )


def run():
    if not cameras_available():
        raise SystemExit(
            "No camera detected. Check the CSI ribbon is seated (contacts toward "
            "the HDMI ports) and that rpicam-hello --list-cameras finds it."
        )

    platter = Platter(
        pins=config.MOTOR_PINS,
        steps_per_rev=config.STEPS_PER_REV,
        stops=config.STOPS,
        state_path=config.STATE_PATH,
        step_delay=config.STEP_DELAY,
        ramp_steps=config.RAMP_STEPS,
    )

    lights = None
    if config.LIGHTS_ENABLED:
        from rig.lights import Lights
        lights = Lights(pin=config.LIGHT_PIN,
                        schedule=config.LIGHT_SCHEDULE,
                        settle_seconds=config.LIGHT_SETTLE_SECONDS,
                        active_high=config.LIGHT_ACTIVE_HIGH)

    print(f"Resuming at stop {platter.stop} "
          f"(cycle {platter.cycle}, angle {platter.angle:.1f} deg)")

    with platter:
        # Tick against a fixed origin rather than sleeping a fixed amount, so
        # capture times stay on the interval instead of walking by the length
        # of each move.
        origin = time.monotonic()
        tick = 0

        while True:
            wait = origin + tick * config.INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            path = frame_path(platter.cycle, platter.angle_index, timestamp)

            # Bring the lamp in line with the day/night schedule, then ensure
            # light for the exposure itself -- lit() is a no-op when the
            # schedule already has it on, and pulses it when it does not.
            if lights:
                lights.apply_schedule()

            with (lights.lit() if lights else nullcontext()):
                captured = capture_still(path, config.CAPTURE_WIDTH,
                                         config.CAPTURE_HEIGHT,
                                         settle_ms=config.SETTLE_MS)

            if captured:
                print(f"{timestamp}  cycle {platter.cycle} "
                      f"angle {platter.angle_index:02d} ({platter.angle:6.1f} deg)  {path}")
                if config.PUBLISH:
                    # Never let the website take the rig down with it: a failed
                    # push is a website problem, not a reason to stop capturing.
                    try:
                        from publish import publish
                        publish(path)
                    except Exception as error:
                        print(f"  publish failed (continuing): {error}")
            else:
                # A failed frame must not desynchronise the platter from the
                # schedule, so the index still happens.
                print(f"{timestamp}  capture failed at angle {platter.angle_index:02d}")

            platter.index(config.DIRECTION)
            tick += 1


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
