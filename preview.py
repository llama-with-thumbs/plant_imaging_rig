"""Publish what the camera sees, on a loop, without moving anything.

For setting the rig up rather than running it: aim the camera, move the
backdrop, reposition the subject, and watch the result on the live site instead
of guessing. Nothing rotates and no frames are kept.

    python preview.py                 every 45 s until Ctrl-C
    python preview.py --interval 90   slower, for a long adjustment session
    python preview.py --once          a single frame, then exit
    python preview.py --no-lights     leave the lamp alone

The lamp is held on for the whole session rather than pulsed per frame, since
the point is to see the scene as the capture loop will light it.

Expect roughly a minute between pressing go and seeing it: the push is quick,
but GitHub Pages takes its time rebuilding. Refresh the page -- it also
re-fetches on its own every 60 s.

    https://llama-with-thumbs.github.io/plant_imaging_rig/
"""

import argparse
import time

import config
from publish import publish
from rig.capture import capture_still, capture_usb

SITE = "https://llama-with-thumbs.github.io/plant_imaging_rig/"
TEMP_FRAME = "/tmp/plant_rig_preview.jpg"
TEMP_RULER = "/tmp/plant_rig_preview_ruler.jpg"


def shoot_and_publish():
    frame = capture_still(TEMP_FRAME, config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT,
                          settle_ms=config.SETTLE_MS,
                          roi=config.CROP_ROI, mode=config.SENSOR_MODE)
    if not frame:
        print("  capture failed")
        return False
    ruler = capture_usb(TEMP_RULER, device=config.USB_DEVICE,
                        width=config.USB_WIDTH, height=config.USB_HEIGHT,
                        skip_frames=25, crop=config.USB_CROP)
    try:
        meta = publish(frame, ruler)
    except Exception as error:
        # A failed push is a website problem; keep previewing regardless.
        print(f"  publish failed: {error}")
        return False
    print(f"  published {meta['captured']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=45.0,
                        help="seconds between frames (default 45)")
    parser.add_argument("--once", action="store_true", help="one frame, then exit")
    parser.add_argument("--no-lights", action="store_true",
                        help="do not touch the lamp")
    args = parser.parse_args()

    lights = None
    if not args.no_lights and config.LIGHTS_ENABLED:
        from rig.lights import Lights
        lights = Lights(pin=config.LIGHT_PIN, schedule=config.LIGHT_SCHEDULE,
                        settle_seconds=config.LIGHT_SETTLE_SECONDS,
                        active_high=config.LIGHT_ACTIVE_HIGH)
        lights.on()
        print(f"lamp on, settling {config.LIGHT_SETTLE_SECONDS}s")
        time.sleep(config.LIGHT_SETTLE_SECONDS)

    print(SITE)
    print("Ctrl-C to stop\n")
    try:
        n = 0
        while True:
            n += 1
            print(f"frame {n}")
            shoot_and_publish()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if lights:
            # Hand the lamp back to the schedule rather than leaving it forced on.
            lights.apply_schedule()
            lights.close()


if __name__ == "__main__":
    main()
