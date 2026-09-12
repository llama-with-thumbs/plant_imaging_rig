"""Move the platter by a raw number of steps. Runs on the Pi.

Deliberately tiny. The zero check needs two things from the rig -- a frame from
the secondary camera and a motor move -- and neither needs OpenCV, which is not
installed on the Pi and is not worth installing there when the machine driving
it already has it. So the Pi grabs and moves; the correlation happens elsewhere.

Uses its own state file. The capture runs keep a stop counter interpreted
against a stop count, and nudging the platter by a raw step count has no
business writing to that.
"""

import argparse
import os
import sys
import time

RIG = "/home/pi/Documents/plant_imaging_rig"
sys.path.insert(0, RIG)
import config
from rig.lights import Lights
from rig.motor import Platter

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int)
    ap.add_argument("--light", action="store_true", help="hold the lights on")
    a = ap.parse_args()

    lights = None
    if a.light:
        lights = Lights(pin=config.LIGHT_PIN, schedule=(),
                        settle_seconds=config.LIGHT_SETTLE_SECONDS,
                        active_high=config.LIGHT_ACTIVE_HIGH)
        lights.on()
    p = Platter(pins=config.MOTOR_PINS, steps_per_rev=config.STEPS_PER_REV,
                stops=36, state_path=os.path.join(HERE, "zero_platter.json"),
                step_delay=config.STEP_DELAY, ramp_steps=config.RAMP_STEPS)
    try:
        with p:
            p.step(a.steps)
        time.sleep(0.8)
        print("stepped %+d" % a.steps)
    finally:
        if lights:
            lights.off()


if __name__ == "__main__":
    main()
