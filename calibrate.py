"""Measure how many half-steps one platter revolution actually takes.

The motor's own shaft is about 4076 half-steps per turn, but the belt between
the motor pulley and the platter rim adds a reduction that is not known from
the parts list -- it has to be measured on the assembled rig.  Getting it wrong
is not cosmetic: the 24 stops are derived from this number, so an error here
shows up as the plant appearing to creep around between revolutions.

Typical session, with a pencil mark on the platter rim lined up against a fixed
reference on the arm:

    python calibrate.py reset
    python calibrate.py nudge 4000        # most of a turn, then creep up on it
    python calibrate.py nudge 60
    python calibrate.py nudge 12          # until the mark lines up again
    python calibrate.py status            # -> put this total in config.py

Or, if one turn is awkward to judge, run a known number of steps and report the
angle actually swept:

    python calibrate.py measure 2000
"""

import argparse

import config
from rig.motor import MOTOR_HALF_STEPS_PER_REV, Platter

# Kept apart from the run position so calibrating never disturbs an experiment.
CALIBRATION_STATE = "state/calibrate.json"


def open_platter(steps_per_rev=None):
    return Platter(
        pins=config.MOTOR_PINS,
        steps_per_rev=steps_per_rev or config.STEPS_PER_REV,
        stops=config.STOPS,
        state_path=CALIBRATION_STATE,
        step_delay=config.STEP_DELAY,
        ramp_steps=config.RAMP_STEPS,
    )


def cmd_reset(args):
    with open_platter() as platter:
        platter.reset()
        platter.release()
    print("Counter zeroed. Line the mark up with your reference and start nudging.")


def cmd_nudge(args):
    with open_platter() as platter:
        platter.step(args.steps)
        platter.release()
        platter.save()
        print(f"Moved {args.steps:+d}. Total since reset: {platter.steps} half-steps.")


def cmd_status(args):
    with open_platter() as platter:
        total = platter.steps
    print(f"Total since reset: {total} half-steps")
    if total:
        belt_ratio = total / MOTOR_HALF_STEPS_PER_REV
        print(f"If that is exactly one platter revolution:")
        print(f"  STEPS_PER_REV = {total}.0")
        print(f"  implied belt reduction = {belt_ratio:.4f}:1")
        print(f"  one {360 / config.STOPS:.1f} deg stop = {total / config.STOPS:.2f} half-steps")


def cmd_measure(args):
    with open_platter() as platter:
        print(f"Running {args.steps} half-steps...")
        platter.step(args.steps)
        platter.release()
        platter.save()

    reply = input("Degrees the platter actually swept (blank to abort): ").strip()
    if not reply:
        print("Aborted.")
        return
    try:
        degrees = float(reply)
    except ValueError:
        print(f"Not a number: {reply!r}")
        return
    if degrees <= 0:
        print("Needs to be greater than zero.")
        return

    steps_per_rev = args.steps * 360.0 / degrees
    print(f"\n  STEPS_PER_REV = {steps_per_rev:.1f}")
    print(f"  implied belt reduction = {steps_per_rev / MOTOR_HALF_STEPS_PER_REV:.4f}:1")
    print(f"  one {360 / config.STOPS:.1f} deg stop = {steps_per_rev / config.STOPS:.2f} half-steps")
    print("\nPut that STEPS_PER_REV into config.py. Accuracy scales with the")
    print("angle swept, so re-measure over a full turn if this looks marginal.")


def cmd_index(args):
    """Exercise the real indexing path with whatever config.py currently says."""
    with open_platter() as platter:
        for _ in range(args.count):
            platter.index()
            print(f"stop {platter.stop:4d}  angle {platter.angle:7.2f} deg  "
                  f"steps {platter.steps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reset", help="treat the current position as zero").set_defaults(
        func=cmd_reset)

    nudge = subparsers.add_parser("nudge", help="move N half-steps (negative to reverse)")
    nudge.add_argument("steps", type=int)
    nudge.set_defaults(func=cmd_nudge)

    subparsers.add_parser("status", help="show the running total and what it implies").set_defaults(
        func=cmd_status)

    measure = subparsers.add_parser("measure", help="run N steps, then report the angle swept")
    measure.add_argument("steps", type=int)
    measure.set_defaults(func=cmd_measure)

    index = subparsers.add_parser("index", help="perform N indexes using config.py")
    index.add_argument("count", type=int, nargs="?", default=1)
    index.set_defaults(func=cmd_index)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
