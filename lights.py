"""Control and inspect the rig lighting by hand.

    python lights.py status          what the schedule wants, and what is on
    python lights.py on              force on, and leave it on
    python lights.py off             force off
    python lights.py auto            follow the schedule right now
    python lights.py pulse           on, settle, off -- what a night capture does
    python lights.py preview         the next 24 h of scheduled on/off

The schedule lives in config.py as LIGHT_SCHEDULE, a list of ("HH:MM", "HH:MM")
windows in the Pi's local time; windows may cross midnight.  Edit it there and
the capture loop picks the change up on its next tick.

This tool drives the pin with `pinctrl` rather than gpiozero, because gpiozero
releases its pins when the process exits -- which would switch the lamp back
off the instant `lights.py on` returned.  pinctrl sets the pin and leaves it
set.  While app.py is running it owns the schedule, so a manual override here
lasts only until its next tick.
"""

import argparse
import subprocess
import time
from datetime import datetime, timedelta

import config
from rig.lights import in_window


def set_pin(on):
    """Drive the control pin and leave it driven after this process exits."""
    level = "dh" if on == config.LIGHT_ACTIVE_HIGH else "dl"
    subprocess.run(["pinctrl", "set", str(config.LIGHT_PIN), "op", level],
                   check=True, capture_output=True)


def read_pin():
    """True if the lamp is currently being driven on, None if pin is an input."""
    out = subprocess.run(["pinctrl", "get", str(config.LIGHT_PIN)],
                         capture_output=True, text=True).stdout
    if " op " not in out:
        return None                      # not driven at all -> relay open
    high = "| hi" in out
    return high == config.LIGHT_ACTIVE_HIGH


def scheduled_state(now=None):
    now = now or datetime.now()
    minutes = now.hour * 60 + now.minute
    return any(in_window(minutes, start, end) for start, end in config.LIGHT_SCHEDULE)


def describe(state):
    return "ON" if state else "OFF" if state is not None else "not driven (OFF)"


def cmd_status(args):
    now = datetime.now()
    windows = ", ".join(f"{a}-{b}" for a, b in config.LIGHT_SCHEDULE) or "none"
    print(f"time now      : {now:%Y-%m-%d %H:%M} (local)")
    print(f"schedule      : {windows}")
    print(f"schedule wants: {describe(scheduled_state(now))}")
    print(f"lamp is       : {describe(read_pin())}")
    print(f"control pin   : BCM {config.LIGHT_PIN} "
          f"(active {'high' if config.LIGHT_ACTIVE_HIGH else 'low'})")
    print(f"settle        : {config.LIGHT_SETTLE_SECONDS}s before a capture in the dark")


def cmd_on(args):
    set_pin(True)
    print("Lamp ON. app.py will restore the schedule on its next tick.")


def cmd_off(args):
    set_pin(False)
    print("Lamp OFF. app.py will restore the schedule on its next tick.")


def cmd_auto(args):
    state = scheduled_state()
    set_pin(state)
    print(f"Following schedule: lamp {describe(state)}")


def cmd_pulse(args):
    was_on = read_pin()
    print(f"on -> settle {config.LIGHT_SETTLE_SECONDS}s -> off")
    set_pin(True)
    time.sleep(config.LIGHT_SETTLE_SECONDS)
    print("  lit and settled; this is when the shutter would fire")
    set_pin(bool(was_on))
    print(f"Done, restored to {describe(bool(was_on))}.")


def cmd_preview(args):
    start = datetime.now().replace(second=0, microsecond=0)
    previous = scheduled_state(start)
    print(f"{start:%H:%M}  {describe(previous)}  (now)")
    for step in range(1, 24 * 60 + 1):
        moment = start + timedelta(minutes=step)
        state = scheduled_state(moment)
        if state != previous:
            print(f"{moment:%H:%M}  {describe(state)}")
            previous = state
    lit = sum(1 for step in range(24 * 60)
              if scheduled_state(start + timedelta(minutes=step)))
    print(f"\nLit {lit // 60}h {lit % 60}m of every 24h, plus a "
          f"{config.LIGHT_SETTLE_SECONDS}s pulse per capture in the dark")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in [
        ("status", cmd_status, "show schedule and current state"),
        ("on", cmd_on, "force the lamp on"),
        ("off", cmd_off, "force the lamp off"),
        ("auto", cmd_auto, "follow the schedule now"),
        ("pulse", cmd_pulse, "on, settle, off -- as a capture in the dark does"),
        ("preview", cmd_preview, "print the next 24 h of scheduled changes"),
    ]:
        sub.add_parser(name, help=help_text).set_defaults(func=func)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
