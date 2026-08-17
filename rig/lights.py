"""Mains lighting control through a relay with a DC control input.

A GPIO pin drives the relay's control terminal; the 110 V side stays sealed
inside the relay box and is never touched by this code or by the Pi.

Two behaviours are layered, because they answer different needs:

* A **schedule** keeps the light on through part of the day and off at night,
  which is what the plant needs -- continuous light is bad for most species.
* A **capture pulse** lights the subject briefly for any photograph taken
  outside that window, so night frames are still exposed like day frames.

The pulse exists because timelapse and photogrammetry are both unforgiving of
inconsistent lighting.  LEDs also drift in brightness and colour for a while
after switch-on, so the pulse waits out a settle period before the shutter --
without it the first frames of a revolution would not match the last.
"""

import time
from contextlib import contextmanager
from datetime import datetime

from gpiozero import OutputDevice


def _minutes(hhmm):
    """'HH:MM' -> minutes since midnight."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def in_window(now_minutes, start, end):
    """Whether now falls inside a start..end window, midnight crossings included."""
    start_m, end_m = _minutes(start), _minutes(end)
    if start_m == end_m:
        return False
    if start_m < end_m:
        return start_m <= now_minutes < end_m
    return now_minutes >= start_m or now_minutes < end_m   # e.g. 22:00 -> 06:00


class Lights:
    """A relay-switched lamp with a daily schedule and a capture pulse."""

    def __init__(self, pin, schedule=(), settle_seconds=20, active_high=True):
        self.schedule = list(schedule)
        self.settle_seconds = settle_seconds
        self._device = OutputDevice(pin, active_high=active_high,
                                    initial_value=False)

    # -- state ------------------------------------------------------------

    @property
    def is_on(self):
        return bool(self._device.value)

    def on(self):
        self._device.on()

    def off(self):
        self._device.off()

    # -- schedule ---------------------------------------------------------

    def scheduled_state(self, now=None):
        """Whether the schedule wants the light on at `now` (local time)."""
        now = now or datetime.now()
        minutes = now.hour * 60 + now.minute
        return any(in_window(minutes, start, end) for start, end in self.schedule)

    def apply_schedule(self, now=None):
        """Bring the light in line with the schedule.  Returns the new state."""
        wanted = self.scheduled_state(now)
        if wanted != self.is_on:
            self.on() if wanted else self.off()
        return wanted

    # -- capture ----------------------------------------------------------

    @contextmanager
    def lit(self):
        """Guarantee light for the duration of the block.

        If the lamp is already on -- daytime, per the schedule -- nothing is
        switched and no settle time is spent, since it has long since
        stabilised.  If it is off, it is turned on, allowed to settle, and
        returned to off afterwards, so a night capture does not disturb the
        plant's dark period any longer than the exposure needs.
        """
        was_on = self.is_on
        if not was_on:
            self.on()
            time.sleep(self.settle_seconds)
        try:
            yield
        finally:
            if not was_on:
                self.off()

    # -- lifecycle --------------------------------------------------------

    def close(self):
        """Release the pin.  The relay falls open, so the lamp fails to OFF."""
        self._device.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
