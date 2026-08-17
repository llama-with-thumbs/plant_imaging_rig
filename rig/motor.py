"""Turntable control for the plant imaging rig.

The platter is turned by a 28BYJ-48 unipolar stepper through a ULN2003 board,
belt-reduced down to the platter rim.  Positioning is open loop, so nothing
here can measure where the platter actually is -- every move is counted in
half-steps and the running position is written to disk so a restart picks up
where the last run left off.

Two details matter more than they look:

* One revolution is not a whole number of steps, and 24 stops divide it even
  less evenly.  Each stop is therefore derived from an absolute target rather
  than by repeatedly adding a rounded increment, so rounding error never
  accumulates -- 24 indexes always close the circle exactly.
* The coils are released between moves.  Holding them costs ~200 mA and warms
  both motor and driver for days on end, and the 1:64 gearbox holds a level
  platter perfectly well on its own.
"""

import json
import os
import time

from gpiozero import OutputDevice

# Half-step sequence for the 28BYJ-48.  Walking it backwards reverses travel.
HALF_STEP_SEQUENCE = [
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
]

# Half-steps per revolution of the motor's own output shaft: 8 phases x 8
# teeth x the 63.68:1 gearbox.  The belt reduction to the platter sits on top
# of this, so it is only a starting point -- run calibrate.py for the real one.
MOTOR_HALF_STEPS_PER_REV = 4076.0


class Platter:
    """A belt-driven turntable that indexes in fixed angular stops."""

    # How much slower the first and last steps of a move are than the middle.
    # Starting a loaded belt at full rate is the easiest way to lose steps.
    RAMP_FACTOR = 3.0

    def __init__(self, pins, steps_per_rev, stops, state_path,
                 step_delay=0.002, ramp_steps=24):
        self.steps_per_rev = float(steps_per_rev)
        self.stops = int(stops)
        self.state_path = state_path
        self.step_delay = step_delay
        self.ramp_steps = ramp_steps

        self._coils = [OutputDevice(pin, initial_value=False) for pin in pins]

        state = self._load_state()
        self.stop = state["stop"]
        self.steps = state["steps"]
        self.phase = state["phase"]

    # -- position ---------------------------------------------------------

    def _target_steps(self, stop):
        """Absolute step count at which the given stop index sits.

        Deriving every stop from stop 0 is what keeps the rounding error
        bounded at half a step forever instead of compounding once per index.
        """
        return round(stop * self.steps_per_rev / self.stops)

    @property
    def angle(self):
        """Current platter angle in degrees, 0-360."""
        return (self.steps / self.steps_per_rev * 360.0) % 360.0

    @property
    def cycle(self):
        """How many complete revolutions have been made.

        Counted by magnitude so that indexing backwards -- which is how this
        rig runs, see config.DIRECTION -- still numbers its cycles 0, 1, 2
        rather than 0, -1, -2 and its output directories with them.
        """
        return abs(self.stop) // self.stops

    @property
    def angle_index(self):
        """Which stop within the current revolution, 0..stops-1."""
        return self.stop % self.stops

    # -- movement ---------------------------------------------------------

    def _apply(self, pattern):
        for coil, value in zip(self._coils, pattern):
            coil.value = bool(value)

    def _delay_for(self, i, count):
        """Step delay at position i of a count-step move, ramped at both ends."""
        ramp = min(self.ramp_steps, count // 2)
        if ramp <= 0:
            return self.step_delay
        into = min(i, count - 1 - i)
        if into >= ramp:
            return self.step_delay
        factor = self.RAMP_FACTOR - (self.RAMP_FACTOR - 1.0) * (into / ramp)
        return self.step_delay * factor

    def step(self, count):
        """Move `count` half-steps; negative counts run backwards."""
        if count == 0:
            return
        direction = 1 if count > 0 else -1
        count = abs(count)

        # Schedule against a running deadline so a slow iteration is absorbed
        # by the next one rather than stretching the whole move.
        deadline = time.perf_counter()
        for i in range(count):
            self.phase = (self.phase + direction) % len(HALF_STEP_SEQUENCE)
            self._apply(HALF_STEP_SEQUENCE[self.phase])
            deadline += self._delay_for(i, count)
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        self.steps += count * direction

    def index(self, count=1):
        """Advance `count` stops (15 degrees each at 24 stops) and settle."""
        target_stop = self.stop + count
        self.step(self._target_steps(target_stop) - self.steps)
        self.stop = target_stop
        self.release()
        self.save()

    def release(self):
        """Drop all four coils.  Phase is kept so the next move resumes cleanly."""
        self._apply((0, 0, 0, 0))

    # -- persistence ------------------------------------------------------

    def _load_state(self):
        try:
            with open(self.state_path) as handle:
                state = json.load(handle)
            return {
                "stop": int(state["stop"]),
                "steps": int(state["steps"]),
                "phase": int(state["phase"]),
            }
        except (OSError, ValueError, KeyError):
            return {"stop": 0, "steps": 0, "phase": 0}

    def save(self):
        """Write position atomically, so a power cut cannot leave a half file."""
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = self.state_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump({"stop": self.stop, "steps": self.steps,
                       "phase": self.phase, "angle": round(self.angle, 3)},
                      handle, indent=2)
        os.replace(temporary, self.state_path)

    def reset(self):
        """Declare the current physical position to be stop 0."""
        self.stop = 0
        self.steps = 0
        self.save()

    # -- lifecycle --------------------------------------------------------

    def close(self):
        self.release()
        for coil in self._coils:
            coil.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
