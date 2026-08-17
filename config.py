"""Rig settings.

This is tracked in git on purpose -- it holds no secrets, and the calibration
figure below is a measured property of this particular rig that is worth
keeping under version control.
"""

# --- Turntable -----------------------------------------------------------

# ULN2003 IN1-IN4, in BCM numbering.  Inherited from the original motor_run.py.
MOTOR_PINS = (5, 6, 13, 26)

# Half-steps for one full turn of the *platter*, including the belt reduction.
# Measured on the assembled rig: 80000 half-steps produced exactly two platter
# revolutions.  That is a 9.81:1 reduction, which agrees with a 20-tooth GT2
# pulley driving the ~125 mm platter rim.
#
# Note this does not divide evenly by 24 (1666.67 steps per stop), which is why
# rig/motor.py derives each stop from an absolute target instead of adding a
# rounded increment -- 24 stops still sum to exactly 40000.
STEPS_PER_REV = 40000.0

# Stops per revolution.  24 gives the 15 degree index the rig was designed for.
STOPS = 24

# Seconds per half-step.  1.5 ms ran cleanly for 450 s continuous during
# calibration, so 2 ms keeps a margin of torque in hand for the sake of runs
# that last weeks -- a skipped step is unrecoverable in an open loop, and one
# index still takes only ~3.3 s.
#
# Belt tension matters more than speed here: over-tensioned, this rig lost
# roughly 70% of its steps at every delay tried.
STEP_DELAY = 0.002

# Steps spent ramping in and out of each move.  0 disables ramping.
RAMP_STEPS = 24

# Where the platter position is remembered between runs.
STATE_PATH = "state/platter.json"

# --- Capture -------------------------------------------------------------

INTERVAL_SECONDS = 5 * 60          # one stop every five minutes -> 2 h per rev

# Full native resolution of the IMX477 (Raspberry Pi HQ Camera) fitted to this
# rig.  bio-chart's 2592x1944 belonged to an earlier, smaller sensor.
CAPTURE_WIDTH = 4056
CAPTURE_HEIGHT = 3040
SETTLE_MS = 2000                   # exposure/white-balance settling before the shot
OUTPUT_DIR = "captured_images"

# --- Experiment metadata -------------------------------------------------

SUBJECT = "Amaryllis"
NOTES = ""
