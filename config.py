"""Rig settings.

This is tracked in git on purpose -- it holds no secrets, and the calibration
figure below is a measured property of this particular rig that is worth
keeping under version control.
"""

# --- Turntable -----------------------------------------------------------

# ULN2003 IN1-IN4, in BCM numbering.  Inherited from the original motor_run.py.
MOTOR_PINS = (5, 6, 13, 26)

# Half-steps for one full turn of the *platter*, including the belt reduction.
# 4076 is the motor's own shaft only, and is certainly wrong for the platter --
# run calibrate.py and put the measured value here.
STEPS_PER_REV = 4076.0

# Stops per revolution.  24 gives the 15 degree index the rig was designed for.
STOPS = 24

# Seconds per half-step.  The original script used 0.001, which is at the edge
# of what a 28BYJ-48 will pull under belt tension; 0.002 trades speed for not
# silently losing steps, which open-loop positioning cannot recover from.
STEP_DELAY = 0.002

# Steps spent ramping in and out of each move.  0 disables ramping.
RAMP_STEPS = 24

# Where the platter position is remembered between runs.
STATE_PATH = "state/platter.json"

# --- Capture -------------------------------------------------------------

INTERVAL_SECONDS = 5 * 60          # one stop every five minutes -> 2 h per rev
CAPTURE_WIDTH = 2592
CAPTURE_HEIGHT = 1944
SETTLE_MS = 2000                   # exposure/white-balance settling before the shot
OUTPUT_DIR = "captured_images"

# --- Experiment metadata -------------------------------------------------

SUBJECT = "Amaryllis"
NOTES = ""
