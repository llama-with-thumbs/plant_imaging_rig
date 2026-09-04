"""Rig settings.

This is tracked in git on purpose -- it holds no secrets, and the calibration
figure below is a measured property of this particular rig that is worth
keeping under version control.
"""

# --- Turntable -----------------------------------------------------------

# ULN2003 IN1-IN4, in BCM numbering.  Inherited from the original motor_run.py.
MOTOR_PINS = (5, 6, 13, 26)

# Half-steps for one full turn of the *platter*, including the belt reduction.
#
# WARNING: this number is not currently meaningful, and no value of it would be.
# Measured 2026-09-04 with a rigid test object photographed every 15 degrees
# through 480 degrees of commanded rotation: the platter returned to its
# starting pose at 450 degrees commanded, i.e. it delivers ~80% of what it is
# told.  But the loss is not a fixed ratio -- three consecutive revolutions at
# the corrected figure landed in three different places, differing from each
# other by 6-7 grey levels against a 0.7 noise floor, and worsening each time.
#
# Variable slip cannot be calibrated away, because software has no way to know
# how much was lost.  Fixing it needs positive engagement (gears or a worm
# drive) or an encoder on the platter axis to close the loop.  Until then,
# treat platter angle as an estimate, not a measurement.
#
# The 40000 below came from an earlier count of "two turns per 80000 steps".
# That was taken while the platter was not moving the pot at all, so it is
# unreliable; it is kept only so the indexing arithmetic has something to run
# against.
#
# Note it does not divide evenly by 24 (1666.67 steps per stop), which is why
# rig/motor.py derives each stop from an absolute target instead of adding a
# rounded increment -- 24 stops still sum to exactly 40000.
STEPS_PER_REV = 40000.0

# Stops per revolution.  24 gives the 15 degree index the rig was designed for.
STOPS = 24

# Which way the platter indexes.  The two directions are NOT equivalent on this
# rig: 24 forward indexes finished 45 degrees short of a full turn (12.5% of
# steps lost to belt slip), while 24 reverse indexes returned to the mark
# exactly.  A torque shortfall would have cost both directions equally, so the
# asymmetry is in the drive -- most likely unequal wrap angle over the two
# spring idlers, giving less grip one way round.
#
# Rotation direction is arbitrary for photogrammetry, so index the way that
# holds position.  Revisit if the idler geometry is ever made symmetric.
DIRECTION = -1

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

# --- Lighting ------------------------------------------------------------

# BCM pin driving the relay module's S (signal) terminal.  The lamps are 12 V
# DC LED bars (2 x Litever 5 W, ~0.83 A total) fed by their own adapter; the
# relay breaks the 12 V positive line, so nothing mains-side is switched here.
#
# The fitted module (Inland/SONGLE SRD-05VDC-SL-C) has no optocoupler and
# triggers on a HIGH input -- measured, not assumed: driving BCM 17 high lit
# the module's LED and pulled the coil in.
#
# BCM 17 pairs with that deliberately.  It boots as an input with a pull-DOWN,
# so it idles low and the lamps stay dark until software drives them.  Pairing
# an active-high module with a pull-UP pin such as BCM 4 would instead switch
# the lamps on at every power-up.
LIGHT_PIN = 17              # physical pin 11
LIGHT_ACTIVE_HIGH = True    # measured: coil pulls in on a HIGH input

# Daily on-windows in the Pi's LOCAL time, as ("HH:MM", "HH:MM") pairs.  A
# window may cross midnight ("22:00", "06:00").  Outside these hours the lamp
# is off except for a brief pulse around each capture.
#   python lights.py preview   to check what a change actually does
LIGHT_SCHEDULE = [("08:00", "20:00")]

# Seconds to wait after switching on before capturing in the dark.  LEDs shift
# in brightness and colour temperature for a while after switch-on, and
# photogrammetry is unforgiving of frames that do not match each other.
LIGHT_SETTLE_SECONDS = 20

# Set False to ignore the lamp entirely (no relay fitted).
LIGHTS_ENABLED = True

# --- Publishing ----------------------------------------------------------

# Push each frame to the GitHub Pages site as it is captured.  Requires the
# deploy key described in the README; leave False until that is set up, or the
# capture loop will log a publish failure every tick.
#
# Note the site is PUBLIC, as is the repository serving it.
PUBLISH = False

# --- Experiment metadata -------------------------------------------------

SUBJECT = "Amaryllis"
NOTES = ""
