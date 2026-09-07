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
# Set 2026-09-04 after a seized tensioner idler was found and freed.  With the
# idler locked, the belt had been skidding across it instead of rolling, which
# is what produced every symptom we chased for days: ~70% step loss when
# over-tensioned, a 12.5% loss in one direction but not the other, and a
# scatter of ~27 degrees between supposedly identical revolutions.  None of it
# was the motor, the driver, or the software.
#
# With the idler free, 40000 half-steps over-rotated the platter by 15 degrees
# (375 instead of 360), giving 40000 * 360/375 = 38400.  That divides evenly
# into 24 stops of exactly 1600 steps, which is a good sign in itself -- the
# earlier figures never did.
#
# Verify after any mechanical change: mark the rim, command one revolution,
# and check it returns.  rig/motor.py derives each stop from an absolute target
# rather than adding a rounded increment, so 24 stops sum to exactly this
# number even when it does not divide evenly.
# Refined 2026-09-04: at 38400 the platter ran 4.5 deg short per revolution
# (45 deg cumulative over ten), so 38400 * 360/355.5 = 38886.
#
# Refined again 2026-09-06, this time instrumented rather than eyeballed.  A USB
# camera watches a ruler on the platter rim, and cross-correlating that strip
# between frames measures displacement to a fraction of a pixel.  Calibrating
# against known 2-degree steps gave 5.271 px/deg, so the +17 px of drift over
# ten revolutions is 0.322 deg per revolution of OVER-rotation:
#   38886 * 360/360.322 = 38851
#
# 0.322 deg/rev sounds negligible but accumulates: at one orbit per hour it is
# ~8 deg a day, ~54 deg a week, which would ruin a long run.
STEPS_PER_REV = 38851.0

# Stops per revolution.  24 gives the 15 degree index the rig was designed for.
STOPS = 24

# Which way the platter indexes.  Rotation direction is arbitrary for
# photogrammetry, so this is free to choose.
#
# It was originally set from an apparent asymmetry -- forward losing 45 degrees
# per revolution while reverse seemed exact.  That was the seized idler, which
# dragged differently in each direction; both readings were artefacts of it and
# neither should be trusted.  Worth re-testing forward now that the fault is
# fixed, though there is no reason to expect a difference any more.
DIRECTION = -1

# Seconds per half-step.  1.5 ms ran cleanly for 450 s continuous even while
# the tensioner idler was still seized, so with that fault fixed it has margin
# to spare, and a revolution takes 58 s rather than 78 s.
#
# Faster is probably available now, but only raise it alongside a repeatability
# run: a skipped step is unrecoverable in an open loop.
STEP_DELAY = 0.0015

# Steps spent ramping in and out of each move.  0 disables ramping.
RAMP_STEPS = 48

# Where the platter position is remembered between runs.
STATE_PATH = "state/platter.json"

# --- Capture -------------------------------------------------------------

INTERVAL_SECONDS = 5 * 60          # one stop every five minutes -> 2 h per rev

# The subject sits against a beige backdrop occupying about a seventh of the
# very wide lens's view, so the camera crops to it in-pipeline rather than
# capturing 12 MP of mostly living room.  Measured from the backdrop's own
# edges: x 1472..2482, y 1325..2795 of the 4056x3040 frame, plus a small margin.
#
# As fractions of the frame, "x,y,w,h":
CROP_ROI = "0.355,0.424,0.265,0.516"

# Forcing the full sensor mode matters whenever CROP_ROI is set: asked for a
# small output the camera otherwise selects the binned 2028x1520 mode and
# quietly halves the detail inside the crop.
SENSOR_MODE = "4056:3040:12:P"

# Output size of the cropped frame, matching the ROI 1:1 so no scaling happens.
# rpicam-still rejects odd numbers, hence 1076 rather than the measured 1075.
# Files land at ~280 KB instead of 1.7 MB, which matters over weeks of capture.
CAPTURE_WIDTH = 1076
CAPTURE_HEIGHT = 1570
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

# --- Position readout -----------------------------------------------------

# A second, USB camera watches a ruler sticker on the platter rim, giving a
# direct reading of where the platter actually stopped rather than an inference
# from how much the subject changed.  Max resolution on this camera is 1280x960.
USB_DEVICE = "/dev/video2"
USB_WIDTH = 1280
USB_HEIGHT = 960

# Set False to ignore the lamp entirely (no relay fitted).
LIGHTS_ENABLED = True

# --- Publishing ----------------------------------------------------------

# Push each frame to the GitHub Pages site as it is captured.  Requires the
# deploy key described in the README; leave False until that is set up, or the
# capture loop will log a publish failure every tick.
#
# Note the site is PUBLIC, as is the repository serving it.
PUBLISH = True

# --- Experiment metadata -------------------------------------------------

SUBJECT = "Amaryllis"
NOTES = ""
