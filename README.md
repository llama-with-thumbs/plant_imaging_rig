# Plant Imaging Rig

Long-duration turntable photography of a growing plant, for reconstructing
animated 3D models of slow biological processes.

Conventional photogrammetry surrounds the subject with many synchronised
cameras. This rig uses one fixed camera and turns the subject instead, which
costs a fraction as much and suits experiments that run for days or weeks
rather than seconds.

The platter indexes **15° every five minutes** and a frame is captured at each
stop, so a full revolution — **24 angles** — completes every **two hours**.

## Hardware

| Part | Detail |
|---|---|
| Controller | Raspberry Pi 4 Model B |
| Camera | CSI camera module, captured via `rpicam-still` |
| Motor | 28BYJ-48 unipolar stepper, 1:64 gearbox |
| Driver | ULN2003 board, IN1–IN4 on BCM **5, 6, 13, 26** |
| Drive | GT2 belt from motor pulley to platter rim, spring-loaded idlers |
| Platter | Lathe-turned stand on bearings, timing belt bonded around the rim |

## Layout

```
app.py           the capture loop: shoot, index, repeat
calibrate.py     measure the platter's true steps-per-revolution
config.py        pins, calibration, interval, capture settings
rig/motor.py     counted-step turntable control with persistent position
rig/capture.py   rpicam-still wrapper
```

## Setup

`gpiozero` and `lgpio` are Raspberry Pi OS system packages and are already
installed. A plain virtualenv **will not see them**, so either use the system
interpreter directly, or build the venv with system packages visible:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

That is the one difference from the older `bio-chart` and `SporeScope` setups,
neither of which needed GPIO.

## Calibration — do this first

`STEPS_PER_REV` in `config.py` ships as `4076.0`, which is the motor's own
output shaft only. The belt reduction to the platter multiplies it, so **the
shipped value is wrong for the assembled rig** and must be measured once.

Mark the platter rim and line it up against a fixed point on the arm, then:

```bash
python calibrate.py reset
python calibrate.py nudge 4000      # most of a turn
python calibrate.py nudge 60        # creep up on the mark
python calibrate.py nudge 12
python calibrate.py status          # total half-steps = STEPS_PER_REV
```

Put that total into `config.py`, then confirm the real indexing path moves in
even fifteenths of a circle:

```bash
python calibrate.py index 24        # should return to 0.00 deg
```

Why this matters: one revolution is not a whole number of steps and 24 stops
divide it even less evenly, so `rig/motor.py` derives each stop from an
absolute target rather than adding a rounded increment each time. Rounding
error therefore stays under half a step forever instead of compounding into
visible creep between revolutions.

## Running

```bash
tmux
source .venv/bin/activate
python app.py
```

`Ctrl+B` then `D` to detach, `tmux attach` to come back.

Frames land one directory per revolution, so a turntable set stays together:

```
captured_images/cycle_00007/angle_13_2026-08-17T04-25-11Z.jpg
```

Platter position is written to `state/platter.json` after every index, so a
restart — or a power cut — resumes at the right angle instead of silently
starting a new revolution mid-turn.

## Notes

- The coils are released between moves. Holding them draws ~200 mA and warms
  the motor and driver for nothing; the 1:64 gearbox holds a level platter on
  its own.
- `STEP_DELAY` is 2 ms rather than the 1 ms used in early testing. A 28BYJ-48
  is marginal at 1 ms under belt tension, and because positioning is open loop
  a skipped step is a permanent error that nothing downstream can detect.
- **The rig indexes in one direction only, and it matters.** 24 forward indexes
  finished 45° short of a full revolution — 12.5% of steps lost — while 24
  reverse indexes returned to the mark exactly. Both directions execute
  identically in software (verified: 1667/1666 alternating, summing to 40000),
  so the asymmetry is mechanical, most likely unequal wrap angle over the two
  spring idlers. `config.DIRECTION` selects the direction that holds position.
- **Belt tension is the dominant variable.** Over-tensioned, this rig lost
  roughly 70% of its commanded steps at every step delay tried; no software
  setting compensated. If positions start drifting, check tension before
  touching `STEP_DELAY`.
- There is no homing switch. Position is only ever relative to wherever the
  platter was when `state/platter.json` was created.
