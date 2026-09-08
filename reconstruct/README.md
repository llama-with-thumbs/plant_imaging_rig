# Reconstruction

Builds a 3D model from a turntable orbit, without photogrammetry.

Photogrammetry solves for camera poses by matching features between images,
which fails on a glossy object against a plain backdrop. That step is also
unnecessary here: the turntable is calibrated, so every view's angle is *known*.
Each silhouette carves away what it rules out; what survives all 36 is the model.

## Pipeline

```bash
python segment_green.py              # orbit36g/*.jpg -> masks_g/*.png   (chroma key)
python solve_geom.py                 # solve rotation axis + camera pitch
python persp.py                      # solve perspective strength
python robust.py                     # choose the voting threshold
python build_final.py _v3            # carve at full resolution -> hull + raw mesh
python process_mesh.py _v3 60000     # clean, smooth, decimate -> printable STL
python render_mesh.py bottle_v3.stl  # shaded previews, no display needed
python compare_models.py             # reprojection IoU against all 36 views
```

## The camera is solved, not assumed

Three parameters describe how the rig sees the object, and all three are
fitted rather than guessed:

| | value | how |
|---|---|---|
| rotation axis | 522.8 px | joint search |
| camera pitch | −15.75° | joint search; matches an independent ellipse fit of 15–17° |
| perspective | camera 431 mm away | solved as inverse distance |

**Scored by held-out validation, not by volume.** Carve from the even-numbered
views, project into the odd ones, and measure IoU against their real
silhouettes. Maximising carved volume is the obvious objective and the wrong
one — it has no interior optimum for pitch and peaks at an implausible 30°,
because a wrong geometry inflates volume as readily as it shrinks it.

Result on all 36 views:

```
orthographic, level camera        IoU 0.7799
+ solved axis and pitch           IoU 0.8173
+ perspective                     IoU 0.8257
+ robust voting (32 of 36)        IoU 0.8545
+ corrected platter angles        IoU 0.8564   (+9.8% overall)
```

## The platter does not stop where it is told

The carve assumes view m was taken at exactly m x 360/36 degrees, which comes
from counting motor steps through a belt. `refine_views.py` tests that: build
the hull from the other 35 views, then ask which angle makes it best explain
the held-out mask. That never uses view m's own assumed angle.

The corrections come out structured rather than random -- a 1- and 2-per-rev
sinusoid explains 66% of their variance, amplitude about 2 degrees, which is
what eccentricity and belt error look like. Three checks, because the effect is
small enough to be an artifact:

- **dose-response** -- half the correction gives half the gain (0.8555 vs 0.8564);
- **sign control** -- the reversed correction makes things *worse* (0.8506), so
  this is not the score rewarding self-consistency;
- **held out** -- fitting the curve on the 18 even views predicts the 18 odd
  views' own measured corrections at r = 0.79, and improves them by the same
  amount.

So it is real, and worth the five parameters. But it is only +0.002 IoU against
robust voting's +0.029, so platter indexing is a minor contributor and not the
explanation for the residual gap.

`probe_angle.py` is why the per-view corrections themselves are not applied
directly: the objective moves about 0.001 of IoU per degree and has no sharp
peak, so 36 free parameters slide along an almost flat surface and rail at
whatever search bound they are given. Only the smooth 5-parameter curve is
trustworthy.

## Don't require unanimity

Strict carving is an intersection, so it is maximally sensitive to the worst
mask in the set: one view clipping the object by a few pixels deletes that
material permanently, however many other views disagree. `robust.py` counts
votes instead and keeps voxels that enough views accept.

Allowing 2 dissenters in 18 held-out views scored best (0.8349 strict -> 0.8521),
and the same ratio on the full set -- 32 of 36 -- gives 0.8545. It also recovered
a real topological feature the strict carve had filled in: the model comes out
genus 1, which is correct, because a trigger sprayer has a finger loop.

## How much headroom is left

`ceiling.py` runs the whole pipeline on synthetic silhouettes generated
analytically, with no camera error, no segmentation error and no lens. It
scores **0.985-0.993**, so the metric's ceiling is ~0.99 and the real bottle's
0.85 leaves genuine room.

What the remaining gap is *not*:

- **not resolution** -- a 280-cubed grid scores the same as a 180-cubed one;
- **not lens distortion** -- fitting a radial term changes IoU by 0.001 across
  a wide range of k1, because the crop is 980 px of a 4056 px frame taken near
  the centre, where a fisheye is mildest;
- **not the scoring method** -- see the ceiling above.

It is **silhouette inconsistency**. Adding views keeps shrinking the hull long
after it should plateau: from 18 to 36 views the real data loses 4.4% of its
volume where synthetic consistent silhouettes lose 0.7%. Leave-one-out shows
every view has 13-20% of its mask unexplained by the hull built from the other
35, worst at the edge-on angles (100-110 and 280-310 degrees). Better masks, or
a subject that cannot shift on the platter, is where the next real gain is.

## Notes that cost time to learn

**Resolution does not improve fidelity; geometry does.** A 280³ grid scores the
same IoU as a 180³ one (0.8078 vs 0.8085). Higher resolution buys a smoother
surface and nothing else — spend the effort on the camera model.

**Score reprojections with the voxel's footprint, not its centre.** Projecting
centres leaves the image striped: at zero pitch every voxel of a given height
lands on one row, so a 150-row grid fills 150 of 1392 image rows. That artifact
alone made zero pitch score 0.076 while every other value scored 0.30, which
looks exactly like a real signal and is not.

**Segment on hue, not brightness.** See `segment_green.py`. A dark-bodied,
white-headed object cannot be separated from a light or dark backdrop by any
single brightness threshold.

**Voxels are not cubes.** The grid spans ±radius across but the whole object
height vertically. Treating indices as isotropic stretches the model — 1.34×
here. `build_final.py` writes both voxel dimensions to `geometry_*.json`.

**Anything rotating *with* the subject must be excluded explicitly.** The cork
stand is not green, so a chroma key keeps it, and since the bottle stands on it
"largest connected component" cannot separate them. Ordinary background is
averaged away by other views; a plinth is carved into the model.

**Taubin smoothing, not Laplacian.** Laplacian shrinks a closed surface toward
its centroid on every iteration, which on a model whose scale is the point
quietly costs millimetres. Taubin alternates signs to cancel that drift.

## Known limits

Concavities always fill — a visual hull is the intersection of silhouette
cones, so the dish behind the trigger is solid. Scale needs one real
measurement from outside (`OBJECT_HEIGHT_MM`). And what remains after the
camera solve is mostly lens distortion, which needs a checkerboard calibration
rather than more fitting.
