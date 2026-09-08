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
+ perspective                     IoU 0.8257   (+5.9%)
```

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
