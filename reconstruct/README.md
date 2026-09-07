# Reconstruction

Builds a 3D model from a turntable orbit, without photogrammetry.

Standard photogrammetry solves for where each camera was by matching features
between images. On a glossy bottle against a featureless black screen there is
nothing to match, and it fails. This rig makes that step unnecessary: the
turntable is calibrated, so every view's angle is *known*. That allows the older
and far more robust visual-hull approach.

## Pipeline

```bash
python segment.py     # orbit36/*.jpg  ->  masks/*.png
python carve.py       # masks          ->  hull_voxels.npy + bottle.obj
python render.py      # voxels         ->  shaded previews from 8 angles
python export_stl.py  # voxels         ->  bottle.stl
```

## Notes that cost time to learn

**Segment with a per-image threshold, not a fixed one.** The camera auto-exposes
every frame, so the same black screen read V=42 in one view and V=78 in another
as the bottle turned more of its bright face to the lens. A fixed threshold
worked on half the views and swallowed the backdrop on the rest. Otsu adapts.

**Check the top edge of the silhouettes.** The object does not move vertically,
so the top of every mask should sit at the same row. A spread of a few pixels
means good masks; a large spread means the segmentation is wandering.

**Cut the stand off.** It turns with the object, so it is not background — it
carves into the model as a plinth unless explicitly removed.

**A visual hull cannot see concavities.** The intersection of silhouette cones
fills any dish or waist that is never on the outline. Inherent, not a bug.

**Scale has to come from outside.** The mesh is in image pixels; only a real
measurement of the object turns it into millimetres. Set `OBJECT_HEIGHT_MM` in
`export_stl.py`.
