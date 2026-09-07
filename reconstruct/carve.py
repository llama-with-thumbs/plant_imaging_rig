"""Carve a 3D model out of 36 silhouettes.

Standard photogrammetry has to solve for where each camera was, by matching
features between images.  On a glossy bottle against a featureless black screen
there is nothing to match, and it fails.

This rig makes that step unnecessary: the turntable is calibrated, so every
view's angle is known exactly rather than estimated.  That allows the older and
far more robust approach -- carve away every voxel that any silhouette says
cannot be part of the object, and keep what survives all 36.

Geometry.  The camera is fixed and the object turns, which is equivalent to a
camera orbiting a stationary object.  A voxel at (x, y, z) seen at angle theta
projects to column  u = axis + x*cos(theta) + z*sin(theta)  and row  v = top + y.

Projection is orthographic.  The real camera is perspective, so distant parts of
the object are drawn slightly too large -- at roughly 60 cm from a 25 cm object
that is a low-tens-of-percent effect, and correcting it would need a lens
calibration this rig has not had.  The result is a good shape, not a metrology
report.

The method also cannot see concavities: a visual hull is the intersection of
silhouette cones, so the dish under the trigger and the waist behind the label
come out filled.  That is inherent, not a bug.
"""

import glob
import os

import numpy as np
from PIL import Image
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
MASKS = os.path.join(HERE, "masks")

GRID_XZ = 180          # voxels across the turntable
GRID_Y = 260           # voxels vertically
PAD = 1.10             # grid a little wider than the widest silhouette


def load_masks():
    files = sorted(glob.glob(os.path.join(MASKS, "v*.png")))
    masks = [np.asarray(Image.open(f).convert("L")) > 127 for f in files]
    angles = np.deg2rad(np.arange(len(masks)) * 360.0 / len(masks))
    return masks, angles


def geometry(masks):
    """Rotation axis column, and the object's vertical extent."""
    # Averaged over a full revolution, the silhouette centroid sits on the axis:
    # an object mounted off-centre swings symmetrically either side of it.
    cxs, tops, bots, halfw = [], [], [], []
    for m in masks:
        ys, xs = np.where(m)
        cxs.append(xs.mean())
        tops.append(ys.min())
        bots.append(ys.max())
        halfw.append(max(xs.max() - xs.mean(), xs.mean() - xs.min()))
    axis = float(np.mean(cxs))
    top, bottom = int(min(tops)), int(max(bots))
    radius = float(max(halfw)) * PAD
    return axis, top, bottom, radius


def carve(masks, angles, axis, top, bottom, radius):
    h, w = masks[0].shape
    height = bottom - top

    xs = np.linspace(-radius, radius, GRID_XZ)
    zs = np.linspace(-radius, radius, GRID_XZ)
    ys = np.linspace(0, height, GRID_Y)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

    # Start solid, then remove anything any single view rules out.
    keep = np.ones(X.size, dtype=bool)
    for i, (m, th) in enumerate(zip(masks, angles)):
        u = axis + X * np.cos(th) + Z * np.sin(th)
        v = top + Y
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        hit = np.zeros(X.size, dtype=bool)
        hit[inside] = m[vi[inside], ui[inside]]
        keep &= hit
        if i % 6 == 0:
            print("   view %2d/%d  surviving voxels %d" % (i, len(masks), keep.sum()),
                  flush=True)
    return keep.reshape(GRID_XZ, GRID_Y, GRID_XZ)


def write_obj(path, verts, faces):
    with open(path, "w") as fh:
        fh.write("# Plant Imaging Rig - visual hull from 36 silhouettes\n")
        for v in verts:
            fh.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for f in faces:
            fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))


if __name__ == "__main__":
    masks, angles = load_masks()
    print("views: %d" % len(masks))
    axis, top, bottom, radius = geometry(masks)
    print("rotation axis at column %.1f;  object rows %d..%d;  radius %.1f px"
          % (axis, top, bottom, radius))

    vol = carve(masks, angles, axis, top, bottom, radius)
    filled = int(vol.sum())
    print("\nvoxels kept: %d of %d (%.1f%%)"
          % (filled, vol.size, 100.0 * filled / vol.size))
    if filled == 0:
        raise SystemExit("nothing survived carving")

    np.save(os.path.join(HERE, "hull_voxels.npy"), vol)

    # Pad so marching cubes closes the surface at the grid boundary.
    padded = np.pad(vol.astype(np.float32), 1)
    verts, faces, normals, _ = measure.marching_cubes(padded, level=0.5)

    # Voxel indices -> pixel-scaled object coordinates, and flip Y so the
    # bottle stands upright rather than on its head.
    sx = (2 * radius) / (GRID_XZ - 1)
    sy = (bottom - top) / (GRID_Y - 1)
    verts_scaled = np.column_stack([
        (verts[:, 0] - 1) * sx - radius,
        (bottom - top) - (verts[:, 1] - 1) * sy,
        (verts[:, 2] - 1) * sx - radius,
    ])

    write_obj(os.path.join(HERE, "bottle.obj"), verts_scaled, faces)
    print("mesh: %d vertices, %d triangles -> bottle.obj" % (len(verts), len(faces)))
    np.savez(os.path.join(HERE, "mesh.npz"), verts=verts_scaled, faces=faces,
             normals=normals)
