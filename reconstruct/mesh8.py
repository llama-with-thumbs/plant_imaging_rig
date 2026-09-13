"""Build the surface from the vote field instead of from a thresholded blob.

The dents in the body are not carved by the masks -- only 0.3% of the pixels
the chroma key removes lie on the silhouette rim, so the outlines are clean.
They are made when the mesh is built.

The old path threw away almost everything it knew. `votes` counts how many of
the 36 views accept each voxel, which is a smooth scalar field; thresholding it
to a binary blob discards that and leaves a staircase, marching cubes turns the
staircase into facets, and smoothing afterwards can only blur what is already
wrong. Near the surface the count changes gradually, so taking the isosurface
of the count *directly* puts the boundary between voxels rather than on them,
and the steps never appear.

Three further things, in the order they matter:

  * Grey closing fills the craters. A cave is a handful of voxels a few views
    disagreed about; a closing over a small ball removes anything smaller than
    the ball without touching the outline.
  * Smooth before decimating, not after. Quadric decimation preserves what
    looks like detail, and to it surface noise is detail -- so it spends
    triangles on the dents and then they survive the smoothing.
  * Fewer, better triangles. Once the surface is smooth a far lower budget
    holds the shape, because the triangles go on the shape instead of the noise.
"""

import json
import math
import os

import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage
from skimage import measure

import carve8
import persp
import robust

HERE = os.path.dirname(os.path.abspath(__file__))
NXZ, NY, VOTES = 260, 380, 32

# The geometry to scale the mesh with. Set this before building.
#
# It used to be read from geometry_g8.json every time, a file written by
# whichever carve last ran. That is a global with no owner: carving one subject
# and meshing another silently scaled the second by the first one's millimetres
# per pixel -- a 95 mm figure came out 195 mm tall, exactly the height of the
# object measured the run before. Passing it in makes the dependency explicit
# and impossible to get stale.
GEOM = None


def _geom():
    if GEOM is not None:
        return GEOM
    return json.load(open(os.path.join(HERE, "geometry_g8.json")))


def vote_field():
    cache = os.path.join(HERE, "votes_g8.npy")
    if os.path.exists(cache):
        v = np.load(cache)
        if v.shape == (NXZ, NY, NXZ):
            return v
    masks, ang = carve8.load_masks()
    g = carve8.geometry(masks)
    persp.CU, persp.CV = g["cu"], g["cv"]
    v, _, _, _ = robust.vote_volume(masks, ang, g["axis"], g["top"], g["height_px"],
                                    g["radius"], g["pitch"], g["k"], NXZ, NY)
    np.save(cache, v)
    return v


def roughness(tm):
    """How far each vertex sits from the average of its neighbours.

    A smooth surface has vertices close to their neighbours' mean; dents and
    staircase steps push them off it. Reported relative to the mesh size so the
    number can be compared between meshes of different triangle counts.
    """
    import collections
    nbr = collections.defaultdict(set)
    f = tm.faces
    for a, b, c in f:
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
    V = tm.vertices
    d = np.array([np.linalg.norm(V[i] - V[list(n)].mean(axis=0))
                  for i, n in nbr.items() if n])
    return float(d.mean() / np.ptp(V, axis=0).max() * 1000)   # per mille of size


def prepare_field(votes, close_radius, sigma):
    """The smoothed, padded vote field and the pad offset, shared by all backends."""
    pad = int(close_radius + 3 * sigma + 3)
    field = np.pad(votes.astype(np.float32), pad)
    if close_radius:
        r = close_radius
        zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
        ball = (xx * xx + yy * yy + zz * zz) <= r * r
        field = ndimage.grey_closing(field, footprint=ball)
    if sigma:
        field = ndimage.gaussian_filter(field, sigma)
    return field, pad


def to_mm(tm, pad):
    """Map grid coordinates to millimetres, anchored to the axis and grid top."""
    g = _geom()
    px_xz = (2 * g["radius"]) / (NXZ - 1)
    px_y = g["height_px"] / (NY - 1)
    mm = g["mm_per_px"]
    V = np.asarray(tm.vertices) - pad
    tm.vertices = np.column_stack([
        (V[:, 0] - (NXZ - 1) / 2.0) * px_xz * mm,
        (g["height_px"] - V[:, 1] * px_y) * mm,
        (V[:, 2] - (NXZ - 1) / 2.0) * px_xz * mm,
    ])
    return tm


def build(votes, close_radius, sigma, tris, label):
    # Pad with empty space FIRST. The object reaches the edge of the voxel grid
    # -- it stands on the platter, so its base is the last row -- and both the
    # closing and the blur push values outward from there. Padding afterwards is
    # too late: the field already reads VOTES on the array face, the isosurface
    # is cut by the boundary, and the mesh comes out open along the whole base.
    # The margin has to clear the closing ball and the blur's reach.
    pad = int(close_radius + 3 * sigma + 3)
    field = np.pad(votes.astype(np.float32), pad)
    if close_radius:
        # grey closing: fill craters smaller than the ball, leave the outline
        r = close_radius
        zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
        ball = (xx * xx + yy * yy + zz * zz) <= r * r
        field = ndimage.grey_closing(field, footprint=ball)
    if sigma:
        field = ndimage.gaussian_filter(field, sigma)
    face_max = max(field[0].max(), field[-1].max(), field[:, 0].max(),
                   field[:, -1].max(), field[:, :, 0].max(), field[:, :, -1].max())
    assert face_max < VOTES, "field reaches %.1f on the boundary, surface would be cut" % face_max
    # Take the isosurface half a vote BELOW the threshold, never on it. The
    # votes are integers, so a level of exactly 33 coincides with ~120,000
    # samples; marching cubes has no consistent way to triangulate a cell whose
    # corners sit exactly on the level, and the surface shatters -- 8005
    # fragments and nothing watertight. Half a step off, it is 6 components and
    # the body closes. "Keep voxels with >= 33 votes" is the same set either way.
    verts, faces, _, _ = measure.marching_cubes(field, level=VOTES - 0.5)

    g = _geom()
    px_xz = (2 * g["radius"]) / (NXZ - 1)
    px_y = g["height_px"] / (NY - 1)
    mm = g["mm_per_px"]

    # Anchor the mesh to the voxel grid, not to its own extent.
    #
    # Centring x on the vertex centroid and measuring y from the lowest vertex
    # loses the only two things that tie the model to the camera: the grid
    # centre IS the rotation axis, and grid row 0 IS the top of the object. A
    # mesh re-centred on itself cannot be projected back into the photographs,
    # which is why the end-to-end silhouette score read 23% too fat and 14% too
    # thin at the same time -- the shape was right and simply in the wrong place.
    v = verts - pad                              # undo the padding offset
    x_px = (v[:, 0] - (NXZ - 1) / 2.0) * px_xz   # 0 on the rotation axis
    z_px = (v[:, 2] - (NXZ - 1) / 2.0) * px_xz
    y_px = v[:, 1] * px_y                        # 0 at the top of the grid
    pts = np.column_stack([
        x_px * mm,
        (g["height_px"] - y_px) * mm,            # y measured up from the base
        z_px * mm,
    ])

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pts.astype(np.float64)),
                                  o3d.utility.Vector3iVector(faces.astype(np.int32)))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m = m.filter_smooth_taubin(number_of_iterations=20)     # smooth BEFORE decimating
    d = m.simplify_quadric_decimation(target_number_of_triangles=tris)
    d.remove_duplicated_vertices(); d.remove_degenerate_triangles()
    d.remove_unreferenced_vertices()
    # process=True merges coincident vertices. Without it the decimator's seam
    # duplicates leave the mesh looking like thousands of tiny gaps, which reads
    # as "not watertight" and defeats fill_holes -- there is no single hole to
    # fill, just unmerged seams.
    tm = trimesh.Trimesh(np.asarray(d.vertices), np.asarray(d.triangles), process=True)
    tm.merge_vertices()
    parts = tm.split(only_watertight=False)
    if len(parts) > 1:
        tm = max(parts, key=lambda p: len(p.faces))
    if not tm.is_watertight:
        trimesh.repair.fill_holes(tm)
        trimesh.repair.fix_winding(tm)
        trimesh.repair.fix_normals(tm)
    bb = tm.bounds[1] - tm.bounds[0]
    print("  %-22s %6d tris  rough %5.2f  bbox %.1f x %.1f x %.1f  vol %4.0f cm3  genus %+d"
          % (label, len(tm.faces), roughness(tm), bb[0], bb[1], bb[2],
             tm.volume / 1000 if tm.is_watertight else -1,
             (2 - tm.euler_number) // 2), flush=True)
    return tm


if __name__ == "__main__":
    votes = vote_field()
    print("vote field %s, range %d..%d\n" % (votes.shape, votes.min(), votes.max()))
    print("  variant                 tris        roughness   size (mm)            volume  genus")
    build(votes, 0, 0.0, 60000, "binary-ish (as now)")
    build(votes, 0, 1.2, 24000, "soft field, sigma 1.2")
    build(votes, 2, 1.6, 16000, "closed r2, sigma 1.6")
    best = build(votes, 3, 2.0, 12000, "closed r3, sigma 2.0")
    build(votes, 3, 2.6, 8000, "closed r3, sigma 2.6")
    best.export(os.path.join(HERE, "clean_g8.stl"))
    print("\nwrote clean_g8.stl")
