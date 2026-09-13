"""Colour the model from the photographs.

The camera is known exactly -- where the axis is, how far away, how much it
tilts -- so every vertex can be projected into every photograph and the pixel it
lands on read off. No feature matching, no texture unwrapping: the same
projection that carved the shape also says where each point appears.

What takes care is deciding WHICH photographs a vertex may be coloured from.
Project a vertex into all 36 and most of those pixels are wrong: the ones where
the vertex is on the far side of the object, and the ones where it is at such a
grazing angle that a single pixel spans centimetres of surface. Two tests:

  * facing. The surface normal must point towards that camera. A vertex on the
    back of the bag is not coloured from the front view, however cleanly it
    projects there.
  * occlusion. Facing the camera is not enough -- a vertex in a hollow faces the
    camera and is still hidden. Each candidate view gets a depth buffer, and a
    vertex further from the camera than what that view can see is rejected.

Surviving views are averaged, weighted by how square-on the surface is to each,
so a view looking straight at a patch dominates one that grazes it. Weighting
rather than picking the single best view avoids visible seams where the winner
changes from one triangle to the next.
"""

import glob
import json
import math
import os

import cv2
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))


def project(P, g, ang):
    """Object-space points (mm) into image coordinates for one view angle."""
    mm = g["mm_per_px"]
    X = P[:, 0] / mm
    Y = (g["height_px"] * mm - P[:, 1]) / mm
    Z = P[:, 2] / mm
    p = math.radians(g["pitch"])
    ct, st = math.cos(ang), math.sin(ang)
    lat = X * ct + Z * st
    d = -X * st + Z * ct
    s = 1.0 + g["k"] * d
    u = g["cu"] + (g["axis"] + lat - g["cu"]) / s
    v = g["cv"] + (g["top"] + Y * math.cos(p) + d * math.sin(p) - g["cv"]) / s
    return u, v, d


def view_dir(ang, g):
    """Unit vector from the object towards the camera, in object space."""
    p = math.radians(g["pitch"])
    # the camera looks along +d after rotation; towards it is -d
    ct, st = math.cos(ang), math.sin(ang)
    v = np.array([st, -math.sin(p), -ct], dtype=np.float64)
    return v / np.linalg.norm(v)


def colourise(mesh_path, frames_dir, out_ply, shrink=4):
    import carve8, persp
    masks, ang = carve8.load_masks()
    g = carve8.geometry(masks)
    persp.CU, persp.CV = g["cu"], g["cv"]

    tm = trimesh.load(mesh_path)
    tm.merge_vertices()
    V = np.asarray(tm.vertices, dtype=np.float64)
    N = np.asarray(tm.vertex_normals, dtype=np.float64)

    files = sorted(glob.glob(os.path.join(HERE, frames_dir, "v*.jpg")))
    if not files:
        raise SystemExit("no colour frames in %s" % frames_dir)
    n = len(files)

    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    for i, f in enumerate(files):
        img = cv2.rotate(cv2.imread(f), cv2.ROTATE_90_COUNTERCLOCKWISE)
        H0, W0 = img.shape[:2]
        H, W = masks[0].shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        a = ang[i]

        u, v, d = project(V, g, a)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # facing: the normal must point towards this camera
        cam = view_dir(a, g)
        facing = N @ cam
        ok = inside & (facing > 0.15)
        if not ok.any():
            continue

        # occlusion: keep only the nearest surface per pixel, at reduced
        # resolution so a whole triangle is not lost to a one-pixel argument
        sw, sh = W // shrink, H // shrink
        depth = np.full((sh, sw), np.inf)
        su = np.clip(ui // shrink, 0, sw - 1)
        sv = np.clip(vi // shrink, 0, sh - 1)
        order = np.argsort(d)                       # nearest camera first
        oi = order[ok[order]]
        np.minimum.at(depth, (sv[oi], su[oi]), d[oi])
        visible = ok & (d <= depth[sv, su] + 1.5 * shrink)

        idx = np.where(visible)[0]
        if not len(idx):
            continue
        px = img[vi[idx], ui[idx]].astype(np.float64)[:, ::-1]   # BGR -> RGB
        w = facing[idx] ** 2                        # square-on views dominate
        acc[idx] += px * w[:, None]
        wsum[idx] += w

    seen = wsum > 0
    col = np.zeros((len(V), 3))
    col[seen] = acc[seen] / wsum[seen][:, None]
    # a vertex no view could see borrows from its neighbours
    if (~seen).any():
        nbr = tm.vertex_neighbors
        for _ in range(6):
            missing = np.where(~seen)[0]
            if not len(missing):
                break
            for j in missing:
                have = [k for k in nbr[j] if seen[k]]
                if have:
                    col[j] = col[have].mean(axis=0)
                    seen[j] = True
    print("coloured %d of %d vertices from %d views" % (int(seen.sum()), len(V), n))

    tm.visual = trimesh.visual.ColorVisuals(
        mesh=tm, vertex_colors=np.clip(col, 0, 255).astype(np.uint8))
    tm.export(out_ply)
    print("wrote %s (%.2f MB)" % (out_ply, os.path.getsize(out_ply) / 1e6))
    return tm


if __name__ == "__main__":
    import sys
    obj = json.load(open(os.path.join(HERE, "object.json")))
    mesh = sys.argv[1] if len(sys.argv) > 1 else "bag.stl"
    frames = sys.argv[2] if len(sys.argv) > 2 else obj["frames"] + "/colour"
    colourise(os.path.join(HERE, mesh), frames, os.path.join(HERE, "bag_colour.ply"))
