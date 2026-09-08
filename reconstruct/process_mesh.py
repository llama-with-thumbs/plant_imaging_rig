"""Turn a raw iso-surface into a mesh a slicer will accept.

Marching cubes over a binary volume gives a staircase with a triangle per voxel
face and no topological guarantees. Four things fix that, in this order:

1. remove duplicate and degenerate geometry, so later steps operate on a clean
   manifold rather than working around slivers;
2. Taubin smoothing rather than Laplacian -- Laplacian smoothing shrinks a
   closed surface toward its centroid a little on every iteration, which on a
   model whose scale is the point would quietly cost millimetres. Taubin
   alternates a positive and a negative step to cancel that drift;
3. quadric decimation, which collapses edges in flat regions and keeps them
   where curvature lives, unlike uniform subsampling;
4. a watertightness check, because a slicer needs a closed solid and an STL
   that merely looks right can still have holes.

Both open3d and trimesh are used: open3d for the filters, trimesh for the
verdict on whether the result is actually printable.
"""

import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))


def report(mesh, label):
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         process=False)
    bb = tm.bounds[1] - tm.bounds[0]
    print("  %-22s %7d tris  %7d verts  bbox %5.1f x %5.1f x %5.1f mm"
          % (label, len(tm.faces), len(tm.vertices), *bb))
    return tm


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_v2"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 60000

    verts = np.load(os.path.join(HERE, "verts%s.npy" % tag)).astype(np.float64)
    faces = np.load(os.path.join(HERE, "faces%s.npy" % tag)).astype(np.int32)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces))
    print("mesh pipeline:")
    report(m, "raw")

    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    report(m, "cleaned")

    m = m.filter_smooth_taubin(number_of_iterations=12)
    report(m, "taubin x12")

    if len(m.triangles) > target:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target)
        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
    tm = report(m, "decimated")

    m.compute_vertex_normals()

    print()
    print("  watertight        :", tm.is_watertight)
    print("  winding consistent:", tm.is_winding_consistent)
    print("  euler number      :", tm.euler_number)
    print("  volume            : %.1f cm3" % (tm.volume / 1000.0) if tm.is_watertight
          else "  volume            : n/a (not watertight)")

    if not tm.is_watertight:
        print("  repairing...")
        tm.fill_holes()
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fix_winding(tm)
        print("  watertight after repair:", tm.is_watertight)
        if tm.is_watertight:
            print("  volume            : %.1f cm3" % (tm.volume / 1000.0))

    out = os.path.join(HERE, "bottle%s.stl" % tag)
    tm.export(out)
    print("\nwrote %s  (%.2f MB, %d triangles)"
          % (os.path.basename(out), os.path.getsize(out) / 1e6, len(tm.faces)))

    obj = os.path.join(HERE, "bottle%s.obj" % tag)
    tm.export(obj)
    print("wrote %s" % os.path.basename(obj))

    json.dump({"triangles": int(len(tm.faces)), "vertices": int(len(tm.vertices)),
               "watertight": bool(tm.is_watertight),
               "volume_cm3": float(tm.volume / 1000.0) if tm.is_watertight else None,
               "bbox_mm": [float(x) for x in (tm.bounds[1] - tm.bounds[0])]},
              open(os.path.join(HERE, "mesh%s.json" % tag), "w"), indent=2)


if __name__ == "__main__":
    main()
