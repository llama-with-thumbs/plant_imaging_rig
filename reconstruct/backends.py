"""Alternative ways to turn the vote field into a surface, and repairs for it.

Marching cubes is not the only option and it has a specific weakness here: it
places vertices on grid edges, so the triangulation inherits the grid's axes and
the result is dense, anisotropic, and full of tiny handles wherever the field
wobbles across the isolevel. Two alternatives, both from libraries already
carrying better machinery than the hand-rolled path:

  poisson  -- open3d's screened Poisson. Fits an implicit function to an
              oriented point cloud, so the surface is smooth by construction
              rather than smoothed afterwards, and it is watertight by
              construction too. It will happily invent surface where there is no
              data, which is normally a hazard; here the hull is closed and
              densely sampled, so there is nowhere for it to invent.

  remesh   -- pymeshlab's isotropic explicit remeshing. Rebuilds the surface
              out of near-equilateral triangles of a chosen edge length, which
              is exactly what "fewer triangles but still recognisable" wants:
              quadric decimation spends its budget where the mesh happens to be
              dense, remeshing spends it evenly over the shape.

The repairs afterwards fix things that are wrong for reasons outside the mesher:

  * the base must be flat, because the bottle stands on a platter. The carve
    ends at the stand cut, so the bottom is a ragged approximation of a plane
    that no amount of smoothing will flatten. Snapping it is both more honest
    and more printable.
  * small shells are never real at this scale, and neither are handles thinner
    than the voxel the field was sampled on.
"""

import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage
from skimage import measure


def _o3d(tm):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)))


def _tri(m):
    t = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=True)
    t.merge_vertices()
    return t


def surface_from_field(field, level):
    verts, faces, normals, _ = measure.marching_cubes(field, level=level)
    return verts, faces, normals


def build_poisson(field, level, tris, depth=8):
    """Screened Poisson from the isosurface's own points and normals.

    Marching cubes already computes a normal at every vertex from the field
    gradient, so the oriented cloud Poisson needs is free and exact -- no
    estimating normals from neighbours and no orientation ambiguity.
    """
    verts, faces, normals = surface_from_field(field, level)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(verts.astype(np.float64))
    n = -normals.astype(np.float64)            # marching cubes points into the solid
    pc.normals = o3d.utility.Vector3dVector(n)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, width=0, scale=1.1, linear_fit=False)
    # No density trimming. The usual reason to trim is that Poisson invents
    # surface where the cloud is sparse -- but this cloud is a closed, densely
    # and evenly sampled isosurface, so there is nowhere sparse to invent into,
    # and trimming only punches holes in a mesh that was watertight.
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    if tris and len(m.triangles) > tris:
        m = m.simplify_quadric_decimation(target_number_of_triangles=tris)
    return _tri(m), verts


def build_remesh(field, level, tris, taubin=12):
    """Isotropic remeshing to a target triangle count, via pymeshlab."""
    import pymeshlab
    verts, faces, _ = surface_from_field(field, level)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts.astype(np.float64), faces.astype(np.int32)))
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("apply_coord_hc_laplacian_smoothing")
    # edge length that lands near the requested triangle count: for a closed
    # surface of area A, n triangles of side L cover about n * L^2 * sqrt(3)/4
    area = trimesh.Trimesh(verts, faces, process=False).area
    L = float(np.sqrt(4.0 * area / (max(tris, 100) * np.sqrt(3.0))))
    ms.apply_filter("meshing_isotropic_explicit_remeshing",
                    targetlen=pymeshlab.PureValue(L), iterations=6,
                    adaptive=False, checksurfdist=False)
    for _ in range(taubin // 6):
        ms.apply_filter("apply_coord_taubin_smoothing")
    m = ms.current_mesh()
    tm = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    tm.merge_vertices()
    return tm, verts


def repair(tm, flatten_base=True, base_tol_mm=1.5):
    """Fix what is wrong for reasons the mesher is not responsible for."""
    notes = []
    parts = tm.split(only_watertight=False)
    if len(parts) > 1:
        tm = max(parts, key=lambda p: len(p.faces))
        notes.append("dropped %d stray shell(s)" % (len(parts) - 1))

    m = _o3d(tm)
    before = len(m.triangles)
    m.remove_duplicated_vertices(); m.remove_duplicated_triangles()
    m.remove_degenerate_triangles(); m.remove_non_manifold_edges()
    m.remove_unreferenced_vertices()
    if len(m.triangles) != before:
        notes.append("removed %d bad triangles" % (before - len(m.triangles)))
    tm = _tri(m)

    if flatten_base and len(tm.vertices):
        # The bottle stands on a flat platter, so its underside is a plane. The
        # carve stops at the stand cut and leaves a ragged approximation of one.
        y = tm.vertices[:, 1]
        floor = y.min()
        near = y < floor + base_tol_mm
        if near.sum() > 8:
            tm.vertices[near, 1] = floor
            notes.append("flattened %d base vertices" % int(near.sum()))

    if not tm.is_watertight:
        trimesh.repair.fill_holes(tm)
        trimesh.repair.fix_winding(tm)
        trimesh.repair.fix_normals(tm)
        notes.append("filled holes" if tm.is_watertight else "STILL OPEN")
    else:
        trimesh.repair.fix_normals(tm)
    return tm, notes
