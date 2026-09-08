"""Final carve: solved perspective camera, corrected platter angles, robust voting."""
import json, math, os, sys
import numpy as np
from scipy import ndimage
from skimage import measure
import persp, robust

OBJECT_HEIGHT_MM = 270.0
NXZ = int(os.environ.get("NXZ", 280)); NY = int(os.environ.get("NY", 400))
VOTES = int(os.environ.get("VOTES", 32))

g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
masks, angles = persp.load()
corr = np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"])
angles = angles + np.deg2rad(corr)

print("carving %dx%dx%d from %d views, keeping voxels %d views accept" % (NXZ, NY, NXZ, len(masks), VOTES))
print("  axis %.2f  pitch %+.2f  k %.6f  platter correction +-%.2f deg"
      % (axis, pitch, k, np.abs(corr).max()), flush=True)
votes, xs, ys, zs = robust.vote_volume(masks, angles, axis, top, height, radius, pitch, k, NXZ, NY)
vol = votes >= VOTES
print("  kept %d of %d (%.2f%%)" % (vol.sum(), vol.size, 100*vol.sum()/vol.size), flush=True)

lbl, n = ndimage.label(vol)
if n > 1:
    sizes = ndimage.sum(vol, lbl, range(1, n+1))
    vol = lbl == (1 + int(np.argmax(sizes)))
    print("  largest of %d components: %d voxels" % (n, vol.sum()))
np.save("hull_v5.npy", vol)

px_xz = (2*radius)/(NXZ-1); px_y = height/(NY-1); mm = OBJECT_HEIGHT_MM/height
smooth = ndimage.gaussian_filter(vol.astype(np.float32), 1.2)
verts, faces, _, _ = measure.marching_cubes(np.pad(smooth, 2), level=0.5)
print("  marching cubes: %d verts, %d tris" % (len(verts), len(faces)))
v = verts
pts = np.column_stack([(v[:,0]-v[:,0].mean())*px_xz*mm,
                       (np.ptp(v[:,1])-(v[:,1]-v[:,1].min()))*px_y*mm,
                       (v[:,2]-v[:,2].mean())*px_xz*mm])
pts[:,1] -= pts[:,1].min()
np.save("verts_v5.npy", pts.astype(np.float32)); np.save("faces_v5.npy", faces.astype(np.int32))
bb = pts.max(axis=0)-pts.min(axis=0)
print("  bounding box %.1f x %.1f x %.1f mm" % tuple(bb))
