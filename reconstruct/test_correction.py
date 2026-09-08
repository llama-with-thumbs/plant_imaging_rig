"""Does the smooth angle correction actually help the finished model?

72 free parameters can always improve a fit. Five cannot, unless they describe
something real. So take the smooth 1- and 2-per-rev curve fitted to the
leave-one-out corrections, rebuild the hull with it, and score all 36 views.
"""
import json, math
import numpy as np
import persp, robust

masks, angles = persp.load()
g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
c = json.load(open("view_corrections.json"))
smooth = np.array(c["smooth_d_angle_deg"])
raw = np.array(c["d_angle_deg"])
NXZ, NY = 170, 230

print("full-set reprojection IoU, robust carve 32 of 36\n")
print("  correction applied            | voxels  |  IoU")
for name, corr in (("none (published v4)", np.zeros(len(masks))),
                   ("smooth 1+2 per rev, 5 params", smooth),
                   ("per-view raw, 36 params", raw),
                   ("smooth scaled x0.5", smooth*0.5),
                   ("smooth reversed (control)", -smooth)):
    a = angles + np.deg2rad(corr)
    votes, xs, ys, zs = robust.vote_volume(masks, a, axis, top, height, radius,
                                           pitch, k, NXZ, NY)
    vol = votes >= 32
    s = robust.score(vol, xs, ys, zs, masks, a, axis, top, pitch, k, radius, height)
    print("  %-29s | %7d | %.4f" % (name, vol.sum(), s), flush=True)
