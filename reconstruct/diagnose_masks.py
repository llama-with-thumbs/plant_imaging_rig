"""Where exactly do the worst views disagree with the hull -- and which way?

"Silhouette inconsistency" is not actionable until you know the sign. If the
mask is fatter than the hull, the key is spilling and the fix is a tighter key.
If the hull is fatter than the mask, the mask is clipping the object and the
fix is the opposite. Split the disagreement both ways, per view.
"""
import json, math
import numpy as np, cv2
import persp, robust
from refine_views import project, solidify

masks, angles = persp.load()
g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
corr = np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
angles = angles + corr
n = len(masks)

print("leave-one-out: hull from the other 35, compared with the held-out mask\n")
print(" view | ang | mask px | mask-not-hull | hull-not-mask | verdict")
rows = []
for m in range(n):
    keep = np.array([i for i in range(n) if i != m])
    votes, xs, ys, zs = robust.vote_volume(masks[keep], angles[keep], axis, top,
                                           height, radius, pitch, k, 140, 190)
    vol = votes >= int(round(32/36*35))
    img = solidify(project(vol, xs, ys, zs, angles[m], axis, top, pitch, k, masks[m].shape),
                   radius, height)
    mk = masks[m]
    fat = np.logical_and(mk, ~img).sum()      # mask has it, hull doesn't
    thin = np.logical_and(img, ~mk).sum()     # hull has it, mask doesn't
    tot = mk.sum()
    rows.append((m, fat/tot, thin/tot))
    v = "mask fatter" if fat > 2*thin else ("hull fatter" if thin > 2*fat else "mixed")
    print(" %4d | %3.0f | %7d | %6.1f%%       | %6.1f%%       | %s"
          % (m, math.degrees(angles[m])%360, tot, 100*fat/tot, 100*thin/tot, v), flush=True)

R = np.array([(r[1], r[2]) for r in rows])
print("\nmean: mask-not-hull %.1f%%   hull-not-mask %.1f%%" % (100*R[:,0].mean(), 100*R[:,1].mean()))
worst = np.argsort(-(R[:,0]+R[:,1]))[:6]
print("worst views: %s" % ", ".join("%d (%.0f deg)" % (i, math.degrees(angles[i])%360) for i in worst))
np.save("mask_residual.npy", R)
