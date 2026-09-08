"""Re-solve the camera now that the carve is robust.

axis, pitch and k were all fitted while the carve was a strict intersection.
That matters: a strict carve shrinks to fit the worst mask, so the geometry that
best explained it is the geometry that best accommodated the clipping. With
voting in the loop the objective is different, so the optimum can move.

Three global parameters, coordinate descent, scored on the eighteen views not
used to build the hull.
"""
import json, math
import numpy as np
import persp, robust

masks, angles0 = persp.load()
g = json.load(open("solved_persp.json"))
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
corr = np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
angles = angles0 + corr
n = len(masks)
tr = np.arange(0, n, 2); te = np.arange(1, n, 2)
NXZ, NY = 140, 190

def evaluate(axis, pitch, k):
    votes, xs, ys, zs = robust.vote_volume(masks[tr], angles[tr], axis, top, height,
                                           radius, pitch, k, NXZ, NY)
    vol = votes >= int(round(32/36*len(tr)))
    if vol.sum() < 1000:
        return 0.0
    return robust.score(vol, xs, ys, zs, masks[te], angles[te], axis, top, pitch, k,
                        radius, height)

cur = [g["axis"], g["pitch_deg"], g["k"]]
best = evaluate(*cur)
print("starting from the strict-carve solution: axis %.2f  pitch %+.2f  k %.5f"
      % tuple(cur))
print("held-out IoU %.4f\n" % best, flush=True)

grids = [("axis",  np.arange(-6, 6.1, 1.5)),
         ("pitch", np.arange(-2.0, 2.01, 0.5)),
         ("k",     np.arange(-2e-4, 2.01e-4, 5e-5))]
for rnd in range(2):
    for idx, (name, deltas) in enumerate(grids):
        base = cur[idx]
        row = []
        for d in deltas:
            trial = list(cur); trial[idx] = base + d
            s = evaluate(*trial)
            row.append(s)
            if s > best:
                best, cur = s, trial
        star = deltas[int(np.argmax(row))]
        print("  round %d  %-5s best offset %+.5g -> IoU %.4f  (now axis %.2f pitch %+.2f k %.5f)"
              % (rnd+1, name, star, best, cur[0], cur[1], cur[2]), flush=True)

print("\nre-solved: axis %.2f  pitch %+.2f  k %.5f   held-out IoU %.4f" % (cur[0], cur[1], cur[2], best))
print("shift from the strict-carve solution: axis %+.2f px, pitch %+.2f deg, k %+.5f"
      % (cur[0]-g["axis"], cur[1]-g["pitch_deg"], cur[2]-g["k"]))
json.dump({"axis": cur[0], "pitch_deg": cur[1], "k": cur[2], "top": top,
           "bottom": bottom, "radius": radius, "heldout_iou": best},
          open("solved_persp_v6.json", "w"), indent=2)
