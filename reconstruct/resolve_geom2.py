"""Run the re-solve to convergence, and ask whether the answer is physical."""
import json, math
import numpy as np
import persp, robust

masks, angles0 = persp.load()
g = json.load(open("solved_persp.json"))
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
MM_PER_PX = 270.0 / height
angles = angles0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n = len(masks); tr = np.arange(0,n,2); te = np.arange(1,n,2)

def evaluate(axis, pitch, k):
    votes, xs, ys, zs = robust.vote_volume(masks[tr], angles[tr], axis, top, height,
                                           radius, pitch, k, 140, 190)
    vol = votes >= int(round(32/36*len(tr)))
    if vol.sum() < 1000: return 0.0
    return robust.score(vol, xs, ys, zs, masks[te], angles[te], axis, top, pitch, k,
                        radius, height)

cur = [524.32, -11.75, 0.00075]; best = evaluate(*cur)
grids = [("axis", np.arange(-4,4.1,1.0)), ("pitch", np.arange(-3,3.01,0.75)),
         ("k", np.arange(-2e-4,2.01e-4,5e-5))]
print("continuing coordinate descent to convergence\n")
for rnd in range(6):
    moved = False
    for idx,(name,deltas) in enumerate(grids):
        base = cur[idx]; row=[]
        for d in deltas:
            t=list(cur); t[idx]=base+d
            s=evaluate(*t); row.append(s)
            if s>best+1e-5: best, cur, moved = s, t, True
        print("  r%d %-5s peak at %+.5g   IoU %.4f   axis %.2f pitch %+.2f k %.5f  (camera %.0f mm)"
              % (rnd+1,name,deltas[int(np.argmax(row))],best,cur[0],cur[1],cur[2],
                 MM_PER_PX/cur[2]), flush=True)
    if not moved:
        print("\nconverged after %d rounds" % (rnd+1)); break

print("\nfinal: axis %.2f  pitch %+.2f  k %.6f" % tuple(cur))
print("implied camera distance %.0f mm   (was %.0f mm under the strict carve)"
      % (MM_PER_PX/cur[2], MM_PER_PX/g["k"]))
print("held-out IoU %.4f" % best)
json.dump({"axis":cur[0],"pitch_deg":cur[1],"k":cur[2],"top":top,"bottom":bottom,
           "radius":radius,"heldout_iou":best}, open("solved_persp_v6.json","w"), indent=2)
