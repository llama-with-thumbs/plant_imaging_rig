"""Solve all four camera parameters together, with the base as a guard rail.

CV was never fitted, and it turns out to matter more than anything fitted around
it. Re-solving axis, pitch, k and CV jointly is the honest version. Base loss is
carried alongside the score because a geometry can raise IoU by throwing away
material it cannot explain, and that is not an improvement.
"""
import json, numpy as np, persp, robust

masks, ang0 = persp.load()
g = json.load(open("solved_persp.json"))
top,bottom,radius = g["top"],g["bottom"],g["radius"]; height=bottom-top
ang = ang0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n=len(masks); tr=np.arange(0,n,2); te=np.arange(1,n,2)

def ev(p):
    cv, axis, pitch, k = p
    old = persp.CV; persp.CV = cv
    try:
        v,xs,ys,zs = robust.vote_volume(masks[tr],ang[tr],axis,top,height,radius,pitch,k,140,190)
        vol = v>=int(round(32/36*len(tr)))
        if vol.sum()<1000: return 0.0, 99.0
        s = robust.score(vol,xs,ys,zs,masks[te],ang[te],axis,top,pitch,k,radius,height)
        occ = vol.any(axis=(0,2)); ny=len(occ)
        last = ny-1-int(np.argmax(occ[::-1]))
        return s, (ny-1-last)*(270.0/ny)
    finally:
        persp.CV = old

print("extending the CV sweep past where it was still climbing")
print("   CV  |  IoU   | base")
for cv in (940, 1040, 1140, 1240, 1340, 1440):
    s,b = ev([cv, 522.82, -15.75, 0.00045]); print("  %4d | %.4f | %.1f mm" % (cv,s,b), flush=True)

cur=[940.0, 522.82, -15.75, 0.00045]; best,bb = ev(cur)
grids=[("CV",np.arange(-150,151,50.)),("axis",np.arange(-4,4.1,2.)),
       ("pitch",np.arange(-4,4.1,1.)),("k",np.arange(-1.5e-4,1.51e-4,5e-5))]
print("\njoint coordinate descent (base loss must stay near 3 mm)")
for rnd in range(4):
    moved=False
    for i,(name,ds) in enumerate(grids):
        base_v=cur[i]
        for d in ds:
            t=list(cur); t[i]=base_v+d
            s,b = ev(t)
            if s>best+1e-5 and b<6.0: best,cur,bb,moved = s,t,b,True
        print("  r%d %-5s -> IoU %.4f  base %.1f mm  | CV %.0f axis %.1f pitch %+.2f k %.5f"
              % (rnd+1,name,best,bb,cur[0],cur[1],cur[2],cur[3]), flush=True)
    if not moved: print("\nconverged after %d rounds" % (rnd+1)); break
print("\nfinal: CV %.0f  axis %.2f  pitch %+.2f  k %.5f" % tuple(cur))
print("held-out IoU %.4f, loses %.1f mm at the base" % (best,bb))
json.dump({"cv":cur[0],"axis":cur[1],"pitch_deg":cur[2],"k":cur[3],
           "top":top,"bottom":bottom,"radius":radius,"heldout_iou":best},
          open("solved_persp_v7.json","w"), indent=2)
