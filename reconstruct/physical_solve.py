"""Use the principal point the capture chain actually implies, and re-solve.

CU 528 / CV 440 were arithmetic done against the wrong crop origin, and without
accounting for the 0.91 rescale from the 1072x1674 ROI to the stored 980x1520.
The correct values are CU 541, CV 367. Pin those and refit axis, pitch and k.
"""
import json, numpy as np, persp, robust

masks, ang0 = persp.load()
g = json.load(open("solved_persp.json"))
top,bottom,radius = g["top"],g["bottom"],g["radius"]; height=bottom-top
ang = ang0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n=len(masks); tr=np.arange(0,n,2); te=np.arange(1,n,2)

def ev(axis,pitch,k,cu,cv,grid=(140,190)):
    ou,ov = persp.CU, persp.CV; persp.CU, persp.CV = cu, cv
    try:
        v,xs,ys,zs = robust.vote_volume(masks[tr],ang[tr],axis,top,height,radius,pitch,k,*grid)
        vol = v>=int(round(32/36*len(tr)))
        if vol.sum()<1000: return 0.0, 99.0
        s = robust.score(vol,xs,ys,zs,masks[te],ang[te],axis,top,pitch,k,radius,height)
        occ=vol.any(axis=(0,2)); ny=len(occ); last=ny-1-int(np.argmax(occ[::-1]))
        return s, (ny-1-last)*(270.0/ny)
    finally:
        persp.CU, persp.CV = ou, ov

CU, CV = 541.0, 367.0
cur=[522.82,-15.75,0.00045]; best,bb = ev(*cur,CU,CV)
print("principal point pinned at the capture-chain values CU %.0f CV %.0f" % (CU,CV))
print("starting IoU %.4f (base loss %.1f mm)\n" % (best,bb))
grids=[("axis",np.arange(-6,6.1,2.)),("pitch",np.arange(-6,6.1,1.5)),
       ("k",np.arange(-2e-4,2.01e-4,5e-5))]
for rnd in range(4):
    moved=False
    for i,(name,ds) in enumerate(grids):
        b0=cur[i]
        for d in ds:
            t=list(cur); t[i]=b0+d
            s,bl = ev(*t,CU,CV)
            if s>best+1e-5 and bl<6.0: best,cur,bb,moved = s,t,bl,True
        print("  r%d %-5s -> IoU %.4f  base %.1f mm | axis %.1f pitch %+.2f k %.5f"
              % (rnd+1,name,best,bb,cur[0],cur[1],cur[2]), flush=True)
    if not moved: print("\nconverged after %d rounds" % (rnd+1)); break

print("\nphysical principal point + refitted geometry:")
print("  axis %.2f  pitch %+.2f  k %.5f   held-out IoU %.4f  base loss %.1f mm"
      % (cur[0],cur[1],cur[2],best,bb))
json.dump({"cu":CU,"cv":CV,"axis":cur[0],"pitch_deg":cur[1],"k":cur[2],
           "top":top,"bottom":bottom,"radius":radius,"heldout_iou":best},
          open("solved_persp_v7.json","w"),indent=2)
