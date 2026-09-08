"""Fit the principal point, which every solve so far has taken on trust.

CU, CV came from arithmetic -- sensor centre minus crop origin -- and were then
held fixed while axis, pitch and k were fitted around them. That is the riskiest
assumption in the model, because the object sits about 1000 px below CV, so the
perspective divide is extrapolating a long way and any error in CV is amplified
all the way down the bottle. It also predicts the base landing 250 px below
where any mask actually has content.
"""
import json, numpy as np, persp, robust

masks, ang0 = persp.load()
g = json.load(open("solved_persp.json"))
top,bottom,radius = g["top"],g["bottom"],g["radius"]; height=bottom-top
ang = ang0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n=len(masks); tr=np.arange(0,n,2); te=np.arange(1,n,2)
AXIS, PITCH, K = 522.82, -15.75, 0.00045          # the published v5 geometry

def ev(cv, axis=AXIS, pitch=PITCH, k=K):
    old = persp.CV; persp.CV = cv
    try:
        v,xs,ys,zs = robust.vote_volume(masks[tr],ang[tr],axis,top,height,radius,pitch,k,140,190)
        vol = v>=int(round(32/36*len(tr)))
        if vol.sum()<1000: return 0.0, 99.0
        s = robust.score(vol,xs,ys,zs,masks[te],ang[te],axis,top,pitch,k,radius,height)
        occ = vol.any(axis=(0,2)); ny=len(occ); mm=270.0/ny
        last = ny-1-int(np.argmax(occ[::-1]))
        return s, (ny-1-last)*mm
    finally:
        persp.CV = old

print("principal point CV, with the published v5 axis/pitch/k held fixed")
print("(CV = 440 is the value taken from crop arithmetic)\n")
print("   CV  | held-out IoU | lost at base")
best=None
for cv in (140, 240, 340, 440, 540, 640, 740, 840, 940):
    s,b = ev(cv)
    mark = "   <- assumed" if cv==440 else ""
    print("  %4d | %.4f       | %5.1f mm%s" % (cv,s,b,mark), flush=True)
    if best is None or s>best[0]: best=(s,cv,b)
print("\nbest CV %d: IoU %.4f, loses %.1f mm at the base" % (best[1],best[0],best[2]))
