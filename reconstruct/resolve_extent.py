"""Re-solve the vertical extent under the new pitch.

top and bottom fix where the object sits in the frame and therefore the mm
scale. They were fitted under the old 15.75 deg pitch, and the base loss says
they no longer fit. Score each candidate two ways: how well it predicts the
held-out views, and how much of the 270 mm the carve actually spans -- a
geometry that is right should not need to throw the base away.
"""
import json, numpy as np, persp, robust
from scipy import ndimage

masks, ang0 = persp.load()
g = json.load(open("solved_persp.json"))
radius = g["radius"]; axis, pitch, k = 523.32, -7.0, 0.00070
ang = ang0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n=len(masks); tr=np.arange(0,n,2); te=np.arange(1,n,2)

print(" top | bottom | held-out IoU | span of 270 mm | lost at base")
best=None
for top in (37, 47, 57):
    for bottom in (1439, 1459, 1479, 1499):
        height = bottom-top
        v,xs,ys,zs = robust.vote_volume(masks[tr],ang[tr],axis,top,height,radius,pitch,k,140,190)
        vol = v>=int(round(32/36*len(tr)))
        if vol.sum()<1000: continue
        s = robust.score(vol,xs,ys,zs,masks[te],ang[te],axis,top,pitch,k,radius,height)
        occ = vol.any(axis=(0,2)); ny=len(occ); mm=270.0/ny
        first=int(np.argmax(occ)); last=ny-1-int(np.argmax(occ[::-1]))
        span=(last-first)*mm; base=(ny-1-last)*mm
        print("  %3d | %6d | %.4f       | %6.1f mm      | %5.1f mm" % (top,bottom,s,span,base), flush=True)
        if best is None or s>best[0]: best=(s,top,bottom,span,base)
print("\nbest held-out: top %d bottom %d -> IoU %.4f, spans %.1f mm, loses %.1f mm at base"
      % (best[1],best[2],best[0],best[3],best[4]))
