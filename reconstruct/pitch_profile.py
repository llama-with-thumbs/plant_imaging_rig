"""Measure the platter ellipse cleanly, then ask what the carve thinks of it."""
import glob, math, json
import numpy as np, cv2
import persp, robust

# ---- 1. the physical measurement, with the backdrop excluded ----
vals = []
for f in sorted(glob.glob("orbit36g/v*.jpg")):
    bgr = cv2.imread(f); h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0].astype(int), hsv[:,:,1].astype(int), hsv[:,:,2].astype(int)
    y0 = int(h*0.86)
    green = (((H>=35)&(H<=90)&(S>=60)&(V>=40))[y0:]).astype(np.uint8)
    cork  = (((H>4)&(H<34)&(S>50)&(V>40))[y0:]).astype(np.uint8)
    # the disc is the green blob that does NOT run to either side border
    nlab, lbl, st, cen = cv2.connectedComponentsWithStats(green, 8)
    cand = [i for i in range(1, nlab)
            if st[i,cv2.CC_STAT_LEFT] > 2
            and st[i,cv2.CC_STAT_LEFT]+st[i,cv2.CC_STAT_WIDTH] < w-2
            and st[i,cv2.CC_STAT_AREA] > 3000]
    if not cand: continue
    i = max(cand, key=lambda j: st[j,cv2.CC_STAT_AREA])
    disc = (lbl==i)
    major = st[i,cv2.CC_STAT_WIDTH]; cx = int(cen[i][0])
    colg = disc[:, max(0,cx-8):cx+9].mean(axis=1) > .5
    colc = cork[:, max(0,cx-8):cx+9].mean(axis=1) > .5
    gy, cy = np.where(colg)[0], np.where(colc)[0]
    if not len(gy) or not len(cy): continue
    minor = cy.min() - gy.min()
    if 5 < minor < major:
        vals.append((major, minor, math.degrees(math.asin(minor/major))))
A = np.array(vals)
print("platter ellipse over %d views: major %.0f +- %.0f px, minor %.0f +- %.0f px"
      % (len(A), A[:,0].mean(), A[:,0].std(), A[:,1].mean(), A[:,1].std()))
meas = A[:,2].mean()
print("MEASURED camera elevation: %.2f deg  (sd %.2f)\n" % (meas, A[:,2].std()))

# ---- 2. what the carve prefers, at each pitch, with axis and k free ----
masks, ang0 = persp.load()
g = json.load(open("solved_persp.json"))
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom-top; MM = 270.0/height
ang = ang0 + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n=len(masks); tr=np.arange(0,n,2); te=np.arange(1,n,2)
def ev(axis,pitch,k):
    v,xs,ys,zs = robust.vote_volume(masks[tr],ang[tr],axis,top,height,radius,pitch,k,140,190)
    vol = v>=int(round(32/36*len(tr)))
    if vol.sum()<1000: return 0.0
    return robust.score(vol,xs,ys,zs,masks[te],ang[te],axis,top,pitch,k,radius,height)

print(" pitch | best axis | best k    | camera  | held-out IoU")
prof=[]
for p in [-16.0,-13.0,-10.0,-8.0,-7.0,-6.0,-4.0,-2.0]:
    ba,bk,bs = 523.32, 0.00070, -1
    for axis in [521.3,523.3,525.3]:
        for k in [0.00040,0.00055,0.00070,0.00085]:
            s = ev(axis,p,k)
            if s>bs: bs,ba,bk = s,axis,k
    prof.append((p,bs))
    flag = "   <-- measured" if abs(p-(-meas))<1.0 else ""
    print("  %5.1f | %9.1f | %.5f   | %4.0f mm | %.4f%s" % (p,ba,bk,MM/bk,bs,flag), flush=True)
P=np.array(prof)
print("\ncarve's own best pitch: %.1f deg (IoU %.4f)" % (P[np.argmax(P[:,1]),0], P[:,1].max()))
print("at the measured pitch  : IoU %.4f" % P[np.argmin(np.abs(P[:,0]+meas)),1])
