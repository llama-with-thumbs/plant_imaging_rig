"""Measure the camera elevation from the platter, not from the carve.

The platter is a circle. A circle seen from elevation alpha projects to an
ellipse whose minor/major ratio is sin(alpha), so the camera angle is readable
straight off the photograph and owes nothing to the reconstruction.

The green platter disc sits on the cork stand. Its widest horizontal extent is
the major axis, 2R. Down the centre column, the top of the disc is the *back*
edge of its top face and the green-to-cork seam is the *front* edge of the same
face, so their separation is the minor axis, 2R*sin(alpha).
"""
import glob, math
import numpy as np, cv2

vals = []
for f in sorted(glob.glob("orbit36g/v*.jpg"))[:12]:
    bgr = cv2.imread(f); h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0].astype(int), hsv[:,:,1].astype(int), hsv[:,:,2].astype(int)
    band = slice(int(h*0.86), h)                       # platter region only
    green = ((H>=35)&(H<=90)&(S>=60)&(V>=40))[band]
    cork  = ((H>4)&(H<34)&(S>50)&(V>40))[band]
    # major axis: widest run of disc green in the band
    widths = green.sum(axis=1)
    if widths.max() < 50: continue
    r_wide = int(np.argmax(widths)); major = int(widths[r_wide])
    cols = np.where(green[r_wide])[0]; cx = int(cols.mean())
    col_g = green[:, max(0,cx-6):cx+7].mean(axis=1) > .5
    col_c = cork[:,  max(0,cx-6):cx+7].mean(axis=1) > .5
    gy = np.where(col_g)[0]; cy = np.where(col_c)[0]
    if not len(gy) or not len(cy): continue
    y_back = gy.min()                                  # back edge of the top face
    y_seam = cy.min()                                  # green meets cork = front edge
    minor = y_seam - y_back
    if minor <= 0 or minor > major: continue
    a = math.degrees(math.asin(min(1.0, minor/major)))
    vals.append(a)
    print("  %s  major %4d px  minor %3d px  ->  elevation %5.2f deg" % (f[-7:], major, minor, a))

v = np.array(vals)
print("\nmeasured camera elevation: %.2f deg  (sd %.2f over %d views)" % (v.mean(), v.std(), len(v)))
print("carve solved under strict intersection : 15.75 deg")
print("carve re-solved under robust voting    :  3.50 deg")
