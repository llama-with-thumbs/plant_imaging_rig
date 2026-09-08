"""Fit the correction curve on half the views, test it on the other half."""
import json, math
import numpy as np
import persp, robust
from refine_views import fourier_fit

masks, angles = persp.load()
g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
raw = np.array(json.load(open("view_corrections.json"))["d_angle_deg"])
n = len(masks)
NXZ, NY = 170, 230

# fit the 1+2 per-rev curve using only the even views' corrections,
# then evaluate the curve on the odd views, which had no say in it
ev = np.arange(0, n, 2); od = np.arange(1, n, 2)
t = np.arange(n) * 2*np.pi/n
A = np.column_stack([np.ones(n), np.cos(t), np.sin(t), np.cos(2*t), np.sin(2*t)])
coef, *_ = np.linalg.lstsq(A[ev], raw[ev], rcond=None)
pred = A @ coef
print("curve fitted on 18 even views, applied to all; scored on the 18 odd views")
print("predicted odd-view corrections vs their own measured values:")
print("  correlation %.3f" % np.corrcoef(pred[od], raw[od])[0,1])
print()
print("  correction        |  IoU on held-out odd views")
for name, corr in (("none", np.zeros(n)), ("predicted curve", pred), ("reversed (control)", -pred)):
    a = angles + np.deg2rad(corr)
    votes, xs, ys, zs = robust.vote_volume(masks, a, axis, top, height, radius,
                                           pitch, k, NXZ, NY)
    vol = votes >= 32
    s = robust.score(vol, xs, ys, zs, masks[od], a[od], axis, top, pitch, k, radius, height)
    print("  %-17s | %.4f" % (name, s), flush=True)
