"""Does the angle objective have a real peak, or is it flat?"""
import json, math, os
import numpy as np
import persp, robust
from refine_views import project, solidify, iou, NXZ, NY

masks, angles = persp.load()
g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
n = len(masks)

SWEEP = np.arange(-8.0, 8.01, 1.0)
print("held-out IoU vs angle correction, wide window")
print("(a real indexing error shows an interior peak; a flat curve rails at the edge)\n")
print("  view |" + "".join("%7.0f" % d for d in SWEEP))
for m in (0, 9, 18, 27):
    keep = np.array([i for i in range(n) if i != m])
    votes, xs, ys, zs = robust.vote_volume(masks[keep], angles[keep], axis, top,
                                           height, radius, pitch, k, NXZ, NY)
    vol = votes >= int(round(32/36*len(keep)))
    row = []
    for dd in SWEEP:
        img = project(vol, xs, ys, zs, angles[m]+math.radians(dd), axis, top, pitch, k, masks[m].shape)
        row.append(iou(solidify(img, radius, height), masks[m]))
    star = int(np.argmax(row))
    print("  %4d |" % m + "".join(("%7.3f" % v) for v in row) + "   peak at %+.0f deg" % SWEEP[star])
