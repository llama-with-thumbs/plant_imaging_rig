"""Show, on the photograph, where mask and hull disagree."""
import json, math
import numpy as np, cv2
import persp, robust
from refine_views import project, solidify

masks, angles = persp.load()
g = json.load(open("solved_persp.json"))
axis, pitch, k = g["axis"], g["pitch_deg"], g["k"]
top, bottom, radius = g["top"], g["bottom"], g["radius"]
height = bottom - top
angles = angles + np.deg2rad(np.array(json.load(open("view_corrections.json"))["smooth_d_angle_deg"]))
n = len(masks)
PICK = [0, 10, 18, 28]
tiles = []
for m in PICK:
    keep = np.array([i for i in range(n) if i != m])
    votes, xs, ys, zs = robust.vote_volume(masks[keep], angles[keep], axis, top,
                                           height, radius, pitch, k, 140, 190)
    vol = votes >= int(round(32/36*35))
    hull = solidify(project(vol, xs, ys, zs, angles[m], axis, top, pitch, k, masks[m].shape),
                    radius, height)
    img = cv2.imread("orbit36g/v%02d.jpg" % m).astype(np.float32)
    mk = masks[m]
    img[np.logical_and(mk, ~hull)] = img[np.logical_and(mk, ~hull)]*.35 + np.array([60,200,255])*.65   # mask only: amber
    img[np.logical_and(hull, ~mk)] = img[np.logical_and(hull, ~mk)]*.35 + np.array([255,120,60])*.65   # hull only: blue
    img[np.logical_and(hull, mk)]  = img[np.logical_and(hull, mk)]*.75 + np.array([255,255,255])*.25
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.putText(img, "v%02d  %.0f deg" % (m, math.degrees(angles[m])%360), (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    tiles.append(img)
sheet = np.hstack(tiles)
cv2.putText(sheet, "amber = mask only (hull too thin)   blue = hull only (hull too fat)",
            (12, sheet.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2)
cv2.imwrite("overlay.jpg", np.clip(sheet,0,255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote overlay.jpg", sheet.shape)
