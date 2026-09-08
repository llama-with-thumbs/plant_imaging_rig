"""Render a triangle mesh to shaded images, without a display server.

open3d's visualiser wants a window, which there isn't one of here, so this
rasterises directly: project the triangles, z-buffer them, and shade from the
face normals.  Flat shading per triangle is honest for this model -- it is a
carved hull, and pretending otherwise with interpolated normals would hide the
faceting rather than show it.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from numba import njit

W, H = 460, 700


@njit(cache=True)
def raster(tri2d, depth, shade, zbuf, out):
    n = tri2d.shape[0]
    h, w = zbuf.shape
    for i in range(n):
        x0, y0 = tri2d[i, 0, 0], tri2d[i, 0, 1]
        x1, y1 = tri2d[i, 1, 0], tri2d[i, 1, 1]
        x2, y2 = tri2d[i, 2, 0], tri2d[i, 2, 1]
        minx = int(max(0.0, min(x0, min(x1, x2))))
        maxx = int(min(w - 1.0, max(x0, max(x1, x2)) + 1))
        miny = int(max(0.0, min(y0, min(y1, y2))))
        maxy = int(min(h - 1.0, max(y0, max(y1, y2)) + 1))
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                fx = px + 0.5
                fy = py + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inv
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * depth[i, 0] + w1 * depth[i, 1] + w2 * depth[i, 2]
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    out[py, px] = shade[i]


def render(verts, faces, yaw_deg, pitch_deg=12.0, tint=(0.32, 0.44, 0.86)):
    c = verts.mean(axis=0)
    P = verts - c
    a = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = np.deg2rad(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    P = P @ R.T @ Rx.T

    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) * 1.12
    s = min(W / span, H / span)
    u = W / 2.0 + P[:, 0] * s
    v = H / 2.0 - P[:, 1] * s
    d = P[:, 2]

    tri = np.stack([np.column_stack([u, v])[faces[:, k]] for k in range(3)], axis=1)
    dep = np.stack([d[faces[:, k]] for k in range(3)], axis=1)

    e1 = P[faces[:, 1]] - P[faces[:, 0]]
    e2 = P[faces[:, 2]] - P[faces[:, 0]]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln == 0, 1, ln)
    key = np.array([0.42, 0.62, 0.66]); key /= np.linalg.norm(key)
    fill = np.array([-0.7, 0.15, 0.5]); fill /= np.linalg.norm(fill)
    lam = np.clip(nrm @ key, 0, 1) * 0.78 + np.clip(nrm @ fill, 0, 1) * 0.22
    shade = np.clip(0.18 + 0.86 * lam, 0, 1).astype(np.float32)

    zbuf = np.full((H, W), np.inf, np.float32)
    out = np.zeros((H, W), np.float32)
    raster(tri.astype(np.float32), dep.astype(np.float32), shade, zbuf, out)

    rgb = np.zeros((H, W, 3), np.float32)
    for i in range(3):
        rgb[:, :, i] = out * tint[i]
    rgb[~np.isfinite(zbuf)] = (0.07, 0.08, 0.07)
    return (np.clip(rgb, 0, 1) ** (1 / 1.9) * 255).astype(np.uint8)


def sheet(verts, faces, path, angles=(0, 45, 90, 135, 180, 225, 270, 315), cols=4):
    tiles = [(a, Image.fromarray(render(verts, faces, a))) for a in angles]
    tw, th = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw + (cols + 1) * 10, rows * (th + 24) + 10), (18, 18, 22))
    d = ImageDraw.Draw(im)
    for k, (a, t) in enumerate(tiles):
        r, c = divmod(k, cols)
        x, y = 10 + c * (tw + 10), 10 + r * (th + 24)
        im.paste(t, (x, y))
        d.text((x + 4, y + th + 6), "%d deg" % a, fill=(226, 226, 232))
    im.save(path, quality=90)
    return im


if __name__ == "__main__":
    import trimesh
    HERE = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else "bottle_v2.stl"
    out = sys.argv[2] if len(sys.argv) > 2 else "sheet_v2.jpg"
    m = trimesh.load(os.path.join(HERE, src))
    print("%s: %d tris" % (src, len(m.faces)))
    sheet(np.asarray(m.vertices), np.asarray(m.faces), os.path.join(HERE, out))
    print("wrote", out)
