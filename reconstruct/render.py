"""Render the carved hull from several viewpoints.

Rasterising 162k triangles through matplotlib is slow and looks flat, so this
projects the surviving voxels directly and keeps the nearest one per pixel -- a
z-buffer in numpy.  Shading comes from the gradient of that depth map, which
gives a real sense of form without needing a lighting model or the mesh at all.
"""

import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_W, OUT_H = 420, 620


def surface_points(vol):
    """Voxels on the boundary -- the interior is invisible and just costs time."""
    er = ndimage.binary_erosion(vol, np.ones((3, 3, 3), bool))
    shell = vol & ~er
    x, y, z = np.where(shell)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def render(x, y, z, yaw_deg, pitch_deg=14.0):
    cx, cz = x.mean(), z.mean()
    X, Z = x - cx, z - cz
    # Voxel index 0 is the TOP image row, so Y must be negated or the bottle
    # renders standing on its trigger.
    Y = -(y - y.mean())

    a = np.deg2rad(yaw_deg)
    Xr = X * np.cos(a) + Z * np.sin(a)
    Zr = -X * np.sin(a) + Z * np.cos(a)

    p = np.deg2rad(pitch_deg)
    Yr = Y * np.cos(p) - Zr * np.sin(p)
    Dr = Y * np.sin(p) + Zr * np.cos(p)          # depth toward the viewer

    # One uniform scale for both axes, chosen so the object just fits. Scaling
    # u and v differently is what turns a bottle into a sliver.
    scale = min(OUT_W / (np.ptp(Xr) * 1.15), OUT_H / (np.ptp(Yr) * 1.10))
    u = OUT_W / 2.0 + Xr * scale
    v = OUT_H / 2.0 - Yr * scale
    ui = np.clip(np.round(u).astype(int), 0, OUT_W - 1)
    vi = np.clip(np.round(v).astype(int), 0, OUT_H - 1)

    depth = np.full((OUT_H, OUT_W), np.inf, np.float32)
    np.minimum.at(depth, (vi, ui), -Dr)          # nearest surface wins

    # Voxels are discrete, so projection leaves pinholes between them; a
    # grey-scale erosion closes them without eating the true silhouette.
    holes = ~np.isfinite(depth)
    if holes.any():
        filled = ndimage.grey_erosion(np.where(holes, np.inf, depth), size=3)
        depth = np.where(holes & np.isfinite(filled), filled, depth)

    hit = np.isfinite(depth)
    if not hit.any():
        return np.zeros((OUT_H, OUT_W, 3), np.uint8)

    d = depth.copy()
    d[~hit] = depth[hit].max()
    d = ndimage.median_filter(d, 3)

    # Shade from the depth gradient: a surface tipping away darkens.
    gy, gx = np.gradient(d)
    nz = np.ones_like(d) * 2.2
    n = np.dstack([-gx, -gy, nz])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    light = np.array([-0.35, -0.55, 0.76])
    lam = np.clip((n * light).sum(axis=2), 0, 1)

    dd = d[hit]
    lo, hi = np.percentile(dd, 2), np.percentile(dd, 98)
    prox = np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1)

    shade = 0.16 + 0.68 * lam + 0.16 * prox
    shade = np.clip(shade, 0, 1)

    rgb = np.zeros((OUT_H, OUT_W, 3), np.float32)
    tint = np.array([0.42, 0.55, 0.92])          # a nod to the actual bottle
    for c in range(3):
        rgb[:, :, c] = shade * tint[c]
    rgb[~hit] = np.array([0.09, 0.10, 0.09])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    vol = np.load(os.path.join(HERE, "hull_voxels.npy"))
    print("volume", vol.shape, "filled", int(vol.sum()))
    # Imperfect masks leave a few detached crumbs floating beside the object;
    # the model is the one body they are not part of.
    lbl, n = ndimage.label(vol)
    if n > 1:
        sizes = ndimage.sum(vol, lbl, range(1, n + 1))
        vol = lbl == (1 + int(np.argmax(sizes)))
        print("kept largest of %d components: %d voxels (dropped %d)"
              % (n, int(vol.sum()), n - 1))
        np.save(os.path.join(HERE, "hull_voxels_clean.npy"), vol)
    x, y, z = surface_points(vol)
    print("surface voxels:", x.size)

    views = [0, 45, 90, 135, 180, 225, 270, 315]
    tiles = []
    for a in views:
        img = render(x, y, z, a)
        Image.fromarray(img).save(os.path.join(HERE, "render_%03d.png" % a))
        tiles.append((a, Image.fromarray(img)))
        print("  rendered %3d deg" % a)

    from PIL import ImageDraw
    cols = 4
    tw, th = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 10,
                              rows * (th + 24) + 10), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    for k, (a, im) in enumerate(tiles):
        r, c = divmod(k, cols)
        px, py = 10 + c * (tw + 10), 10 + r * (th + 24)
        sheet.paste(im, (px, py))
        d.text((px + 4, py + th + 6), "%d deg" % a, fill=(226, 226, 232))
    sheet.save(os.path.join(HERE, "hull_sheet.jpg"), quality=90)
    print("wrote hull_sheet.jpg", sheet.size)
