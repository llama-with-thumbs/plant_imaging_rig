"""One attempt at a better model, scored end to end and published.

Run unattended on a timer. Each call takes the next untried configuration,
builds a model, scores it against the photographs, records the result, and
republishes the gallery. Nothing is deleted: a worse attempt is still recorded,
because knowing a direction does not work is the point of trying it.

The score is the silhouette of the **mesh**, not of the voxel hull. That
distinction matters here. Every earlier number in this project measured the
hull, but the hull is not what gets published -- smoothing, closing and
decimation all happen afterwards, and all three can move the outline. Sampling
the finished surface and projecting it through the same camera is the only
measure of the thing a person actually looks at.

Ranked on that silhouette match, with two gates rather than a weighted score:
a model that is not watertight is not publishable, and one whose bounding box
misses the tape measure by more than 8% is not the right object. Within those,
fewer triangles wins ties, since the ask was explicitly for a lighter mesh.
"""

import glob
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = os.path.join(HERE, "pages")
MODELS = os.path.join(PAGES, "models")
LEDGER = os.path.join(HERE, "attempts.json")
OBJ_H_MM, OBJ_W_MM = 270.0, 115.0
KEEP_PUBLISHED = 8            # bound the size of the gh-pages branch


# ----------------------------------------------------------------- the plan
def plan():
    """Configurations to try, in order. Cheapest and most promising first.

    Mesh settings come first because they reuse the cached vote field and cost
    about a minute; anything that changes the carve has to recount 36 views
    over the whole grid.
    """
    out = []
    for thr in (33, 32, 34):
        for close in (1, 2, 3):
            for sigma in (1.2, 1.6, 2.0):
                for tris in (16000, 10000, 24000):
                    out.append(dict(kind="mesh", votes=thr, close=close,
                                    sigma=sigma, tris=tris, nxz=260, ny=380))
    # then a finer carve, which is where genuinely new information can come from
    for thr in (33, 32):
        for close in (1, 2):
            for sigma in (1.4, 1.8):
                out.append(dict(kind="carve", votes=thr, close=close,
                                sigma=sigma, tris=16000, nxz=320, ny=460))
    return out


def key_of(cfg):
    return "v%d_c%d_s%.1f_t%d_%dx%d" % (cfg["votes"], cfg["close"], cfg["sigma"],
                                        cfg["tris"], cfg["nxz"], cfg["ny"])


# ------------------------------------------------------------------ scoring
def mesh_silhouette_iou(tm, masks, ang, g):
    """Project the finished surface into every view and compare with the mask.

    Points sampled over the surface, pushed through the same perspective
    camera the carve used, then closed into a solid region. Sampling beats
    rasterising here: it needs no triangle clipping, and at 400k points the
    outline is denser than the mask's own pixels.
    """
    pts, _ = trimesh.sample.sample_surface(tm, 400000)
    mm = g["mm_per_px"]
    X = pts[:, 0] / mm
    Y = (OBJ_H_MM - pts[:, 1]) / mm          # mesh y is up from the base
    Z = pts[:, 2] / mm
    p = math.radians(g["pitch"])
    cos_p, sin_p = math.cos(p), math.sin(p)
    h, w = masks[0].shape
    ious, fats, thins = [], [], []
    ker = np.ones((5, 5), np.uint8)
    for m in range(len(masks)):
        ct, st = math.cos(ang[m]), math.sin(ang[m])
        lat = X * ct + Z * st
        d = -X * st + Z * ct
        s = 1.0 + g["k"] * d
        ok = s > 0.2
        u = g["cu"] + (g["axis"] + lat - g["cu"]) / s
        v = g["cv"] + (g["top"] + Y * cos_p + d * sin_p - g["cv"]) / s
        ui = np.rint(u).astype(np.int32); vi = np.rint(v).astype(np.int32)
        ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        img = np.zeros((h, w), np.uint8)
        img[vi[ok], ui[ok]] = 1
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker)
        img = ndimage.binary_fill_holes(img.astype(bool))
        mk = masks[m]
        inter = np.logical_and(img, mk).sum(); union = np.logical_or(img, mk).sum()
        ious.append(inter / union if union else 0.0)
        tot = max(1, mk.sum())
        fats.append(np.logical_and(img, ~mk).sum() / tot)
        thins.append(np.logical_and(mk, ~img).sum() / tot)
    F, T = np.array(fats), np.array(thins)
    edge = [8, 9, 10, 26, 27, 28]; broad = [0, 1, 17, 18, 19, 35]
    return dict(iou=float(np.mean(ious)),
                fat=float(F.mean()), thin=float(T.mean()),
                fat_edge=float(F[edge].mean()), fat_broad=float(F[broad].mean()))


# -------------------------------------------------------------------- build
def run(cfg):
    sys.path.insert(0, HERE)
    import carve8, mesh8, persp, robust

    masks, ang = carve8.load_masks()
    g = carve8.geometry(masks)
    persp.CU, persp.CV = g["cu"], g["cv"]

    cache = os.path.join(HERE, "votes_%dx%d.npy" % (cfg["nxz"], cfg["ny"]))
    if os.path.exists(cache):
        votes = np.load(cache)
    else:
        votes, _, _, _ = robust.vote_volume(masks, ang, g["axis"], g["top"],
                                            g["height_px"], g["radius"],
                                            g["pitch"], g["k"], cfg["nxz"], cfg["ny"])
        np.save(cache, votes)

    mesh8.NXZ, mesh8.NY, mesh8.VOTES = cfg["nxz"], cfg["ny"], cfg["votes"]
    tm = mesh8.build(votes, cfg["close"], cfg["sigma"], cfg["tris"], key_of(cfg))

    bb = tm.bounds[1] - tm.bounds[0]
    rec = dict(cfg)
    rec["key"] = key_of(cfg)
    rec["when"] = time.strftime("%Y-%m-%d %H:%M")
    rec["tris"] = int(len(tm.faces))
    rec["watertight"] = bool(tm.is_watertight)
    rec["genus"] = int((2 - tm.euler_number) // 2)
    rec["volume_cm3"] = float(tm.volume / 1000) if tm.is_watertight else None
    rec["w_mm"], rec["h_mm"], rec["d_mm"] = [float(x) for x in bb]
    rec["h_err"] = float(bb[1] / OBJ_H_MM - 1)
    rec["w_err"] = float(max(bb[0], bb[2]) / OBJ_W_MM - 1)
    rec.update(mesh_silhouette_iou(tm, masks, ang, g))
    rec["ok"] = bool(rec["watertight"]
                     and abs(rec["h_err"]) < 0.08 and abs(rec["w_err"]) < 0.08)
    return tm, rec


# ------------------------------------------------------------------ gallery
def render_gallery(records):
    pub = [r for r in records if r.get("published")]
    pub.sort(key=lambda r: (-r["iou"], r["tris"]))
    best = pub[0]["key"] if pub else None
    rows = []
    for r in sorted(records, key=lambda r: (-r["iou"], r["tris"])):
        badge = ""
        if r["key"] == best:
            badge = '<span class="tag best">best</span>'
        elif not r["ok"]:
            badge = '<span class="tag bad">rejected</span>'
        link = ('<a href="model.html?m=%s">view</a> &middot; '
                '<a href="models/%s.stl" download>stl</a>' % (r["key"], r["key"])) \
            if r.get("published") else '<span class="dim">not published</span>'
        rows.append(
            "<tr%s><td class=\"k\">%s%s</td><td class=\"n\">%.4f</td>"
            "<td class=\"n\">%s</td><td class=\"n\">%.0f&times;%.0f&times;%.0f</td>"
            "<td class=\"n\">%+.1f%% / %+.1f%%</td><td class=\"n\">%.0f%%</td>"
            "<td class=\"n\">%d</td><td class=\"n\">%s</td><td>%s</td><td class=\"dim\">%s</td></tr>"
            % (' class="is-best"' if r["key"] == best else "",
               r["key"], badge, r["iou"], "{:,}".format(r["tris"]),
               r["w_mm"], r["h_mm"], r["d_mm"], 100 * r["h_err"], 100 * r["w_err"],
               100 * r["fat_edge"], r["genus"],
               "yes" if r["watertight"] else "NO", link, r["when"]))
    tpl = open(os.path.join(HERE, "models.tpl.html"), encoding="utf-8").read()
    html = tpl.replace("{{ROWS}}", "\n".join(rows))
    html = html.replace("{{COUNT}}", str(len(records)))
    html = html.replace("{{PUBLISHED}}", str(len(pub)))
    html = html.replace("{{BEST}}", "%.4f" % pub[0]["iou"] if pub else "&mdash;")
    html = html.replace("{{UPDATED}}", time.strftime("%Y-%m-%d %H:%M"))
    open(os.path.join(PAGES, "models.html"), "w", encoding="utf-8").write(html)


def publish(records):
    """Keep the best few on the branch so it does not grow without bound."""
    keep = sorted([r for r in records if r["ok"]],
                  key=lambda r: (-r["iou"], r["tris"]))[:KEEP_PUBLISHED]
    keepset = {r["key"] for r in keep}
    for r in records:
        r["published"] = r["key"] in keepset
    for f in glob.glob(os.path.join(MODELS, "*.stl")):
        if os.path.splitext(os.path.basename(f))[0] not in keepset:
            os.remove(f)
    if keep:
        import shutil
        shutil.copy(os.path.join(MODELS, keep[0]["key"] + ".stl"),
                    os.path.join(PAGES, "bottle.stl"))
    render_gallery(records)
    subprocess.run(["git", "add", "-A"], cwd=PAGES, check=True)
    subprocess.run(["git", "commit", "-q", "--amend", "-m", "site: 3D model viewer"],
                   cwd=PAGES, check=True)
    subprocess.run(["git", "push", "-q", "--force", "origin", "gh-pages"],
                   cwd=PAGES, check=True)


def main():
    os.makedirs(MODELS, exist_ok=True)
    records = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    done = {r["key"] for r in records}
    todo = [c for c in plan() if key_of(c) not in done]
    if not todo:
        print("every planned configuration has been tried (%d records)" % len(records))
        return
    cfg = todo[0]
    print("attempt %d of %d: %s" % (len(done) + 1, len(done) + len(todo), key_of(cfg)),
          flush=True)
    tm, rec = run(cfg)
    print("  iou %.4f  tris %d  watertight %s  genus %d  bbox %.0f x %.0f x %.0f"
          % (rec["iou"], rec["tris"], rec["watertight"], rec["genus"],
             rec["w_mm"], rec["h_mm"], rec["d_mm"]))
    print("  height %+.1f%%  width %+.1f%%  edge-on fat %.0f%%  -> %s"
          % (100 * rec["h_err"], 100 * rec["w_err"], 100 * rec["fat_edge"],
             "publishable" if rec["ok"] else "rejected"))
    if rec["ok"]:
        tm.export(os.path.join(MODELS, rec["key"] + ".stl"))
    records.append(rec)
    json.dump(records, open(LEDGER, "w"), indent=2)
    publish(records)
    ranked = sorted([r for r in records if r["ok"]], key=lambda r: (-r["iou"], r["tris"]))
    if ranked:
        b = ranked[0]
        print("  best so far: %s  iou %.4f  %d tris" % (b["key"], b["iou"], b["tris"]))


if __name__ == "__main__":
    main()
