"""Measure the lens with a checkerboard instead of inferring it from silhouettes.

Every camera parameter this project has used so far was fitted indirectly, by
asking which values best explained the silhouettes they were meant to explain.
That process will happily absorb model error into whichever parameter is
loosest -- it put the principal point 370 px away from where the capture chain
says it is, and pulled the camera pitch to -15 deg when the tape says 0.75.

A checkerboard measures the optics directly. Known geometry, known square size,
no reconstruction involved.

It fits both camera models and reports each one's reprojection error, because
which model is right is itself the open question. The wide-angle is a circular
fisheye, where the standard pinhole-plus-radial model breaks down badly off
axis; the 8 mm is close enough to rectilinear that the standard model should
win. Whichever gives the lower error is the one to carve with.

    python calibrate_lens.py calib/            # a directory of board images
    python calibrate_lens.py calib/ --squares 38

Writes camera.json, and undistort.py reads it to rectify the orbit frames.
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

# Detection flags. NORMALIZE_IMAGE matters under the rig's LED bars, which are
# bright in the middle of the board and fall off at the edges.
FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
         cv2.CALIB_CB_FAST_CHECK)
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)


def find_board_size(paths, probe=6):
    """Work out the board's inner-corner count instead of asking for it.

    Counting squares on a printed board is a classic off-by-one -- OpenCV wants
    inner corners, which is one less than the squares in each direction. Trying
    the plausible sizes on a few images is faster than getting it wrong.
    """
    sizes = [(c, r) for c in range(3, 14) for r in range(3, 14) if c >= r]
    sizes.sort(key=lambda s: -(s[0] * s[1]))          # prefer the largest that works
    for path in paths[:probe]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, None, fx=0.35, fy=0.35)
        for size in sizes:
            ok, _ = cv2.findChessboardCorners(small, size, FLAGS)
            if ok:
                return size
    return None


def collect(paths, size, square_mm):
    objp = np.zeros((size[0] * size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= square_mm
    objpoints, imgpoints, used, shape = [], [], [], None
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        shape = img.shape[::-1]
        ok, corners = cv2.findChessboardCorners(img, size, FLAGS)
        if not ok:
            print("  %-28s no board found" % os.path.basename(path))
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), CRIT)
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(os.path.basename(path))
        print("  %-28s ok" % os.path.basename(path))
    return objpoints, imgpoints, used, shape


def spread(imgpoints, shape):
    """How much of the frame the board actually visited.

    Calibration is only as good as its coverage: boards clustered in the middle
    leave the distortion terms unconstrained exactly where distortion lives.
    """
    grid = np.zeros((3, 3), bool)
    for c in imgpoints:
        for x, y in c.reshape(-1, 2):
            gx = min(2, int(3 * x / shape[0]))
            gy = min(2, int(3 * y / shape[1]))
            grid[gy, gx] = True
    return grid


def fit_standard(objpoints, imgpoints, shape):
    err, K, D, rv, tv = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return err, K, D, rv, tv


def fit_fisheye(objpoints, imgpoints, shape):
    n = len(objpoints)
    obj = [o.reshape(1, -1, 3) for o in objpoints]
    img = [i.reshape(1, -1, 2) for i in imgpoints]
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rv = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    tv = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_FIX_SKEW)
    err, K, D, rv, tv = cv2.fisheye.calibrate(obj, img, shape, K, D, rv, tv,
                                              flags, CRIT)
    return err, K, D, rv, tv


def describe(K, shape, sensor_mm=(6.287, 4.712)):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    mm_per_px = sensor_mm[0] / shape[0]
    return {
        "fx_px": float(fx), "fy_px": float(fy),
        "cx_px": float(cx), "cy_px": float(cy),
        "f_mm": float(fx * mm_per_px),
        "cx_offset_px": float(cx - shape[0] / 2),
        "cy_offset_px": float(cy - shape[1] / 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--squares", type=float, default=38.0,
                    help="side of one square in mm (default 38)")
    ap.add_argument("--size", type=str, default=None,
                    help="inner corners as CxR, e.g. 6x9; auto-detected if omitted")
    ap.add_argument("--out", default="camera.json")
    a = ap.parse_args()

    paths = sorted(sum([glob.glob(os.path.join(a.folder, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")], []))
    if not paths:
        sys.exit("no images in %s" % a.folder)
    print("%d images in %s" % (len(paths), a.folder))

    if a.size:
        size = tuple(int(v) for v in a.size.lower().split("x"))
    else:
        print("looking for the board...")
        size = find_board_size(paths)
        if size is None:
            sys.exit("could not find a checkerboard. Pass --size CxR (inner corners, "
                     "one less than the squares each way).")
    print("board: %d x %d inner corners, %.1f mm squares\n" % (size[0], size[1], a.squares))

    objpoints, imgpoints, used, shape = collect(paths, size, a.squares)
    print("\n%d of %d images usable" % (len(used), len(paths)))
    if len(used) < 8:
        sys.exit("need at least 8 good views; tilt and move the board more.")

    g = spread(imgpoints, shape)
    print("frame coverage (3x3 zones): %d of 9 visited" % g.sum())
    if g.sum() < 6:
        print("  WARNING: the board stayed in too few parts of the frame.")
        print("  Distortion is estimated worst where the board never went.")
        for row in g:
            print("   " + " ".join("##" if v else " ." for v in row))

    print("\nfitting both camera models\n")
    results = {}
    try:
        err, K, D, _, _ = fit_standard(objpoints, imgpoints, shape)
        results["standard"] = (err, K, D)
        print("  standard (pinhole + radial/tangential): reprojection error %.4f px" % err)
    except cv2.error as e:
        print("  standard model failed: %s" % str(e).split("\n")[0])
    try:
        err, K, D, _, _ = fit_fisheye(objpoints, imgpoints, shape)
        results["fisheye"] = (err, K, D)
        print("  fisheye  (equidistant + 4 terms)     : reprojection error %.4f px" % err)
    except cv2.error as e:
        print("  fisheye model failed: %s" % str(e).split("\n")[0])

    if not results:
        sys.exit("both models failed to fit")

    best = min(results, key=lambda k: results[k][0])
    err, K, D = results[best]
    print("\n-> %s model wins (%.4f px)" % (best, err))
    if err > 1.0:
        print("   NOTE: over 1 px is high. Usually blur, or the board not being flat.")

    info = describe(K, shape)
    print("\nmeasured optics")
    print("  focal length : %.0f px  = %.2f mm on the sensor" % (info["fx_px"], info["f_mm"]))
    print("  principal pt : %.1f, %.1f  (%+.1f, %+.1f from the image centre)"
          % (info["cx_px"], info["cy_px"], info["cx_offset_px"], info["cy_offset_px"]))
    if best == "standard":
        halfx = math.degrees(2 * math.atan(shape[0] / (2 * info["fx_px"])))
        print("  horizontal FoV: %.1f deg" % halfx)

    out = {
        "model": best,
        "image_size": [int(shape[0]), int(shape[1])],
        "K": K.tolist(),
        "D": np.asarray(D).ravel().tolist(),
        "reprojection_error_px": float(err),
        "square_mm": a.squares,
        "board_inner_corners": list(size),
        "images_used": used,
        **info,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
