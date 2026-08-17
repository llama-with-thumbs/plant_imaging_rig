"""Publish the most recent frame to the GitHub Pages site.

The site lives on an orphan `gh-pages` branch holding exactly one commit. Every
publish amends that commit and force-pushes, so a photo that updates every few
minutes never accumulates history -- the repository stays the size of one
image rather than growing by 200 KB a tick forever.

Run it standalone, or let app.py call it after each capture by setting
PUBLISH = True in config.py.

Requires push access from the Pi -- see the README for the deploy key setup.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from PIL import Image

import config

PAGES_URL = "git@github.com-plantrig:llama-with-thumbs/plant_imaging_rig.git"
PAGES_DIR = os.path.expanduser("~/.cache/plant-rig-pages")
BRANCH = "gh-pages"

# Full frames are ~1.8 MB; the site wants something a browser can load over a
# phone connection, and detail beyond this is not visible on a web page anyway.
WEB_MAX_EDGE = 1800
WEB_QUALITY = 82


def run(args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check,
                          capture_output=True, text=True)


def newest_capture(root):
    """Most recently modified frame under the capture directory."""
    newest, newest_mtime = None, -1
    for directory, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(directory, name)
            mtime = os.path.getmtime(path)
            if mtime > newest_mtime:
                newest, newest_mtime = path, mtime
    return newest


def ensure_clone():
    """A local checkout of just the gh-pages branch, created once."""
    if os.path.isdir(os.path.join(PAGES_DIR, ".git")):
        return
    os.makedirs(os.path.dirname(PAGES_DIR), exist_ok=True)
    run(["git", "clone", "--quiet", "--branch", BRANCH, "--single-branch",
         PAGES_URL, PAGES_DIR])


def platter_state():
    try:
        with open(config.STATE_PATH) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def publish(source_image):
    ensure_clone()

    # Start from the published commit rather than whatever is lying around, so
    # a half-finished previous run cannot wedge this one.
    run(["git", "fetch", "--quiet", "origin", BRANCH], cwd=PAGES_DIR)
    run(["git", "reset", "--hard", "--quiet", f"origin/{BRANCH}"], cwd=PAGES_DIR)

    image = Image.open(source_image)
    image.thumbnail((WEB_MAX_EDGE, WEB_MAX_EDGE), Image.LANCZOS)
    image.save(os.path.join(PAGES_DIR, "latest.jpg"),
               quality=WEB_QUALITY, optimize=True)

    state = platter_state()
    meta = {
        "captured": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "angle": state.get("angle"),
        "cycle": state.get("stop", 0) // config.STOPS if state else None,
        "camera": f"IMX477 ({config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT})",
        "source": os.path.basename(source_image),
    }
    with open(os.path.join(PAGES_DIR, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    run(["git", "add", "-A"], cwd=PAGES_DIR)
    # --amend keeps the branch at a single commit; -f is therefore required and
    # is safe precisely because nothing else ever writes to this branch.
    run(["git", "commit", "--quiet", "--amend", "-m",
         f"site: frame at {meta['captured']}"], cwd=PAGES_DIR)
    run(["git", "push", "--quiet", "--force", "origin", BRANCH], cwd=PAGES_DIR)

    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?",
                        help="frame to publish (default: newest in OUTPUT_DIR)")
    parser.add_argument("--capture", action="store_true",
                        help="take a fresh frame instead of using an existing one")
    args = parser.parse_args()

    if args.capture:
        from rig.capture import capture_still
        source = capture_still("/tmp/plant_rig_publish.jpg",
                               config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT,
                               settle_ms=config.SETTLE_MS)
        if not source:
            raise SystemExit("Capture failed; nothing published.")
    else:
        source = args.image or newest_capture(config.OUTPUT_DIR)
        if not source:
            raise SystemExit(f"No frames found under {config.OUTPUT_DIR}.")

    meta = publish(source)
    print(f"Published {meta['source']} at {meta['captured']}")
    print("https://llama-with-thumbs.github.io/plant_imaging_rig/")


if __name__ == "__main__":
    main()
