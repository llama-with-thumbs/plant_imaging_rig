"""Camera capture, wrapping the rpicam-still CLI.

rpicam-still is used rather than a Python binding for the same reason bio-chart
and SporeScope did it: it is the tool that ships with the OS, and shelling out
means the camera stack is never held open between the shots that are half an
hour apart.
"""

import os
import subprocess


def cameras_available():
    """True if libcamera can see a camera at all.

    Worth calling before a long run: with the CSI ribbon unseated every capture
    fails identically, and the error is much clearer here than 288 times a day.
    """
    try:
        result = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "no cameras available" not in result.stdout.lower()


def capture_still(image_path, width=None, height=None, settle_ms=2000,
                  extra_args=None, timeout=60):
    """Capture one frame to image_path.  Returns the path, or None on failure.

    settle_ms is handed to rpicam-still as its run time before the shot: the
    auto exposure and white balance need a moment to converge, and skipping it
    is what makes consecutive frames flicker in the finished timelapse.
    """
    directory = os.path.dirname(image_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    command = ["rpicam-still", "-o", image_path, "-n", "--timeout", str(settle_ms)]
    if width and height:
        command += ["--width", str(width), "--height", str(height)]
    if extra_args:
        command += list(extra_args)

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode(errors="replace").strip()
        print(f"Capture failed: {detail or error}")
        return None
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Capture failed: {error}")
        return None

    return image_path
