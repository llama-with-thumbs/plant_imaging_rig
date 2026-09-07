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


def capture_usb(image_path, device="/dev/video2", width=1280, height=960,
                skip_frames=12, timeout=30):
    """Capture from the USB webcam, which watches the ruler on the platter rim.

    This is the rig's position readout: the CSI camera photographs the subject,
    this one photographs a scale attached to the turntable, so platter angle can
    be read directly rather than inferred from how much the subject changed.

    skip_frames discards the first frames of the stream -- a UVC webcam opens
    with its auto exposure wide open and takes a moment to settle, and the first
    frame is usually unreadable.
    """
    directory = os.path.dirname(image_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    command = ["fswebcam", "-d", device, "-r", f"{width}x{height}",
               "--no-banner", "-S", str(skip_frames), image_path]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", b"") or b""
        print(f"USB capture failed: {detail.decode(errors='replace').strip() or error}")
        return None
    return image_path


def capture_still(image_path, width=None, height=None, settle_ms=2000,
                  roi=None, mode=None, extra_args=None, timeout=60):
    """Capture one frame to image_path.  Returns the path, or None on failure.

    settle_ms is handed to rpicam-still as its run time before the shot: the
    auto exposure and white balance need a moment to converge, and skipping it
    is what makes consecutive frames flicker in the finished timelapse.

    roi crops in the camera pipeline rather than afterwards, as "x,y,w,h" in
    fractions of the frame -- so a backdrop occupying a quarter of the view
    yields a quarter-sized file with no cropping step and no wasted pixels.

    mode forces a sensor mode such as "4056:3040:12:P".  It matters whenever roi
    is used: asked for a small output, the camera otherwise picks a binned mode
    and quietly halves the detail inside the crop.

    Note rpicam-still rejects odd width or height.
    """
    directory = os.path.dirname(image_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    command = ["rpicam-still", "-o", image_path, "-n", "--timeout", str(settle_ms)]
    if width and height:
        command += ["--width", str(width), "--height", str(height)]
    if roi:
        command += ["--roi", roi]
    if mode:
        command += ["--mode", mode]
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
