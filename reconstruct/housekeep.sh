#!/bin/bash
# Free space on the Pi, but only when it is actually short.
#
# Every capture set is copied to the workstation as soon as it is pulled, so
# the Pi's copies are redundant -- but they are also the only copy if a pull
# ever failed, so this deletes on pressure rather than on a schedule, and
# always keeps the two newest sets.
MIN_FREE_GB=${1:-4}
free_gb=$(df --output=avail -BG /home/pi | tail -1 | tr -dc '0-9')
echo "free: ${free_gb}G (threshold ${MIN_FREE_GB}G)"
if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then echo "nothing to do"; exit 0; fi
cd /home/pi/captures 2>/dev/null || exit 0
ls -1dt */ 2>/dev/null | tail -n +3 | while read d; do
  echo "removing $d ($(du -sh "$d" | cut -f1))"
  rm -rf "$d"
done
echo "free after: $(df -h /home/pi | tail -1 | awk '{print $4}')"
