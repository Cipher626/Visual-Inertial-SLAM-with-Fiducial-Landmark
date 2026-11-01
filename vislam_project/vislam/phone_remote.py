"""Start/stop OpenCamera Sensors recordings from the laptop over Wi-Fi.
Protocol re-implemented from the app's api_client (TCP port 6969, server version v.0.1.1).
Enable it on the phone: Settings > 'Enable remote recording control' (restart the app), put the app in video
mode, and connect phone and laptop to the same Wi-Fi (a phone hotspot works)."""
import re
import shutil
import socket
import subprocess
from pathlib import Path

PORT = 6969
SUPPORTED = ("v.0.1.1",)


class OpenCameraRemote:
    def __init__(self, host, port=PORT, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(30.0)

    def _request(self, msg):
        f = self.sock.makefile("rw")
        f.write(msg + "\n")
        f.flush()
        status = f.readline().strip()
        version = f.readline().strip()
        if version not in SUPPORTED:
            raise RuntimeError(f"Unsupported OpenCamera Sensors server version '{version}' (status '{status}')")
        if status == "ERROR":
            raise RuntimeError(f.readline().strip())
        return f

    def start(self):
        """Returns frame phase [ns], mean frame duration [ns], exposure time [ns]."""
        f = self._request("video_start")
        phase = int(f.readline())
        duration = float(f.readline())
        exposure = int(f.readline())
        f.readline()
        return {"phase_ns": phase, "frame_duration_ns": duration, "fps": 1e9 / duration if duration else None,
                "exposure_ns": exposure}

    def stop(self):
        f = self._request("video_stop")
        while True:
            line = f.readline()
            if not line or line.strip() == "end":
                break

    def close(self):
        self.sock.close()


def adb_available():
    return shutil.which("adb") is not None


def adb_pull_latest(dst, phone_dir="/sdcard/DCIM/OpenCamera"):
    """Copy the newest video and its sensor folder (named by the video's yyyyMMdd_HHmmss) from the phone
    over USB. Needs Android 'USB debugging' and the adb tool (Android SDK platform-tools) on PATH."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    ls = subprocess.run(["adb", "shell", "ls", phone_dir], capture_output=True, text=True, check=True).stdout.split()
    vids = sorted(n for n in ls if n.endswith(".mp4"))
    if not vids:
        raise FileNotFoundError(f"No videos in {phone_dir}")
    vid = vids[-1]                                  # VID_yyyyMMdd_HHmmss.mp4 sorts chronologically
    subprocess.run(["adb", "pull", f"{phone_dir}/{vid}", str(dst)], check=True, capture_output=True)
    m = re.search(r"\d{8}_\d{6}", vid)
    if m and m.group(0) in ls:
        subprocess.run(["adb", "pull", f"{phone_dir}/{m.group(0)}", str(dst)], check=True, capture_output=True)
    else:
        raise FileNotFoundError(f"Sensor folder for {vid} not found on the phone (was IMU recording enabled?)")
    return dst / vid
