"""Make a degraded copy of a dataset to stress-test the pipeline (e.g. a camera blackout to force the
inertial fallback, motion blur, or low light). Sensor files are copied unchanged.

    python tools/degrade.py data/sim_loop data/sim_blackout --blackout 20 22
    python tools/degrade.py data/sim_loop data/sim_dark --gain 0.35 --noise 6
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vislam.dataset import load_dataset, open_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blackout", nargs=2, type=float, metavar=("T0", "T1"), help="black frames between T0 and T1 [s]")
    ap.add_argument("--blur", type=int, default=0, help="horizontal motion-blur kernel length [px]")
    ap.add_argument("--gain", type=float, default=1.0, help="brightness gain (<1 = darker)")
    ap.add_argument("--noise", type=float, default=0.0, help="extra Gaussian noise sigma [grey levels]")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("video.mp4"))
    ds = load_dataset(src)
    cap = open_video(ds.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(dst / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), False)
    rng = np.random.default_rng(0)
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        t = ds.frame_t[min(k, len(ds.frame_t) - 1)]
        if a.blur > 1:
            g = cv2.filter2D(g, -1, np.ones((1, a.blur), np.float32) / a.blur)
        g = g * a.gain + (rng.normal(0, a.noise, g.shape) if a.noise else 0)
        if a.blackout and a.blackout[0] <= t <= a.blackout[1]:
            g[:] = 3
        vw.write(np.clip(g, 0, 255).astype(np.uint8))
        k += 1
    vw.release()
    import json
    meta = json.loads((dst / "meta.json").read_text()) if (dst / "meta.json").exists() else {}
    meta["degraded_from"] = str(src)
    meta["degradation"] = {k: v for k, v in vars(a).items() if k not in ("src", "dst")}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
