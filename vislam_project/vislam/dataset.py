"""Canonical dataset format + importer for OpenCamera Sensors recordings.

data/<name>/
    video.mp4      raw video (frames in sensor orientation; auto-rotation is disabled on read)
    frames.csv     t_ns              one row per video frame, same clock as IMU
    gyro.csv       t_ns,x,y,z        rad/s, phone IMU frame
    accel.csv      t_ns,x,y,z        m/s^2, phone IMU frame
    meta.json      free-form metadata (source, notes, experiment type, ...)
Optional: groundtruth.csv (simulation only), checkpoints.csv (id,x,y in room frame)."""
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    root: Path
    video: Path
    frame_t: np.ndarray            # seconds, relative to t0
    gyro: np.ndarray               # (N,4) t[s], x, y, z
    accel: np.ndarray              # (N,4) t[s], x, y, z
    t0_ns: int
    meta: dict = field(default_factory=dict)

    @property
    def duration(self):
        return float(self.frame_t[-1] - self.frame_t[0]) if len(self.frame_t) else 0.0


def _numeric_csv(path, ncols):
    df = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    arr = df.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < ncols:
        raise ValueError(f"{path}: expected >= {ncols} numeric columns, got {arr.shape}")
    return arr[:, :ncols]


def load_dataset(root):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}
    video = root / meta.get("video", "video.mp4")
    frames_ns = _numeric_csv(root / "frames.csv", 1)[:, 0].astype(np.int64)
    gyro = _numeric_csv(root / "gyro.csv", 4)
    accel = _numeric_csv(root / "accel.csv", 4)
    t0 = int(frames_ns[0])
    gyro = gyro[np.argsort(gyro[:, 0])]
    accel = accel[np.argsort(accel[:, 0])]
    to_s = lambda ns: (np.asarray(ns, np.float64) - t0) * 1e-9
    gyro[:, 0] = to_s(gyro[:, 0])
    accel[:, 0] = to_s(accel[:, 0])
    return Dataset(root, video, to_s(frames_ns), gyro, accel, t0, meta)


def safe_name(name):
    """Dataset / run names become folder names: no path separators or characters Windows rejects."""
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(name)).strip("_.") or "unnamed"


def open_video(path):
    # Prefer the FFmpeg backend explicitly: its per-frame times (CAP_PROP_POS_MSEC) and orientation handling are
    # what the importer and calibration were tested with; other backends (e.g. Windows MSMF) may differ.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # keep raw sensor orientation (consistent with calibration)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    return cap


def iter_frames(ds, stride=1, t_start=None, t_end=None, scale=1.0):
    """Yields (frame_index, t_seconds, gray_image). Stops at the shorter of video / timestamp list."""
    cap = open_video(ds.video)
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok or idx >= len(ds.frame_t):
                break
            t = ds.frame_t[idx]
            if idx % stride == 0 and (t_start is None or t >= t_start) and (t_end is None or t <= t_end):
                ok, img = cap.retrieve()
                if not ok:
                    break
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if scale != 1.0:
                    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                yield idx, float(t), gray
            if t_end is not None and t > t_end:
                break
            idx += 1
    finally:
        cap.release()


def video_frame_count(path):
    cap = open_video(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frame(ds, index):
    cap = open_video(ds.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def dataset_report(ds, count_video_frames=True):
    """Sensor status / sync sanity numbers shown in the UI."""
    def rate(t):
        return float(1.0 / np.median(np.diff(t))) if len(t) > 2 else 0.0

    def max_gap(t):
        return float(np.max(np.diff(t))) if len(t) > 2 else 0.0

    rep = {
        "duration_s": ds.duration,
        "n_frame_timestamps": int(len(ds.frame_t)),
        "camera_rate_hz": rate(ds.frame_t),
        "camera_max_gap_ms": 1e3 * max_gap(ds.frame_t),
        "gyro_rate_hz": rate(ds.gyro[:, 0]),
        "accel_rate_hz": rate(ds.accel[:, 0]),
        "gyro_max_gap_ms": 1e3 * max_gap(ds.gyro[:, 0]),
        "imu_covers_video": bool(ds.gyro[0, 0] <= ds.frame_t[0] + 0.05 and ds.gyro[-1, 0] >= ds.frame_t[-1] - 0.05),
        "imu_start_offset_s": float(ds.gyro[0, 0] - ds.frame_t[0]),
        "imu_end_offset_s": float(ds.gyro[-1, 0] - ds.frame_t[-1]),
    }
    if count_video_frames:
        n = ds.meta.get("n_video_frames") or video_frame_count(ds.video)   # counted once at import
        rep["n_video_frames"] = n
        rep["frame_count_match"] = n == len(ds.frame_t)
    if "frame_alignment" in ds.meta:
        rep["frame_alignment"] = ds.meta["frame_alignment"]
    return rep


def _find_one(folder, patterns, exclude=()):
    for pat in patterns:
        hits = [p for p in sorted(folder.rglob(pat)) if not any(e in p.name for e in exclude)]
        if hits:
            return hits[-1]
    return None


def video_frame_times(path):
    """Presentation time [s] of every frame as stored in the video container (relative to the first frame).
    Android's encoder keeps the camera's capture times here, so dropped frames show up as gaps."""
    cap = open_video(path)
    t = []
    while cap.grab():
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3)
    cap.release()
    t = np.asarray(t, float)
    return t - t[0] if len(t) else t


def align_frames(pts, ts, max_shift=40, tol=0.008):
    """Match video frames (container times `pts`, s, relative) to the timestamp file `ts` (s).
    Finds the constant offset c with ts ~= pts + c that matches the most frames (the pattern of gaps makes
    this unambiguous when frames were dropped), then gives each video frame its nearest timestamp.
    Returns (per-frame timestamps, report dict)."""
    pts, ts = np.asarray(pts, float), np.asarray(ts, float)
    cands = {round(ts[i] - pts[j], 4) for i in range(min(max_shift, len(ts))) for j in range(min(max_shift, len(pts)))}
    scored = []
    for c in cands:
        q = pts + c
        idx = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
        nn = np.where(np.abs(ts[idx - 1] - q) <= np.abs(ts[idx] - q), idx - 1, idx)
        r = np.abs(ts[nn] - q)
        frac = float(np.mean(r < tol))
        scored.append((frac, -float(np.median(r)), c))
    scored.sort(reverse=True)
    frac, _, c = scored[0]
    period = float(np.median(np.diff(ts)))
    rivals = [x for x in scored[1:] if abs(x[2] - c) > 0.5 * period]
    ambiguous = bool(rivals and rivals[0][0] >= frac - 0.002)
    if ambiguous:          # no gap pattern to lock onto: fall back to pairing from the first frame
        c = float(ts[0] - pts[0])
    q = pts + c
    idx = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    nn = np.where(np.abs(ts[idx - 1] - q) <= np.abs(ts[idx] - q), idx - 1, idx)
    r = np.abs(ts[nn] - q)
    inl = r < tol
    if inl.sum() > 3:
        c = float(np.median(ts[nn[inl]] - pts[inl]))
        q = pts + c
    out = np.where(inl, ts[nn], q)                     # unmatched frames get the container time + offset
    rep = {"method": "container frame times", "matched_fraction": frac, "ambiguous": ambiguous,
           "first_timestamp_index": int(nn[0]), "median_residual_ms": 1e3 * float(np.median(r)),
           "n_video": int(len(pts)), "n_timestamps": int(len(ts)), "n_unmatched": int((~inl).sum()),
           "timestamps_without_video_frame": int(len(ts) - len(np.unique(nn[inl])))}
    return out, rep


def _xyz_t(arr):
    """OpenCamera Sensors rows are  x, y, z[, extra values...], t_ns : first three columns + last column."""
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, -1].astype(np.int64)


def ingest_opencamera(src, dst_root, name=None, notes="", video=None, extra_meta=None):
    """Import one OpenCamera Sensors recording into the canonical format.
    Phone layout (as written by the app's code): DCIM/OpenCamera/VID_<date>.mp4 and DCIM/OpenCamera/<date>/
    containing VID_<date>gyro.csv, VID_<date>accel.csv, VID_<date>magnetic.csv (no underscore before the
    sensor name, unlike the app's README) and VID_<date>_imu_timestamps.csv. Underscored variants are
    accepted too. `src` may be a folder holding one or many recordings (newest video used) or the .mp4 itself."""
    clean = lambda p: Path(str(p).strip().strip('"').strip("'").strip())   # Windows "Copy as path" adds quotes
    src = clean(src)
    video = clean(video) if video is not None else None
    if not src.exists():
        raise FileNotFoundError(f"Path not found: {src}")
    if video is None and src.suffix.lower() == ".mp4":
        video, src = src, src.parent
    if video is None:
        videos = sorted(src.rglob("*.mp4"), key=lambda p: (p.name, p.stat().st_mtime))
        if not videos:
            raise FileNotFoundError(f"No .mp4 found under {src}")
        video = videos[-1]
    video = Path(video)
    stem = video.stem
    search_root = src if src.exists() else video.parent
    gyro = _find_one(search_root, [f"{stem}gyro.csv", f"{stem}_gyro.csv"])
    accel = _find_one(search_root, [f"{stem}accel.csv", f"{stem}_accel.csv"])
    stamps = _find_one(search_root, [f"{stem}_imu_timestamps.csv", f"{stem}imu_timestamps.csv", f"{stem}_timestamps.csv"])
    warnings = []
    if not (gyro and accel and stamps):   # fall back to any CSVs, but say so
        gyro = gyro or _find_one(search_root, ["*gyro.csv"])
        accel = accel or _find_one(search_root, ["*accel.csv"])
        stamps = stamps or _find_one(search_root, ["*imu_timestamps.csv", "*_timestamps.csv"], exclude=("recsync",))
        warnings.append("CSV names did not match the video name; check that they belong to the same recording")
    if not (gyro and accel and stamps):
        raise FileNotFoundError(f"Missing CSVs for {video.name}: gyro={gyro}, accel={accel}, timestamps={stamps}. "
                                "Enable 'Enable sync video IMU recording' in OpenCamera Sensors (IMU settings).")
    name = safe_name(name or stem)
    dst = Path(dst_root) / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, dst / "video.mp4")
    gx, gy, gz, gt = _xyz_t(_numeric_csv(gyro, 4))
    ax, ay, az, at = _xyz_t(_numeric_csv(accel, 4))
    f = _numeric_csv(stamps, 1)
    if len(gt) < 10 or len(at) < 10 or len(f) < 10:
        raise ValueError(f"Too few rows: gyro {len(gt)}, accel {len(at)}, frame timestamps {len(f)}")
    pd.DataFrame({"t_ns": gt, "x": gx, "y": gy, "z": gz}).to_csv(dst / "gyro.csv", index=False)
    pd.DataFrame({"t_ns": at, "x": ax, "y": ay, "z": az}).to_csv(dst / "accel.csv", index=False)
    ts_ns = f[:, 0].astype(np.int64)
    pts = video_frame_times(dst / "video.mp4")
    alignment = {"method": "in order (counts match)", "n_video": int(len(pts)), "n_timestamps": int(len(ts_ns))}
    if len(pts) != len(ts_ns) or np.any(np.abs(np.diff(pts) - np.diff((ts_ns[:len(pts)] - ts_ns[0]) * 1e-9)) > 0.008):
        t_al, alignment = align_frames(pts, (ts_ns - ts_ns[0]) * 1e-9)
        ts_ns = (np.round(t_al * 1e9) + ts_ns[0]).astype(np.int64)
        if alignment["ambiguous"]:
            warnings.append("frame/timestamp alignment is ambiguous (video frame times are evenly spaced); "
                            "frames were matched by best guess - check the sync check after processing")
        elif alignment["matched_fraction"] < 0.98:
            warnings.append(f"only {100 * alignment['matched_fraction']:.1f} % of video frames matched a timestamp")
    pd.DataFrame({"t_ns": ts_ns}).to_csv(dst / "frames.csv", index=False)
    meta = {"video": "video.mp4", "source": "OpenCamera Sensors", "original_video": video.name,
            "gyro_file": gyro.name, "accel_file": accel.name, "timestamps_file": stamps.name, "notes": notes,
            "frame_alignment": alignment, "n_video_frames": int(len(pts)), "import_warnings": warnings}
    meta.update(extra_meta or {})
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return dst


def make_excerpt(ds, out_root, seconds=20.0, scale=0.5, calib_files=()):
    """Small, self-contained copy of the first `seconds` of a dataset (video downscaled by `scale`) plus the
    calibration files, zipped, so a recording can be shared for debugging without uploading the full video."""
    import zipfile
    name = f"{ds.root.name}_first{int(seconds)}s"
    dst = Path(out_root) / name
    dst.mkdir(parents=True, exist_ok=True)
    cap = open_video(ds.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(np.searchsorted(ds.frame_t, ds.frame_t[0] + seconds))
    writer = None
    for k in range(n):
        ok, img = cap.read()
        if not ok:
            n = k
            break
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(dst / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
    cap.release()
    if writer is not None:
        writer.release()
    t_end = ds.frame_t[n - 1] + 1.0
    to_ns = lambda t: (np.round(t * 1e9) + ds.t0_ns).astype(np.int64)
    pd.DataFrame({"t_ns": to_ns(ds.frame_t[:n])}).to_csv(dst / "frames.csv", index=False)
    for nm, arr in (("gyro", ds.gyro), ("accel", ds.accel)):
        a = arr[arr[:, 0] <= t_end]
        pd.DataFrame({"t_ns": to_ns(a[:, 0]), "x": a[:, 1], "y": a[:, 2], "z": a[:, 3]}).to_csv(dst / f"{nm}.csv", index=False)
    meta = dict(ds.meta, video="video.mp4", excerpt={"seconds": seconds, "scale": scale, "frames": int(n)})
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    for f in calib_files:
        f = Path(f)
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if f.name == "camera.json" and scale != 1.0:      # keep intrinsics consistent with the smaller video
            K = np.array(d["K"], float)
            K[:2] *= scale
            d["K"] = K.tolist()
            d["image_size"] = [int(round(v * scale)) for v in d["image_size"]]
            d.pop("per_view_rms_px", None)
        (dst / f.name).write_text(json.dumps(d, indent=2))
    zpath = Path(out_root) / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dst.iterdir()):
            z.write(p, f"{name}/{p.name}")
    return zpath


def list_datasets(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    return sorted(p.name for p in data_root.iterdir() if (p / "frames.csv").exists())
