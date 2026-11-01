"""Camera intrinsics from a ChArUco video, and camera-IMU time offset + rotation from gyro vs visual rotation."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .dataset import iter_frames
from .geometry import kabsch_vectors
from .imu import ImuData
from .markers import get_dictionary

DEFAULT_BOARD = {"squares_x": 5, "squares_y": 7, "square_m": 0.035, "marker_m": 0.026, "dictionary": "DICT_5X5_100"}


def make_board(cfg=None):
    c = {**DEFAULT_BOARD, **(cfg or {})}
    board = cv2.aruco.CharucoBoard((c["squares_x"], c["squares_y"]), c["square_m"], c["marker_m"],
                                   get_dictionary(c["dictionary"]))
    return board, c


def detect_board(detector, board, gray, min_corners=8):
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None, None
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    if obj is None or len(obj) < min_corners:
        return None, None
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def calibrate_camera_charuco(ds, board_cfg=None, stride=5, max_views=60, progress=None):
    """Intrinsics from a video of the printed ChArUco board (move the phone, tilt it, cover the whole image)."""
    board, cfg = make_board(board_cfg)
    detector = cv2.aruco.CharucoDetector(board)
    objs, imgs, size = [], [], None
    n_total = max(1, len(ds.frame_t) // stride)
    for k, (idx, t, gray) in enumerate(iter_frames(ds, stride=stride)):
        size = gray.shape[::-1]
        obj, img = detect_board(detector, board, gray, min_corners=12)
        if obj is not None:
            objs.append(obj)
            imgs.append(img)
        if progress:
            progress(min(1.0, k / n_total))
    if len(objs) < 8:
        raise RuntimeError(f"Board found in only {len(objs)} frames; need >= 8. Check board settings/printing.")
    if len(objs) > max_views:  # spread views over the recording
        sel = np.linspace(0, len(objs) - 1, max_views).astype(int)
        objs, imgs = [objs[i] for i in sel], [imgs[i] for i in sel]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None)
    per_view = []
    for o, i, r, t in zip(objs, imgs, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(size), "rms_px": float(rms),
            "n_views": len(objs), "per_view_rms_px": per_view, "board": cfg, "source": str(ds.root)}


def board_orientations(ds, cam, board_cfg=None, stride=1, progress=None):
    """Camera orientation R_BC (board frame <- camera) per frame where the board is seen."""
    board, _ = make_board(board_cfg)
    detector = cv2.aruco.CharucoDetector(board)
    K, dist = np.array(cam["K"]), np.array(cam["dist"])
    times, Rs = [], []
    n_total = max(1, len(ds.frame_t) // stride)
    for k, (idx, t, gray) in enumerate(iter_frames(ds, stride=stride)):
        obj, img = detect_board(detector, board, gray, min_corners=10)
        if obj is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                times.append(t)
                Rs.append(cv2.Rodrigues(rvec)[0].T)
        if progress:
            progress(min(1.0, k / n_total))
    return np.array(times), np.array(Rs)


def visual_angular_velocity(times, R_WC, min_dt=0.0, max_dt=0.4):
    """Body-frame (camera) angular velocity between orientation pairs (k, j), j the first sample at least
    `min_dt` after k. A longer baseline averages out per-frame pose noise (use ~0.2 s for single markers)."""
    times = np.asarray(times, float)
    t_mid, w, pairs = [], [], []
    for k in range(len(times) - 1):
        j = int(np.searchsorted(times, times[k] + min_dt, side="left"))
        j = max(j, k + 1)
        if j >= len(times):
            break
        dt = times[j] - times[k]
        if 0 < dt <= max_dt:
            rv = Rot.from_matrix(R_WC[k].T @ R_WC[j]).as_rotvec()
            t_mid.append(0.5 * (times[k] + times[j]))
            w.append(rv / dt)
            pairs.append((times[k], times[j]))
    return np.array(t_mid), np.array(w).reshape(-1, 3), np.array(pairs).reshape(-1, 2)


def estimate_cam_imu(times, R_WC, imu: ImuData, td_range=0.3, td_step=0.001, min_dt=0.0, max_dt=0.12):
    """Time offset td (frame at camera time t was captured at IMU time t + td) by maximising the
    correlation of |omega| from vision and gyro, then camera<-IMU rotation by Kabsch on omega vectors.
    `times`/`R_WC` may also be lists of sequences (e.g. one per marker); pairs never cross sequences."""
    if len(times) and np.ndim(times[0]) > 0:
        parts = [visual_angular_velocity(t, R, min_dt, max_dt) for t, R in zip(times, R_WC) if len(t) > 2]
        w_c = np.vstack([p[1] for p in parts]) if parts else np.zeros((0, 3))
        pairs = np.vstack([p[2] for p in parts]) if parts else np.zeros((0, 2))
    else:
        _, w_c, pairs = visual_angular_velocity(times, R_WC, min_dt, max_dt)
    if len(w_c) < 30:
        raise RuntimeError("Not enough consecutive visual orientations for camera-IMU calibration.")
    dts = pairs[:, 1] - pairs[:, 0]
    mag_c = np.linalg.norm(w_c, axis=1)

    def gyro_avg(td):
        dR = imu.delta_R(pairs[:, 0] + td, pairs[:, 1] + td)
        return Rot.from_matrix(dR).as_rotvec() / dts[:, None]

    cands = np.arange(-td_range, td_range + 1e-9, td_step)
    scores = []
    for td in cands:
        mag_i = np.linalg.norm(gyro_avg(td), axis=1)
        scores.append(np.corrcoef(mag_c, mag_i)[0, 1])
    scores = np.nan_to_num(np.array(scores), nan=-1)
    k = int(np.argmax(scores))
    td = cands[k]
    if 0 < k < len(cands) - 1:  # parabolic refinement
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            td += 0.5 * (y0 - y2) / den * td_step
    w_i = gyro_avg(td)
    use = (mag_c > 0.2) & (np.linalg.norm(w_i, axis=1) > 0.2)
    if use.sum() < 20:
        use = np.ones(len(w_c), bool)
    R_CI = kabsch_vectors(w_i[use], w_c[use])
    for _ in range(2):   # trim the worst 20 % (blurred / mis-detected frames) and refit
        e = np.linalg.norm(w_c[use] - w_i[use] @ R_CI.T, axis=1)
        keep = np.where(use)[0][e <= np.quantile(e, 0.8)]
        if len(keep) >= 20:
            R_CI = kabsch_vectors(w_i[keep], w_c[keep])
    resid = w_c[use] - w_i[use] @ R_CI.T
    return {"td_s": float(td), "R_CI": R_CI.tolist(), "corr_peak": float(scores[k]),
            "omega_rms_resid_rad_s": float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
            "n_pairs": int(use.sum()), "td_curve": {"td": cands.tolist(), "corr": scores.tolist()}}


def sync_diagnose(times, R_WC, imu, window_s=6.0, step_s=3.0, n_frames=None, min_dt=0.1, max_dt=0.3):
    """Separates the usual causes of a bad camera-IMU result:
    - offset that changes during the recording (frames and timestamps paired with a shift that varies),
    - constant offset of several frames (video started k frames after the timestamp file),
    - noisy visual rotation (blur, rolling shutter, board too small or lost)."""
    times = np.asarray(times, float)
    R_WC = np.asarray(R_WC)
    out = {"n_board_frames": int(len(times))}
    if n_frames:
        out["board_detected_fraction"] = len(times) / n_frames
    glob = estimate_cam_imu(times, R_WC, imu, td_range=0.3, min_dt=min_dt, max_dt=max_dt)
    out["global"] = {k: glob[k] for k in ("td_s", "corr_peak", "omega_rms_resid_rad_s", "n_pairs")}
    out["R_CI"] = glob["R_CI"]
    wins = []
    t0 = times[0]
    while t0 + window_s <= times[-1] + 1e-9:
        m = (times >= t0) & (times < t0 + window_s)
        if m.sum() > 30:
            try:
                r = estimate_cam_imu(times[m], R_WC[m], imu, td_range=0.3, min_dt=min_dt, max_dt=max_dt)
                _, w_c, _ = visual_angular_velocity(times[m], R_WC[m], min_dt, max_dt)
                wins.append({"t_start": float(t0), "td_ms": 1e3 * r["td_s"], "corr": r["corr_peak"],
                             "mean_rate_rad_s": float(np.linalg.norm(w_c, axis=1).mean()) if len(w_c) else 0.0})
            except RuntimeError:
                pass
        t0 += step_s
    out["windows"] = wins
    good = [w for w in wins if w["corr"] > 0.7 and w["mean_rate_rad_s"] > 0.3]
    period = float(np.median(np.diff(times))) if len(times) > 2 else 1 / 30
    verdicts = []
    if n_frames and len(times) / n_frames < 0.6:
        verdicts.append("board found in only {:.0f} % of frames: keep it in view, slow down, shorten exposure".format(
            100 * len(times) / n_frames))
    if len(good) >= 2:
        tds = np.array([w["td_ms"] for w in good])
        spread = tds.max() - tds.min()
        out["window_td_spread_ms"] = float(spread)
        if spread > 20:
            verdicts.append(f"time offset changes during the recording ({tds.min():.0f} to {tds.max():.0f} ms): frames "
                            "and timestamps are paired with a shift that varies (dropped video frames)")
        elif abs(np.median(tds)) > 40:
            k = np.median(tds) / (1e3 * period)
            verdicts.append(f"constant offset of {np.median(tds):.0f} ms (~{k:.1f} frames): the video probably starts "
                            "a few frames after the timestamp file; this shift differs per recording, so the offset "
                            "must be estimated for every recording, not copied from calibration")
        if np.median([w["corr"] for w in good]) < 0.85:
            verdicts.append("correlation is low even inside short windows: the visual rotation is noisy (motion blur, "
                            "rolling shutter, board small in the image)")
    else:
        verdicts.append("too few windows with strong rotation and good correlation: rotate more briskly (about all "
                        "three axes) for at least 20 s while the board stays sharp and in view")
    if not verdicts:
        verdicts.append("consistent: offset stable across the recording and correlation high")
    out["verdicts"] = verdicts
    # time series for plotting
    _, w_c, pairs = visual_angular_velocity(times, R_WC, min_dt, max_dt)
    dR = imu.delta_R(pairs[:, 0] + glob["td_s"], pairs[:, 1] + glob["td_s"])
    w_i = Rot.from_matrix(dR).as_rotvec() / (pairs[:, 1] - pairs[:, 0])[:, None]
    out["series"] = {"t": (0.5 * pairs.sum(1)).tolist(), "vision": np.linalg.norm(w_c, axis=1).tolist(),
                     "gyro": np.linalg.norm(w_i, axis=1).tolist()}
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())
