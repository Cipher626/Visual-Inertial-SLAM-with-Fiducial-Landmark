"""Evaluation metrics (all positions in metres, angles in degrees)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .geometry import umeyama, inv_T


def path_length(P):
    P = np.asarray(P)
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0


def _stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {}
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(e.mean()), "median": float(np.median(e)),
            "max": float(e.max()), "n": int(len(e))}


def ate(P_est, P_gt, align=True):
    """Absolute trajectory error after (optional) rigid alignment."""
    P_est, P_gt = np.asarray(P_est), np.asarray(P_gt)
    if align:
        _, R, t = umeyama(P_est, P_gt)
        P_est = P_est @ R.T + t
    else:
        R, t = np.eye(3), np.zeros(3)
    out = _stats(np.linalg.norm(P_est - P_gt, axis=1))
    out["alignment"] = {"R": R.tolist(), "t": t.tolist()} if align else None
    return out


def orientation_error(R_est, R_gt, R_align=np.eye(3)):
    E = np.einsum("nji,njk->nik", np.einsum("ij,njk->nik", R_align, R_est), R_gt)
    ang = np.degrees(Rot.from_matrix(E).magnitude())
    yaw = []
    for Re, Rg in zip(np.einsum("ij,njk->nik", R_align, R_est), R_gt):
        fe, fg = Re[:, 2], Rg[:, 2]          # camera forward axis projected on the floor
        a = np.arctan2(fe[1], fe[0]) - np.arctan2(fg[1], fg[0])
        yaw.append(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))
    return {"orientation": _stats(ang), "heading_abs": _stats(np.abs(yaw))}


def static_mean_positions(t, P, intervals):
    out = []
    for a, b in intervals:
        m = (t >= a) & (t <= b)
        if m.any():
            out.append(P[m].mean(0))
    return out


def endpoint_drift(t, P, intervals):
    """Closed-loop drift: distance between mean positions in the first and last still periods
    (the protocol starts and ends with the phone at the same spot). Falls back to first/last pose."""
    ps = static_mean_positions(t, P, intervals) if intervals else []
    if len(ps) >= 2:
        d = float(np.linalg.norm(ps[-1] - ps[0]))
        src = "static periods"
    else:
        d = float(np.linalg.norm(P[-1] - P[0]))
        src = "first/last pose"
    L = path_length(P)
    return {"endpoint_drift_m": d, "path_length_m": L, "drift_per_m": d / L if L > 0 else np.nan, "source": src}


def marker_map_errors(est, ref, align=True):
    common = sorted(set(est) & set(ref))
    if len(common) < (3 if align else 1):
        return {"n_common": len(common)}
    Pe = np.array([est[m][:3, 3] for m in common])
    Pr = np.array([ref[m][:3, 3] for m in common])
    R, t = np.eye(3), np.zeros(3)
    if align:
        _, R, t = umeyama(Pe, Pr)
    pos = np.linalg.norm(Pe @ R.T + t - Pr, axis=1)
    scale = umeyama(Pe, Pr, with_scale=True)[0] if len(common) >= 3 else None   # >1: map too small vs tape
    ang = [np.degrees(np.arccos(np.clip((R @ est[m][:3, 2]) @ ref[m][:3, 2], -1, 1))) for m in common]
    return {"n_common": len(common), "aligned": align, "scale_vs_survey": scale,
            "position": _stats(pos), "normal_angle": _stats(ang),
            "per_marker": {str(m): {"pos_err_m": float(p), "normal_err_deg": float(a)} for m, p, a in zip(common, pos, ang)},
            "alignment": {"R": R.tolist(), "t": t.tolist()}}


def marker_world_estimates(poses, obs):
    """World position of each marker implied by each single observation: W_T_M = W_T_C C_T_M."""
    per = {}
    for i, o, t in obs:
        p = (poses[i] @ o.T_CM)[:3, 3]
        per.setdefault(o.marker_id, []).append((t, p))
    return per


def loop_closure_residuals(poses, obs, min_gap_s=10.0, window=5):
    """For markers re-observed after >= min_gap_s: distance between the marker position implied by the
    first sightings and by the last sightings. Large under odometry alone (drift), small after closure."""
    out = {}
    for m, lst in marker_world_estimates(poses, obs).items():
        ts = np.array([a for a, _ in lst])
        if ts[-1] - ts[0] < min_gap_s:
            continue
        P = np.array([p for _, p in lst])
        first = np.median(P[:window], axis=0)
        last = np.median(P[-window:], axis=0)
        out[str(m)] = {"gap_s": float(ts[-1] - ts[0]), "residual_m": float(np.linalg.norm(last - first))}
    vals = [v["residual_m"] for v in out.values()]
    return {"per_marker": out, "summary": _stats(vals)}


def observation_consistency(poses, obs):
    """Spread (RMS distance to the median) of per-observation marker positions. Measures how consistent
    repeated observations are given the trajectory."""
    per = {}
    for m, lst in marker_world_estimates(poses, obs).items():
        P = np.array([p for _, p in lst])
        per[str(m)] = float(np.sqrt(np.mean(np.sum((P - np.median(P, 0)) ** 2, axis=1))))
    return {"per_marker_rms_m": per, "summary": _stats(list(per.values()))}


def detection_rate(poses, markers, K, image_size, detected_by_node, max_range=6.0, max_angle_deg=60.0, margin=10):
    """Detected / expected marker sightings. 'Expected' = marker centre and corners project inside the
    image, within range and facing the camera, according to the optimised poses and map."""
    w, h = image_size
    exp = det = 0
    for i, T in enumerate(poses):
        T_CW = inv_T(T)
        for m, T_WM in markers.items():
            T_CM = T_CW @ T_WM
            p = T_CM[:3, 3]
            if p[2] <= 0.2 or np.linalg.norm(p) > max_range:
                continue
            cosang = -T_CM[:3, 2] @ (p / np.linalg.norm(p))
            if cosang < np.cos(np.radians(max_angle_deg)):
                continue
            uv = K @ (p / p[2])
            if not (margin <= uv[0] < w - margin and margin <= uv[1] < h - margin):
                continue
            exp += 1
            det += int(m in detected_by_node.get(i, ()))
    return {"expected": exp, "detected": det, "rate": det / exp if exp else np.nan}


def checkpoint_errors(t, P, intervals, checkpoints, T_align=None):
    """Match the i-th still period to the i-th surveyed checkpoint (x, y in room frame), report XY error."""
    ps = static_mean_positions(t, P, intervals)
    if T_align is not None:
        ps = [T_align[:3, :3] @ p + T_align[:3, 3] for p in ps]
    n = min(len(ps), len(checkpoints))
    errs = {}
    for k in range(n):
        cp = checkpoints.iloc[k]
        errs[str(cp["id"])] = float(np.hypot(ps[k][0] - cp["x"], ps[k][1] - cp["y"]))
    return {"n_static": len(ps), "n_checkpoints": int(len(checkpoints)), "count_mismatch": len(ps) != len(checkpoints),
            "per_checkpoint_xy_m": errs,
            "summary": _stats(list(errs.values()))}
