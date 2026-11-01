"""End-to-end processing: dataset -> front-end -> baseline -> marker factor graph -> metrics -> files."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from .backend import Graph, optimize, corner_reprojection_errors
from .calibration import estimate_cam_imu
from .dataset import load_dataset, iter_frames, open_video, safe_name
from .frontend import FrontEnd
from .geometry import inv_T, marker_pose_from_survey, make_T
from .imu import ImuData, R_CI_DEFAULT
from .markers import MarkerDetector
from . import metrics as M

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- map files
def load_map(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for k, v in d["markers"].items():
        out[int(k)] = np.array(v["T_WM"]) if "T_WM" in v else marker_pose_from_survey(v["position"], v["normal"])
    return out, d


def save_map(path, markers, dictionary, size, frame, extra=None):
    d = {"dictionary": dictionary, "marker_size": size, "frame": frame,
         "markers": {str(m): {"position": T[:3, 3].tolist(), "normal": T[:3, 2].tolist(), "T_WM": T.tolist()}
                     for m, T in sorted(markers.items())}}
    if extra:
        d.update(extra)
    Path(path).write_text(json.dumps(d, indent=2))


def find_file(name, ds_root):
    for p in (Path(ds_root) / name, ROOT / "calib" / name, ROOT / "room" / name):
        if p.exists():
            return p
    return None


def _traj_df(records_or_t, poses, extra=None):
    q = Rot.from_matrix(poses[:, :3, :3]).as_quat()
    df = pd.DataFrame({"t": records_or_t, "x": poses[:, 0, 3], "y": poses[:, 1, 3], "z": poses[:, 2, 3],
                       "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]})
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def load_traj(path):
    df = pd.read_csv(path)
    R = Rot.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    T = np.array([make_T(r, p) for r, p in zip(R, df[["x", "y", "z"]].values)])
    return df, T


SCAN_DICTS = ["DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL",
              "DICT_APRILTAG_36h11"]


def scan_markers(ds, stride=15, scale=1.0):
    """Which marker families appear in the video at all, and how bright it is (used to explain failures)."""
    import cv2
    dets = {d: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, d))) for d in SCAN_DICTS}
    found, bright = {}, []
    for _, _, g in iter_frames(ds, stride=stride, scale=scale):
        bright.append(float(g.mean()))
        for name, d in dets.items():
            corners, ids, _ = d.detectMarkers(g)
            if ids is not None:
                for c_, i in zip(corners, ids.ravel()):
                    side = float(np.mean(np.linalg.norm(c_.reshape(4, 2) - np.roll(c_.reshape(4, 2), 1, 0), axis=1)))
                    e = found.setdefault(name, {})
                    e[int(i)] = max(e.get(int(i), 0.0), side)
    return found, (float(np.mean(bright)) if bright else 0.0)


def bootstrap_failure_reason(ds, cfg, bd, map_ids, scale):
    """Turn the front-end's bootstrap bookkeeping into one specific, actionable sentence."""
    c, mk = cfg["frontend"], cfg["markers"]
    atts = bd.get("attempts", [])
    extra = {}
    if bd.get("frames_with_marker", 0) == 0:
        found, bright = scan_markers(ds, scale=scale)
        extra = {"markers_in_video_any_dictionary": {k: {str(i): round(v) for i, v in d.items()} for k, d in found.items()},
                 "mean_brightness": bright}
        if found:
            desc = "; ".join(f"{k}: ids {sorted(d)[:12]}" for k, d in found.items())
            return (f"no {mk['dictionary']} marker with an allowed ID (sidebar 'IDs placed in the room': "
                    f"{mk.get('valid_ids')}) was seen, but the video contains {desc}. 4x4 markers are also found by "
                    "larger 4x4 dictionaries, so check the IDs; 5x5 markers are the calibration board. Set the "
                    "dictionary and IDs to what you printed."), extra
        if bright < 45:
            return (f"no marker found and the video is very dark (mean brightness {bright:.0f}/255). Raise ISO or add "
                    "light while keeping exposure at 8 ms or less."), extra
        return ("no marker found in any frame, in any common dictionary: start the recording facing a marker that is "
                "sharp and fills at least ~3 % of the image width."), extra
    max_side = max(bd["max_side_px"].values()) if bd.get("max_side_px") else 0.0
    if map_ids is not None:
        seen = set(bd.get("max_side_px", {}))
        if not seen & set(map_ids):
            return f"markers {sorted(seen)} were seen but none of them is in the loaded map {sorted(map_ids)}.", extra
    if not atts:
        return (f"markers were seen (largest {max_side:.0f} px wide) but none reached the {c['boot_min_side_px']} px "
                "needed to start: stand closer to a marker at the start, print bigger markers, or lower "
                "frontend.boot_min_side_px in config.json."), extra
    from collections import Counter
    best_base = max(a["max_baseline_m"] for a in atts)
    best_pts = max(a["max_points"] for a in atts)
    ends = Counter(a["end"] for a in atts if a["end"] != "running").most_common(2)
    why = ", ".join(f"{e} ({n}x)" for e, n in ends) or "recording ended"
    if best_base < c["boot_baseline_m"]:
        return (f"{bd.get('n_attempts', len(atts))} start attempts, but the phone never moved "
                f"{100 * c['boot_baseline_m']:.0f} cm sideways while the start marker stayed in view (best "
                f"{100 * best_base:.0f} cm). Attempts ended because: {why}. Hold still facing the marker, then "
                "side-step ~20 cm slowly without turning the phone, keeping the marker in view."), extra
    if best_pts < c["boot_min_points"]:
        nf = max(a["n_features"] for a in atts)
        return (f"the side-step was seen (up to {100 * best_base:.0f} cm) but only {best_pts} 3D points could be "
                f"built (needs {c['boot_min_points']}; at most {nf} trackable features near the start). The area "
                "around the start marker lacks texture or the images are blurred: start facing a textured area."), extra
    return f"{len(atts)} start attempts failed ({why}).", extra


def estimate_recording_td(records, imu, sc):
    """Time offset of one recording from its own visual-odometry rotation vs the gyro."""
    vo = [r for r in records if r.T_WC is not None and r.status == "vo"]
    if len(vo) < 60:
        return {"reliable": False, "reason": "too few visually tracked frames"}
    t = np.array([r.t for r in vo])
    R = np.array([r.T_WC[:3, :3] for r in vo])
    try:
        chk = estimate_cam_imu(t, R, imu, td_range=sc.get("td_search_ms", 300.0) / 1e3, min_dt=0.1, max_dt=0.3)
    except RuntimeError as e:
        return {"reliable": False, "reason": str(e)}
    ok = chk["corr_peak"] >= sc.get("min_corr", 0.85) and chk["n_pairs"] >= 60
    return {"td_recording_s": chk["td_s"], "corr": chk["corr_peak"], "n_pairs": chk["n_pairs"],
            "omega_rms_resid_rad_s": chk["omega_rms_resid_rad_s"], "reliable": bool(ok),
            **({} if ok else {"reason": f"correlation {chk['corr_peak']:.2f} from {chk['n_pairs']} rotation samples "
                                         f"(needs >= {sc.get('min_corr', 0.85)} and >= 60)"})}


# ----------------------------------------------------------------------------- main entry
def run(dataset_dir, cfg, mode="slam", map_file=None, out_root=None, progress=None, log=print, tag=None):
    """mode: 'slam'   - markers are unknown landmarks, estimated jointly with the trajectory
             'anchor' - surveyed marker map (room frame) used as priors
             'reloc'  - map produced by an earlier run is loaded; the platform relocalises in it"""
    t_start = time.perf_counter()
    ds = load_dataset(dataset_dir)
    out = Path(out_root or ROOT / "runs") / ds.root.name / safe_name(tag or mode)
    out.mkdir(parents=True, exist_ok=True)
    mk, fe_cfg, be = cfg["markers"], cfg["frontend"], cfg["backend"]
    warnings = []

    cam_path = find_file("camera.json", ds.root)
    if cam_path is None:
        raise FileNotFoundError("camera.json not found (dataset folder or calib/). Run camera calibration first.")
    cam = json.loads(cam_path.read_text())
    s = float(cfg["processing"].get("resize", 1.0))
    # pixel thresholds are tuned for full-resolution video: scale them with any resize / downscaled excerpt
    px = s * float(ds.meta.get("excerpt", {}).get("scale", 1.0))
    if px != 1.0:
        fe_cfg = {**fe_cfg, **{k: fe_cfg[k] * px for k in ("boot_min_side_px", "boot_parallax_px", "kf_parallax_px")}}
        mk = {**mk, "min_side_px": mk["min_side_px"] * px}
    K = np.array(cam["K"], float)
    K[:2] *= s
    dist = np.array(cam["dist"], float)
    img_size = (int(round(cam["image_size"][0] * s)), int(round(cam["image_size"][1] * s)))
    # the video must have the resolution and orientation the camera was calibrated at, or every
    # pixel-to-ray conversion is silently wrong
    cap = open_video(ds.video)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read frames from {ds.video}")
    vw, vh = int(round(first.shape[1] * s)), int(round(first.shape[0] * s))
    if (vw, vh) != img_size:
        if abs(vw / vh - img_size[0] / img_size[1]) > 0.01:
            raise ValueError(f"video frames are {first.shape[1]}x{first.shape[0]} but camera.json was calibrated at "
                             f"{cam['image_size'][0]}x{cam['image_size'][1]} (different shape or orientation). "
                             "Recalibrate with the same video resolution.")
        r = vw / img_size[0]
        K[:2] *= r
        warnings.append(f"video is {first.shape[1]}x{first.shape[0]}, calibration {cam['image_size'][0]}x"
                        f"{cam['image_size'][1]}: intrinsics scaled by {r:.3f}. Recalibrating at this resolution is better.")
        img_size = (vw, vh)
    ci_path = find_file("camera_imu.json", ds.root)
    if ci_path is not None:
        ci = json.loads(ci_path.read_text())
        R_CI, td = np.array(ci["R_CI"]), float(ci["td_s"])
    else:
        R_CI, td = R_CI_DEFAULT, 0.0
        warnings.append("camera_imu.json not found: using default R_CI and td = 0")

    warnings += [f"import: {w}" for w in ds.meta.get("import_warnings", [])]
    imu = ImuData(ds.gyro, ds.accel)
    static = imu.static_intervals(min_dur=cfg["evaluation"]["static_min_s"])
    if static and static[0][0] < 3.0:
        imu.estimate_gyro_bias(static[0])
    else:
        warnings.append("no still period at the start: gyro bias not estimated")

    known_map, map_meta = {}, {}
    if mode in ("anchor", "reloc"):
        if map_file is None:
            map_file = find_file("markers_surveyed.json", ds.root) if mode == "anchor" else None
        if map_file is None:
            raise FileNotFoundError(f"mode '{mode}' needs a marker map file")
        known_map, map_meta = load_map(map_file)
    size = float(map_meta.get("marker_size", mk["marker_size_m"]))
    dictionary = map_meta.get("dictionary", mk["dictionary"])
    valid = mk.get("valid_ids") or None
    if valid and known_map:
        valid = sorted(set(valid) | set(known_map))
    det = MarkerDetector(dictionary, size, K, dist, mk["min_side_px"], mk["max_reproj_px"], mk["max_range_m"],
                         valid_ids=valid, error_correction_rate=mk.get("error_correction_rate", 0.6))

    # ---------------- front-end, then a per-recording camera-IMU time offset
    # The offset between frame timestamps and IMU time is not fixed across recordings on every phone (e.g. the
    # video can start a few frames after the timestamp file), so each recording measures its own offset from
    # visual-odometry rotation vs gyro and, if it differs from the one in use, the front-end is run again.
    stride = int(cfg["processing"]["stride"])
    n_exp = len(ds.frame_t) // stride
    sc = cfg.get("sync", {})

    grav = None
    if static and static[0][0] < 3.0:
        grav = (0.5 * (static[0][0] + static[0][1]), imu.mean_specific_force(*static[0]))

    def run_frontend(td_use, label, p0, p1, cached_obs=None):
        fe = FrontEnd(fe_cfg, K, dist, imu, R_CI, td_use, det, mode=mode, known_map=known_map, gravity_ref=grav)
        return fe.run(iter_frames(ds, stride=stride, scale=s), n_expected=n_exp,
                      progress=(lambda p: progress(p0 + (p1 - p0) * p, label)) if progress else None,
                      cached_obs=cached_obs)

    log("front-end: visual odometry + gyro + marker detection")
    res = run_frontend(td, "front-end", 0.0, 0.45)
    detect_ms = res.stats["t_detect_ms_per_frame"]          # a 2nd pass reuses these detections
    sync_info = {"td_calibration_s": td, "td_used_s": td, "source": "calibration", "rerun": False}
    if sc.get("per_recording_td", True):
        est = estimate_recording_td(res.records, imu, sc)
        sync_info.update(est)
        if est.get("reliable"):
            sync_info["td_used_s"], sync_info["source"] = est["td_recording_s"], "this recording"
            if abs(est["td_recording_s"] - td) * 1e3 > sc.get("rerun_if_diff_ms", 10.0):
                log(f"time offset of this recording {1e3 * est['td_recording_s']:.1f} ms vs {1e3 * td:.1f} ms in use: "
                    "running the front-end again")
                td = est["td_recording_s"]
                res = run_frontend(td, "front-end (2nd pass)", 0.45, 0.8, cached_obs=res.frame_obs)
                sync_info["rerun"] = True
            td = est["td_recording_s"]
        else:
            warnings.append("time offset of this recording could not be measured reliably "
                            f"({est.get('reason', 'low correlation')}); using the calibrated value. Turn the phone more "
                            "during the walk (look left and right at corners).")
    recs = [r for r in res.records if r.T_WC is not None]
    if len(recs) < 10:
        bdiag = res.stats.get("bootstrap_diagnostics", {})
        msg, extra = bootstrap_failure_reason(ds, {**cfg, "frontend": fe_cfg, "markers": mk}, bdiag,
                                              set(known_map) if known_map else None, s)
        (out / "bootstrap_diagnostics.json").write_text(json.dumps({"reason": msg, **bdiag, **extra}, indent=2, default=str))
        raise RuntimeError(f"Bootstrap never succeeded: {msg} (details: {out / 'bootstrap_diagnostics.json'})")
    node_of = {r.k: i for i, r in enumerate(recs)}
    T0 = np.array([r.T_WC for r in recs])
    times = np.array([r.t for r in recs])
    status = [r.status for r in recs]
    obs = [(node_of[k], o, recs[node_of[k]].t) for k, o in res.observations if k in node_of]
    counts = {}
    for _, o, _ in obs:
        counts[o.marker_id] = counts.get(o.marker_id, 0) + 1
    rare = sorted(m for m, n in counts.items() if n < mk.get("min_sightings", 1))
    if rare:   # an ID seen only a handful of times is more likely a false detection than a landmark
        warnings.append(f"ignored marker ids seen fewer than {mk.get('min_sightings')} times: {rare}")
        obs = [x for x in obs if x[1].marker_id not in rare]

    # ---------------- baseline map: first sighting, W_T_M = W_T_C C_T_M (handout eq.)
    base_map = {}
    for i, o, _ in obs:
        base_map.setdefault(o.marker_id, T0[i] @ o.T_CM)

    # ---------------- factor graph
    log("back-end: factor graph with marker corner constraints")
    g = Graph(T0=T0)
    d2r = np.pi / 180
    seg = 0                  # VO segment index: a new unknown scale starts after every re-initialisation
    for i in range(1, len(recs)):
        Tij = inv_T(T0[i - 1]) @ T0[i]
        dn = np.linalg.norm(Tij[:3, 3])
        if recs[i].reinit:
            seg += 1
        if status[i] == "imu":
            g.odo.append((i - 1, i, Tij, be["sigma_imu_rot_deg"] * d2r, be["sigma_imu_trans_m"] + be["sigma_imu_trans_frac"] * dn, -1))
        elif status[i] == "init":   # bootstrap poses come straight from the marker (metric)
            g.odo.append((i - 1, i, Tij, be["sigma_vo_rot_deg"] * d2r, be["sigma_vo_trans_m"] + be["sigma_vo_trans_frac"] * dn, -1))
        else:   # segment 0 inherits metric scale from the bootstrap marker; later segments get a free scale
            g.odo.append((i - 1, i, Tij, be["sigma_vo_rot_deg"] * d2r, be["sigma_vo_trans_m"] + be["sigma_vo_trans_frac"] * dn,
                          seg if seg > 0 else -1))
    g.scale_prior = {s: be["sigma_log_scale_reinit"] for s in range(1, seg + 1)}
    dR = imu.delta_R(times[:-1] + td, times[1:] + td)
    for i in range(1, len(recs)):
        g.gyro.append((i - 1, i, R_CI @ dR[i - 1] @ R_CI.T, be["sigma_gyro_rot_deg"] * d2r))
    best = {}
    for i, o, _ in obs:
        if o.marker_id not in best or o.side_px > best[o.marker_id][1].side_px:
            best[o.marker_id] = (i, o)
    g.marker_init = {m: T0[i] @ o.T_CM for m, (i, o) in best.items()}
    if mode == "slam":
        g.pose_prior.append((0, T0[0], 1e-4, 1e-4))
    else:
        for m, T in known_map.items():
            if m in g.marker_init:
                g.marker_init[m] = T
                g.marker_prior[m] = (T, be["sigma_map_prior_deg"] * d2r, be["sigma_map_prior_m"])
    g.obs = [(i, o.marker_id, o.corners_norm) for i, o, _ in obs]
    f_px = float(K[0, 0])
    t_be = time.perf_counter()
    poses, markers, info1 = optimize(g, f_px, size, be["sigma_corner_px"], be["huber_scale"], be["max_iter"])
    if progress:
        progress(0.9, "back-end")
    err = corner_reprojection_errors(poses, markers, g.obs, f_px, size)
    keep = err < be["outlier_reproj_px"]
    n_rej = int((~keep).sum())
    if n_rej:
        g.obs = [o for o, k in zip(g.obs, keep) if k]
        g.T0, g.marker_init = poses, markers
        poses, markers, info2 = optimize(g, f_px, size, be["sigma_corner_px"], be["huber_scale"], be["max_iter"])
    else:
        info2 = None
    last = info2 or info1
    if not last["converged"]:
        warnings.append(f"back-end stopped after {last['iterations']} iterations without converging; "
                        "raise backend.max_iter in config.json")
    if last.get("markers_without_observations"):
        warnings.append(f"markers {last['markers_without_observations']} lost all observations to outlier "
                        "rejection and were left out of the map")
    t_be = time.perf_counter() - t_be
    err_final = corner_reprojection_errors(poses, markers, g.obs, f_px, size)
    log(f"back-end done: {info1['n_marker_obs']} marker observations, {n_rej} rejected")

    # ---------------- save trajectories / maps / observations
    extra = {"frame_idx": [r.frame_idx for r in recs], "status": status}
    _traj_df(times, T0, extra).to_csv(out / "trajectory_baseline.csv", index=False)
    _traj_df(times, poses, extra).to_csv(out / "trajectory_optimized.csv", index=False)
    if mode == "anchor" or (mode == "reloc" and str(map_meta.get("frame", "")).startswith("room")):
        frame_name = "room"
    elif mode == "reloc":
        frame_name = str(map_meta.get("frame", "map"))
    else:
        frame_name = "slam (gravity-aligned, origin at first camera)"
    save_map(out / "map.json", markers, dictionary, size, frame_name, {"source_run": str(out)})
    save_map(out / "map_baseline.json", base_map, dictionary, size, frame_name)
    pd.DataFrame([{"node": i, "t": t, "marker_id": o.marker_id, "side_px": o.side_px, "range_m": float(np.linalg.norm(o.T_CM[:3, 3])),
                   "pnp_reproj_px": o.reproj_px, "ambiguity": o.ambiguity} for i, o, t in obs]).to_csv(out / "observations.csv", index=False)
    pd.DataFrame([vars(r) | {"T_WC": None} for r in res.records]).drop(columns=["T_WC"]).to_csv(out / "frontend.csv", index=False)

    # ---------------- metrics
    met = {"mode": mode, "dataset": ds.root.name, "warnings": warnings, "frontend": res.stats,
           "backend": {"round1": info1, "round2": info2, "rejected_obs": n_rej, "time_s": t_be,
                       "corner_reproj_px": M._stats(err_final)},
           "timing": {}}
    for name, P in (("baseline", T0), ("optimized", poses)):
        met[name] = {"closed_loop": M.endpoint_drift(times, P[:, :3, 3], static),
                     "loop_closure_residual": M.loop_closure_residuals(P, obs),
                     "observation_consistency": M.observation_consistency(P, obs)}
    det_by_node = {}
    for i, o, _ in obs:
        det_by_node.setdefault(i, set()).add(o.marker_id)
    met["detection"] = M.detection_rate(poses, markers, K, img_size, det_by_node, mk["max_range_m"])
    met["detection"]["frames_with_marker"] = len(det_by_node) / max(len(recs), 1)
    met["bootstrap"] = res.stats.get("bootstrap")
    if mode in ("anchor", "reloc") and recs:
        first = next((r for r in recs), None)
        met["relocalization"] = {"time_to_first_fix_s": float(first.t - ds.frame_t[0]) if first else None,
                                 "marker_id": res.stats.get("bootstrap", {}) and res.stats["bootstrap"].get("marker_id")}

    # metrics need an alignment to the room frame unless the run is already expressed in it
    in_room_frame = mode == "anchor" or (mode == "reloc" and str(map_meta.get("frame", "")).startswith("room"))
    met["frame"] = "room" if in_room_frame else "local (aligned to reference for metrics)"
    surveyed_path = find_file("markers_surveyed.json", ds.root)
    surveyed = load_map(surveyed_path)[0] if surveyed_path else None
    if surveyed:
        align = not in_room_frame
        met["baseline"]["marker_map"] = M.marker_map_errors(base_map, surveyed, align=align)
        met["optimized"]["marker_map"] = M.marker_map_errors(markers, surveyed, align=align)
        sc_ = met["optimized"]["marker_map"].get("scale_vs_survey")
        if align and sc_ and abs(sc_ - 1) > 0.03:   # metric scale comes only from the marker size in SLAM mode
            pct = 100 * (1 / sc_ - 1)
            warnings.append(f"the estimated marker map is {abs(pct):.1f} % {'larger' if pct > 0 else 'smaller'} than "
                            f"the tape survey: check the marker size ({1e3 * size:.0f} mm used) and the survey")
    gt_path = ds.root / "groundtruth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        gt_t = (gt["t_ns"].values - ds.t0_ns) * 1e-9
        idx = [int(np.argmin(np.abs(gt_t - t))) for t in times]
        Pg = gt[["x", "y", "z"]].values[idx]
        Rg = Rot.from_quat(gt[["qx", "qy", "qz", "qw"]].values[idx]).as_matrix()
        for name, P in (("baseline", T0), ("optimized", poses)):
            a = M.ate(P[:, :3, 3], Pg, align=not in_room_frame)
            Ra = np.array(a["alignment"]["R"]) if a.get("alignment") else np.eye(3)
            met[name]["ate"] = a
            met[name]["orientation"] = M.orientation_error(P[:, :3, :3], Rg, Ra)
            e = np.linalg.norm((P[:, :3, 3] @ Ra.T + (np.array(a["alignment"]["t"]) if a.get("alignment") else 0)) - Pg, axis=1)
            pd.DataFrame({"t": times, f"err_{name}": e}).to_csv(out / f"error_{name}.csv", index=False)
    cp_path = ds.root / "checkpoints.csv" if (ds.root / "checkpoints.csv").exists() else None   # per recording
    if cp_path is not None:
        cps = pd.read_csv(cp_path)
        for name, P, mp in (("baseline", T0, base_map), ("optimized", poses, markers)):
            T_al = None
            if not in_room_frame and surveyed:
                al = met[name].get("marker_map", {}).get("alignment")
                if al:
                    T_al = make_T(np.array(al["R"]), np.array(al["t"]))
            if in_room_frame or T_al is not None:
                met[name]["checkpoints"] = M.checkpoint_errors(times, P[:, :3, 3], static, cps, T_al)
                if name == "optimized" and met[name]["checkpoints"]["count_mismatch"]:
                    warnings.append(f"checkpoints: {met[name]['checkpoints']['n_static']} still periods detected but "
                                    f"{met[name]['checkpoints']['n_checkpoints']} checkpoints listed; they are matched "
                                    "in order, so an extra or missing pause shifts every match")

    # camera-IMU sync check on this dataset: VO orientations vs gyro
    vo = [i for i, st in enumerate(status) if st == "vo"]
    if len(vo) > 60:
        try:
            chk = estimate_cam_imu(times[vo], T0[vo][:, :3, :3], imu, td_range=0.3, min_dt=0.1, max_dt=0.3)
            chk.pop("td_curve", None)
            chk["R_CI_diff_deg"] = float(np.degrees(Rot.from_matrix(np.array(chk["R_CI"]).T @ R_CI).magnitude()))
            chk["td_used_s"] = td
            met["sync_check"] = chk
            if chk["corr_peak"] > 0.8 and abs(chk["td_s"] - td) > 0.02:
                warnings.append(f"sync check: the final trajectory implies a time offset of {1e3 * chk['td_s']:.0f} ms, "
                                f"more than 20 ms from the {1e3 * td:.0f} ms used")
        except RuntimeError as e:
            met["sync_check"] = {"error": str(e)}
    met["timing"] = {"total_s": time.perf_counter() - t_start, "frames": len(res.records),
                     "processing_fps": len(res.records) / max(time.perf_counter() - t_start, 1e-6),
                     "detect_ms_per_frame": detect_ms, "vo_ms_per_frame": res.stats["t_vo_ms_per_frame"],
                     "backend_s": t_be}
    if met["optimized"]["closed_loop"]["source"] != "static periods":
        warnings.append("start-to-end distance uses the first and last pose: no still period found at both ends")
    met["marker_size_m"] = size
    met["sync"] = sync_info
    met["calibration_used"] = {"camera": str(cam_path), "camera_imu": str(ci_path) if ci_path else None,
                               "td_s": td, "gyro_bias": imu.bg.tolist(), "static_intervals": static}
    (out / "metrics.json").write_text(json.dumps(met, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if progress:
        progress(1.0, "done")
    log(f"saved results to {out}")
    return out, met
