"""Front-end: monocular KLT/PnP visual odometry, aided by gyro attitude, with inertial propagation
when vision fails. Metric scale and the initial pose come from one ArUco marker during bootstrap.
Every processed frame also yields ArUco observations for the back-end."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import inv_T, make_T, T_from_rvec_tvec, rvec_tvec_from_T, gravity_aligned_R_WC, rot_angle_deg
from .imu import propagate_accel


@dataclass
class FrameRecord:
    k: int                      # processed-frame counter
    frame_idx: int              # index in the video
    t: float                    # camera time [s]
    T_WC: np.ndarray = None     # None while waiting for bootstrap
    status: str = "wait"        # wait | init | vo | imu
    n_tracks: int = 0
    n_map: int = 0
    n_inliers: int = 0
    is_kf: bool = False
    reinit: bool = False        # visual tracking restarted here (new unknown monocular scale)


@dataclass
class FrontEndResult:
    records: list
    observations: list          # (k, MarkerObs)
    stats: dict = field(default_factory=dict)
    frame_obs: list = field(default_factory=list)   # all detections per processed frame (reused by a 2nd pass)


class FrontEnd:
    def __init__(self, cfg, K, dist, imu, R_CI, td, detector, mode="slam", known_map=None, gravity_ref=None):
        self.c = cfg
        self.K, self.dist = np.asarray(K, float), np.asarray(dist, float).ravel()
        self.f = float(self.K[0, 0])
        self.imu, self.R_CI, self.td = imu, np.asarray(R_CI, float), float(td)
        self.det = detector
        self.mode = mode
        self.known_map = known_map or {}       # id -> T_WM (anchor / relocalisation modes)
        self.accel_bias = np.zeros(3)
        self.gravity_ref = gravity_ref   # (IMU time, specific force) from the initial still period

    # ------------------------------------------------------------------ helpers
    def _undist(self, pts):
        if len(pts) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist).reshape(-1, 2)

    def _gyro_R_cam(self, t_a, t_b):
        """Camera rotation from t_a to t_b predicted by the gyro: R_Ca_Cb."""
        dR = self.imu.delta_R(np.array([t_a + self.td]), np.array([t_b + self.td]))[0]
        return self.R_CI @ dR @ self.R_CI.T

    def _new_features(self, gray, existing):
        n_need = self.c["max_features"] - len(existing)
        if n_need < 10:
            return np.zeros((0, 2), np.float32)
        mask = np.full(gray.shape, 255, np.uint8)
        for p in existing:
            cv2.circle(mask, (int(p[0]), int(p[1])), self.c["min_feature_dist"], 0, -1)
        pts = cv2.goodFeaturesToTrack(gray, n_need, 0.01, self.c["min_feature_dist"], mask=mask, blockSize=7)
        return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32)

    def _add_tracks(self, pts, kf_id, T_WC):
        n = len(pts)
        if n == 0:
            return
        self.pts = np.vstack([self.pts, pts])
        self.X = np.vstack([self.X, np.full((n, 3), np.nan)])
        self.anc_norm = np.vstack([self.anc_norm, self._undist(pts)])
        self.anc_px = np.vstack([self.anc_px, pts])
        self.anc_kf = np.r_[self.anc_kf, np.full(n, kf_id)]
        self.kf_px = np.vstack([self.kf_px, pts])

    def _reset_tracks(self):
        self.pts = np.zeros((0, 2), np.float32)
        self.X = np.zeros((0, 3))
        self.anc_norm = np.zeros((0, 2))
        self.anc_px = np.zeros((0, 2), np.float32)
        self.anc_kf = np.zeros(0, int)
        self.kf_px = np.zeros((0, 2), np.float32)

    def _keep(self, m):
        for name in ("pts", "X", "anc_norm", "anc_px", "anc_kf", "kf_px"):
            setattr(self, name, getattr(self, name)[m])

    def _track(self, prev, cur):
        if len(self.pts) == 0:
            return
        p0 = self.pts.reshape(-1, 1, 2)
        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        h, w = cur.shape
        p1 = p1.reshape(-1, 2)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.c["klt_fb_px"]) & \
             (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        self.pts = p1.astype(np.float32)
        self._keep(ok)

    def _triangulate(self, T_WC):
        """Triangulate tracks that have no 3D point, between their anchor keyframe and the current pose."""
        cand = np.where(np.isnan(self.X[:, 0]))[0]
        if len(cand) == 0:
            return 0
        cur_norm = self._undist(self.pts[cand])
        T_CW2 = inv_T(T_WC)
        P2 = T_CW2[:3]
        added = 0
        for kf in np.unique(self.anc_kf[cand]):
            sel = cand[self.anc_kf[cand] == kf]
            n2 = cur_norm[self.anc_kf[cand] == kf]
            T_WC1 = self.kf_poses[kf]
            T_CW1 = inv_T(T_WC1)
            if np.linalg.norm(T_WC1[:3, 3] - T_WC[:3, 3]) < self.c["min_tri_baseline_m"]:
                continue
            n1 = self.anc_norm[sel]
            Xh = cv2.triangulatePoints(T_CW1[:3], P2, n1.T, n2.T)
            X = (Xh[:3] / Xh[3]).T
            c1 = X @ T_CW1[:3, :3].T + T_CW1[:3, 3]
            c2 = X @ T_CW2[:3, :3].T + T_CW2[:3, 3]
            e1 = np.linalg.norm(c1[:, :2] / c1[:, 2:3] - n1, axis=1) * self.f
            e2 = np.linalg.norm(c2[:, :2] / c2[:, 2:3] - n2, axis=1) * self.f
            r1 = X - T_WC1[:3, 3]
            r2 = X - T_WC[:3, 3]
            cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            good = (c1[:, 2] > 0.1) & (c2[:, 2] > 0.1) & (c1[:, 2] < self.c["max_depth_m"]) & \
                   (c2[:, 2] < self.c["max_depth_m"]) & (e1 < self.c["tri_reproj_px"]) & \
                   (e2 < self.c["tri_reproj_px"]) & (ang > self.c["min_tri_angle_deg"])
            self.X[sel[good]] = X[good]
            added += int(good.sum())
        return added

    def _make_keyframe(self, gray, T_WC, rec, triangulate=True):
        kf_id = len(self.kf_poses)
        self.kf_poses.append(T_WC.copy())
        self.kf_times.append(rec.t)
        self.last_kf_k = rec.k
        rec.is_kf = True
        if triangulate:   # never triangulate from inertially propagated poses (scale/position too uncertain)
            self._triangulate(T_WC)
        # tracks without 3D whose anchor is very old and still not triangulated are re-anchored here
        stale = np.isnan(self.X[:, 0]) & (self.anc_kf < kf_id - self.c["reanchor_after_kfs"])
        if stale.any():
            self.anc_kf[stale] = kf_id
            self.anc_norm[stale] = self._undist(self.pts[stale])
            self.anc_px[stale] = self.pts[stale]
        self.kf_px = self.pts.copy()
        self._add_tracks(self._new_features(gray, self.pts), kf_id, T_WC)

    def _reinit(self, t):
        """Marker-free recovery after visual tracking is lost: two-view relative pose (essential matrix)
        between the keyframe most current tracks started in and the current frame, checked against the gyro.
        Monocular vision cannot see scale, so the translation length is set from the walking speed measured
        by VO before the loss; the back-end later corrects it wherever markers are seen."""
        c = self.c
        no3d = np.isnan(self.X[:, 0])
        if no3d.sum() < c["boot_min_points"]:
            return None
        anchors, counts = np.unique(self.anc_kf[no3d], return_counts=True)
        kf = int(anchors[np.argmax(counts)])
        sel = np.where(no3d & (self.anc_kf == kf))[0]
        if len(sel) < c["boot_min_points"] or self.kf_times[kf] >= t:
            return None
        if np.median(np.linalg.norm(self.pts[sel] - self.anc_px[sel], axis=1)) < c["boot_parallax_px"]:
            return None
        n1, n2 = self.anc_norm[sel], self._undist(self.pts[sel])
        E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / self.f)
        if E is None or E.shape != (3, 3):
            return None
        n_in, R, tvec, _ = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)   # x_cur = R x_kf + t
        if n_in < c["boot_min_points"]:
            return None
        R_gyro = self._gyro_R_cam(self.kf_times[kf], t)                        # R_Ckf_Ccur
        if rot_angle_deg(R_gyro.T @ R.T) > c["gate_rot_deg"]:
            return None
        speed = float(np.clip(self.ref_speed, c["reinit_speed_min"], c["reinit_speed_max"]))
        T_cur_kf = make_T(R, tvec.ravel() * speed * (t - self.kf_times[kf]))
        return self.kf_poses[kf] @ inv_T(T_cur_kf)

    def _initial_pose(self, ob, t):
        """Pose of the bootstrap frame. slam: gravity-aligned, origin at the camera.
        anchor/reloc: W_T_C = W_T_M (C_T_M)^-1 from the known map (eq. in the handout)."""
        if self.mode == "slam":
            if self.gravity_ref is not None:
                # 'up' measured while still, rotated to the bootstrap time with the gyro: unaffected by the
                # acceleration of a phone that is already moving when the bootstrap happens
                t_s, f_s = self.gravity_ref
                dR = self.imu.delta_R(np.array([t_s]), np.array([t + self.td]))[0]      # R_Is_Ib
                f = dR.T @ f_s
            else:
                f = self.imu.mean_specific_force(t + self.td - 0.25, t + self.td + 0.25)   # 'up' in IMU frame
            up_c = self.R_CI @ (f / np.linalg.norm(f))
            return make_T(gravity_aligned_R_WC(up_c), np.zeros(3))
        return self.known_map[ob.marker_id] @ inv_T(ob.T_CM)

    # ------------------------------------------------------------------ main loop
    def run(self, frames, n_expected=None, progress=None, cached_obs=None):
        c = self.c
        records, observations, frame_obs = [], [], []
        self._reset_tracks()
        self.kf_poses, self.kf_times = [], []
        self.ref_speed, n_reinit = 0.5, 0
        bd = {"frames": 0, "frames_with_marker": 0, "max_side_px": {}, "attempts": []}
        state, prev_gray, prev_T, prev_t = "wait", None, None, None
        vel = np.zeros(3)
        boot = None
        t_detect = t_vo = 0.0
        allowed = set(self.known_map) if self.mode != "slam" else None
        for k, (idx, t, gray) in enumerate(frames):
            rec = FrameRecord(k, idx, t)
            t0 = time.perf_counter()
            obs = cached_obs[k] if cached_obs is not None and k < len(cached_obs) else self.det.detect(gray)
            frame_obs.append(obs)
            t_detect += time.perf_counter() - t0
            t0 = time.perf_counter()
            if state == "boot":
                self._track(prev_gray, gray)
                ref = [o for o in obs if o.marker_id == boot["id"]]
                att = bd["attempts"][-1]
                restart = None
                if not ref:
                    restart = "marker left the view"
                elif len(self.pts) < c["boot_min_points"]:   # can no longer reach the point count
                    restart = "features lost (fast rotation or blur)"
                elif rot_angle_deg(boot["T_C0M"][:3, :3] @ ref[0].T_CM[:3, :3].T) > c.get("boot_max_rot_deg", 25.0):
                    restart = "rotated too far from the reference view"
                if restart:          # discard this attempt; a new one can start on this very frame
                    att["end"] = restart
                    state = "wait"
                    for r in records[boot["k0"]:]:
                        r.T_WC, r.status = None, "wait"
                    observations = [(kk, o) for kk, o in observations if kk < boot["k0"]]
                else:
                    T_C0C = boot["T_C0M"] @ inv_T(ref[0].T_CM)
                    T_WC = boot["T_WC0"] @ T_C0C
                    rec.T_WC, rec.status = T_WC, "init"
                    par = np.median(np.linalg.norm(self.pts - self.anc_px, axis=1)) if len(self.pts) else 0
                    base = float(np.linalg.norm(T_C0C[:3, 3]))
                    att["max_baseline_m"] = max(att["max_baseline_m"], base)
                    att["max_parallax_px"] = max(att["max_parallax_px"], float(par))
                    if base > c["boot_baseline_m"] and par > c["boot_parallax_px"]:
                        self._triangulate(T_WC)
                        n = int((~np.isnan(self.X[:, 0])).sum())      # all points so far, not just this frame's
                        att["max_points"] = max(att["max_points"], n)
                        if n >= c["boot_min_points"]:
                            att["end"] = "success"
                            state = "track"
                            rec.status = "vo"
                            vel = (T_WC[:3, 3] - prev_T[:3, 3]) / max(t - prev_t, 1e-3)
                            self.ref_speed = max(float(np.linalg.norm(vel)), 0.3)
                            self._make_keyframe(gray, T_WC, rec)
                            self.boot_info = {"marker_id": boot["id"], "t": t, "n_points": n}
            if state == "wait":
                bd["frames"] += 1
                if obs:
                    bd["frames_with_marker"] += 1
                for o in obs:
                    bd["max_side_px"][o.marker_id] = max(bd["max_side_px"].get(o.marker_id, 0.0), float(o.side_px))
                cands = [o for o in obs if o.side_px >= c["boot_min_side_px"] and (allowed is None or o.marker_id in allowed)]
                if cands:
                    ob = max(cands, key=lambda o: o.side_px)
                    T_WC = self._initial_pose(ob, t)
                    self._reset_tracks()
                    self.kf_poses, self.kf_times = [T_WC.copy()], [t]
                    self._add_tracks(self._new_features(gray, self.pts), 0, T_WC)
                    boot = {"id": ob.marker_id, "T_C0M": ob.T_CM, "T_WC0": T_WC, "k0": k, "t0": t}
                    bd["attempts"].append({"t": float(t), "marker_id": ob.marker_id, "n_features": int(len(self.pts)),
                                           "max_baseline_m": 0.0, "max_parallax_px": 0.0, "max_points": 0, "end": "running"})
                    self.last_kf_k = k
                    rec.T_WC, rec.status, rec.is_kf = T_WC, "init", True
                    state = "boot"
            elif state == "track" and rec.T_WC is None:   # normal tracking (skipped on the bootstrap-success frame)
                self._track(prev_gray, gray)
                dt = max(t - prev_t, 1e-3)
                R_pred = prev_T[:3, :3] @ self._gyro_R_cam(prev_t, t)
                p_pred = prev_T[:3, 3] + vel * dt
                T_pred = make_T(R_pred, p_pred)
                has3d = ~np.isnan(self.X[:, 0])
                ok = False
                if has3d.sum() >= c["pnp_min_points"]:
                    obj = self.X[has3d]
                    img = self._undist(self.pts[has3d])
                    rvec, tvec = rvec_tvec_from_T(inv_T(T_pred))
                    succ, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img, np.eye(3), None, rvec, tvec, useExtrinsicGuess=True,
                        iterationsCount=100, reprojectionError=c["pnp_reproj_px"] / self.f, confidence=0.99,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    if succ and inl is not None and len(inl) >= c["pnp_min_inliers"]:
                        inl = inl.ravel()
                        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], np.eye(3), None, rvec, tvec)
                        T_WC = inv_T(T_from_rvec_tvec(rvec, tvec))
                        d_rot = rot_angle_deg(R_pred.T @ T_WC[:3, :3])
                        d_pos = np.linalg.norm(T_WC[:3, 3] - p_pred)
                        if d_rot < c["gate_rot_deg"] and d_pos < c["gate_pos_m"] and len(inl) / len(obj) > c["pnp_min_inlier_ratio"]:
                            ok = True
                            rec.n_inliers = len(inl)
                            outl = np.ones(len(obj), bool)
                            outl[inl] = False
                            idx3d = np.where(has3d)[0]
                            self._keep(~np.isin(np.arange(len(self.pts)), idx3d[outl]))
                T_re = None
                if not ok and has3d.sum() < c["pnp_min_points"]:
                    T_re = self._reinit(t)
                    if T_re is not None and self._triangulate(T_re) >= c["boot_min_points"]:
                        T_WC = T_re
                        kf_t = self.kf_times[int(np.bincount(self.anc_kf[~np.isnan(self.X[:, 0])]).argmax())]
                        vel = (T_WC[:3, 3] - prev_T[:3, 3]) / dt if t - kf_t <= 0 else vel
                        n_reinit += 1
                    else:
                        T_re = None
                if ok:
                    a = c["vel_smoothing"]
                    vel = a * vel + (1 - a) * (T_WC[:3, 3] - prev_T[:3, 3]) / dt
                    self.ref_speed = 0.95 * self.ref_speed + 0.05 * float(np.linalg.norm(vel))
                    rec.status = "vo"
                elif T_re is not None:
                    rec.status = "imu"          # the edge into this frame came through the lost period
                    rec.reinit = True
                    T_WC = T_re
                else:
                    if c["lost_motion_model"] == "accel":
                        p, vel = propagate_accel(self.imu, prev_T[:3, 3], vel, prev_T[:3, :3], self.R_CI,
                                                 prev_t, t, self.accel_bias, self.td)
                    else:
                        vel = vel * np.exp(-dt / c["lost_vel_decay_s"])
                        p = p_pred
                    T_WC = make_T(R_pred, p)
                    rec.status = "imu"
                rec.T_WC = T_WC
                par = np.median(np.linalg.norm(self.pts - self.kf_px, axis=1)) if len(self.pts) else 0
                n3d = int((~np.isnan(self.X[:, 0])).sum())
                since = k - self.last_kf_k
                if since >= c["kf_min_gap"] and (par > c["kf_parallax_px"] or n3d < c["kf_min_map_points"]
                                                 or rec.status == "imu" or since >= c["kf_max_gap"]):
                    self._make_keyframe(gray, T_WC, rec, triangulate=(rec.status == "vo"))
            t_vo += time.perf_counter() - t0
            rec.n_tracks = len(self.pts)
            rec.n_map = int((~np.isnan(self.X[:, 0])).sum()) if len(self.X) else 0
            records.append(rec)
            if rec.T_WC is not None:
                observations.extend((k, o) for o in obs)
            prev_gray = gray
            if rec.T_WC is not None:
                prev_T, prev_t = rec.T_WC, t
            if progress and n_expected:
                progress(min(1.0, (k + 1) / n_expected))
        n = max(len(records), 1)
        stats = {"n_frames": len(records), "n_keyframes": len(self.kf_poses),
                 "frac_vo": sum(r.status == "vo" for r in records) / n,
                 "frac_imu": sum(r.status == "imu" for r in records) / n,
                 "frac_wait": sum(r.status in ("wait", "init") for r in records) / n,
                 "t_detect_ms_per_frame": 1e3 * t_detect / n, "t_vo_ms_per_frame": 1e3 * t_vo / n,
                 "n_reinit": n_reinit, "bootstrap": getattr(self, "boot_info", None),
                 "bootstrap_diagnostics": {**bd, "attempts": bd["attempts"][-30:], "n_attempts": len(bd["attempts"])}}
        return FrontEndResult(records, observations, stats, frame_obs)
