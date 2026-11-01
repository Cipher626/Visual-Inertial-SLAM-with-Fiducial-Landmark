"""Back-end: batch factor-graph optimisation over camera poses X_k, marker poses L_i and one
log-scale s_j per visual-odometry segment (monocular VO cannot observe metric scale; markers can):

    min  sum_k ||r_odo(X_k, X_k+1, s_seg(k))||^2 + sum_k ||r_gyro(X_k, X_k+1)||^2
       + sum_(k,i) rho(||r_corner(X_k, L_i, z_ki)||^2) + priors

r_corner reprojects the 4 marker corners (known marker size) into the camera and compares them with the
detected corners, which avoids the planar-pose ambiguity of using C_T_M directly.
Solved with Levenberg-Marquardt (sparse normal equations, Huber weights via IRLS, grouped finite-difference
Jacobian). Poses are parameterised as small corrections around their initial value: R = R0 Exp(d_rot), t = t0 + d_t."""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as Rot

from .geometry import make_T
from .markers import marker_object_points


@dataclass
class Graph:
    T0: np.ndarray                                 # (N,4,4) initial T_WC
    odo: list = field(default_factory=list)        # (i, j, T_ij, sig_rot_rad, sig_t_m, segment or -1)
    gyro: list = field(default_factory=list)       # (i, j, R_ij, sig_rot_rad)
    obs: list = field(default_factory=list)        # (i, marker_id, corners_norm (4,2))
    marker_init: dict = field(default_factory=dict)    # id -> T_WM
    marker_prior: dict = field(default_factory=dict)   # id -> (T_WM, sig_rot_rad, sig_t_m)
    pose_prior: list = field(default_factory=list)     # (i, T_WC, sig_rot_rad, sig_t_m)
    scale_prior: dict = field(default_factory=dict)    # segment -> sigma of log-scale (prior mean 0 = scale 1)


def _rv(R):
    return Rot.from_matrix(R).as_rotvec()


def optimize(g: Graph, f_px, marker_size, sigma_px=1.0, huber=2.0, max_iter=40, verbose=0):
    N = len(g.T0)
    # a marker with no observation and no prior has nothing constraining it: leaving it in would make the
    # normal equations singular and the solve silently produce garbage
    seen = {o[1] for o in g.obs}
    mids = sorted(m for m in g.marker_init if m in seen or m in g.marker_prior)
    mcol = {m: N + j for j, m in enumerate(mids)}
    M = len(mids)
    segs = sorted(set(o[5] for o in g.odo if len(o) > 5 and o[5] >= 0))
    scol = {s: k for k, s in enumerate(segs)}
    S = len(segs)
    R0 = np.concatenate([g.T0[:, :3, :3], np.array([g.marker_init[m][:3, :3] for m in mids]).reshape(-1, 3, 3)])
    t0 = np.concatenate([g.T0[:, :3, 3], np.array([g.marker_init[m][:3, 3] for m in mids]).reshape(-1, 3)])
    nv = N + M                       # 6-DoF blocks; scale scalars follow at x[6*nv:]

    oi = np.array([o[0] for o in g.odo], int)
    oj = np.array([o[1] for o in g.odo], int)
    oR = np.array([o[2][:3, :3] for o in g.odo]).reshape(-1, 3, 3)
    ot = np.array([o[2][:3, 3] for o in g.odo]).reshape(-1, 3)
    osr = np.array([o[3] for o in g.odo])
    ost = np.array([o[4] for o in g.odo])
    oseg = np.array([scol.get(o[5], -1) if len(o) > 5 else -1 for o in g.odo], int)
    gi = np.array([x[0] for x in g.gyro], int)
    gj = np.array([x[1] for x in g.gyro], int)
    gR = np.array([x[2] for x in g.gyro]).reshape(-1, 3, 3)
    gs = np.array([x[3] for x in g.gyro])
    obs = [o for o in g.obs if o[1] in mcol]
    bi = np.array([o[0] for o in obs], int)
    bm = np.array([mcol[o[1]] for o in obs], int)
    bz = np.array([o[2] for o in obs]).reshape(-1, 4, 2)
    corners = marker_object_points(marker_size)
    mp = [(mcol[m], v) for m, v in g.marker_prior.items() if m in mcol]
    pp = list(g.pose_prior)
    sp = [(scol[s], sig) for s, sig in g.scale_prior.items() if s in scol]

    def unpack(x):
        d = x[:6 * nv].reshape(nv, 6)
        return R0 @ Rot.from_rotvec(d[:, :3]).as_matrix(), t0 + d[:, 3:], x[6 * nv:]

    def residuals(x):
        R, t, ls = unpack(x)
        out = []
        if len(oi):
            Rrel = np.transpose(R[oi], (0, 2, 1)) @ R[oj]
            trel = np.einsum("nji,nj->ni", R[oi], t[oj] - t[oi])
            scale = np.where(oseg >= 0, np.exp(ls[np.maximum(oseg, 0)] if S else 0.0), 1.0)
            Re = np.transpose(oR, (0, 2, 1)) @ Rrel
            te = np.einsum("nji,nj->ni", oR, trel - scale[:, None] * ot)
            out += [(Rot.from_matrix(Re).as_rotvec() / osr[:, None]).ravel(), (te / ost[:, None]).ravel()]
        if len(gi):
            Re = np.transpose(gR, (0, 2, 1)) @ (np.transpose(R[gi], (0, 2, 1)) @ R[gj])
            out.append((Rot.from_matrix(Re).as_rotvec() / gs[:, None]).ravel())
        if len(bi):
            Pw = np.einsum("nij,kj->nki", R[bm], corners) + t[bm][:, None, :]
            Pc = np.einsum("nji,nkj->nki", R[bi], Pw - t[bi][:, None, :])
            z = np.maximum(Pc[..., 2:3], 0.05)
            out.append(((Pc[..., :2] / z - bz) * (f_px / sigma_px)).ravel())
        for col, (T, sr, st) in mp:
            out += [_rv(T[:3, :3].T @ R[col]) / sr, (t[col] - T[:3, 3]) / st]
        for i, T, sr, st in pp:
            out += [_rv(T[:3, :3].T @ R[i]) / sr, (t[i] - T[:3, 3]) / st]
        for k, sig in sp:
            out.append(np.array([ls[k] / sig]))
        return np.concatenate(out) if out else np.zeros(0)

    # ---- which variables each residual row depends on (same order as residuals()).
    # Variable ids: 0..nv-1 are 6-DoF blocks, nv+k is the log-scale of segment k.
    rows = []
    for a, b in zip(oi, oj):
        rows += [(a, b)] * 3
    for a, b, s in zip(oi, oj, oseg):
        rows += [(a, b) if s < 0 else (a, b, nv + s)] * 3
    for a, b in zip(gi, gj):
        rows += [(a, b)] * 3
    for a, b in zip(bi, bm):
        rows += [(a, b)] * 8
    for col, _ in mp:
        rows += [(col,)] * 6
    for i, *_ in pp:
        rows += [(i,)] * 6
    for k, _ in sp:
        rows += [(nv + k,)]
    nres = len(rows)
    nvar = nv + S
    var_rows = [[] for _ in range(nvar)]
    for r, vs in enumerate(rows):
        for v in set(vs):
            var_rows[v].append(r)
    var_rows = [np.array(v, int) for v in var_rows]
    # colour classes: variables in one class never share a residual -> perturb them together.
    # Consecutive nodes share odometry rows (3 classes); markers never share rows with each other; segment
    # scales never share rows with each other.
    colour = np.r_[np.arange(N) % 3, np.full(M, 3), np.full(S, 4)]
    width = np.r_[np.full(nv, 6), np.ones(S, int)]
    first = np.r_[6 * np.arange(nv), 6 * nv + np.arange(S)].astype(int)
    h = 1e-6

    def jacobian(x, r0):
        Ri, Ci, V = [], [], []
        n_eval = 0
        for c in range(5):
            vs = np.where(colour == c)[0]
            if len(vs) == 0:
                continue
            for d in range(int(width[vs].max())):
                vs_d = vs[width[vs] > d]
                xp = x.copy()
                xp[first[vs_d] + d] += h
                diff = (residuals(xp) - r0) / h
                n_eval += 1
                for v in vs_d:
                    rr = var_rows[v]
                    if len(rr):
                        Ri.append(rr)
                        Ci.append(np.full(len(rr), first[v] + d))
                        V.append(diff[rr])
        J = coo_matrix((np.concatenate(V), (np.concatenate(Ri), np.concatenate(Ci))), shape=(nres, 6 * nv + S)).tocsr()
        return J, n_eval

    def robust(r):
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        return w, np.where(a <= huber, 0.5 * r * r, huber * a - 0.5 * huber * huber).sum()

    x = np.zeros(6 * nv + S)
    r = residuals(x)
    w, cost = robust(r)
    cost0, lam, n_eval, it, rel, step = cost, 1e-4, 1, 0, 1.0, np.ones(1)
    rel_hist = []
    improved = True
    for it in range(max_iter):
        J, ne = jacobian(x, r)
        n_eval += ne
        H = (J.T @ J.multiply(w[:, None]).tocsr()).tocsc()
        gvec = J.T @ (w * r)
        dH = H.diagonal() + 1e-9
        improved = False
        for _ in range(10):
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # a singular / ill-conditioned solve counts as a failed step
                try:
                    step = spsolve(H + diags(lam * dH), -gvec)
                except Exception:
                    step = None
            if step is None or not np.all(np.isfinite(step)):
                lam *= 5
                continue
            xn = x + step
            rn = residuals(xn)
            n_eval += 1
            wn, cn = robust(rn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                x, r, w, cost = xn, rn, wn, cn
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 5
        if verbose:
            print(f"it {it}: cost {cost:.3f} lambda {lam:.1e}")
        if not improved or rel < 1e-6 or np.max(np.abs(step)) < 1e-9:
            break
        rel_hist.append(rel)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("back-end produced non-finite values; check the marker size and calibration")
    R, t, ls = unpack(x)
    poses = np.array([make_T(R[k], t[k]) for k in range(N)])
    markers = {m: make_T(R[mcol[m]], t[mcol[m]]) for m in mids}
    scales = {int(s): float(np.exp(ls[scol[s]])) for s in segs}
    info = {"cost_initial": float(cost0), "cost_final": float(cost), "iterations": it + 1, "residual_evals": int(n_eval),
            # "converged" = no meaningful progress left: the last 3 iterations together lowered the cost < 0.1 %
            "converged": bool(not improved or rel < 1e-6 or sum(rel_hist[-3:]) < 1e-3),
            "last_relative_decrease": float(rel), "n_poses": N, "n_markers": M, "n_marker_obs": len(obs),
            "n_residuals": nres, "segment_scales": scales,
            "markers_without_observations": sorted(set(g.marker_init) - set(mids))}
    return poses, markers, info


def corner_reprojection_errors(poses, markers, obs, f_px, marker_size):
    """Pixel RMS reprojection error per observation (i, id, corners_norm)."""
    corners = marker_object_points(marker_size)
    errs = []
    for i, m, z in obs:
        if m not in markers:
            errs.append(np.nan)
            continue
        T_CW = np.linalg.inv(poses[i])
        Pc = (T_CW @ markers[m] @ np.c_[corners, np.ones(4)].T)[:3].T
        if np.any(Pc[:, 2] <= 0):
            errs.append(np.inf)
            continue
        e = (Pc[:, :2] / Pc[:, 2:3] - z) * f_px
        errs.append(float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))))
    return np.array(errs)
