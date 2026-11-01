"""Synthetic dataset with perfect ground truth: a textured box room with ArUco markers on the walls,
rendered by ray casting, plus a phone-like IMU (bias, noise, camera-IMU rotation, time offset).
Use it to validate the whole pipeline before trusting results on real recordings.

    python tools/simulate.py --out data/sim_loop
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vislam.imu import R_CI_DEFAULT, GRAVITY  # noqa: E402

ROOM = np.array([6.0, 5.0, 2.8])
MARKERS = {  # id: (center, normal into room)
    0: ((6.0, 2.0, 1.45), (-1, 0, 0)), 1: ((6.0, 3.4, 1.55), (-1, 0, 0)),
    2: ((4.6, 5.0, 1.40), (0, -1, 0)), 3: ((2.0, 5.0, 1.50), (0, -1, 0)),
    4: ((0.0, 3.2, 1.50), (1, 0, 0)), 5: ((0.0, 1.6, 1.40), (1, 0, 0)),
    6: ((5.0, 0.0, 1.50), (0, 1, 0)),
}


def make_texture(h, w, rng):
    tex = np.zeros((h, w), np.float32)
    for cell, amp in ((6, 0.25), (24, 0.35), (96, 0.4)):
        small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        tex += amp * cv2.resize(small, (w + 2 * cell, h + 2 * cell), interpolation=cv2.INTER_CUBIC)[cell:cell + h, cell:cell + w]
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    tex = (40 + 170 * tex).astype(np.uint8)
    for _ in range(int(h * w / 4000)):  # sharp blobs give trackable corners
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(4, 18))
        cv2.rectangle(tex, (x, y), (x + s, y + int(rng.integers(4, 18))), int(rng.integers(0, 255)), -1)
    return tex


class Room:
    """Faces: 0:x=0, 1:x=Lx, 2:y=0, 3:y=Ly, 4:z=0 floor, 5:z=Lz ceiling. Texture (u,v) per face."""

    def __init__(self, res=400, marker_size=0.16, dictionary="DICT_4X4_50", seed=1):
        self.res, rng = res, np.random.default_rng(seed)
        L = ROOM
        dims = [(L[1], L[2]), (L[1], L[2]), (L[0], L[2]), (L[0], L[2]), (L[0], L[1]), (L[0], L[1])]
        self.dims = dims
        texs = [make_texture(int(b * res), int(a * res), rng) for a, b in dims]
        width = max(t.shape[1] for t in texs)
        self.offsets, rows = [], []
        y = 0
        for t in texs:
            self.offsets.append(y)
            pad = np.zeros((t.shape[0], width), np.uint8)
            pad[:, :t.shape[1]] = t
            rows.append(pad)
            y += t.shape[0]
        self.atlas = np.vstack(rows)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        for mid, (c, n) in MARKERS.items():
            face = self._face_of(np.array(c, float), np.array(n, float))
            px = int(round(marker_size * res))
            img = cv2.aruco.generateImageMarker(d, mid, px)
            border = int(px * 0.25)
            img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
            u, v = self._uv(face, np.array(c, float)[None])
            u0, v0 = int(round(u[0] - img.shape[1] / 2)), int(round(v[0] - img.shape[0] / 2)) + self.offsets[face]
            self.atlas[v0:v0 + img.shape[0], u0:u0 + img.shape[1]] = img

    @staticmethod
    def _face_of(c, n):
        ax = int(np.argmax(np.abs(n)))
        return 2 * ax + (1 if c[ax] > 0.5 * ROOM[ax] else 0)

    def _uv(self, face, P):
        """Texture pixel coords for world points P on `face`. For walls, u grows to the right of a viewer
        standing inside the room and v grows downwards, so markers pasted upright read correctly."""
        L, r = ROOM, self.res
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if face == 0:
            u, v = y, L[2] - z
        elif face == 1:
            u, v = L[1] - y, L[2] - z
        elif face == 2:
            u, v = L[0] - x, L[2] - z
        elif face == 3:
            u, v = x, L[2] - z
        elif face == 4:
            u, v = x, L[1] - y
        else:
            u, v = x, y
        return u * r, v * r

    def render(self, R_WC, t_WC, K, size):
        w, h = size
        if not hasattr(self, "_rays") or self._rays_key != (w, h, tuple(K.ravel())):
            u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
            pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 0)
            self._rays = np.linalg.inv(K) @ pix
            self._rays_key = (w, h, tuple(K.ravel()))
        d = R_WC @ self._rays
        o = np.asarray(t_WC, float)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.where(d > 0, (ROOM[:, None] - o) / d, np.where(d < 0, (0 - o) / d, np.inf))
        ax = np.argmin(tb, axis=0)
        tm = tb[ax, np.arange(tb.shape[1])]
        P = (o + d * tm).T
        face = 2 * ax + (d[ax, np.arange(d.shape[1])] > 0)
        mapx = np.zeros(len(P), np.float32)
        mapy = np.zeros(len(P), np.float32)
        for f in range(6):
            m = face == f
            if m.any():
                uu, vv = self._uv(f, P[m])
                mapx[m] = uu
                mapy[m] = vv + self.offsets[f]
        img = cv2.remap(self.atlas, mapx.reshape(h, w), mapy.reshape(h, w), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        return img


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def trajectory(t, T_total, still=2.0, laps=1.0, phase=0.0, twist=0.0):
    """Camera pose: ellipse around the room centre, camera looking roughly outward."""
    start = still + twist                                           # wrist twists happen before walking
    s = smoothstep((t - start) / (T_total - still - start))
    env = smoothstep(t - start) * smoothstep(T_total - still - t)   # 0 while standing still / twisting
    tw = np.where((t >= still) & (t <= still + twist), 0.5 * np.sin(np.pi * (t - still) / max(twist, 1e-6)) ** 2, 0.0)
    twist_yaw = 0.5 * np.sin(2 * np.pi * 2.0 * (t - still)) * tw / 0.5 if twist > 0 else 0.0   # 3-4 twists of ~+-0.5 rad
    phi = phase + 2 * np.pi * laps * s
    cx, cy, rx, ry = 3.0, 2.5, 1.5, 1.0
    p = np.stack([cx + rx * np.cos(phi), cy + ry * np.sin(phi), 1.40 + 0.03 * env * np.sin(2 * np.pi * 1.8 * t)], -1)
    yaw = phi + 0.20 * env * np.sin(2 * np.pi * 0.15 * t) + twist_yaw
    pitch = 0.05 * env * np.sin(2 * np.pi * 0.4 * t)
    roll = 0.04 * env * np.sin(2 * np.pi * 0.7 * t + 1.0)
    R_list = []
    for y, pi_, r in zip(np.atleast_1d(yaw), np.atleast_1d(pitch), np.atleast_1d(roll)):
        f = np.array([np.cos(y), np.sin(y), 0.0])
        x = np.array([np.sin(y), -np.cos(y), 0.0])
        yv = np.cross(f, x)
        R0 = np.column_stack([x, yv, f])                       # R_WC for a level camera
        R_list.append(R0 @ Rot.from_euler("xz", [pi_, r]).as_matrix())
    return p, np.array(R_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim_loop")
    ap.add_argument("--duration", type=float, default=46.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--td", type=float, default=0.015, help="true camera-IMU time offset [s]")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--phase", type=float, default=0.0, help="start angle on the ellipse [rad] (use a different value for a repeat run)")
    ap.add_argument("--laps", type=float, default=1.0, help="1.0 = closed loop ending where it started")
    ap.add_argument("--twist", type=float, default=0.0, help="seconds of wrist twists after the initial still period")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    W, H = 640, 480
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    room = Room()
    R_CI = Rot.from_rotvec([0.02, -0.015, 0.01]).as_matrix() @ R_CI_DEFAULT   # slightly off the default

    # ----- IMU at 200 Hz (true time), from a 1 kHz reference trajectory -----
    tf = np.arange(0, args.duration + 0.5, 0.001)
    P, R_WC = trajectory(tf, args.duration, laps=args.laps, phase=args.phase, twist=args.twist)
    R_WI = R_WC @ R_CI
    acc_w = np.gradient(np.gradient(P, tf, axis=0), tf, axis=0)
    rel = Rot.from_matrix(np.transpose(R_WI[:-1], (0, 2, 1)) @ R_WI[1:]).as_rotvec() / 0.001
    w_i = np.vstack([rel, rel[-1:]])
    f_i = np.einsum("nji,nj->ni", R_WI, acc_w - np.array([0, 0, -GRAVITY]))
    sel = np.arange(0, len(tf), 5)
    bg, ba = rng.normal(0, 0.004, 3), rng.normal(0, 0.04, 3)
    gyro = w_i[sel] + bg + rng.normal(0, 0.003, (len(sel), 3))
    acc = f_i[sel] + ba + rng.normal(0, 0.03, (len(sel), 3))
    t_ns_imu = (tf[sel] * 1e9).astype(np.int64) + 10_000_000_000

    # ----- camera frames (stamped on a clock that lags IMU time by td) -----
    tc = np.arange(0, args.duration, 1.0 / args.fps)
    Pc, Rc = trajectory(tc, args.duration, laps=args.laps, phase=args.phase, twist=args.twist)
    vw = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H), False)
    for k, (p, R) in enumerate(zip(Pc, Rc)):
        img = room.render(R, p, K, (W, H)).astype(np.float32)
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        if k % 100 == 0:
            print(f"rendered {k}/{len(tc)}")
    vw.release()
    t_ns_cam = ((tc - args.td) * 1e9).astype(np.int64) + 10_000_000_000

    import pandas as pd
    pd.DataFrame({"t_ns": t_ns_cam}).to_csv(out / "frames.csv", index=False)
    pd.DataFrame({"t_ns": t_ns_imu, "x": gyro[:, 0], "y": gyro[:, 1], "z": gyro[:, 2]}).to_csv(out / "gyro.csv", index=False)
    pd.DataFrame({"t_ns": t_ns_imu, "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2]}).to_csv(out / "accel.csv", index=False)
    q = Rot.from_matrix(Rc).as_quat()
    pd.DataFrame({"t_ns": t_ns_cam, "x": Pc[:, 0], "y": Pc[:, 1], "z": Pc[:, 2],
                  "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]}).to_csv(out / "groundtruth.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"video": "video.mp4", "source": "simulation", "experiment": "closed loop",
                                               "true_td_s": args.td, "true_R_CI": R_CI.tolist(),
                                               "true_gyro_bias": bg.tolist(), "true_accel_bias": ba.tolist()}, indent=2))
    (out / "camera.json").write_text(json.dumps({"K": K.tolist(), "dist": [0, 0, 0, 0, 0], "image_size": [W, H],
                                                 "rms_px": 0.0, "source": "simulation"}, indent=2))
    (out / "camera_imu.json").write_text(json.dumps({"td_s": args.td, "R_CI": R_CI.tolist(), "source": "simulation (true values)"}, indent=2))
    survey = {"dictionary": "DICT_4X4_50", "marker_size": 0.16, "frame": "room",
              "markers": {str(k): {"position": list(c), "normal": list(n)} for k, (c, n) in MARKERS.items()}}
    (out / "markers_surveyed.json").write_text(json.dumps(survey, indent=2))
    # checkpoints: positions where the simulated walker is standing still (start / end)
    pd.DataFrame({"id": ["start", "end"], "x": [Pc[0, 0], Pc[-1, 0]], "y": [Pc[0, 1], Pc[-1, 1]]}).to_csv(out / "checkpoints.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
