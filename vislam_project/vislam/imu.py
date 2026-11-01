"""Inertial processing: static detection, bias/gravity estimation, gyro attitude integration,
and (optional) accelerometer propagation."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

GRAVITY = 9.81

# Default camera<-IMU rotation for an Android back camera with SENSOR_ORIENTATION = 90 and raw
# (un-rotated) video frames: x_c = -y_i, y_c = -x_i, z_c = -z_i.  Replace with calibrated value.
R_CI_DEFAULT = np.array([[0.0, -1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


class ImuData:
    def __init__(self, gyro, accel, gyro_bias=None):
        self.gyro = self._clean(gyro)
        self.accel = self._clean(accel)
        self.bg = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
        self._build_attitude()

    @staticmethod
    def _clean(a):
        """Sort by time, drop non-finite rows and repeated timestamps (attitude interpolation needs strictly
        increasing times; phones occasionally deliver duplicates)."""
        a = np.asarray(a, float)
        a = a[np.all(np.isfinite(a), axis=1)]
        a = a[np.argsort(a[:, 0], kind="stable")]
        keep = np.r_[True, np.diff(a[:, 0]) > 0]
        return a[keep]

    # ---------- static detection / biases ----------
    def static_intervals(self, min_dur=1.0, win=0.25, gyro_thr=0.05, acc_std_thr=0.25):
        """Intervals where the phone is held still (gyro norm small, accel steady)."""
        t = self.gyro[:, 0]
        if len(t) < 10:
            return []
        edges = np.arange(t[0], t[-1], win)
        flags = []
        for a in edges:
            g = self.gyro[(t >= a) & (t < a + win), 1:]
            m = (self.accel[:, 0] >= a) & (self.accel[:, 0] < a + win)
            acc = self.accel[m, 1:]
            ok = len(g) > 2 and len(acc) > 2 and np.median(np.linalg.norm(g - self.bg, axis=1)) < gyro_thr \
                and np.linalg.norm(acc, axis=1).std() < acc_std_thr
            flags.append(ok)
        out, start = [], None
        for a, f in zip(edges, flags):
            if f and start is None:
                start = a
            if not f and start is not None:
                if a - start >= min_dur:
                    out.append((float(start), float(a)))
                start = None
        if start is not None and edges[-1] + win - start >= min_dur:
            out.append((float(start), float(edges[-1] + win)))
        return out

    def estimate_gyro_bias(self, interval):
        m = (self.gyro[:, 0] >= interval[0]) & (self.gyro[:, 0] <= interval[1])
        self.bg = self.gyro[m, 1:].mean(0)
        self._build_attitude()
        return self.bg

    def mean_specific_force(self, t_a, t_b):
        m = (self.accel[:, 0] >= t_a) & (self.accel[:, 0] <= t_b)
        if m.sum() < 3:
            m = np.argsort(np.abs(self.accel[:, 0] - 0.5 * (t_a + t_b)))[:10]
        return self.accel[m, 1:].mean(0)

    # ---------- attitude ----------
    def _build_attitude(self):
        t = self.gyro[:, 0]
        w = self.gyro[:, 1:] - self.bg
        q = [(1.0, 0.0, 0.0, 0.0)]
        dts = np.diff(t)
        wm = 0.5 * (w[1:] + w[:-1])
        dq = Rot.from_rotvec(wm * dts[:, None]).as_quat()  # x,y,z,w
        cur = q[0]
        for x, y, z, ww in dq:
            cur = _quat_mul(cur, (ww, x, y, z))
            q.append(cur)
        q = np.array(q)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        self._t = t
        self._slerp = Slerp(t, Rot.from_quat(q[:, [1, 2, 3, 0]]))

    def R_I(self, times):
        """Orientation of the IMU at `times` relative to the first gyro sample (R_I0_I(t))."""
        times = np.clip(np.asarray(times, float), self._t[0], self._t[-1])
        return self._slerp(times)

    def delta_R(self, t_a, t_b):
        """Rotation of IMU frame at t_b expressed in IMU frame at t_a (R_Ia_Ib). Vectorised."""
        Ra, Rb = self.R_I(t_a), self.R_I(t_b)
        return (Ra.inv() * Rb).as_matrix()

    def gyro_at(self, t):
        t = np.asarray(t, float)
        return np.column_stack([np.interp(t, self.gyro[:, 0], self.gyro[:, k]) for k in (1, 2, 3)]) - self.bg

    def accel_at(self, t):
        t = np.asarray(t, float)
        return np.column_stack([np.interp(t, self.accel[:, 0], self.accel[:, k]) for k in (1, 2, 3)])


def propagate_accel(imu, p, v, R_WC_a, R_CI, t_a, t_b, accel_bias=None, td=0.0, n_steps=None):
    """Strapdown propagation of position/velocity between camera times t_a and t_b using the
    accelerometer, with attitude from gyro integration. World z is up, gravity = (0,0,-g)."""
    ba = np.zeros(3) if accel_bias is None else accel_bias
    g_w = np.array([0.0, 0.0, -GRAVITY])
    if n_steps is None:
        n_steps = max(2, int((t_b - t_a) * 200))
    ts = np.linspace(t_a, t_b, n_steps + 1)
    R_WI_a = R_WC_a @ R_CI
    dR = imu.delta_R(np.full(len(ts), t_a + td), ts + td)          # R_Ia_I(t)
    f = imu.accel_at(ts + td) - ba
    p, v = p.copy(), v.copy()
    for k in range(n_steps):
        dt = ts[k + 1] - ts[k]
        a_w = R_WI_a @ dR[k] @ f[k] + g_w
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
    return p, v
