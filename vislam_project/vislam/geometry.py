"""Rigid-body helpers. Convention: T_AB maps points from frame B to frame A (p_A = T_AB p_B).
Camera frame C follows OpenCV (x right, y down, z forward). World W is z-up."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotvec_to_R(v):
    return Rot.from_rotvec(v).as_matrix()


def R_to_rotvec(R):
    return Rot.from_matrix(R).as_rotvec()


def T_from_rvec_tvec(rvec, tvec):
    return make_T(rotvec_to_R(np.ravel(rvec)), np.ravel(tvec))


def rvec_tvec_from_T(T):
    return R_to_rotvec(T[:3, :3]).reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def rot_angle_deg(R):
    return float(np.degrees(np.linalg.norm(R_to_rotvec(R))))


def project_to_SO3(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def umeyama(src, dst, with_scale=False):
    """Least-squares (s, R, t) with dst ~= s * R @ src + t. src, dst: (N,3)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    H = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def kabsch_vectors(a, b):
    """Rotation R minimising sum ||b_k - R a_k||^2 (no translation). a, b: (N,3)."""
    H = np.asarray(a).T @ np.asarray(b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def marker_pose_from_survey(center, normal, up=(0.0, 0.0, 1.0)):
    """T_WM for an upright marker on a wall. Marker frame (OpenCV ArUco): x right,
    y up along the marker, z out of the marker (towards the room). `normal` points into the room."""
    z = np.asarray(normal, float)
    z /= np.linalg.norm(z)
    y = np.asarray(up, float)
    y = y - (y @ z) * z
    if np.linalg.norm(y) < 1e-6:  # marker on floor/ceiling: pick world x as marker y
        y = np.array([1.0, 0.0, 0.0]) - z[0] * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return make_T(np.column_stack([x, y, z]), center)


def gravity_aligned_R_WC(up_c):
    """World z = measured 'up' (in camera coords); world x = camera forward projected on the
    horizontal plane. Returns R_WC (rows are world axes in camera coordinates)."""
    z = np.asarray(up_c, float)
    z /= np.linalg.norm(z)
    fwd = np.array([0.0, 0.0, 1.0])
    x = fwd - (fwd @ z) * z
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])
