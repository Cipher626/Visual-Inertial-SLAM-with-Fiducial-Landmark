"""ArUco detection and camera-to-marker pose (C_T_M) estimation."""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import T_from_rvec_tvec


def get_dictionary(name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_object_points(size):
    """Corner order used by ArUco detection and SOLVEPNP_IPPE_SQUARE: TL, TR, BR, BL; z out of marker."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass
class MarkerObs:
    marker_id: int
    corners_px: np.ndarray      # (4,2) detected (distorted) pixels
    corners_norm: np.ndarray    # (4,2) undistorted normalised image coordinates
    T_CM: np.ndarray            # 4x4 marker pose in camera frame
    side_px: float
    reproj_px: float
    ambiguity: float            # error ratio of the two IPPE solutions (close to 1 => orientation ambiguous)


class MarkerDetector:
    def __init__(self, dictionary, marker_size, K, dist, min_side_px=20.0, max_reproj_px=3.0, max_range_m=8.0,
                 valid_ids=None, error_correction_rate=0.6):
        self.size = float(marker_size)
        self.K, self.dist = np.asarray(K, float), np.asarray(dist, float).ravel()
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.errorCorrectionRate = float(error_correction_rate)   # lower = fewer false IDs from texture
        self.valid_ids = set(valid_ids) if valid_ids else None
        self.detector = cv2.aruco.ArucoDetector(get_dictionary(dictionary), params)
        self.obj = marker_object_points(self.size)
        self.min_side_px, self.max_reproj_px, self.max_range = min_side_px, max_reproj_px, max_range_m

    def detect(self, gray, allowed_ids=None):
        corners, ids, _ = self.detector.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, mid in zip(corners, ids.ravel()):
            mid = int(mid)
            if (allowed_ids is not None and mid not in allowed_ids) or (self.valid_ids and mid not in self.valid_ids):
                continue
            c = c.reshape(4, 2).astype(np.float64)
            side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
            if side < self.min_side_px:
                continue
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, c, self.K, self.dist,
                                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n < 1:
                continue
            errs = np.ravel(errs) if errs is not None else np.zeros(n)
            best = int(np.argmin(errs))
            T = T_from_rvec_tvec(rvecs[best], tvecs[best])
            proj, _ = cv2.projectPoints(self.obj, rvecs[best], tvecs[best], self.K, self.dist)
            reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - c) ** 2, axis=1))))
            if reproj > self.max_reproj_px or T[2, 3] <= 0 or np.linalg.norm(T[:3, 3]) > self.max_range:
                continue
            amb = float(errs[best] / max(errs[1 - best], 1e-12)) if n == 2 else 0.0
            norm = cv2.undistortPoints(c.reshape(-1, 1, 2), self.K, self.dist).reshape(4, 2)
            out.append(MarkerObs(mid, c, norm, T, side, reproj, amb))
        return out


def draw_detections(img, obs, K=None, dist=None, size=None):
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for o in obs:
        pts = o.corners_px.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        cv2.putText(vis, f"id {o.marker_id}  {np.linalg.norm(o.T_CM[:3, 3]):.2f} m", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if K is not None and size is not None:
            rvec = cv2.Rodrigues(o.T_CM[:3, :3])[0]
            cv2.drawFrameAxes(vis, K, dist, rvec, o.T_CM[:3, 3], size * 0.6)
    return vis
