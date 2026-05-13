import math
from typing import Sequence, Tuple

import numpy as np


def rpy_to_matrix(rpy_xyz: Sequence[float]) -> np.ndarray:
    r, p, y = float(rpy_xyz[0]), float(rpy_xyz[1]), float(rpy_xyz[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    return tf


def xyz_rpy_to_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    tf = rpy_to_matrix(rpy)
    tf[:3, 3] = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64)
    return tf


def rotation_matrix_to_quatd(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 column-vector rotation matrix to (w, x, y, z) tuple."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return w, x, y, z


def quat_multiply_xyzw(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Quaternion multiply in xyzw order: result = q1 * q2."""
    x1, y1, z1, w1 = [float(v) for v in q1]
    x2, y2, z2, w2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_normalize_xyzw(q: Sequence[float]) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(qn))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return qn / n
