from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def count_descendants(stage, prim_path: str) -> int:
    from pxr import Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0
    count = 0
    for _ in Usd.PrimRange(prim):
        count += 1
    return max(0, count - 1)


def set_prim_pose(stage, prim_path: str, pos_xyz: Sequence[float], quat_xyzw: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom

    x, y, z, w = (
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for pose set: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    mat = Gf.Matrix4d()
    mat.SetTranslateOnly(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
    mat.SetRotateOnly(Gf.Rotation(Gf.Quatd(float(w), float(x), float(y), float(z))))

    xform.AddTransformOp().Set(mat)


def get_prim_world_pose(stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for world pose query: {prim_path}")

    xf = UsdGeom.Xformable(prim)
    mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    tr = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    pos = np.array([float(tr[0]), float(tr[1]), float(tr[2])], dtype=np.float64)
    quat_xyzw = np.array([float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())], dtype=np.float64)
    return pos, quat_xyzw


def get_prim_world_bbox_center(stage, prim_path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for bbox query: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned = world_bound.ComputeAlignedBox()
    v_min = aligned.GetMin()
    v_max = aligned.GetMax()
    center = 0.5 * (
        np.array([float(v_min[0]), float(v_min[1]), float(v_min[2])], dtype=np.float64)
        + np.array([float(v_max[0]), float(v_max[1]), float(v_max[2])], dtype=np.float64)
    )
    return center


def get_prim_world_aabb(stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for bbox query: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned = world_bound.ComputeAlignedBox()
    v_min = aligned.GetMin()
    v_max = aligned.GetMax()
    bb_min = np.array([float(v_min[0]), float(v_min[1]), float(v_min[2])], dtype=np.float64)
    bb_max = np.array([float(v_max[0]), float(v_max[1]), float(v_max[2])], dtype=np.float64)
    return bb_min, bb_max


def aabbs_overlap(a_min: Sequence[float], a_max: Sequence[float], b_min: Sequence[float], b_max: Sequence[float]) -> bool:
    a_min = np.asarray(a_min, dtype=np.float64)
    a_max = np.asarray(a_max, dtype=np.float64)
    b_min = np.asarray(b_min, dtype=np.float64)
    b_max = np.asarray(b_max, dtype=np.float64)
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


def world_pos_to_frame_pos(stage, frame_prim_path: str, world_pos_xyz: Sequence[float]) -> np.ndarray:
    from pxr import Gf, Usd, UsdGeom

    frame_prim = stage.GetPrimAtPath(frame_prim_path)
    if not frame_prim.IsValid():
        raise RuntimeError(f"Invalid frame prim path for conversion: {frame_prim_path}")

    frame_xf = UsdGeom.Xformable(frame_prim)
    frame_world = frame_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_frame = frame_world.GetInverse()
    local = world_to_frame.Transform(Gf.Vec3d(float(world_pos_xyz[0]), float(world_pos_xyz[1]), float(world_pos_xyz[2])))
    return np.array([float(local[0]), float(local[1]), float(local[2])], dtype=np.float64)


def find_robot_hand_prim(stage, robot_root_prim_path: str) -> Optional[str]:
    from pxr import Usd

    root = stage.GetPrimAtPath(robot_root_prim_path)
    if not root.IsValid():
        return None

    candidates_exact = ["panda_hand", "hand", "ee_link", "gripper_base", "panda_link8"]
    candidates_partial = ["hand", "gripper", "ee", "wrist"]

    for prim in Usd.PrimRange(root):
        name_l = prim.GetName().lower()
        if name_l in candidates_exact:
            return str(prim.GetPath())

    for prim in Usd.PrimRange(root):
        name_l = prim.GetName().lower()
        if any(tok in name_l for tok in candidates_partial):
            return str(prim.GetPath())

    return None


def find_articulation_root_prim(stage, root_prim_path: str) -> Optional[str]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return None

    if root.HasAPI(UsdPhysics.ArticulationRootAPI):
        return str(root.GetPath())

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def hide_collision_geometry(stage, root_prim_path: str) -> int:
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return 0

    hidden = 0
    for prim in Usd.PrimRange(root):
        name_l = prim.GetName().lower()
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)

        parent = prim.GetParent()
        while not has_collision_api and parent and parent.IsValid() and parent != root.GetParent():
            has_collision_api = parent.HasAPI(UsdPhysics.CollisionAPI)
            parent = parent.GetParent()

        name_looks_like_collision = any(tok in name_l for tok in ["collision", "collider", "convex"])
        is_imageable = UsdGeom.Imageable(prim)
        purpose_is_non_render = False
        if is_imageable:
            purpose_attr = is_imageable.GetPurposeAttr()
            purpose_val = purpose_attr.Get() if purpose_attr else None
            purpose_is_non_render = purpose_val in {"proxy", "guide"}

        if not (has_collision_api or name_looks_like_collision or purpose_is_non_render):
            continue

        if is_imageable:
            is_imageable.MakeInvisible()
            hidden += 1

    return hidden
