import argparse
import json
import math
import pathlib
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JSON_FILENAME = "isaac_objects_for_moveit.json"
CAMERA_PRIM_PATH = "/Franka/panda_hand/FrankaCamera"
VIEW_CANDIDATES_PATH = WORKSPACE_ROOT / "view_candidates" / "view_candidates.json"
NUM_CANDIDATES = 100
SETTLE_FRAMES = 5
JOINT_REACH_MAX_STEPS = 180
JOINT_REACH_TOL_RAD = 2e-3
MAX_OBJECT_CLASSES = 15


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_scene_info(scene_key: str) -> dict:
    with SCENE_INFO_PATH.open(encoding="utf-8") as f:
        scene_info_all = json.load(f)
    if scene_key not in scene_info_all:
        available = ", ".join(sorted(scene_info_all.get("key", [])))
        raise KeyError(f"Unknown scene '{scene_key}'. Available: {available}")
    return scene_info_all[scene_key]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB+depth views from robot arm candidates.")
    parser.add_argument("--scene", type=str, default=None, help="Scene key from scene_info.json, e.g. 01.")
    parser.add_argument("--scene-num", type=str, default=None, help="Scene index in dataset root, e.g. 001.")
    parser.add_argument("--scene-json", type=str, default=None, help="Explicit path to isaac_objects_for_moveit.json.")
    parser.add_argument("--base-usd", type=str, default=None)
    parser.add_argument("--robot-prim-path", type=str, default="/Franka")
    parser.add_argument("--camera-prim-path", type=str, default=CAMERA_PRIM_PATH)
    parser.add_argument("--candidates-json", type=str, default=str(VIEW_CANDIDATES_PATH))
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()
scene_info = _load_scene_info(ARGS.scene) if ARGS.scene else None

if ARGS.scene_json:
    scene_json_path = _resolve_repo_path(ARGS.scene_json)
elif scene_info is not None and ARGS.scene_num is not None:
    try:
        scene_num = int(ARGS.scene_num)
    except ValueError as exc:
        raise ValueError(f"Invalid --scene-num: {ARGS.scene_num}") from exc
    dataset_root = _resolve_repo_path(str(scene_info.get("dataset_root")))
    scene_json_path = dataset_root / f"scene_{scene_num:03d}" / JSON_FILENAME
else:
    scene_json_path = None

if scene_info is not None and ARGS.base_usd is None:
    ARGS.base_usd = str(_resolve_repo_path(str(scene_info.get("scene_usd"))))

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.usd
import PIL.Image
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(world: World, n: int) -> None:
    for _ in range(n):
        # Sensor annotators need render ticks even in long capture loops.
        world.step(render=True)


def _quat_xyzw_to_rotmat(q_xyzw) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy)],
        [2*(xy + wz),       1 - 2*(xx + zz),  2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float64)


def _build_c2w(position, quat_xyzw) -> list[list[float]]:
    """Build 4x4 camera-to-world matrix from position and xyzw quaternion."""
    R = _quat_xyzw_to_rotmat(quat_xyzw)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = np.array(position, dtype=np.float64)
    return mat.tolist()


def _get_prim_world_pose(stage: Usd.Stage, prim_path: str):
    """Return (position list, quaternion_xyzw list) for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None
    m = omni.usd.get_world_transform_matrix(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotation().GetQuat()
    im = q.GetImaginary()
    pos = [float(t[0]), float(t[1]), float(t[2])]
    quat_xyzw = [float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())]
    return pos, quat_xyzw


def _save_rgb(rgba: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgba is None:
        carb.log_warn(f"RGB data is None, skipping save: {path}")
        return
    arr = np.asarray(rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        img = PIL.Image.fromarray(arr[:, :, :3].astype(np.uint8), "RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = PIL.Image.fromarray(arr.astype(np.uint8), "RGB")
    else:
        carb.log_warn(f"Unexpected RGB shape {arr.shape}, skipping: {path}")
        return
    img.save(str(path))


def _save_depth(depth: np.ndarray, path: Path, expected_hw: tuple[int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth is None:
        carb.log_warn(f"Depth data is None, skipping save: {path}")
        return

    arr = np.asarray(depth)
    if arr.size == 0:
        carb.log_warn(f"Depth data is empty, skipping save: {path}")
        return

    # Depth annotator can return HxW, HxWx1, or flattened H*W depending on backend.
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 1 and expected_hw is not None and arr.size == expected_hw[0] * expected_hw[1]:
        arr = arr.reshape(expected_hw)

    if arr.ndim != 2:
        carb.log_warn(f"Unexpected depth shape {arr.shape}, skipping: {path}")
        return

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to 16-bit PNG (meters * 1000 → millimeters, clipped at 65535)
    arr_mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
    img = PIL.Image.fromarray(arr_mm)
    img.save(str(path))


def _is_empty_frame(frame) -> bool:
    if frame is None:
        return True
    arr = np.asarray(frame)
    return arr.size == 0


def _capture_rgbd_with_retry(camera: Camera, world: World, max_retry_frames: int = 20) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fetch camera RGB/depth and wait for a few frames if annotators return empty arrays."""
    rgba = camera.get_rgba()
    depth = camera.get_depth()
    if not _is_empty_frame(rgba) and not _is_empty_frame(depth):
        return rgba, depth

    for _ in range(max_retry_frames):
        world.step(render=True)
        rgba = camera.get_rgba()
        depth = camera.get_depth()
        if not _is_empty_frame(rgba) and not _is_empty_frame(depth):
            return rgba, depth
    return None, None


def _set_semantic_class_label(prim: Usd.Prim, class_label: str) -> None:
    if not prim.IsValid():
        return
    sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem_api.CreateSemanticTypeAttr().Set("class")
    sem_api.CreateSemanticDataAttr().Set(class_label)


def _build_object_class_mapping(
    all_objects_data: list[dict],
    target_prim_path: str,
    max_classes: int = MAX_OBJECT_CLASSES,
) -> tuple[dict[str, tuple[int, str]], list[dict], dict[str, int]]:
    prim_paths = []
    for obj in all_objects_data:
        p = obj.get("prim_path")
        if p and p not in prim_paths:
            prim_paths.append(p)

    ordered_paths = []
    if target_prim_path in prim_paths:
        ordered_paths.append(target_prim_path)
    for p in prim_paths:
        if p != target_prim_path:
            ordered_paths.append(p)
    ordered_paths = ordered_paths[:max_classes]

    prim_to_class = {}
    class_entries = []
    label_to_class_id = {}
    for class_id, prim_path in enumerate(ordered_paths, start=1):
        class_label = f"class_{class_id:02d}"
        prim_to_class[prim_path] = (class_id, class_label)
        label_to_class_id[class_label] = class_id
        class_entries.append(
            {
                "class_id": class_id,
                "prim_path": prim_path,
                "label": class_label,
                "is_target": prim_path == target_prim_path,
            }
        )
    return prim_to_class, class_entries, label_to_class_id


def _apply_object_class_semantics(stage: Usd.Stage, prim_to_class: dict[str, tuple[int, str]]) -> None:
    for prim_path, (_, class_label) in prim_to_class.items():
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            _set_semantic_class_label(prim, class_label)


def _extract_semantic_payload(semantic_frame) -> tuple[np.ndarray | None, dict | None]:
    if semantic_frame is None:
        return None, None
    if isinstance(semantic_frame, dict):
        data = semantic_frame.get("data")
        info = semantic_frame.get("info")
        return (np.asarray(data) if data is not None else None), (info if isinstance(info, dict) else None)
    return np.asarray(semantic_frame), None


def _collect_labels_from_info_entry(entry) -> list[str]:
    labels = []
    if isinstance(entry, str):
        labels.append(entry)
    elif isinstance(entry, dict):
        for value in entry.values():
            if isinstance(value, str):
                labels.append(value)
            elif isinstance(value, (list, tuple)):
                labels.extend([v for v in value if isinstance(v, str)])
    elif isinstance(entry, (list, tuple)):
        labels.extend([v for v in entry if isinstance(v, str)])
    return labels


def _build_semantic_id_to_class_map(info: dict | None, label_to_class_id: dict[str, int]) -> dict[int, int]:
    sid_to_class = {}
    if not info:
        return sid_to_class

    id_maps = []
    for key in ["idToLabels", "idToSemantics"]:
        value = info.get(key)
        if isinstance(value, dict):
            id_maps.append(value)

    for id_map in id_maps:
        for sid_key, payload in id_map.items():
            try:
                sid = int(sid_key)
            except Exception:
                continue
            class_id = 0
            for label in _collect_labels_from_info_entry(payload):
                if label in label_to_class_id:
                    class_id = int(label_to_class_id[label])
                    break
            sid_to_class[sid] = class_id
    return sid_to_class


def _semantic_to_class_mask(
    semantic_data: np.ndarray | None,
    expected_hw: tuple[int, int],
    sid_to_class: dict[int, int],
) -> np.ndarray | None:
    if semantic_data is None:
        return None
    arr = np.asarray(semantic_data)
    if arr.size == 0:
        return None

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 1 and arr.size == expected_hw[0] * expected_hw[1]:
        arr = arr.reshape(expected_hw)

    if arr.ndim != 2:
        return None

    arr = np.asarray(arr, dtype=np.uint32)
    class_mask = np.zeros(arr.shape, dtype=np.uint8)
    if not sid_to_class:
        return class_mask

    flat = arr.reshape(-1)
    out = np.zeros(flat.shape, dtype=np.uint8)
    unique_ids = np.unique(flat)
    for sid in unique_ids:
        class_id = sid_to_class.get(int(sid), 0)
        if class_id > 0:
            out[flat == sid] = np.uint8(class_id)
    class_mask[:, :] = out.reshape(arr.shape)
    return class_mask


def _save_class_mask(class_mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if class_mask is None:
        carb.log_warn(f"Class mask is None, skipping save: {path}")
        return
    arr = np.asarray(class_mask, dtype=np.uint8)
    if arr.ndim != 2:
        carb.log_warn(f"Unexpected class mask shape {arr.shape}, skipping: {path}")
        return
    img = PIL.Image.fromarray(arr)
    img.save(str(path))


def _build_intrinsics_payload(camera: Camera, stage: Usd.Stage, camera_prim_path: str) -> dict:
    width, height = [int(v) for v in camera.get_resolution()]

    K = None
    try:
        K_raw = camera.get_intrinsics_matrix()
        K = np.asarray(K_raw, dtype=np.float64)
    except Exception:
        focal_length = float(camera.get_focal_length())
        horizontal_aperture = float(camera.get_horizontal_aperture())
        vertical_aperture = float(camera.get_vertical_aperture())
        fx = width * focal_length / horizontal_aperture if abs(horizontal_aperture) > 1e-12 else 0.0
        fy = height * focal_length / vertical_aperture if abs(vertical_aperture) > 1e-12 else 0.0
        cx = width * 0.5
        cy = height * 0.5
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    projection = None
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if camera_prim.IsValid():
        proj_attr = camera_prim.GetAttribute("cameraProjectionType")
        if proj_attr and proj_attr.IsValid():
            projection = proj_attr.Get()

    return {
        "camera_prim_path": camera_prim_path,
        "projection": projection,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K.tolist(),
    }


def _capture_rgbd_semantic_with_retry(
    camera: Camera,
    world: World,
    max_retry_frames: int = 20,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]:
    for _ in range(max_retry_frames + 1):
        rgba = camera.get_rgba()
        depth = camera.get_depth()
        semantic_frame = camera.get_current_frame().get("semantic_segmentation")
        semantic_data, semantic_info = _extract_semantic_payload(semantic_frame)
        if not _is_empty_frame(rgba) and not _is_empty_frame(depth) and not _is_empty_frame(semantic_data):
            return rgba, depth, semantic_data, semantic_info
        world.step(render=True)
    return None, None, None, None


def _move_and_wait_joint_target(
    world: World,
    robot: SingleArticulation,
    controller,
    goal_q: np.ndarray,
    check_indices: list[int],
    max_steps: int = JOINT_REACH_MAX_STEPS,
    tol: float = JOINT_REACH_TOL_RAD,
) -> tuple[bool, float]:
    """Drive joints to target and verify convergence before capture."""
    goal_q = np.asarray(goal_q, dtype=np.float64)

    # Try immediate teleport first to avoid long settling tails.
    try:
        robot.set_joint_positions(goal_q)
    except Exception:
        pass

    max_err = float("inf")
    for _ in range(max_steps):
        controller.apply_action(ArticulationAction(joint_positions=goal_q))
        world.step(render=True)
        cur = robot.get_joint_positions()
        if cur is None:
            continue
        cur = np.asarray(cur, dtype=np.float64)
        if check_indices:
            err = np.abs(cur[check_indices] - goal_q[check_indices])
        else:
            err = np.abs(cur - goal_q)
        max_err = float(np.max(err))
        if max_err <= tol:
            return True, max_err

    # One more hard-set fallback, then check once.
    try:
        robot.set_joint_positions(goal_q)
        world.step(render=True)
        cur = robot.get_joint_positions()
        if cur is not None:
            cur = np.asarray(cur, dtype=np.float64)
            if check_indices:
                err = np.abs(cur[check_indices] - goal_q[check_indices])
            else:
                err = np.abs(cur - goal_q)
            max_err = float(np.max(err))
            if max_err <= tol:
                return True, max_err
    except Exception:
        pass

    return False, max_err


def _clear_children(stage: Usd.Stage, root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())


def _ensure_xform(stage: Usd.Stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        UsdGeom.Xform.Define(stage, path)


def _reset_xform_to_origin(stage: Usd.Stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    t_op = xform.AddTranslateOp()
    r_op = xform.AddOrientOp()
    s_op = xform.AddScaleOp()
    t_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    r_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    s_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _reload_objects_from_usda(
    stage: Usd.Stage,
    objects_usda: Path,
    object_root: str,
    all_objects_data: list[dict],
) -> None:
    _clear_children(stage, object_root)
    _ensure_xform(stage, object_root)
    _reset_xform_to_origin(stage, object_root)

    obj_stage = Usd.Stage.Open(str(objects_usda))
    if obj_stage is None:
        raise RuntimeError(f"Failed to open objects USDA: {objects_usda}")

    src_layer = obj_stage.GetRootLayer()
    dst_layer = stage.GetRootLayer()
    for obj_data in all_objects_data:
        src_path = obj_data.get("prim_path")
        if not src_path:
            continue
        Sdf.CopySpec(src_layer, src_path, dst_layer, src_path)

    del obj_stage
    for _ in range(30):
        simulation_app.update()


def _get_camera_capture_settings(stage: Usd.Stage, camera_prim_path: str) -> tuple[str | None, tuple[int, int] | None, str | None]:
    """Read projection + linked render product resolution directly from stage USD."""
    cam_prim = stage.GetPrimAtPath(camera_prim_path)
    if not cam_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_prim_path}")

    projection = None
    proj_attr = cam_prim.GetAttribute("cameraProjectionType")
    if proj_attr and proj_attr.IsValid():
        projection = proj_attr.Get()

    matched_render_products = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        rel = prim.GetRelationship("camera")
        if rel is None:
            continue
        targets = rel.GetTargets()
        if targets and str(targets[0]) == camera_prim_path:
            matched_render_products.append(prim)

    chosen_prim = None
    for prim in matched_render_products:
        if prim.GetName() == "Replicator":
            chosen_prim = prim
            break
    if chosen_prim is None and matched_render_products:
        chosen_prim = matched_render_products[0]

    render_product_path = None
    resolution = None
    if chosen_prim is not None:
        render_product_path = str(chosen_prim.GetPath())
        res_attr = chosen_prim.GetAttribute("resolution")
        if res_attr and res_attr.IsValid():
            res_val = res_attr.Get()
            if res_val is not None and len(res_val) == 2:
                resolution = (int(res_val[0]), int(res_val[1]))

    return projection, resolution, render_product_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if scene_json_path is None:
        raise ValueError("Provide --scene and --scene-num, or an explicit --scene-json path.")
    if not scene_json_path.is_file():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

    with scene_json_path.open(encoding="utf-8") as f:
        scene_data = json.load(f)
    object_root = scene_data.get("object_root", "/objects")
    target_prim_path = scene_data.get("target_object_prim_path", "")
    all_objects_data = scene_data.get("objects", [])
    objects_usda = scene_json_path.with_name("isaac_objects.usda")
    if not objects_usda.is_file():
        raise FileNotFoundError(f"Objects USDA not found: {objects_usda}")

    # --- Resolve scene directory for output ---
    if scene_json_path is not None and scene_json_path.is_file():
        scene_dir = scene_json_path.parent  # e.g. dataset/01_robot/scene_001
    elif ARGS.scene and ARGS.scene_num is not None:
        dataset_root = _resolve_repo_path(str(scene_info.get("dataset_root")))
        scene_dir = dataset_root / f"scene_{int(ARGS.scene_num):03d}"
    else:
        raise ValueError("Cannot determine scene directory. Provide --scene + --scene-num or --scene-json.")

    views_dir = scene_dir / "views"
    rgb_dir = views_dir / "rgb"
    depth_dir = views_dir / "depth"
    class_dir = views_dir / "class"
    pose_json_path = views_dir / "pose.json"
    intrinsics_json_path = views_dir / "intrinsics.json"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)

    # --- Load view candidates ---
    candidates_json = _resolve_repo_path(ARGS.candidates_json)
    if not candidates_json.is_file():
        raise FileNotFoundError(f"view_candidates.json not found: {candidates_json}")
    with candidates_json.open(encoding="utf-8") as f:
        candidates_data = json.load(f)
    candidates = candidates_data["candidates"][:NUM_CANDIDATES]
    print(f"[GetViews] Loaded {len(candidates)} candidates from {candidates_json}", flush=True)

    # --- Open USD stage ---
    if ARGS.base_usd is None:
        raise ValueError("Could not determine base USD. Pass --scene or --base-usd.")
    base_usd = Path(ARGS.base_usd).resolve()
    if not base_usd.is_file():
        raise FileNotFoundError(f"Base USD not found: {base_usd}")

    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(str(base_usd)):
        raise RuntimeError(f"Failed to open stage: {base_usd}")
    while usd_context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("USD stage unavailable.")

    projection, stage_resolution, stage_render_product_path = _get_camera_capture_settings(stage, ARGS.camera_prim_path)
    if ARGS.resolution is not None:
        cam_w, cam_h = int(ARGS.resolution[0]), int(ARGS.resolution[1])
    elif stage_resolution is not None:
        cam_w, cam_h = stage_resolution
    else:
        cam_w, cam_h = 1280, 720
    carb.log_info(f"[GetViews] Camera projection from USD: {projection}")
    carb.log_info(f"[GetViews] Capture resolution: {cam_w}x{cam_h}")
    if stage_render_product_path:
        carb.log_info(f"[GetViews] Found stage RenderProduct: {stage_render_product_path}")
    else:
        carb.log_warn("[GetViews] No stage RenderProduct found for camera.")
    carb.log_info("[GetViews] Creating dedicated capture RenderProduct for stable RGB/depth streaming.")

    # Reset object subtree and reload authored object specs from per-scene USDA.
    _reload_objects_from_usda(stage, objects_usda, object_root, all_objects_data)
    prim_to_class, class_entries, label_to_class_id = _build_object_class_mapping(
        all_objects_data=all_objects_data,
        target_prim_path=target_prim_path,
        max_classes=MAX_OBJECT_CLASSES,
    )
    _apply_object_class_semantics(stage, prim_to_class)

    # --- Set up world and robot ---
    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))

    # --- Set up camera ---
    camera = Camera(
        prim_path=ARGS.camera_prim_path,
        name="franka_camera",
        resolution=(cam_w, cam_h),
        render_product_path=None,
    )
    world.scene.add(camera)

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.add_semantic_segmentation_to_frame({"colorize": False})
    world.play()
    _step(world, 10)  # let simulation settle

    intrinsics_data = _build_intrinsics_payload(camera, stage, ARGS.camera_prim_path)
    with intrinsics_json_path.open("w", encoding="utf-8") as f:
        json.dump(intrinsics_data, f, indent=2)
    print(f"[GetViews] Intrinsics → {intrinsics_json_path}", flush=True)

    controller = robot.get_articulation_controller()
    dof_names = robot.dof_names

    # Find arm (non-finger) DOF indices
    arm_joint_ids = [i for i, name in enumerate(dof_names) if "finger" not in name.lower() and "gripper" not in name.lower()]
    print(f"[GetViews] Arm DOF indices: {arm_joint_ids} ({len(arm_joint_ids)} joints)", flush=True)

    pose_records = []
    reached_indices = []
    last_valid_rgba = None
    last_valid_depth = None
    last_valid_class_mask = None

    for idx, candidate in enumerate(candidates):
        rank = candidate.get("rank", idx + 1)
        joint_angles = candidate.get("joint_angles", [])
        cam_position = candidate.get("cam_position")
        cam_quaternion_xyzw = candidate.get("cam_quaternion_xyzw")
        ee_position = candidate.get("ee_position")
        ee_quaternion_xyzw = candidate.get("ee_quaternion_xyzw")

        print(f"[GetViews] {idx + 1}/{len(candidates)} rank={rank}", flush=True)

        # Reset objects every capture attempt to avoid accumulated disturbances.
        _reload_objects_from_usda(stage, objects_usda, object_root, all_objects_data)
        _apply_object_class_semantics(stage, prim_to_class)
        _step(world, 6)

        # --- Teleport robot joints ---
        current_q = robot.get_joint_positions()
        if current_q is None:
            carb.log_warn(f"[GetViews] idx={idx}: Failed to read joint positions, skipping.")
            continue
        goal_q = np.array(current_q, dtype=np.float64, copy=True)
        ja = np.array(joint_angles, dtype=np.float64)
        if ja.shape[0] == len(arm_joint_ids):
            for local_i, joint_i in enumerate(arm_joint_ids):
                goal_q[joint_i] = ja[local_i]
        elif ja.shape[0] == len(dof_names):
            goal_q[:] = ja
        else:
            carb.log_warn(f"[GetViews] idx={idx}: joint_angles length {ja.shape[0]} doesn't match arm ({len(arm_joint_ids)}) or all ({len(dof_names)}) DOFs.")

        reached, max_err = _move_and_wait_joint_target(
            world=world,
            robot=robot,
            controller=controller,
            goal_q=goal_q,
            check_indices=arm_joint_ids,
        )
        if not reached:
            carb.log_warn(
                f"[GetViews] idx={idx}: joint target not reached (max_err={max_err:.6f} rad), skipping capture for this candidate."
            )
            continue
        reached_indices.append(idx)
        _step(world, SETTLE_FRAMES)

        # --- Capture images (with retry for occasional empty annotator frames) ---
        rgba, depth, semantic_data, semantic_info = _capture_rgbd_semantic_with_retry(camera, world, max_retry_frames=24)
        sid_to_class = _build_semantic_id_to_class_map(semantic_info, label_to_class_id)
        class_mask = _semantic_to_class_mask(semantic_data, expected_hw=(cam_h, cam_w), sid_to_class=sid_to_class)
        if rgba is None or depth is None or class_mask is None:
            carb.log_warn(f"[GetViews] idx={idx}: empty frame after retries, using previous valid frame if available.")
            rgba = last_valid_rgba
            depth = last_valid_depth
            class_mask = last_valid_class_mask
        else:
            last_valid_rgba = rgba
            last_valid_depth = depth
            last_valid_class_mask = class_mask

        _save_rgb(rgba, rgb_dir / f"{idx:04d}.png")
        _save_depth(depth, depth_dir / f"{idx:04d}.png", expected_hw=(cam_h, cam_w))
        _save_class_mask(class_mask, class_dir / f"{idx:04d}.png")

        # --- Read actual camera pose from stage ---
        actual_cam_pos, actual_cam_quat_xyzw = _get_prim_world_pose(stage, ARGS.camera_prim_path)

        # Use candidate values if stage query failed
        pose_cam_position = actual_cam_pos if actual_cam_pos is not None else cam_position
        pose_cam_quat_xyzw = actual_cam_quat_xyzw if actual_cam_quat_xyzw is not None else cam_quaternion_xyzw

        cam_matrix = _build_c2w(pose_cam_position, pose_cam_quat_xyzw) if pose_cam_position is not None else None

        pose_records.append({
            "index": idx,
            "cam_position": pose_cam_position,
            "cam_quaternion_xyzw": pose_cam_quat_xyzw,
            "cam_matrix": cam_matrix,
            "ee_position": ee_position,
            "ee_quaternion_xyzw": ee_quaternion_xyzw,
            "joint_angles": joint_angles,
        })

    # --- Save pose.json ---
    output_data = {
        "reached_indices": reached_indices,
        "num_reached": len(reached_indices),
        "class_mapping": class_entries,
        "target_class_id": 1,
        "poses": pose_records,
    }
    with pose_json_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[GetViews] Saved {len(pose_records)} poses to {pose_json_path}", flush=True)
    print(f"[GetViews] Reached indices: {reached_indices}", flush=True)
    print(f"[GetViews] RGB   → {rgb_dir}", flush=True)
    print(f"[GetViews] Depth → {depth_dir}", flush=True)
    print(f"[GetViews] Class → {class_dir}", flush=True)

    simulation_app.close()


main()
