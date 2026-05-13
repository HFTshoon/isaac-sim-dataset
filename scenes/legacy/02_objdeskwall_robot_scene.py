import argparse
import json
import math
import os
import pathlib
import random
import re
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build robot-centric YCB clutter scene and save dataset files.")
	parser.add_argument(
		"--scene-usd",
		type=str,
		default=str(pathlib.Path(__file__).with_name("02_objdeskwall.usda")),
		help="USD scene to open.",
	)
	parser.add_argument("--robot-prim-path", type=str, default="/Franka", help="Franka articulation prim path.")
	parser.add_argument("--ee-frame", type=str, default="right_gripper", help="IK end-effector frame name.")
	parser.add_argument("--headless", action="store_true", help="Run without viewport.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
	parser.add_argument("--approach-distance", type=float, default=0.15, help="Approach offset (m) along -Y from target.")
	parser.add_argument("--dataset-root", type=str, default="/isaac-sim/corl2025/dataset/01_robot", help="Dataset root.")
	return parser.parse_args()


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.client
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver


OBJECT_ROOT = "/objects"
YCB_SUBDIR_CANDIDATES = ["/Isaac/Props/YCB/Axis_Aligned"]

OBJECT_KEY = "009"
TOTAL_OBJECTS = 15
ADDITIONAL_OBJECTS = 14

OBJECT_X_MIN = 0.5
OBJECT_X_MAX = 0.7
OBJECT_Y_MIN = -0.2
OBJECT_Y_MAX = 0.2
OBJECT_RZ_JITTER_DEG = 30.0
Z_MIN = 0.0
Z_MAX = 0.3

PREDEFINED_GRASP_POSE_JSON = pathlib.Path(__file__).with_name("predefined_grasp_pose.json")

ADDITIONAL_SPAWN_XY_MIN_RADIUS = 0.07
ADDITIONAL_SPAWN_XY_MAX_RADIUS = 0.15
ADDITIONAL_SPAWN_Z_MIN = 0.10
ADDITIONAL_SPAWN_Z_MAX = 0.25
PER_OBJECT_SETTLE_MAX_SECONDS = 5.0

SETTLE_POS_EPS = 0.0008
SETTLE_ROT_EPS_RAD = 0.004
SETTLE_STABLE_WINDOW_FRAMES = 90
SETTLE_MAX_SECONDS = 25.0

OBJ_DIR = "/isaac-sim/corl2025/obj"
JSON_FILENAME = "isaac_objects_for_moveit.json"
USDA_FILENAME = "isaac_objects.usda"
SCENE_IMAGE_FILENAME = "sceneCamera.png"
SCENE_CAMERA_PRIM_PATH = "/cameras/sceneCamera"


def _update_frames(count: int) -> None:
	for _ in range(count):
		simulation_app.update()


def _set_viewport_scene_camera(usd_context) -> None:
	if ARGS.headless:
		return
	try:
		from omni.kit.viewport.utility import get_active_viewport_window

		viewport_window = get_active_viewport_window(window_name="Viewport")
		if viewport_window is None:
			carb.log_warn("Viewport window named 'Viewport' was not found.")
			return

		stage = usd_context.get_stage()
		camera_prim = stage.GetPrimAtPath(SCENE_CAMERA_PRIM_PATH) if stage is not None else None
		if camera_prim is None or not camera_prim.IsValid():
			carb.log_warn(f"Scene camera prim not found: {SCENE_CAMERA_PRIM_PATH}")
			return

		viewport_window.viewport_api.camera_path = SCENE_CAMERA_PRIM_PATH
		carb.log_info(f"Viewport 1 camera set to: {SCENE_CAMERA_PRIM_PATH}")
	except Exception as exc:
		carb.log_warn(f"Failed to switch viewport camera: {exc}")


def _capture_scene_camera_image(usd_context, image_path: Path) -> bool:
	if ARGS.headless:
		carb.log_warn("Headless mode is enabled; skipping scene camera capture.")
		return False

	try:
		from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport_window

		viewport_window = get_active_viewport_window(window_name="Viewport")
		if viewport_window is None:
			carb.log_warn("Viewport window named 'Viewport' was not found; skipping scene camera capture.")
			return False

		stage = usd_context.get_stage()
		camera_prim = stage.GetPrimAtPath(SCENE_CAMERA_PRIM_PATH) if stage is not None else None
		if camera_prim is None or not camera_prim.IsValid():
			carb.log_warn(f"Scene camera prim not found: {SCENE_CAMERA_PRIM_PATH}; skipping capture.")
			return False

		viewport_window.viewport_api.camera_path = SCENE_CAMERA_PRIM_PATH
		_update_frames(2)

		capture_viewport_to_file(viewport_window.viewport_api, str(image_path))
		for _ in range(120):
			if image_path.exists():
				break
			_update_frames(1)

		if image_path.exists():
			carb.log_info(f"Saved scene camera image: {image_path}")
			return True

		carb.log_warn(f"Scene camera capture did not produce file in time: {image_path}")
		return False
	except Exception as exc:
		carb.log_warn(f"Failed to capture scene camera image: {exc}")
		return False


def _join_url(a: str, b: str) -> str:
	return a.rstrip("/") + "/" + b.lstrip("/")


def _list_usd_files(folder_url: str) -> list[str]:
	result, entries = omni.client.list(folder_url)
	if result != omni.client.Result.OK:
		return []
	usd_files = []
	for e in entries:
		name = e.relative_path
		if name.startswith("."):
			continue
		if name.lower().endswith((".usd", ".usda", ".usdc")):
			usd_files.append(_join_url(folder_url, name))
	return sorted(usd_files)


def _get_isaac_assets_root() -> str:
	try:
		from isaacsim.storage.native import get_assets_root_path

		root = get_assets_root_path()
	except Exception:
		from omni.isaac.core.utils.nucleus import get_assets_root_path

		root = get_assets_root_path()
	if not root:
		raise RuntimeError("Failed to resolve Isaac Sim assets root.")
	return root


def _find_ycb_assets() -> tuple[list[str], str]:
	root = _get_isaac_assets_root()
	tried = []
	for subdir in YCB_SUBDIR_CANDIDATES:
		folder = _join_url(root, subdir)
		tried.append(folder)
		files = _list_usd_files(folder)
		if files:
			return files, folder
	raise RuntimeError("YCB assets not found. Checked: " + " | ".join(tried))


def _find_object_asset(asset_paths: list[str]) -> str:
	for p in asset_paths:
		name = Path(p).stem.lower()
		if OBJECT_KEY in name:
			return p
	for p in asset_paths:
		name = Path(p).stem.lower()
		if "mustard" in name:
			return p
	raise RuntimeError("Could not find target object asset in YCB folder.")


def _sanitize_name(path_or_name: str) -> str:
	base = Path(path_or_name).stem
	clean = re.sub(r"[^A-Za-z0-9_]+", "_", base)
	if not clean:
		clean = "object"
	if clean[0].isdigit():
		clean = "obj_" + clean
	return clean


def _ensure_xform(stage: Usd.Stage, path: str) -> None:
	prim = stage.GetPrimAtPath(path)
	if not prim.IsValid():
		UsdGeom.Xform.Define(stage, path)


def _clear_children(stage: Usd.Stage, root_path: str) -> None:
	root = stage.GetPrimAtPath(root_path)
	if not root.IsValid():
		return
	for child in list(root.GetChildren()):
		stage.RemovePrim(child.GetPath())


def _ensure_physics_scene(stage: Usd.Stage) -> None:
	for prim in stage.Traverse():
		if prim.IsA(UsdPhysics.Scene):
			return
	UsdPhysics.Scene.Define(stage, "/physicsScene")


def _set_prim_transform(prim: Usd.Prim, translate_xyz: tuple[float, float, float], rotate_xyz_deg: tuple[float, float, float]) -> None:
	xform = UsdGeom.Xformable(prim)
	xform.ClearXformOpOrder()
	t_op = xform.AddTranslateOp()
	r_op = xform.AddRotateXYZOp()
	t_op.Set(Gf.Vec3d(*translate_xyz))
	r_op.Set(Gf.Vec3f(*rotate_xyz_deg))


def _apply_physics_to_object(object_prim: Usd.Prim) -> None:
	if not object_prim.IsValid():
		return
	if not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
		UsdPhysics.RigidBodyAPI.Apply(object_prim)
	rb_api = UsdPhysics.RigidBodyAPI(object_prim)
	rb_api.CreateRigidBodyEnabledAttr(True)

	for prim in Usd.PrimRange(object_prim):
		if not prim.IsA(UsdGeom.Mesh):
			continue
		if not prim.HasAPI(UsdPhysics.CollisionAPI):
			UsdPhysics.CollisionAPI.Apply(prim)
		UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
		if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
			UsdPhysics.MeshCollisionAPI.Apply(prim)
		UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("convexHull")


def _matrix_translation(m: Gf.Matrix4d) -> list[float]:
	t = m.ExtractTranslation()
	return [float(t[0]), float(t[1]), float(t[2])]


def _matrix_quat_xyzw(m: Gf.Matrix4d) -> list[float]:
	q = m.ExtractRotation().GetQuat()
	imag = q.GetImaginary()
	return [float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]


def _matrix_quat_wxyz(m: Gf.Matrix4d) -> list[float]:
	q = m.ExtractRotation().GetQuat()
	imag = q.GetImaginary()
	return [float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return np.array(
		[
			w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
			w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
			w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
			w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
		],
		dtype=np.float64,
	)


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
	w, x, y, z = q
	return np.array(
		[
			[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
			[2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
			[2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
		],
		dtype=np.float64,
	)


def _load_json_loose(path: pathlib.Path) -> dict:
	text = path.read_text(encoding="utf-8")
	# Allow trailing commas in hand-edited json files.
	text = re.sub(r",\s*([}\]])", r"\1", text)
	return json.loads(text)


def _extract_object_key(value: str) -> str:
	m = re.search(r"(\d{3})", value or "")
	if m:
		return m.group(1)
	return (value or "").strip()


def _default_predefined_pose_entry() -> dict:
	return {
		"local_offset_m": [0.0, -0.095, 0.0],
		"local_rpy_deg": [-90.0, 0.0, 0.0],
		"location_z": 0.11,
		"rotation_deg": [-90.0, 0.0, 180.0],
	}


def _select_predefined_pose_entry(object_key: str) -> dict:
	default_pose = _default_predefined_pose_entry()
	if not PREDEFINED_GRASP_POSE_JSON.is_file():
		carb.log_warn(f"Predefined grasp file not found: {PREDEFINED_GRASP_POSE_JSON}. Using defaults.")
		return default_pose

	try:
		data = _load_json_loose(PREDEFINED_GRASP_POSE_JSON)
	except Exception as exc:
		carb.log_warn(f"Failed to parse predefined grasp file: {exc}. Using defaults.")
		return default_pose

	keys = [str(k) for k in data.get("key", [])]
	if object_key not in keys:
		carb.log_warn(f"object_key '{object_key}' not in predefined key list {keys}. Using defaults.")
		return default_pose

	poses = data.get(object_key, [])
	if not isinstance(poses, list) or not poses:
		carb.log_warn(f"No predefined grasp poses for key '{object_key}'. Using defaults.")
		return default_pose

	selected = random.choice(poses)
	if not isinstance(selected, dict):
		carb.log_warn(f"Invalid predefined pose type for key '{object_key}'. Using defaults.")
		return default_pose
	return selected


def _select_predefined_grasp_pose(object_key: str, selected_pose: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
	default_offset = np.asarray((0.0, -0.095, 0.0), dtype=np.float64)
	default_rpy = np.asarray((-90.0, 0.0, 0.0), dtype=np.float64)
	selected = selected_pose if selected_pose is not None else _select_predefined_pose_entry(object_key)
	local_offset = selected.get("local_offset_m", [0.0, -0.095, 0.0])
	local_rpy = selected.get("local_rpy_deg", [-90.0, 0.0, 0.0])
	if len(local_offset) != 3 or len(local_rpy) != 3:
		carb.log_warn(f"Invalid predefined grasp pose format for key '{object_key}'. Using defaults.")
		return default_offset, default_rpy

	carb.log_info(
		f"Using predefined grasp pose for key '{object_key}': offset={local_offset}, rpy_deg={local_rpy}"
	)
	return np.asarray(local_offset, dtype=np.float64), np.asarray(local_rpy, dtype=np.float64)


def _select_predefined_spawn_pose(object_key: str, selected_pose: dict | None = None) -> tuple[float, np.ndarray]:
	default_z = 0.11
	default_rotation = np.asarray((-90.0, 0.0, 180.0), dtype=np.float64)
	selected = selected_pose if selected_pose is not None else _select_predefined_pose_entry(object_key)

	location_z = selected.get("location_z", default_z)
	rotation_deg = selected.get("rotation_deg", list(default_rotation))

	try:
		location_z = float(location_z)
	except Exception:
		carb.log_warn(f"Invalid location_z for key '{object_key}'. Using default {default_z}.")
		location_z = default_z

	if not isinstance(rotation_deg, (list, tuple)) or len(rotation_deg) != 3:
		carb.log_warn(f"Invalid rotation_deg for key '{object_key}'. Using defaults.")
		return location_z, default_rotation

	return location_z, np.asarray(rotation_deg, dtype=np.float64)


def _compute_grasp_pose_from_object(
	stage: Usd.Stage, object_prim_path: str, object_key: str, selected_pose: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	obj_prim = stage.GetPrimAtPath(object_prim_path)
	if not obj_prim.IsValid():
		raise RuntimeError(f"Invalid object prim path: {object_prim_path}")

	obj_m = omni.usd.get_world_transform_matrix(obj_prim)
	obj_q = np.asarray(_matrix_quat_wxyz(obj_m), dtype=np.float64)
	obj_q = obj_q / np.linalg.norm(obj_q)

	grasp_local_offset_m, grasp_local_rpy_deg = _select_predefined_grasp_pose(object_key, selected_pose=selected_pose)
	local_offset = Gf.Vec3d(*grasp_local_offset_m)
	grasp_world = np.asarray(obj_m.Transform(local_offset), dtype=np.float64)

	grasp_local_q = np.asarray(
		euler_angles_to_quat(np.deg2rad(grasp_local_rpy_deg)),
		dtype=np.float64,
	)
	grasp_local_q = grasp_local_q / np.linalg.norm(grasp_local_q)

	grasp_world_q = _quat_mul_wxyz(obj_q, grasp_local_q)
	grasp_world_q = grasp_world_q / np.linalg.norm(grasp_world_q)

	rot = _quat_to_rotmat_wxyz(grasp_world_q)
	approach_world = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
	return grasp_world, grasp_world_q, approach_world


def _quat_angle_rad(q1: list[float], q2: list[float]) -> float:
	n1 = math.sqrt(sum(v * v for v in q1))
	n2 = math.sqrt(sum(v * v for v in q2))
	if n1 < 1e-12 or n2 < 1e-12:
		return 0.0
	q1n = [v / n1 for v in q1]
	q2n = [v / n2 for v in q2]
	dot = abs(sum(a * b for a, b in zip(q1n, q2n)))
	dot = max(-1.0, min(1.0, dot))
	return 2.0 * math.acos(dot)


def _pose_snapshot(stage: Usd.Stage, prim_paths: list[str]) -> dict[str, dict[str, list[float]]]:
	snap = {}
	for path in prim_paths:
		prim = stage.GetPrimAtPath(path)
		if not prim.IsValid():
			continue
		m = omni.usd.get_world_transform_matrix(prim)
		snap[path] = {"t": _matrix_translation(m), "q": _matrix_quat_wxyz(m)}
	return snap


def _pose_delta(prev: dict[str, dict[str, list[float]]], curr: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
	max_pos = 0.0
	max_rot = 0.0
	for path, c in curr.items():
		if path not in prev:
			continue
		p = prev[path]
		dt = math.sqrt(sum((c["t"][i] - p["t"][i]) ** 2 for i in range(3)))
		dr = _quat_angle_rad(p["q"], c["q"])
		max_pos = max(max_pos, dt)
		max_rot = max(max_rot, dr)
	return max_pos, max_rot


def _wait_until_settled(world: World, stage: Usd.Stage, prim_paths: list[str], max_seconds: float, label: str) -> None:
	fps_assume = 60.0
	max_frames = int(max_seconds * fps_assume)
	stable_count = 0
	prev = _pose_snapshot(stage, prim_paths)
	for frame in range(max_frames):
		world.step(render=not ARGS.headless)
		curr = _pose_snapshot(stage, prim_paths)
		max_pos, max_rot = _pose_delta(prev, curr)
		prev = curr

		if max_pos < SETTLE_POS_EPS and max_rot < SETTLE_ROT_EPS_RAD:
			stable_count += 1
		else:
			stable_count = 0

		if frame % 120 == 0:
			carb.log_info(
				f"[Settle:{label}] frame={frame}, max_pos_delta={max_pos:.6f}, max_rot_delta={max_rot:.6f}, stable_count={stable_count}"
			)

		if stable_count >= SETTLE_STABLE_WINDOW_FRAMES:
			carb.log_info(f"[Settle:{label}] stable")
			return

	carb.log_warn(f"[Settle:{label}] timeout; continuing")


def _get_next_dataset_index(dataset_root: Path) -> int:
	if not dataset_root.exists():
		return 0
	max_idx = -1
	for child in dataset_root.iterdir():
		if not child.is_dir():
			continue
		match = re.fullmatch(r"scene_(\d+)", child.name)
		if match:
			max_idx = max(max_idx, int(match.group(1)))
	return max_idx + 1


def _create_dataset_paths(dataset_root: str) -> tuple[Path, Path, Path]:
	root = Path(dataset_root)
	idx = _get_next_dataset_index(root)
	scene_dir = root / f"scene_{idx:03d}"
	scene_dir.mkdir(parents=True, exist_ok=False)
	return scene_dir, scene_dir / JSON_FILENAME, scene_dir / USDA_FILENAME


def _save_objects_json(stage: Usd.Stage, object_paths: list[str], spawned_assets: dict[str, str], save_path: Path, target_object_key: str = "") -> None:
	objects = []
	for path in object_paths:
		prim = stage.GetPrimAtPath(path)
		if not prim.IsValid():
			continue
		m = omni.usd.get_world_transform_matrix(prim)
		mesh_name = prim.GetName().replace("obj_", "")
		mesh_path = os.path.join(OBJ_DIR, f"{mesh_name}.obj")
		objects.append(
			{
				"id": prim.GetName(),
				"prim_path": path,
				"usd_asset_path": spawned_assets.get(path),
				"frame_id": "world",
				"translation_xyz": _matrix_translation(m),
				"rotation_xyzw": _matrix_quat_xyzw(m),
				"moveit_mesh_path": mesh_path,
				"note": "Fill moveit_mesh_path later.",
			}
		)

	data = {
		"target_object": objects[0]["id"] if objects else "",
		"target_object_prim_path": objects[0]["prim_path"] if objects else "",
		"object_root": OBJECT_ROOT,
		"num_objects": len(objects),
		"objects": objects,
	}
	save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
	carb.log_info(f"Saved JSON: {save_path}")


def _save_objects_usda(stage: Usd.Stage, object_paths: list[str], save_path: Path) -> None:
	objects_prim = stage.GetPrimAtPath(OBJECT_ROOT)
	if not objects_prim.IsValid():
		raise RuntimeError(f"Missing object root: {OBJECT_ROOT}")

	dst_stage = Usd.Stage.CreateNew(str(save_path))
	dst_stage.SetDefaultPrim(dst_stage.DefinePrim(OBJECT_ROOT, "Xform"))

	src_layer = stage.GetRootLayer()
	dst_layer = dst_stage.GetRootLayer()
	for path in object_paths:
		Sdf.CopySpec(src_layer, path, dst_layer, path)
	dst_layer.Save()
	carb.log_info(f"Saved USDA: {save_path}")


def _open_gripper(robot: SingleArticulation, controller) -> None:
	q = robot.get_joint_positions()
	if q is None:
		return
	q = np.array(q, dtype=np.float64, copy=True)
	finger_ids = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	for idx in finger_ids:
		q[idx] = 0.04
	controller.apply_action(ArticulationAction(joint_positions=q))


def _close_gripper(robot: SingleArticulation, controller) -> None:
	q = robot.get_joint_positions()
	if q is None:
		return
	q = np.array(q, dtype=np.float64, copy=True)
	finger_ids = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	for idx in finger_ids:
		q[idx] = 0.0
	controller.apply_action(ArticulationAction(joint_positions=q))


def _spawn_object(
	stage: Usd.Stage,
	prim_path: str,
	asset_path: str,
	xyz: tuple[float, float, float],
	rpy_deg: tuple[float, float, float],
) -> None:
	prim = stage.DefinePrim(prim_path, "Xform")
	prim.GetReferences().AddReference(asset_path)
	_update_frames(1)
	_set_prim_transform(prim, xyz, rpy_deg)
	_apply_physics_to_object(prim)


def _disable_rigidbody(prim: Usd.Prim) -> None:
	if prim.HasAPI(UsdPhysics.RigidBodyAPI):
		UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def _check_objects_z_range(stage: Usd.Stage, prim_paths: list[str], z_min: float = 0.0, z_max: float = 0.15) -> bool:
	"""Check if all objects are within the z range [z_min, z_max]. Returns True if all valid."""
	invalid_objects = []
	for path in prim_paths:
		prim = stage.GetPrimAtPath(path)
		if not prim.IsValid():
			continue
		m = omni.usd.get_world_transform_matrix(prim)
		z = m.ExtractTranslation()[2]
		if z < z_min or z > z_max:
			invalid_objects.append((prim.GetName(), z))

	if invalid_objects:
		carb.log_warn(f"Objects outside Z range [{z_min}, {z_max}]:")
		for obj_name, z_val in invalid_objects:
			carb.log_warn(f"  - {obj_name}: z={z_val:.6f}")
		return False
	return True


def main() -> None:
	if ARGS.seed is not None:
		random.seed(ARGS.seed)
		np.random.seed(ARGS.seed)

	scene_usd = pathlib.Path(ARGS.scene_usd).resolve()
	if not scene_usd.is_file():
		raise FileNotFoundError(f"Scene USD not found: {scene_usd}")

	usd_context = omni.usd.get_context()
	if not usd_context.open_stage(str(scene_usd)):
		raise RuntimeError(f"Failed to open stage: {scene_usd}")
	while usd_context.get_stage_loading_status()[2] > 0:
		simulation_app.update()
	_set_viewport_scene_camera(usd_context)

	stage = usd_context.get_stage()
	if stage is None:
		raise RuntimeError("USD stage is not available after open.")

	_ensure_xform(stage, OBJECT_ROOT)
	_clear_children(stage, OBJECT_ROOT)
	_ensure_physics_scene(stage)

	all_assets, ycb_folder = _find_ycb_assets()
	object_asset = _find_object_asset(all_assets)
	other_assets = [p for p in all_assets if p != object_asset]
	if len(other_assets) < ADDITIONAL_OBJECTS:
		raise RuntimeError(f"Not enough non-target YCB assets: {len(other_assets)} < {ADDITIONAL_OBJECTS}")

	carb.log_info(f"YCB folder: {ycb_folder}")
	carb.log_info(f"Target object asset: {object_asset}")
	selected_pose = _select_predefined_pose_entry(OBJECT_KEY)

	object_x = random.uniform(OBJECT_X_MIN, OBJECT_X_MAX)
	object_y = random.uniform(OBJECT_Y_MIN, OBJECT_Y_MAX)
	object_z, object_rot_deg = _select_predefined_spawn_pose(OBJECT_KEY, selected_pose=selected_pose)
	object_xyz = (object_x, object_y, object_z)
	object_rpy_deg = (
		float(object_rot_deg[0]),
		float(object_rot_deg[1]),
		random.uniform(float(object_rot_deg[2]) - OBJECT_RZ_JITTER_DEG, float(object_rot_deg[2]) + OBJECT_RZ_JITTER_DEG),
	)
	object_name = _sanitize_name(object_asset)
	object_prim_path = f"{OBJECT_ROOT}/{object_name}"

	_spawn_object(stage, object_prim_path, object_asset, object_xyz, object_rpy_deg)

	world = World(stage_units_in_meters=1.0)
	robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
	world.reset()
	world.play()

	for _ in range(30):
		world.step(render=not ARGS.headless)

	ik_solver = KinematicsSolver(robot, end_effector_frame_name=ARGS.ee_frame)
	controller = robot.get_articulation_controller()

	_open_gripper(robot, controller)
	for _ in range(20):
		world.step(render=not ARGS.headless)

	target_pos, grasp_quat, approach_world = _compute_grasp_pose_from_object(
		stage, object_prim_path, OBJECT_KEY, selected_pose=selected_pose
	)
	carb.log_info(
		f"Computed grasp pose from target object frame: pos={np.round(target_pos, 4)}, quat(wxyz)={np.round(grasp_quat, 4)}"
	)
	carb.log_info(
		f"Approach(+Y) in world={np.round(approach_world, 4)} (expected near [0, 0, -1] for current object orientation)"
	)

	ik_action, ik_succ = ik_solver.compute_inverse_kinematics(target_position=target_pos, target_orientation=grasp_quat)
	if not ik_succ:
		raise RuntimeError("IK failed for predefined mustard grasp pose.")
	controller.apply_action(ik_action)
	for _ in range(120):
		world.step(render=not ARGS.headless)

	_close_gripper(robot, controller)
	for _ in range(60):
		world.step(render=not ARGS.headless)

	object_prim = stage.GetPrimAtPath(object_prim_path)
	_disable_rigidbody(object_prim)

	selected_others = random.sample(other_assets, ADDITIONAL_OBJECTS)
	spawned_paths = [object_prim_path]
	spawned_assets = {object_prim_path: object_asset}

	for i, asset_path in enumerate(selected_others, start=1):
		base = _sanitize_name(asset_path)
		prim_path = f"{OBJECT_ROOT}/{base}_{i:02d}"
		radius = random.uniform(ADDITIONAL_SPAWN_XY_MIN_RADIUS, ADDITIONAL_SPAWN_XY_MAX_RADIUS)
		theta = random.uniform(0.0, 2.0 * math.pi)
		ox = object_x + radius * math.cos(theta)
		oy = object_y + radius * math.sin(theta)
		oz = object_z + random.uniform(ADDITIONAL_SPAWN_Z_MIN, ADDITIONAL_SPAWN_Z_MAX)
		rx = random.uniform(-25.0, 25.0)
		ry = random.uniform(-25.0, 25.0)
		rz = random.uniform(0.0, 360.0)

		_spawn_object(stage, prim_path, asset_path, (ox, oy, oz), (rx, ry, rz))
		spawned_paths.append(prim_path)
		spawned_assets[prim_path] = asset_path
		_wait_until_settled(
			world,
			stage,
			spawned_paths,
			max_seconds=PER_OBJECT_SETTLE_MAX_SECONDS,
			label=f"spawn_{i:02d}",
		)

	for _ in range(10):
		world.step(render=not ARGS.headless)

	_wait_until_settled(world, stage, spawned_paths, max_seconds=SETTLE_MAX_SECONDS, label="final")

	timeline = omni.timeline.get_timeline_interface()
	timeline.pause()

	# Validate all objects are within Z range before saving
	if not _check_objects_z_range(stage, spawned_paths, z_min=Z_MIN, z_max=Z_MAX):
		carb.log_error(f"Scene validation failed: objects outside Z range [{Z_MIN}, {Z_MAX}]. Skipping save.")
		return

	scene_dir, json_path, usda_path = _create_dataset_paths(ARGS.dataset_root)
	_save_objects_json(stage, spawned_paths, spawned_assets, json_path, target_object_key=OBJECT_KEY)
	_save_objects_usda(stage, spawned_paths, usda_path)
	_capture_scene_camera_image(usd_context, scene_dir / SCENE_IMAGE_FILENAME)

	carb.log_info(f"DONE. Saved dataset at: {scene_dir}")
	carb.log_info("Exiting Isaac Sim after successful save.")


if __name__ == "__main__":
	try:
		main()
	finally:
		simulation_app.close()
