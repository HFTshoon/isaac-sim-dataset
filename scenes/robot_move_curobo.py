"""
01_base_robot_move.py

Loads a pre-built clutter scene produced by 01_base_robot_scene.py, positions the
Franka at the grasp pose (gripper closed on the target object), then moves it linearly
in each Fibonacci-sphere direction while tracking all other object displacements.

Usage:
	./python.sh corl2025/scenes/robot_move_curobo.py \
		--scene 02 --scene-num 001 --num-directions 50 --move-distance 0.15
"""

import argparse
import json
import math
import pathlib
import random
import re
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_FILENAME = "isaac_objects_for_moveit.json"


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
		raise KeyError(f"Unknown scene '{scene_key}'. Available scenes: {available}")
	return scene_info_all[scene_key]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Move Franka along Fibonacci-sphere directions while tracking object displacements."
	)
	parser.add_argument(
		"--scene",
		type=str,
		default=None,
		help="Scene key from scene_info.json, e.g. 01 or 02.",
	)
	parser.add_argument(
		"--scene-num",
		type=str,
		default=None,
		help="Scene index inside the dataset root, e.g. 000 or 001.",
	)
	parser.add_argument(
		"--scene-json",
		type=str,
		default=None,
		help="Explicit path to isaac_objects_for_moveit.json. Overrides --scene and --scene-num.",
	)
	parser.add_argument(
		"--base-usd",
		type=str,
		default=None,
		help="Explicit base USD. If omitted, it is derived from --scene.",
	)
	parser.add_argument("--robot-prim-path", type=str, default="/Franka")
	parser.add_argument("--ee-frame", type=str, default="right_gripper")
	parser.add_argument("--headless", action="store_true")
	parser.add_argument("--num-directions", type=int, default=64)
	parser.add_argument("--min-z", type=float, default=0.0)
	parser.add_argument("--move-distance", type=float, default=0.15)
	parser.add_argument("--waypoints", type=int, default=80)
	parser.add_argument("--steps-per-waypoint", type=int, default=10)
	parser.add_argument("--settle-frames", type=int, default=100)
	parser.add_argument("--return-frames", type=int, default=120)
	parser.add_argument("--output-json", type=str, default=None)
	parser.add_argument("--viewport1-camera", type=str, default="/cameras/sceneCamera")
	return parser.parse_args()


ARGS = _parse_args()
scene_info = _load_scene_info(ARGS.scene) if ARGS.scene else None

if ARGS.scene_json:
	scene_json_path = _resolve_repo_path(ARGS.scene_json)
elif scene_info is not None and ARGS.scene_num is not None:
	try:
		scene_num = int(ARGS.scene_num)
	except ValueError as exc:
		raise ValueError(f"Invalid --scene-num value: {ARGS.scene_num}") from exc
	dataset_root = _resolve_repo_path(str(scene_info.get("dataset_root")))
	scene_json_path = dataset_root / f"scene_{scene_num:03d}" / JSON_FILENAME
else:
	scene_json_path = None

if scene_info is not None and ARGS.base_usd is None:
	ARGS.base_usd = str(_resolve_repo_path(str(scene_info.get("scene_usd"))))

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState as CuroboJointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import (
	UsdHelper,
	get_capsule_attrs,
	get_cube_attrs,
	get_cylinder_attrs,
	get_mesh_attrs,
	get_prim_world_pose,
	get_sphere_attrs,
)
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


OBJECT_ROOT = "/objects"
OBJECT_KEY = "006"
PREDEFINED_GRASP_POSE_JSON = pathlib.Path(__file__).with_name("predefined_grasp_pose.json")
GRIPPER_CLOSE_VALUE = 0.005
GRIPPER_CLOSE_FRAMES = 40


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return np.array([
		w1*w2 - x1*x2 - y1*y2 - z1*z2,
		w1*x2 + x1*w2 + y1*z2 - z1*y2,
		w1*y2 - x1*z2 + y1*w2 + z1*x2,
		w1*z2 + x1*y2 - y1*x2 + z1*w2,
	], dtype=np.float64)


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
	w, x, y, z = q
	return np.array([
		[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
		[2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
		[2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
	], dtype=np.float64)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
	trace = R[0,0] + R[1,1] + R[2,2]
	if trace > 0:
		s = 0.5 / math.sqrt(trace + 1.0)
		return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s], dtype=np.float64)
	elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
		s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
		return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s], dtype=np.float64)
	elif R[1,1] > R[2,2]:
		s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
		return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s], dtype=np.float64)
	else:
		s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
		return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s], dtype=np.float64)


def _pose_to_matrix44(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
	R = _quat_to_rotmat_wxyz(quat_wxyz)
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = pos
	return T


def _matrix_translation(m: Gf.Matrix4d) -> list[float]:
	t = m.ExtractTranslation()
	return [float(t[0]), float(t[1]), float(t[2])]


def _matrix_quat_wxyz(m: Gf.Matrix4d) -> list[float]:
	q = m.ExtractRotation().GetQuat()
	im = q.GetImaginary()
	return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def _matrix_quat_xyzw(m: Gf.Matrix4d) -> list[float]:
	q = m.ExtractRotation().GetQuat()
	im = q.GetImaginary()
	return [float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())]


def _get_prim_pose_wxyz(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
	prim = stage.GetPrimAtPath(prim_path)
	if not prim.IsValid():
		raise RuntimeError(f"Prim not found: {prim_path}")
	m = omni.usd.get_world_transform_matrix(prim)
	pos = np.array(_matrix_translation(m), dtype=np.float64)
	quat = np.array(_matrix_quat_wxyz(m), dtype=np.float64)
	return pos, quat


def _set_prim_pose_wxyz(stage: Usd.Stage, prim_path: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
	prim = stage.GetPrimAtPath(prim_path)
	if not prim.IsValid():
		return
	xform = UsdGeom.Xformable(prim)
	t_op = None
	q_op = None
	for op in xform.GetOrderedXformOps():
		if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and t_op is None:
			t_op = op
		elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and q_op is None:
			q_op = op
	if t_op is None:
		t_op = xform.AddTranslateOp()
	if q_op is None:
		q_op = xform.AddOrientOp()
	t_op.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
	w, x, y, z = quat_wxyz
	q_op.Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def _set_rigidbody_kinematic(prim: Usd.Prim, enabled: bool) -> None:
	if not prim.IsValid():
		return
	rb = UsdPhysics.RigidBodyAPI.Apply(prim)
	rb.CreateRigidBodyEnabledAttr(True)
	rb.CreateKinematicEnabledAttr(bool(enabled))


def _set_ccd_for_prim(prim: Usd.Prim, enabled: bool) -> None:
	if not prim.IsValid():
		return
	# CCD attribute is provided by PhysX schema (not UsdPhysics.RigidBodyAPI) in this Isaac Sim version.
	try:
		physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
		physx_rb.CreateEnableCCDAttr(bool(enabled))
	except Exception:
		# Fallback for schema/API variation.
		attr = prim.GetAttribute("physxRigidBody:enableCCD")
		if not attr or not attr.IsValid():
			attr = prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool)
		if attr and attr.IsValid():
			attr.Set(bool(enabled))


def _zero_velocity(prim: Usd.Prim) -> None:
	for attr_name in ("physics:velocity", "physics:angularVelocity"):
		attr = prim.GetAttribute(attr_name)
		if attr and attr.IsValid():
			attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


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


def _fibonacci_sphere(n: int, min_z: float = 0.0) -> list[np.ndarray]:
	if n <= 0:
		return []

	# Uniform-on-area sampling on spherical cap z in [min_z, 1].
	# For a sphere, area density is uniform in z, so choose z linearly,
	# and distribute azimuth via golden angle.
	min_z = float(np.clip(min_z, -1.0, 1.0))
	golden_angle = math.pi * (3.0 - math.sqrt(5.0))
	result = []
	for i in range(n):
		z = min_z + (1.0 - min_z) * ((i + 0.5) / n)
		r_xy = math.sqrt(max(0.0, 1.0 - z * z))
		theta = i * golden_angle
		x = r_xy * math.cos(theta)
		y = r_xy * math.sin(theta)
		result.append(np.array([x, y, z], dtype=np.float64))
	return result


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


def _select_predefined_grasp_pose(object_key: str) -> tuple[np.ndarray, np.ndarray]:
	default_offset = np.asarray((0.0, -0.095, 0.0), dtype=np.float64)
	default_rpy = np.asarray((-90.0, 0.0, 0.0), dtype=np.float64)
	if not PREDEFINED_GRASP_POSE_JSON.is_file():
		carb.log_warn(f"Predefined grasp file not found: {PREDEFINED_GRASP_POSE_JSON}. Using defaults.")
		return default_offset, default_rpy

	try:
		data = _load_json_loose(PREDEFINED_GRASP_POSE_JSON)
	except Exception as exc:
		carb.log_warn(f"Failed to parse predefined grasp file: {exc}. Using defaults.")
		return default_offset, default_rpy

	keys = [str(k) for k in data.get("key", [])]
	if object_key not in keys:
		carb.log_warn(f"object_key '{object_key}' not in predefined key list {keys}. Using defaults.")
		return default_offset, default_rpy

	poses = data.get(object_key, [])
	if not isinstance(poses, list) or not poses:
		carb.log_warn(f"No predefined grasp poses for key '{object_key}'. Using defaults.")
		return default_offset, default_rpy

	selected = random.choice(poses)
	local_offset = selected.get("local_offset_m", [0.0, -0.095, 0.0])
	local_rpy = selected.get("local_rpy_deg", [-90.0, 0.0, 0.0])
	if len(local_offset) != 3 or len(local_rpy) != 3:
		carb.log_warn(f"Invalid predefined grasp pose format for key '{object_key}'. Using defaults.")
		return default_offset, default_rpy

	carb.log_info(
		f"Using predefined grasp pose for key '{object_key}': offset={local_offset}, rpy_deg={local_rpy}"
	)
	return np.asarray(local_offset, dtype=np.float64), np.asarray(local_rpy, dtype=np.float64)


def _compute_grasp_pose_from_object(stage: Usd.Stage, object_prim_path: str, object_key: str) -> tuple[np.ndarray, np.ndarray]:
	obj_prim = stage.GetPrimAtPath(object_prim_path)
	if not obj_prim.IsValid():
		raise RuntimeError(f"Invalid object prim: {object_prim_path}")

	obj_m = omni.usd.get_world_transform_matrix(obj_prim)
	obj_q = np.array(_matrix_quat_wxyz(obj_m), dtype=np.float64)
	obj_q /= np.linalg.norm(obj_q)

	grasp_local_offset_m, grasp_local_rpy_deg = _select_predefined_grasp_pose(object_key)
	local_offset = Gf.Vec3d(*grasp_local_offset_m)
	grasp_world_pos = np.array(obj_m.Transform(local_offset), dtype=np.float64)

	grasp_local_q = np.array(
		euler_angles_to_quat(np.deg2rad(grasp_local_rpy_deg)),
		dtype=np.float64,
	)
	grasp_local_q /= np.linalg.norm(grasp_local_q)

	grasp_world_q = _quat_mul_wxyz(obj_q, grasp_local_q)
	grasp_world_q /= np.linalg.norm(grasp_world_q)
	return grasp_world_pos, grasp_world_q


def _open_gripper(robot: SingleArticulation, controller, frames: int = None) -> None:
	if frames is None:
			frames = GRIPPER_CLOSE_FRAMES
	
	q_start = robot.get_joint_positions()
	if q_start is None:
		return
	q_start = np.array(q_start, dtype=np.float64, copy=True)
	
	# Find finger indices
	finger_indices = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	if not finger_indices:
		return
	
	# Get current finger positions
	current_values = [q_start[i] for i in finger_indices]
	
	# Gradually open gripper over frames
	for step in range(frames + 1):
		alpha = step / max(frames, 1)
		q = q_start.copy()
		for i in finger_indices:
			q[i] = current_values[finger_indices.index(i)] * (1 - alpha) + 0.04 * alpha
		controller.apply_action(ArticulationAction(joint_positions=q))
		# Stepping handled by caller


def _close_gripper(robot: SingleArticulation, controller, grip_value: float = None, frames: int = None) -> None:
	if grip_value is None:
			grip_value = GRIPPER_CLOSE_VALUE
	if frames is None:
			frames = GRIPPER_CLOSE_FRAMES
	
	q_start = robot.get_joint_positions()
	if q_start is None:
		return
	q_start = np.array(q_start, dtype=np.float64, copy=True)
	
	# Find finger indices
	finger_indices = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	if not finger_indices:
		return
	
	# Get current finger positions
	current_values = [q_start[i] for i in finger_indices]
	
	# Gradually close gripper over frames
	for step in range(frames + 1):
		alpha = step / max(frames, 1)
		q = q_start.copy()
		for i in finger_indices:
			q[i] = current_values[finger_indices.index(i)] * (1 - alpha) + grip_value * alpha
		controller.apply_action(ArticulationAction(joint_positions=q))
		# Small step in simulation for smooth closing
		if step < frames:  # Don't step on last iteration
			pass  # Stepping handled by caller


def _step(world: World, n: int) -> None:
	for _ in range(n):
		world.step(render=not ARGS.headless)


def _apply_viewport_camera(usd_context) -> None:
	if ARGS.headless:
		return
	try:
		from omni.kit.viewport.utility import get_active_viewport_window
		vp = get_active_viewport_window(window_name="Viewport")
		if vp is None:
			return
		stage = usd_context.get_stage()
		if stage and stage.GetPrimAtPath(ARGS.viewport1_camera).IsValid():
			vp.viewport_api.camera_path = ARGS.viewport1_camera
	except Exception as exc:
		carb.log_warn(f"Viewport camera set failed: {exc}")


def _snapshot_objects(stage: Usd.Stage, prim_paths: list[str]) -> list[dict]:
	result = []
	for path in prim_paths:
		prim = stage.GetPrimAtPath(path)
		if not prim.IsValid():
			continue
		m = omni.usd.get_world_transform_matrix(prim)
		result.append({
			"id": prim.GetName(),
			"prim_path": path,
			"translation_xyz": _matrix_translation(m),
			"rotation_xyzw": _matrix_quat_xyzw(m),
		})
	return result


def _reset_object_to_saved(stage: Usd.Stage, obj_data: dict) -> None:
	prim = stage.GetPrimAtPath(obj_data["prim_path"])
	if not prim.IsValid():
		return
	tx, ty, tz = obj_data["translation_xyz"]
	qx, qy, qz, qw = obj_data["rotation_xyzw"]
	xform = UsdGeom.Xformable(prim)
	xform.ClearXformOpOrder()
	t_op = xform.AddTranslateOp()
	q_op = xform.AddOrientOp()
	t_op.Set(Gf.Vec3f(tx, ty, tz))
	q_op.Set(Gf.Quatf(qw, qx, qy, qz))
	_zero_velocity(prim)


def _reload_all_objects(
	stage: Usd.Stage,
	objects_usda_path: Path,
	all_objects_data: list[dict],
	simulation_app,
) -> None:
	"""Delete all object prims and reload them from objects_usda."""
	# Delete all object prims
	for obj_data in all_objects_data:
		prim_path = obj_data.get("prim_path")
		if not prim_path:
			continue
		prim = stage.GetPrimAtPath(prim_path)
		if prim.IsValid():
			stage.RemovePrim(prim_path)
			carb.log_info(f"Deleted prim: {prim_path}")
	
	# Reload objects from USD
	obj_stage = Usd.Stage.Open(str(objects_usda_path))
	if obj_stage is None:
		raise RuntimeError(f"Failed to reopen objects USDA: {objects_usda_path}")
	
	src_layer = obj_stage.GetRootLayer()
	dst_layer = stage.GetRootLayer()
	copy_count = 0
	for obj_data in all_objects_data:
		src_path = obj_data.get("prim_path")
		if not src_path:
			continue
		try:
			Sdf.CopySpec(src_layer, src_path, dst_layer, src_path)
			copy_count += 1
		except Exception as e:
			carb.log_warn(f"Failed to recopy prim {src_path}: {e}")
	
	del obj_stage
	carb.log_info(f"Reloaded {copy_count}/{len(all_objects_data)} object prims.")
	
	# Update stage
	for _ in range(30):
		simulation_app.update()
	


def _update_target_transform(stage: Usd.Stage, target_prim_path: str, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray, obj_in_ee: np.ndarray) -> None:
	T_ee = _pose_to_matrix44(ee_pos, ee_quat_wxyz)
	T_obj_world = T_ee @ obj_in_ee
	new_pos = T_obj_world[:3, 3]
	new_quat = _rotmat_to_quat_wxyz(T_obj_world[:3, :3])
	_set_prim_pose_wxyz(stage, target_prim_path, new_pos, new_quat)


def _is_collision_enabled(prim: Usd.Prim) -> bool:
	if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
		return False
	attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
	if not attr.IsValid():
		return True
	return attr.Get() is not False


def _has_collision_enabled_ancestor(prim: Usd.Prim, root_path: str) -> bool:
	current = prim
	while current.IsValid() and str(current.GetPath()).startswith(root_path):
		if _is_collision_enabled(current):
			return True
		current = current.GetParent()
	return False


def _merge_world_configs(*worlds: WorldConfig) -> WorldConfig:
	merged = {"cuboid": [], "mesh": [], "sphere": [], "cylinder": [], "capsule": []}
	for world_cfg in worlds:
		if world_cfg is None:
			continue
		for key in merged:
			values = getattr(world_cfg, key, None)
			if values:
				merged[key].extend(values)
	return WorldConfig(**merged)


def _extract_collision_enabled_world(
	stage: Usd.Stage,
	root_path: str,
	reference_prim_path: str,
	ignore_substrings: list[str] | None = None,
) -> WorldConfig:
	root_prim = stage.GetPrimAtPath(root_path)
	if not root_prim.IsValid():
		return WorldConfig(cuboid=[], mesh=[], sphere=[], cylinder=[], capsule=[])

	xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
	transform = None
	if reference_prim_path:
		reference_prim = stage.GetPrimAtPath(reference_prim_path)
		if reference_prim.IsValid():
			transform, _ = get_prim_world_pose(xform_cache, reference_prim, inverse=True)

	obstacles = {"cuboid": [], "mesh": [], "sphere": [], "cylinder": [], "capsule": []}
	for prim in Usd.PrimRange(root_prim):
		prim_path = str(prim.GetPath())
		if ignore_substrings and any(token in prim_path for token in ignore_substrings):
			continue
		if not _has_collision_enabled_ancestor(prim, root_path):
			continue
		if prim.IsA(UsdGeom.Mesh):
			obstacle = get_mesh_attrs(prim, cache=xform_cache, transform=transform)
			if obstacle is not None:
				obstacles["mesh"].append(obstacle)
		elif prim.IsA(UsdGeom.Cube):
			obstacles["cuboid"].append(get_cube_attrs(prim, cache=xform_cache, transform=transform))
		elif prim.IsA(UsdGeom.Sphere):
			obstacles["sphere"].append(get_sphere_attrs(prim, cache=xform_cache, transform=transform))
		elif prim.IsA(UsdGeom.Cylinder):
			obstacles["cylinder"].append(get_cylinder_attrs(prim, cache=xform_cache, transform=transform))
		elif prim.IsA(UsdGeom.Capsule):
			obstacles["capsule"].append(get_capsule_attrs(prim, cache=xform_cache, transform=transform))
	return WorldConfig(**obstacles)


def _build_guard_cuboid_from_prim(
	stage: Usd.Stage,
	prim_path: str,
	reference_prim_path: str,
	padding_xyz: tuple[float, float, float],
	min_dims_xyz: tuple[float, float, float] | None = None,
) -> Cuboid | None:
	prim = stage.GetPrimAtPath(prim_path)
	if not prim.IsValid():
		return None
	bbox_cache = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(),
		[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
	)
	bbox = bbox_cache.ComputeWorldBound(prim)
	aligned = bbox.ComputeAlignedRange()
	if aligned.IsEmpty():
		return None

	min_pt = aligned.GetMin()
	max_pt = aligned.GetMax()
	center_world = Gf.Vec3d(
		0.5 * (min_pt[0] + max_pt[0]),
		0.5 * (min_pt[1] + max_pt[1]),
		0.5 * (min_pt[2] + max_pt[2]),
	)
	dims = [
		float(max_pt[0] - min_pt[0]) + float(padding_xyz[0]),
		float(max_pt[1] - min_pt[1]) + float(padding_xyz[1]),
		float(max_pt[2] - min_pt[2]) + float(padding_xyz[2]),
	]
	if min_dims_xyz is not None:
		dims = [max(dims[i], float(min_dims_xyz[i])) for i in range(3)]

	center = center_world
	if reference_prim_path:
		xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
		reference_prim = stage.GetPrimAtPath(reference_prim_path)
		if reference_prim.IsValid():
			inv_ref_tf, _ = get_prim_world_pose(xform_cache, reference_prim, inverse=True)
			center_h = inv_ref_tf @ np.array(
				[float(center_world[0]), float(center_world[1]), float(center_world[2]), 1.0],
				dtype=np.float64,
			)
			center = center_h[:3]

	return Cuboid(
		name=f"{prim_path}_guard",
		pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
		dims=dims,
	)


def _build_env_guard_world(stage: Usd.Stage, reference_prim_path: str) -> WorldConfig:
	guards = []
	board_guard = _build_guard_cuboid_from_prim(
		stage,
		"/env/env/board_0",
		reference_prim_path,
		padding_xyz=(0.03, 0.05, 0.03),
		min_dims_xyz=(0.22, 0.06, 0.03),
	)
	if board_guard is not None:
		guards.append(board_guard)
	return WorldConfig(cuboid=guards)


def _get_end_effector_pose_wxyz(ik_solver: KinematicsSolver) -> tuple[np.ndarray, np.ndarray]:
	ee_pos, ee_rot = ik_solver.compute_end_effector_pose()
	ee_pos = np.array(ee_pos, dtype=np.float64)
	if isinstance(ee_rot, np.ndarray) and ee_rot.ndim == 2:
		ee_quat = _rotmat_to_quat_wxyz(ee_rot)
	else:
		ee_quat = np.array(ee_rot, dtype=np.float64)
	norm = np.linalg.norm(ee_quat)
	if norm > 1e-12:
		ee_quat /= norm
	return ee_pos, ee_quat


def _build_curobo_world(stage: Usd.Stage, robot_prim_path: str, ignore_paths: list[str]) -> WorldConfig:
	usd_help = UsdHelper()
	usd_help.load_stage(stage)
	stage_obstacles = usd_help.get_obstacles_from_stage(
		only_paths=["/"],
		reference_prim_path=robot_prim_path,
		ignore_substring=ignore_paths + ["/env/env"],
	).get_collision_check_world()
	env_collision_world = _extract_collision_enabled_world(
		stage=stage,
		root_path="/env/env",
		reference_prim_path=robot_prim_path,
		ignore_substrings=ignore_paths,
	).get_collision_check_world()
	env_guard_world = _build_env_guard_world(stage, robot_prim_path).get_collision_check_world()
	world_cfg = _merge_world_configs(stage_obstacles, env_collision_world, env_guard_world).get_collision_check_world()
	carb.log_info(
		"CuRobo world prepared: "
		f"other(mesh={len(stage_obstacles.mesh)}, cuboid={len(stage_obstacles.cuboid)}), "
		f"env_collision(mesh={len(env_collision_world.mesh)}, cuboid={len(env_collision_world.cuboid)}), "
		f"env_guard(cuboid={len(env_guard_world.cuboid)}), "
		f"merged(mesh={len(world_cfg.mesh)}, cuboid={len(world_cfg.cuboid)})"
	)
	return world_cfg


def _create_motion_gen(stage: Usd.Stage, robot_prim_path: str, ignore_paths: list[str]) -> MotionGen:
	setup_curobo_logger("warn")
	tensor_args = TensorDeviceType()
	robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
	world_cfg = _build_curobo_world(stage, robot_prim_path, ignore_paths)
	cache = world_cfg.get_cache_dict()
	cache["obb"] = max(cache.get("obb", 0) + 8, 32)
	cache["mesh"] = max(cache.get("mesh", 0) + 16, 64)
	motion_gen_config = MotionGenConfig.load_from_robot_config(
		robot_cfg,
		world_cfg,
		tensor_args,
		collision_checker_type=CollisionCheckerType.MESH,
		num_trajopt_seeds=12,
		num_graph_seeds=12,
		interpolation_dt=0.05,
		collision_cache=cache,
		collision_activation_distance=0.02,
		collision_max_outside_distance=0.05,
		optimize_dt=True,
		trajopt_tsteps=max(32, min(ARGS.waypoints, 80)),
	)
	motion_gen = MotionGen(motion_gen_config)
	motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
	motion_gen.update_world(world_cfg)
	return motion_gen


def _plan_pose_with_curobo(
	motion_gen: MotionGen,
	robot: SingleArticulation,
	ik_solver: KinematicsSolver,
	target_position: np.ndarray,
	target_orientation_wxyz: np.ndarray,
) -> tuple[bool, list[int] | None, object | None, str | None]:
	sim_js = robot.get_joints_state()
	if sim_js is None:
		return False, None, None, "robot joint state is None"
	if np.any(np.isnan(sim_js.positions)):
		return False, None, None, "NaN in robot joint positions"

	tensor_args = motion_gen.tensor_args
	cu_js = CuroboJointState(
		position=tensor_args.to_device(sim_js.positions),
		velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
		acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
		jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
		joint_names=robot.dof_names,
	)
	cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

	ik_action, ik_ok = ik_solver.compute_inverse_kinematics(
		target_position=np.asarray(target_position, dtype=np.float64),
		target_orientation=np.asarray(target_orientation_wxyz, dtype=np.float64),
	)
	if not ik_ok or ik_action is None or ik_action.joint_positions is None:
		return False, None, None, "IK failed for target pose"

	current_q = robot.get_joint_positions()
	if current_q is None:
		return False, None, None, "Failed to read current robot joint positions"

	goal_joint_positions_robot_order = np.asarray(current_q, dtype=np.float64).copy()
	ik_q = np.asarray(ik_action.joint_positions, dtype=np.float64)
	ik_indices = None
	if getattr(ik_action, "joint_indices", None) is not None:
		ik_indices = np.asarray(ik_action.joint_indices, dtype=np.int64)
	if ik_indices is not None and ik_indices.shape[0] == ik_q.shape[0]:
		for local_i, joint_i in enumerate(ik_indices):
			if 0 <= int(joint_i) < goal_joint_positions_robot_order.shape[0]:
				goal_joint_positions_robot_order[int(joint_i)] = ik_q[local_i]
	else:
		arm_joint_ids = [
			i for i, name in enumerate(robot.dof_names)
			if ("finger" not in name.lower()) and ("gripper" not in name.lower())
		]
		if ik_q.shape[0] == goal_joint_positions_robot_order.shape[0]:
			goal_joint_positions_robot_order[:] = ik_q
		elif ik_q.shape[0] == len(arm_joint_ids):
			for local_i, joint_i in enumerate(arm_joint_ids):
				goal_joint_positions_robot_order[joint_i] = ik_q[local_i]
		else:
			return False, None, None, f"IK goal dof mismatch: got {ik_q.shape[0]}"

	name_to_idx = {name: i for i, name in enumerate(robot.dof_names)}
	goal_joint_positions_curobo = []
	for name in motion_gen.kinematics.joint_names:
		if name not in name_to_idx:
			return False, None, None, f"Missing joint mapping for {name}"
		goal_joint_positions_curobo.append(goal_joint_positions_robot_order[name_to_idx[name]])

	q_goal = tensor_args.to_device(np.asarray(goal_joint_positions_curobo, dtype=np.float32)).view(1, -1)
	goal_state = CuroboJointState.from_position(q_goal, joint_names=motion_gen.kinematics.joint_names)
	plan_attempts = [
		("graph+trajopt", MotionGenPlanConfig(enable_graph=True, enable_graph_attempt=2, max_attempts=4, enable_finetune_trajopt=True, time_dilation_factor=0.5)),
		("trajopt-only", MotionGenPlanConfig(enable_graph=False, enable_graph_attempt=None, max_attempts=6, enable_finetune_trajopt=True, time_dilation_factor=0.6)),
	]
	result = None
	last_status = None
	for attempt_name, plan_config in plan_attempts:
		carb.log_info(f"CuRobo planning attempt: {attempt_name}")
		result = motion_gen.plan_single_js(cu_js.unsqueeze(0), goal_state, plan_config)
		last_status = result.status
		if result.success.item():
			break
		carb.log_warn(f"CuRobo {attempt_name} failed: {result.status}")
	if result is None or not result.success.item():
		return False, None, None, f"CuRobo joint-space planning failed: {last_status}"

	cmd_plan = result.get_interpolated_plan()
	cmd_plan = motion_gen.get_full_js(cmd_plan)
	idx_list = []
	common_js_names = []
	for name in robot.dof_names:
		if name in cmd_plan.joint_names:
			idx_list.append(robot.get_dof_index(name))
			common_js_names.append(name)
	cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
	return True, idx_list, cmd_plan, None


def _execute_curobo_plan(
	world: World,
	ik_solver: KinematicsSolver,
	controller,
	idx_list: list[int],
	cmd_plan,
	stage: Usd.Stage | None = None,
	target_prim_path: str | None = None,
	obj_in_ee: np.ndarray | None = None,
) -> np.ndarray:
	for cmd_idx in range(len(cmd_plan.position)):
		cmd_state = cmd_plan[cmd_idx]
		action = ArticulationAction(
			cmd_state.position.cpu().numpy(),
			cmd_state.velocity.cpu().numpy(),
			joint_indices=idx_list,
		)
		controller.apply_action(action)
		for _ in range(max(1, ARGS.steps_per_waypoint)):
			world.step(render=not ARGS.headless)
			if stage is not None and target_prim_path is not None and obj_in_ee is not None:
				ee_pos_step, ee_quat_step = _get_end_effector_pose_wxyz(ik_solver)
				_update_target_transform(stage, target_prim_path, ee_pos_step, ee_quat_step, obj_in_ee)
	ee_pos_final, _ = ik_solver.compute_end_effector_pose(position_only=True)
	return np.array(ee_pos_final, dtype=np.float64)


def _move_pose_with_curobo(
	world: World,
	motion_gen: MotionGen,
	robot: SingleArticulation,
	ik_solver: KinematicsSolver,
	controller,
	target_position: np.ndarray,
	target_orientation_wxyz: np.ndarray,
	stage: Usd.Stage | None = None,
	target_prim_path: str | None = None,
	obj_in_ee: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, str | None]:
	success, idx_list, cmd_plan, failure_reason = _plan_pose_with_curobo(
		motion_gen,
		robot,
		ik_solver,
		target_position,
		target_orientation_wxyz,
	)
	if not success or idx_list is None or cmd_plan is None:
		ee_pos, _ = ik_solver.compute_end_effector_pose(position_only=True)
		return False, np.array(ee_pos, dtype=np.float64), failure_reason
	final_ee_pos = _execute_curobo_plan(
		world,
		ik_solver,
		controller,
		idx_list,
		cmd_plan,
		stage=stage,
		target_prim_path=target_prim_path,
		obj_in_ee=obj_in_ee,
	)
	return True, final_ee_pos, None


def _move_pose_cartesian(
	world: World,
	robot: SingleArticulation,
	ik_solver: KinematicsSolver,
	controller,
	target_position: np.ndarray,
	target_orientation_wxyz: np.ndarray,
	stage: Usd.Stage | None = None,
	target_prim_path: str | None = None,
	obj_in_ee: np.ndarray | None = None,
	num_waypoints: int = 40,
) -> tuple[bool, np.ndarray, str | None]:
	"""Move end effector linearly in Cartesian space using IK at each waypoint."""
	ee_pos_current, ee_quat_current = _get_end_effector_pose_wxyz(ik_solver)
	
	# Create linear interpolation from current to target
	waypoints = []
	for alpha in np.linspace(0.0, 1.0, num_waypoints):
		pos = ee_pos_current + (target_position - ee_pos_current) * alpha
		# Keep orientation constant (slerp would be better, but linear path is simpler)
		waypoints.append((pos, target_orientation_wxyz))
	
	for waypoint_pos, waypoint_quat in waypoints:
		ik_action, ik_ok = ik_solver.compute_inverse_kinematics(
			target_position=np.asarray(waypoint_pos, dtype=np.float64),
			target_orientation=np.asarray(waypoint_quat, dtype=np.float64),
		)
		if not ik_ok or ik_action is None or ik_action.joint_positions is None:
			ee_pos_final, _ = ik_solver.compute_end_effector_pose(position_only=True)
			return False, np.array(ee_pos_final, dtype=np.float64), "IK failed during Cartesian motion"
		
		# Map IK result (arm joints only) to full robot joint positions
		current_q = robot.get_joint_positions()
		if current_q is None:
			ee_pos_final, _ = ik_solver.compute_end_effector_pose(position_only=True)
			return False, np.array(ee_pos_final, dtype=np.float64), "Failed to read robot joint positions"
		
		goal_joint_positions = np.asarray(current_q, dtype=np.float64).copy()
		ik_q = np.asarray(ik_action.joint_positions, dtype=np.float64)
		ik_indices = getattr(ik_action, "joint_indices", None)
		
		if ik_indices is not None:
			ik_indices = np.asarray(ik_indices, dtype=np.int64)
			if ik_indices.shape[0] == ik_q.shape[0]:
				for local_i, joint_i in enumerate(ik_indices):
					if 0 <= int(joint_i) < goal_joint_positions.shape[0]:
						goal_joint_positions[int(joint_i)] = ik_q[local_i]
		else:
			# Assume IK returns arm joints only; map to arm joint indices
			arm_joint_ids = [
				i for i, name in enumerate(robot.dof_names)
				if ("finger" not in name.lower()) and ("gripper" not in name.lower())
			]
			if ik_q.shape[0] == len(arm_joint_ids):
				for local_i, joint_i in enumerate(arm_joint_ids):
					goal_joint_positions[joint_i] = ik_q[local_i]
		
		action = ArticulationAction(goal_joint_positions)
		controller.apply_action(action)
		
		for _ in range(max(1, ARGS.steps_per_waypoint)):
			world.step(render=not ARGS.headless)
			if stage is not None and target_prim_path is not None and obj_in_ee is not None:
				ee_pos_step, ee_quat_step = _get_end_effector_pose_wxyz(ik_solver)
				_update_target_transform(stage, target_prim_path, ee_pos_step, ee_quat_step, obj_in_ee)
	
	ee_pos_final, _ = ik_solver.compute_end_effector_pose(position_only=True)
	return True, np.array(ee_pos_final, dtype=np.float64), None


def main() -> None:
	try:
		# Setup file-based logging (avoid stdout issues with SimulationApp)
		import logging
		import tempfile, os
		log_file = os.path.join(tempfile.gettempdir(), f'robot_move_curobo_{os.getpid()}.log')
		logging.basicConfig(
			level=logging.INFO,
			format='[%(levelname)s] %(message)s',
			handlers=[logging.FileHandler(log_file, mode='w')],
			force=True
		)
		logger = logging.getLogger()
		
		msg = "[01_base_robot_move] Starting..."
		print(msg, flush=True)
		carb.log_info(msg)
		logger.info(msg)

		# Load scene JSON
		if scene_json_path is None:
			raise ValueError("Provide --scene and --scene-num, or an explicit --scene-json path.")
		if not scene_json_path.is_file():
			raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
		carb.log_info(f"Loading scene JSON: {scene_json_path}")
		print(f"[LOAD] Scene JSON: {scene_json_path}", flush=True)
		
		with scene_json_path.open(encoding="utf-8") as f:
			scene_data = json.load(f)
		target_object = scene_data.get("target_object", "")
		target_prim_path = scene_data.get("target_object_prim_path", f"{OBJECT_ROOT}/{target_object}")
		grasp_object_key = _extract_object_key(target_object) or OBJECT_KEY
		object_root = scene_data.get("object_root", OBJECT_ROOT)
		all_objects_data = scene_data.get("objects", [])
		carb.log_info(f"  target_object: {target_object}")
		print(f"  target_object: {target_object}", flush=True)
		carb.log_info(f"  grasp_object_key: {grasp_object_key}")
		carb.log_info(f"  target_prim_path: {target_prim_path}")
		carb.log_info(f"  num_objects: {len(all_objects_data)}")

		objects_usda = scene_json_path.with_name("isaac_objects.usda")
		if not objects_usda.is_file():
			raise FileNotFoundError(f"Objects USDA not found: {objects_usda}")

		# Open base USD
		if ARGS.base_usd is None:
			raise ValueError("Unable to derive --base-usd. Pass --scene or specify --base-usd explicitly.")
		base_usd = Path(ARGS.base_usd).resolve()
		if not base_usd.is_file():
			raise FileNotFoundError(f"Base USD not found: {base_usd}")
		carb.log_info(f"Opening base USD: {base_usd}")
		print(f"[USD] Opening: {base_usd}", flush=True)
		
		usd_context = omni.usd.get_context()
		if not usd_context.open_stage(str(base_usd)):
			raise RuntimeError(f"Failed to open stage: {base_usd}")
		carb.log_info("Waiting for base USD to load...")
		print("[USD] Waiting for load...", flush=True)
		while usd_context.get_stage_loading_status()[2] > 0:
			simulation_app.update()
		carb.log_info("Base USD loaded.")
		print("[USD] Loaded.", flush=True)

		_apply_viewport_camera(usd_context)

		stage = usd_context.get_stage()
		if stage is None:
			raise RuntimeError("USD stage unavailable.")

		# Clear and load objects
		carb.log_info(f"Clearing existing objects under {object_root}...")
		print(f"[CLEAR] Objects under {object_root}", flush=True)
		_clear_children(stage, object_root)
		_ensure_xform(stage, object_root)

		carb.log_info(f"Opening objects USDA: {objects_usda}")
		print(f"[LOAD] Objects: {objects_usda}", flush=True)
		obj_stage = Usd.Stage.Open(str(objects_usda))
		if obj_stage is None:
			raise RuntimeError(f"Failed to open objects USDA: {objects_usda}")

		src_layer = obj_stage.GetRootLayer()
		dst_layer = stage.GetRootLayer()
		carb.log_info(f"Copying {len(all_objects_data)} object prims...")
		print(f"[COPY] Prims: {len(all_objects_data)}", flush=True)
		copy_count = 0
		for i, obj_data in enumerate(all_objects_data):
			src_path = obj_data.get("prim_path")
			if not src_path:
				continue
			try:
				Sdf.CopySpec(src_layer, src_path, dst_layer, src_path)
				copy_count += 1
			except Exception as e:
				carb.log_warn(f"Failed to copy prim {src_path}: {e}")
		carb.log_info(f"Copied {copy_count}/{len(all_objects_data)} object prims.")
		print(f"[COPY] Done: {copy_count}/{len(all_objects_data)}", flush=True)
		del obj_stage

		carb.log_info("Updating stage...")
		print("[STAGE] Updating...", flush=True)
		for _ in range(30):
			simulation_app.update()

		target_prim = stage.GetPrimAtPath(target_prim_path)
		if not target_prim.IsValid():
			carb.log_error(f"Target prim not found: {target_prim_path}")
			raise RuntimeError(f"Target prim not found: {target_prim_path}")
		carb.log_info(f"Target prim validated: {target_prim_path}")
		print(f"[VALID] Target prim: {target_prim_path}", flush=True)

		# World + robot
		carb.log_info("Creating World and adding robot...")
		print("[WORLD] Creating...", flush=True)
		world = World(stage_units_in_meters=1.0)
		robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
		world.reset()
		world.play()
		_step(world, 30)
		carb.log_info("World created.")
		print("[WORLD] Created.", flush=True)

		ik_solver = KinematicsSolver(robot, end_effector_frame_name=ARGS.ee_frame)
		controller = robot.get_articulation_controller()
		
		# Save initial joint positions (before grasp)
		initial_joint_positions = np.array(robot.get_joint_positions(), dtype=np.float64, copy=True)
		carb.log_info(f"Initial joint positions saved: {np.round(initial_joint_positions, 4)}")
		
		ignore_paths = [
			ARGS.robot_prim_path,
			target_prim_path,
			ARGS.viewport1_camera,
			"/background",
			"/env/small_KLT",
			"/World/defaultGroundPlane",
			"/physicsScene",
			"/cameras",
		]
		motion_gen = _create_motion_gen(stage, ARGS.robot_prim_path, ignore_paths)

		# Move to grasp
		carb.log_info("Computing grasp pose...")
		print("[GRASP] Computing...", flush=True)
		grasp_pos, grasp_quat = _compute_grasp_pose_from_object(stage, target_prim_path, grasp_object_key)
		carb.log_info(f"Grasp pos: {np.round(grasp_pos, 4)}")
		print(f"[GRASP] Pos: {np.round(grasp_pos, 4)}", flush=True)

		_open_gripper(robot, controller)
		for _ in range(GRIPPER_CLOSE_FRAMES + 1):
			_step(world, 1)

		grasp_move_ok, _, grasp_failure_reason = _move_pose_with_curobo(
			world=world,
			motion_gen=motion_gen,
			robot=robot,
			ik_solver=ik_solver,
			controller=controller,
			target_position=grasp_pos,
			target_orientation_wxyz=grasp_quat,
		)
		if not grasp_move_ok:
			raise RuntimeError(f"CuRobo failed for initial grasp pose: {grasp_failure_reason}")
		carb.log_info("CuRobo reached initial grasp pose.")
		print("[GRASP] CuRobo grasp pose reached.", flush=True)

		_close_gripper(robot, controller)
		for _ in range(GRIPPER_CLOSE_FRAMES + 1):
			_step(world, 1)
		carb.log_info("Gripper closed.")
		print("[GRIPPER] Closed.", flush=True)

		target_prim = stage.GetPrimAtPath(target_prim_path)
		if target_prim.IsValid():
			# PhysX does not support kinematic + CCD together.
			_set_ccd_for_prim(target_prim, False)
			_set_rigidbody_kinematic(target_prim, True)
		for obj_data in all_objects_data:
			if obj_data.get("prim_path") == target_prim_path:
				continue
			obj_prim = stage.GetPrimAtPath(obj_data.get("prim_path", ""))
			_set_ccd_for_prim(obj_prim, True)
		carb.log_info("Target object set to kinematic (CCD off); CCD enabled for non-target objects.")
		print("[PHYSICS] Target kinematic (CCD off), non-target CCD on.", flush=True)

		# Compute offset
		carb.log_info("Computing EE-to-object offset...")
		ee_pos_init, ee_quat_init = _get_end_effector_pose_wxyz(ik_solver)

		obj_pos_init, obj_quat_init = _get_prim_pose_wxyz(stage, target_prim_path)

		T_ee_init = _pose_to_matrix44(ee_pos_init, ee_quat_init)
		T_obj_init = _pose_to_matrix44(obj_pos_init, obj_quat_init)
		obj_in_ee = np.linalg.inv(T_ee_init) @ T_obj_init

		carb.log_info(f"EE init pos: {np.round(ee_pos_init, 4)}")
		carb.log_info(f"Obj init pos: {np.round(obj_pos_init, 4)}")

		# Prepare
		all_prim_paths = [obj["prim_path"] for obj in all_objects_data]
		initial_objects_snapshot = _snapshot_objects(stage, all_prim_paths)

		fib_directions = _fibonacci_sphere(ARGS.num_directions, min_z=ARGS.min_z)
		carb.log_info(f"Generated {len(fib_directions)} Fibonacci-sphere directions")
		print(f"[FIB] Directions: {len(fib_directions)}", flush=True)

		direction_results = []

		# Motion loop
		for dir_idx, direction in enumerate(fib_directions):
			carb.log_info(f"[{dir_idx + 1}/{len(fib_directions)}] direction={np.round(direction, 4)}")

			# Reset: Move robot to initial pose first
			controller.apply_action(ArticulationAction(joint_positions=initial_joint_positions))
			_step(world, ARGS.return_frames)
			carb.log_info(f"[Dir {dir_idx}] Robot returned to initial pose.")
			
			# Delete and reload all objects
			_reload_all_objects(stage, objects_usda, all_objects_data, simulation_app)
			carb.log_info(f"[Dir {dir_idx}] All objects reloaded.")
			
			# Reinitialize physics for objects
			_step(world, 30)
			
			# Move robot to grasp pose (CuRobo with collision avoidance)
			grasp_move_ok, _, grasp_failure_reason = _move_pose_with_curobo(
				world=world,
				motion_gen=motion_gen,
				robot=robot,
				ik_solver=ik_solver,
				controller=controller,
				target_position=grasp_pos,
				target_orientation_wxyz=grasp_quat,
			)
			if not grasp_move_ok:
				carb.log_warn(f"[Dir {dir_idx}] Failed to move to grasp pose: {grasp_failure_reason}")
				direction_results.append({
					"index": dir_idx,
					"direction": direction.tolist(),
					"ik_success": False,
					"failure_reason": f"Grasp pose motion failed: {grasp_failure_reason}",
					"objects_before": _snapshot_objects(stage, all_prim_paths),
					"objects_after": _snapshot_objects(stage, all_prim_paths),
				})
				continue
			carb.log_info(f"[Dir {dir_idx}] Robot reached grasp pose.")
			
			# Close gripper
			_close_gripper(robot, controller)
			for _ in range(GRIPPER_CLOSE_FRAMES + 1):
				_step(world, 1)
			
			# Prepare target object as kinematic
			target_prim = stage.GetPrimAtPath(target_prim_path)
			if target_prim.IsValid():
				_set_ccd_for_prim(target_prim, False)
				_set_rigidbody_kinematic(target_prim, True)
			ee_pos_grasp, ee_quat_grasp = _get_end_effector_pose_wxyz(ik_solver)
			_update_target_transform(stage, target_prim_path, ee_pos_grasp, ee_quat_grasp, obj_in_ee)
			_step(world, 10)

			objects_before = _snapshot_objects(stage, all_prim_paths)
			target_move_pos = grasp_pos + direction * ARGS.move_distance
			
			# Use Cartesian motion for pushing
			success, final_ee_pos, failure_reason = _move_pose_cartesian(
				world=world,
				robot=robot,
				ik_solver=ik_solver,
				controller=controller,
				target_position=target_move_pos,
				target_orientation_wxyz=grasp_quat,
				stage=stage,
				target_prim_path=target_prim_path,
				obj_in_ee=obj_in_ee,
			)

			if not success:
				carb.log_warn(f"[Dir {dir_idx}] Cartesian move failed: {failure_reason}")
				direction_results.append({
					"index": dir_idx,
					"direction": direction.tolist(),
					"ik_success": False,
					"failure_reason": failure_reason,
					"objects_before": objects_before,
					"objects_after": _snapshot_objects(stage, all_prim_paths),
				})
				continue

			_step(world, ARGS.settle_frames)
			objects_after = _snapshot_objects(stage, all_prim_paths)
			direction_results.append({
				"index": dir_idx,
				"direction": direction.tolist(),
				"ik_success": True,
				"start_position": grasp_pos.tolist(),
				"end_position": final_ee_pos.tolist(),
				"objects_before": objects_before,
				"objects_after": objects_after,
			})
			carb.log_info(f"[Dir {dir_idx}] Done. EE final pos: {np.round(final_ee_pos, 4)}")

		# Save
		carb.log_info("Pausing simulation...")
		timeline = omni.timeline.get_timeline_interface()
		timeline.pause()

		output_path = Path(ARGS.output_json) if ARGS.output_json else scene_json_path.with_name("motion_tracking.json")
		output_data = {
			"scene_json": str(scene_json_path),
			"target_object": target_object,
			"target_object_prim_path": target_prim_path,
			"grasp_position": grasp_pos.tolist(),
			"grasp_orientation_wxyz": grasp_quat.tolist(),
			"num_directions": len(fib_directions),
			"move_distance": ARGS.move_distance,
			"initial_objects": initial_objects_snapshot,
			"directions": direction_results,
		}
		carb.log_info(f"Saving tracking data to: {output_path}")
		print(f"[SAVE] Tracking: {output_path}", flush=True)
		output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
		carb.log_info(f"DONE. Saved tracking data: {output_path}")
		print(f"[DONE] Tracking saved: {output_path}", flush=True)
		carb.log_info("Exiting Isaac Sim after successful move tracking save.")

	except Exception as exc:
		carb.log_error(f"[01_base_robot_move] Fatal error: {exc}")
		import traceback
		carb.log_error(traceback.format_exc())
		raise


if __name__ == "__main__":
	try:
		main()
	finally:
		simulation_app.close()
