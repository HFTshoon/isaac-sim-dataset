"""
01_base_robot_move.py

Loads a pre-built clutter scene produced by 01_base_robot_scene.py, positions the
Franka at the grasp pose (gripper closed on the target object), then moves it linearly
in each Fibonacci-sphere direction while tracking all other object displacements.

Usage:
    ./python.sh corl2025/scenes/01_base_robot_move.py \
        --scene-json corl2025/dataset/01_robot/scene_000/isaac_objects_for_moveit.json \
        --num-directions 50 --move-distance 0.15 --headless
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


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Move Franka along Fibonacci-sphere directions while tracking object displacements."
	)
	parser.add_argument(
		"--scene-json",
		type=str,
		required=True,
		help="Path to isaac_objects_for_moveit.json produced by 01_base_robot_scene.py.",
	)
	parser.add_argument(
		"--base-usd",
		type=str,
		default=str(pathlib.Path(__file__).with_name("01_base.usda")),
		help="Base USD scene containing the Franka robot.",
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


def _update_target_transform(stage: Usd.Stage, target_prim_path: str, ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray, obj_in_ee: np.ndarray) -> None:
	T_ee = _pose_to_matrix44(ee_pos, ee_quat_wxyz)
	T_obj_world = T_ee @ obj_in_ee
	new_pos = T_obj_world[:3, 3]
	new_quat = _rotmat_to_quat_wxyz(T_obj_world[:3, :3])
	_set_prim_pose_wxyz(stage, target_prim_path, new_pos, new_quat)


def _apply_linear_motion(world: World, ik_solver: KinematicsSolver, controller, start_pos: np.ndarray,
	orientation: np.ndarray, direction: np.ndarray, distance: float, waypoints: int, steps_per_waypoint: int,
	stage: Usd.Stage, target_prim_path: str, obj_in_ee: np.ndarray) -> tuple[bool, np.ndarray]:
	for alpha in np.linspace(0.0, 1.0, waypoints + 1)[1:]:
		wp = start_pos + alpha * direction * distance
		action, success = ik_solver.compute_inverse_kinematics(target_position=wp, target_orientation=orientation)
		if not success:
			carb.log_warn(f"IK failed at alpha={alpha:.3f}")
			ee_pos, _ = ik_solver.compute_end_effector_pose(position_only=True)
			return False, np.array(ee_pos, dtype=np.float64)
		controller.apply_action(action)
		for _ in range(steps_per_waypoint):
			world.step(render=not ARGS.headless)
			ee_pos_step, ee_rot_step = ik_solver.compute_end_effector_pose()
			ee_pos_step = np.array(ee_pos_step, dtype=np.float64)
			# Convert rotation matrix to quaternion if needed
			if isinstance(ee_rot_step, np.ndarray) and ee_rot_step.ndim == 2:
				ee_quat_step = _rotmat_to_quat_wxyz(ee_rot_step)
			else:
				ee_quat_step = np.array(ee_rot_step, dtype=np.float64)
			norm = np.linalg.norm(ee_quat_step)
			if norm > 1e-12:
				ee_quat_step /= norm
			_update_target_transform(stage, target_prim_path, ee_pos_step, ee_quat_step, obj_in_ee)
	ee_pos_final, _ = ik_solver.compute_end_effector_pose(position_only=True)
	return True, np.array(ee_pos_final, dtype=np.float64)


def main() -> None:
	try:
		# Setup file-based logging (avoid stdout issues with SimulationApp)
		import logging
		log_file = '/tmp/01_base_robot_move.log'
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
		scene_json_path = Path(ARGS.scene_json).resolve()
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

		# Move to grasp
		carb.log_info("Computing grasp pose...")
		print("[GRASP] Computing...", flush=True)
		grasp_pos, grasp_quat = _compute_grasp_pose_from_object(stage, target_prim_path, grasp_object_key)
		carb.log_info(f"Grasp pos: {np.round(grasp_pos, 4)}")
		print(f"[GRASP] Pos: {np.round(grasp_pos, 4)}", flush=True)

		_open_gripper(robot, controller)
		for _ in range(GRIPPER_CLOSE_FRAMES + 1):
			_step(world, 1)

		ik_action, ik_ok = ik_solver.compute_inverse_kinematics(target_position=grasp_pos, target_orientation=grasp_quat)
		if not ik_ok:
			raise RuntimeError("IK failed for initial grasp pose.")
		carb.log_info("IK solved for grasp pose.")
		print("[IK] Grasp pose solved.", flush=True)
		controller.apply_action(ik_action)
		_step(world, 120)

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
		ee_pos_init, ee_rot_init = ik_solver.compute_end_effector_pose()
		ee_pos_init = np.array(ee_pos_init, dtype=np.float64)
		# Convert rotation matrix to quaternion if needed
		if isinstance(ee_rot_init, np.ndarray) and ee_rot_init.ndim == 2:
			ee_quat_init = _rotmat_to_quat_wxyz(ee_rot_init)
			carb.log_info(f"  Converted rotation matrix to quaternion")
		else:
			ee_quat_init = np.array(ee_rot_init, dtype=np.float64)
		ee_quat_init /= np.linalg.norm(ee_quat_init)

		obj_pos_init, obj_quat_init = _get_prim_pose_wxyz(stage, target_prim_path)

		T_ee_init = _pose_to_matrix44(ee_pos_init, ee_quat_init)
		T_obj_init = _pose_to_matrix44(obj_pos_init, obj_quat_init)
		obj_in_ee = np.linalg.inv(T_ee_init) @ T_obj_init

		carb.log_info(f"EE init pos: {np.round(ee_pos_init, 4)}")
		carb.log_info(f"Obj init pos: {np.round(obj_pos_init, 4)}")

		# Prepare
		all_prim_paths = [obj["prim_path"] for obj in all_objects_data]
		non_target_data = [o for o in all_objects_data if o["prim_path"] != target_prim_path]
		initial_objects_snapshot = _snapshot_objects(stage, all_prim_paths)

		fib_directions = _fibonacci_sphere(ARGS.num_directions, min_z=ARGS.min_z)
		carb.log_info(f"Generated {len(fib_directions)} Fibonacci-sphere directions")
		print(f"[FIB] Directions: {len(fib_directions)}", flush=True)

		direction_results = []

		# Motion loop
		for dir_idx, direction in enumerate(fib_directions):
			carb.log_info(f"[{dir_idx + 1}/{len(fib_directions)}] direction={np.round(direction, 4)}")

			ik_action_reset, ik_ok_reset = ik_solver.compute_inverse_kinematics(target_position=grasp_pos, target_orientation=grasp_quat)
			if not ik_ok_reset:
				carb.log_warn(f"[Dir {dir_idx}] IK failed for grasp-pose reset; skipping.")
				direction_results.append({"index": dir_idx, "direction": direction.tolist(), "ik_success": False})
				continue
			controller.apply_action(ik_action_reset)
			_step(world, ARGS.return_frames)
			_close_gripper(robot, controller)
			for _ in range(GRIPPER_CLOSE_FRAMES + 1):
				_step(world, 1)

			for obj_data in non_target_data:
				_reset_object_to_saved(stage, obj_data)
			_step(world, 15)

			objects_before = _snapshot_objects(stage, all_prim_paths)

			success, final_ee_pos = _apply_linear_motion(world=world, ik_solver=ik_solver, controller=controller,
				start_pos=grasp_pos, orientation=grasp_quat, direction=direction, distance=ARGS.move_distance,
				waypoints=ARGS.waypoints, steps_per_waypoint=ARGS.steps_per_waypoint,
				stage=stage, target_prim_path=target_prim_path, obj_in_ee=obj_in_ee)

			if not success:
				carb.log_warn(f"[Dir {dir_idx}] Linear IK failed")
				direction_results.append({"index": dir_idx, "direction": direction.tolist(), "ik_success": False,
					"objects_before": objects_before, "objects_after": _snapshot_objects(stage, all_prim_paths)})
				continue

			_step(world, ARGS.settle_frames)
			objects_after = _snapshot_objects(stage, all_prim_paths)
			direction_results.append({
				"index": dir_idx, "direction": direction.tolist(), "ik_success": True,
				"start_position": grasp_pos.tolist(), "end_position": final_ee_pos.tolist(),
				"objects_before": objects_before, "objects_after": objects_after,
			})
			carb.log_info(f"[Dir {dir_idx}] Done. EE final pos: {np.round(final_ee_pos, 4)}")

			# Reset non-target objects before moving robot
			carb.log_info(f"[Dir {dir_idx}] Resetting non-target objects...")
			for obj_data in non_target_data:
				_reset_object_to_saved(stage, obj_data)
			_step(world, 15)

			# Move to initial position (above grasp)
			carb.log_info(f"[Dir {dir_idx}] Moving to initial position...")
			print(f"[Dir {dir_idx}] Initial move start", flush=True)
			initial_pos = grasp_pos + np.array([0.0, 0.0, 0.3], dtype=np.float64)
			ik_action_init, ik_ok_init = ik_solver.compute_inverse_kinematics(
				target_position=initial_pos, target_orientation=grasp_quat
			)
			if ik_ok_init:
				controller.apply_action(ik_action_init)
				_step(world, ARGS.return_frames)
				carb.log_info(f"[Dir {dir_idx}] At initial position.")
				
				# Now reset target object and open gripper
				carb.log_info(f"[Dir {dir_idx}] Resetting target object...")
				target_obj_data = next((o for o in all_objects_data if o["prim_path"] == target_prim_path), None)
				if target_obj_data:
					_set_rigidbody_kinematic(target_prim, False)
					_set_ccd_for_prim(target_prim, True)
					_reset_object_to_saved(stage, target_obj_data)
					_step(world, 15)
				_open_gripper(robot, controller)
				for _ in range(GRIPPER_CLOSE_FRAMES + 1):
					_step(world, 1)
				
				# Return to grasp position
				carb.log_info(f"[Dir {dir_idx}] Returning to grasp position...")
				ik_action_back, ik_ok_back = ik_solver.compute_inverse_kinematics(
					target_position=grasp_pos, target_orientation=grasp_quat
				)
				if ik_ok_back:
					controller.apply_action(ik_action_back)
					_step(world, ARGS.return_frames)
					_set_ccd_for_prim(target_prim, False)
					_set_rigidbody_kinematic(target_prim, True)
					_close_gripper(robot, controller)
					for _ in range(GRIPPER_CLOSE_FRAMES + 1):
						_step(world, 1)
					carb.log_info(f"[Dir {dir_idx}] Back to grasp.")
					print(f"[Dir {dir_idx}] Initial move done", flush=True)
				else:
					carb.log_warn(f"[Dir {dir_idx}] IK failed on return to grasp")
			else:
				carb.log_warn(f"[Dir {dir_idx}] IK failed for initial position")

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
