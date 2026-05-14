import argparse
import json
import math
import pathlib
import random
import re
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_FILENAME = "isaac_objects_for_moveit.json"

OBJECT_ROOT = "/objects"
PREDEFINED_GRASP_POSE_JSON = pathlib.Path(__file__).with_name("predefined_grasp_pose.json")
GRIPPER_CLOSE_VALUE = 0.005
GRIPPER_CLOSE_FRAMES = 40
RESET_FRAMES = 120
GRASP_JITTER_MIN_DEG = 10.0
GRASP_JITTER_DEG = 30.0
MAX_GRASP_ATTEMPTS = 5


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
		description="Load existing scene and validate grasp with local yaw jitter."
	)
	parser.add_argument("--scene", type=str, default=None, help="Scene key from scene_info.json, e.g. 01 or 02.")
	parser.add_argument("--scene-num", type=str, default=None, help="Scene index in dataset root, e.g. 000 or 001.")
	parser.add_argument("--scene-json", type=str, default=None, help="Explicit path to isaac_objects_for_moveit.json.")
	parser.add_argument("--base-usd", type=str, default=None, help="Explicit base USD path.")
	parser.add_argument("--robot-prim-path", type=str, default="/Franka")
	parser.add_argument("--ee-frame", type=str, default="right_gripper")
	parser.add_argument("--headless", action="store_true")
	parser.add_argument("--steps-per-waypoint", type=int, default=10)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--viewport1-camera", type=str, default="/cameras/sceneCamera")
	return parser.parse_args()


ARGS = _parse_args()
if ARGS.seed is not None:
	random.seed(ARGS.seed)
	np.random.seed(ARGS.seed)

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
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

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


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return np.array([
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
	], dtype=np.float64)


def _matrix_translation(m: Gf.Matrix4d) -> list[float]:
	t = m.ExtractTranslation()
	return [float(t[0]), float(t[1]), float(t[2])]


def _matrix_quat_wxyz(m: Gf.Matrix4d) -> list[float]:
	q = m.ExtractRotation().GetQuat()
	im = q.GetImaginary()
	return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]


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


def _load_json_loose(path: pathlib.Path) -> dict:
	text = path.read_text(encoding="utf-8")
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
	return np.asarray(local_offset, dtype=np.float64), np.asarray(local_rpy, dtype=np.float64)


def _compute_grasp_pose_from_object_with_jitter(
	stage: Usd.Stage,
	object_prim_path: str,
	object_key: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
	obj_prim = stage.GetPrimAtPath(object_prim_path)
	if not obj_prim.IsValid():
		raise RuntimeError(f"Invalid object prim: {object_prim_path}")
	obj_m = omni.usd.get_world_transform_matrix(obj_prim)
	obj_q = np.array(_matrix_quat_wxyz(obj_m), dtype=np.float64)
	obj_q /= np.linalg.norm(obj_q)

	grasp_local_offset_m, grasp_local_rpy_deg = _select_predefined_grasp_pose(object_key)
	jitter_magnitude = random.uniform(GRASP_JITTER_MIN_DEG, GRASP_JITTER_DEG)
	jitter_deg = jitter_magnitude * random.choice([-1, 1])
	jittered_local_rpy_deg = np.array(grasp_local_rpy_deg, dtype=np.float64)
	jittered_local_rpy_deg[2] += jitter_deg

	local_offset = Gf.Vec3d(*grasp_local_offset_m)
	grasp_world_pos = np.array(obj_m.Transform(local_offset), dtype=np.float64)
	grasp_local_q = np.array(euler_angles_to_quat(np.deg2rad(jittered_local_rpy_deg)), dtype=np.float64)
	grasp_local_q /= np.linalg.norm(grasp_local_q)
	grasp_world_q = _quat_mul_wxyz(obj_q, grasp_local_q)
	grasp_world_q /= np.linalg.norm(grasp_world_q)
	return grasp_world_pos, grasp_world_q, float(jitter_deg), grasp_local_rpy_deg, jittered_local_rpy_deg


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


def _reload_objects_from_usda(
	stage: Usd.Stage,
	objects_usda: Path,
	object_root: str,
	all_objects_data: list[dict],
) -> None:
	_clear_children(stage, object_root)
	_ensure_xform(stage, object_root)
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


def _is_collision_enabled(prim: Usd.Prim) -> bool:
	from pxr import UsdPhysics
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
	return _merge_world_configs(stage_obstacles, env_collision_world, env_guard_world).get_collision_check_world()


def _create_motion_gen(stage: Usd.Stage, robot_prim_path: str, ignore_paths: list[str]) -> MotionGen:
	setup_curobo_logger("error")
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
		trajopt_tsteps=80,
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
		return False, None, None, "IK failed for jittered grasp pose"

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
		arm_joint_ids = [i for i, name in enumerate(robot.dof_names) if ("finger" not in name.lower()) and ("gripper" not in name.lower())]
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
		return False, None, None, f"CuRobo planning failed: {last_status}"
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
	controller,
	idx_list: list[int],
	cmd_plan,
) -> None:
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


def _open_gripper(robot: SingleArticulation, controller, frames: int | None = None) -> None:
	if frames is None:
		frames = GRIPPER_CLOSE_FRAMES
	q_start = robot.get_joint_positions()
	if q_start is None:
		return
	q_start = np.array(q_start, dtype=np.float64, copy=True)
	finger_indices = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	if not finger_indices:
		return
	current_values = [q_start[i] for i in finger_indices]
	for step in range(frames + 1):
		alpha = step / max(frames, 1)
		q = q_start.copy()
		for i in finger_indices:
			q[i] = current_values[finger_indices.index(i)] * (1 - alpha) + 0.04 * alpha
		controller.apply_action(ArticulationAction(joint_positions=q))


def _close_gripper(robot: SingleArticulation, controller, grip_value: float | None = None, frames: int | None = None) -> None:
	if grip_value is None:
		grip_value = GRIPPER_CLOSE_VALUE
	if frames is None:
		frames = GRIPPER_CLOSE_FRAMES
	q_start = robot.get_joint_positions()
	if q_start is None:
		return
	q_start = np.array(q_start, dtype=np.float64, copy=True)
	finger_indices = [i for i, name in enumerate(robot.dof_names) if "finger" in name.lower()]
	if not finger_indices:
		return
	current_values = [q_start[i] for i in finger_indices]
	for step in range(frames + 1):
		alpha = step / max(frames, 1)
		q = q_start.copy()
		for i in finger_indices:
			q[i] = current_values[finger_indices.index(i)] * (1 - alpha) + grip_value * alpha
		controller.apply_action(ArticulationAction(joint_positions=q))
def main() -> None:
	if scene_json_path is None:
		raise ValueError("Provide --scene and --scene-num, or an explicit --scene-json path.")
	if not scene_json_path.is_file():
		raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
	if ARGS.base_usd is None:
		raise ValueError("Unable to derive --base-usd. Pass --scene or specify --base-usd explicitly.")

	with scene_json_path.open(encoding="utf-8") as f:
		scene_data = json.load(f)
	target_object = scene_data.get("target_object", "")
	target_prim_path = scene_data.get("target_object_prim_path", f"{OBJECT_ROOT}/{target_object}")
	grasp_object_key = _extract_object_key(target_object)
	object_root = scene_data.get("object_root", OBJECT_ROOT)
	all_objects_data = scene_data.get("objects", [])
	objects_usda = scene_json_path.with_name("isaac_objects.usda")
	if not objects_usda.is_file():
		raise FileNotFoundError(f"Objects USDA not found: {objects_usda}")

	base_usd = Path(ARGS.base_usd).resolve()
	if not base_usd.is_file():
		raise FileNotFoundError(f"Base USD not found: {base_usd}")

	usd_context = omni.usd.get_context()
	if not usd_context.open_stage(str(base_usd)):
		raise RuntimeError(f"Failed to open stage: {base_usd}")
	while usd_context.get_stage_loading_status()[2] > 0:
		simulation_app.update()
	_apply_viewport_camera(usd_context)
	stage = usd_context.get_stage()
	if stage is None:
		raise RuntimeError("USD stage unavailable.")

	_reload_objects_from_usda(stage, objects_usda, object_root, all_objects_data)

	target_prim = stage.GetPrimAtPath(target_prim_path)
	if not target_prim.IsValid():
		raise RuntimeError(f"Target prim not found: {target_prim_path}")

	world = World(stage_units_in_meters=1.0)
	robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
	world.reset()
	world.play()
	_step(world, 30)

	ik_solver = KinematicsSolver(robot, end_effector_frame_name=ARGS.ee_frame)
	controller = robot.get_articulation_controller()
	initial_joint_positions = np.array(robot.get_joint_positions(), dtype=np.float64, copy=True)

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

	grasp_success = False
	failure_reason = None
	attempts = []
	for attempt_idx in range(MAX_GRASP_ATTEMPTS):
		controller.apply_action(ArticulationAction(joint_positions=initial_joint_positions))
		_step(world, RESET_FRAMES)
		_open_gripper(robot, controller)
		_step(world, GRIPPER_CLOSE_FRAMES + 1)

		grasp_pos, grasp_quat, jitter_deg, base_local_rpy_deg, jittered_local_rpy_deg = _compute_grasp_pose_from_object_with_jitter(
			stage,
			target_prim_path,
			grasp_object_key,
		)
		carb.log_info(
			f"[GraspJitter] attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}, "
			f"base_local_rpy_deg={base_local_rpy_deg.tolist()}, "
			f"jittered_local_rpy_deg={jittered_local_rpy_deg.tolist()}, jitter_deg={jitter_deg:.3f}"
		)
		print(
			f"[GraspJitter] attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}, "
			f"base_local_rpy_deg={base_local_rpy_deg.tolist()}, "
			f"jittered_local_rpy_deg={jittered_local_rpy_deg.tolist()}, jitter_deg={jitter_deg:.3f}",
			flush=True,
		)

		planned, idx_list, cmd_plan, failure_reason = _plan_pose_with_curobo(
			motion_gen=motion_gen,
			robot=robot,
			ik_solver=ik_solver,
			target_position=grasp_pos,
			target_orientation_wxyz=grasp_quat,
		)
		attempt_success = bool(planned and idx_list is not None and cmd_plan is not None)
		attempts.append({
			"success": attempt_success,
			"jitter_deg": float(jitter_deg),
			"base_local_rpy_deg": [float(v) for v in base_local_rpy_deg],
			"jittered_local_rpy_deg": [float(v) for v in jittered_local_rpy_deg],
			"failure_reason": None if attempt_success else failure_reason,
		})
		if attempt_success:
			_execute_curobo_plan(world, controller, idx_list, cmd_plan)
			_close_gripper(robot, controller)
			_step(world, GRIPPER_CLOSE_FRAMES + 1)
			grasp_success = True
			carb.log_info(f"[GraspValidation] success at attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}")
			print(f"[GraspValidation] success at attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}", flush=True)
			break

		carb.log_warn(f"[GraspValidation] failed at attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}: {failure_reason}")
		print(f"[GraspValidation] failed at attempt {attempt_idx + 1}/{MAX_GRASP_ATTEMPTS}: {failure_reason}", flush=True)

	possible_jitter = {
		"max_abs_jitter_deg": float(GRASP_JITTER_DEG),
		"successful_angles": [
			{
				"jitter_deg": float(attempt["jitter_deg"]),
				"base_local_rpy_deg": attempt["base_local_rpy_deg"],
				"jittered_local_rpy_deg": attempt["jittered_local_rpy_deg"],
			}
			for attempt in attempts
			if attempt["success"]
		],
	}
	possible_jitter_path = scene_json_path.parent / "possible_jitter.json"
	possible_jitter_path.write_text(json.dumps(possible_jitter, indent=2), encoding="utf-8")

	print(f"[DONE] Saved possible jitter angles: {possible_jitter_path}", flush=True)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"[ERROR] robot_scene_grasp_aug failed: {exc}", flush=True)
		traceback.print_exc()
		raise
	finally:
		simulation_app.close()
