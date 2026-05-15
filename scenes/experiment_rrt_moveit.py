import argparse
import json
import math
import os
import pathlib
import random
import re
import traceback
import platform
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
JSON_FILENAME = "isaac_objects_for_moveit.json"
POSSIBLE_JITTER_FILENAME = "possible_jitter.json"

OBJECT_ROOT = "/objects"
PREDEFINED_GRASP_POSE_JSON = pathlib.Path(__file__).with_name("predefined_grasp_pose.json")
GRIPPER_CLOSE_VALUE = 0.005
GRIPPER_CLOSE_FRAMES = 40
MOVE_SUCCESS_POS_TOL_M = 0.02
TRACKING_POS_TOL_M = 0.02
TRACKING_ROT_TOL_DEG = 15.0
OPEN_SPACE_DESTINATION_X_CANDIDATES = (-0.1, -0.2)
OPEN_SPACE_DESTINATION_Y_CANDIDATES = (-0.4, -0.3, 0.3, 0.4)
OPEN_SPACE_DESTINATION_Z_CANDIDATES = (0.3,)
RESET_FRAMES = 120


def _resolve_repo_path(path_str: str) -> Path:
	path = Path(path_str)
	if path.is_absolute():
		return path.resolve()
	return (REPO_ROOT / path).resolve()


def _configure_ros2_environment() -> None:
	if os.environ.get("ROS_DISTRO") is None:
		candidate_distros = []
		for distro in ("humble", "jazzy"):
			if Path(f"/opt/ros/{distro}").is_dir() or (REPO_ROOT / "exts" / "isaacsim.ros2.bridge" / distro / "lib").is_dir():
				candidate_distros.append(distro)
		if candidate_distros:
			os.environ["ROS_DISTRO"] = candidate_distros[0]
		elif platform.system().lower() == "linux":
			os.environ["ROS_DISTRO"] = "humble"
		else:
			os.environ["ROS_DISTRO"] = "humble"
	if os.environ.get("RMW_IMPLEMENTATION") is None:
		os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
	bridge_lib = REPO_ROOT / "exts" / "isaacsim.ros2.bridge" / os.environ["ROS_DISTRO"] / "lib"
	if bridge_lib.is_dir():
		ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
		bridge_path = str(bridge_lib)
		if bridge_path not in ld_library_path.split(":"):
			os.environ["LD_LIBRARY_PATH"] = f"{ld_library_path}:{bridge_path}" if ld_library_path else bridge_path


def _load_scene_info(scene_key: str) -> dict:
	with SCENE_INFO_PATH.open(encoding="utf-8") as f:
		scene_info_all = json.load(f)
	if scene_key not in scene_info_all:
		available = ", ".join(sorted(scene_info_all.get("key", [])))
		raise KeyError(f"Unknown scene '{scene_key}'. Available scenes: {available}")
	return scene_info_all[scene_key]


def _default_output_path(args: argparse.Namespace) -> Path:
	idx = 0
	if args.scene_num is not None:
		idx = int(args.scene_num)
	dataset_name = "dataset"
	output_group = "rrt_moveit_withobj" if args.include_target else "rrt_moveit"
	if args.jitter:
		output_group += "_jit"
	if scene_info is not None:
		dataset_name = Path(str(scene_info.get("dataset_root", "dataset"))).name
	elif scene_json_path is not None:
		dataset_name = scene_json_path.parent.parent.name
	return WORKSPACE_ROOT / "experiment" / output_group / dataset_name / f"scene_{idx:03d}.json"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Grasp with CuRobo, gate by external MoveIt planning (y/n), then validate destination reach and tracking."
	)
	parser.add_argument("--scene", type=str, default=None, help="Scene key from scene_info.json, e.g. 01 or 02.")
	parser.add_argument("--scene-num", type=str, default=None, help="Scene index in dataset root, e.g. 000 or 001.")
	parser.add_argument("--scene-json", type=str, default=None, help="Explicit path to isaac_objects_for_moveit.json.")
	parser.add_argument("--base-usd", type=str, default=None, help="Explicit base USD path.")
	parser.add_argument("--robot-prim-path", type=str, default="/Franka")
	parser.add_argument("--ee-frame", type=str, default="right_gripper")
	parser.add_argument("--headless", action="store_true")
	parser.add_argument("--steps-per-waypoint", type=int, default=10)
	parser.add_argument("--include-target", action="store_true", help="Attach grasped target collision model for open-space CuRobo planning.")
	parser.add_argument("--jitter", action="store_true", help="Use jittered grasp orientation from possible_jitter.json.")
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

_configure_ros2_environment()

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.graph.core as og
import omni.kit.app
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


ROS2_BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
ROS2_GRAPH_PATH = "/World/ROS2BridgeGraph"


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return np.array([
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
	], dtype=np.float64)


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
	w, x, y, z = q
	return np.array([
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	], dtype=np.float64)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
	trace = R[0, 0] + R[1, 1] + R[2, 2]
	if trace > 0:
		s = 0.5 / math.sqrt(trace + 1.0)
		return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s], dtype=np.float64)
	if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
		s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
		return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s], dtype=np.float64)
	if R[1, 1] > R[2, 2]:
		s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
		return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float64)
	s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
	return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float64)


def _pose_to_matrix44(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
	R = _quat_to_rotmat_wxyz(quat_wxyz)
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = pos
	return T


def _quat_angle_error_deg_wxyz(q_a: np.ndarray, q_b: np.ndarray) -> float:
	q_a_n = np.asarray(q_a, dtype=np.float64)
	q_b_n = np.asarray(q_b, dtype=np.float64)
	q_a_n /= max(np.linalg.norm(q_a_n), 1e-12)
	q_b_n /= max(np.linalg.norm(q_b_n), 1e-12)
	dot = float(np.clip(np.abs(np.dot(q_a_n, q_b_n)), -1.0, 1.0))
	return float(np.degrees(2.0 * np.arccos(dot)))


def _evaluate_target_tracking(
	stage: Usd.Stage,
	target_prim_path: str,
	ee_pos_wxyz: tuple[np.ndarray, np.ndarray],
	obj_in_ee: np.ndarray,
) -> tuple[float | None, float | None, str | None]:
	ee_pos, ee_quat = ee_pos_wxyz
	target_prim = stage.GetPrimAtPath(target_prim_path)
	if not target_prim.IsValid():
		return None, None, "Target prim missing during tracking check"
	actual_obj_pos, actual_obj_quat = _get_prim_pose_wxyz(stage, target_prim_path)
	expected_obj_T = _pose_to_matrix44(ee_pos, ee_quat) @ obj_in_ee
	expected_obj_pos = expected_obj_T[:3, 3]
	expected_obj_quat = _rotmat_to_quat_wxyz(expected_obj_T[:3, :3])
	pos_err = float(np.linalg.norm(actual_obj_pos - expected_obj_pos))
	rot_err_deg = _quat_angle_error_deg_wxyz(actual_obj_quat, expected_obj_quat)
	return pos_err, rot_err_deg, None


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
	return np.array(_matrix_translation(m), dtype=np.float64), np.array(_matrix_quat_wxyz(m), dtype=np.float64)


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
	try:
		physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
		physx_rb.CreateEnableCCDAttr(bool(enabled))
	except Exception:
		attr = prim.GetAttribute("physxRigidBody:enableCCD")
		if not attr or not attr.IsValid():
			attr = prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool)
		if attr and attr.IsValid():
			attr.Set(bool(enabled))


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


def _load_jitter_pose_from_scene(scene_json: Path) -> tuple[np.ndarray, float | None]:
	jitter_path = scene_json.with_name(POSSIBLE_JITTER_FILENAME)
	if not jitter_path.is_file():
		raise FileNotFoundError(f"Jitter requested but file not found: {jitter_path}")
	data = _load_json_loose(jitter_path)
	successful = data.get("successful_angles", [])
	if not isinstance(successful, list) or not successful:
		raise ValueError(f"Jitter requested but no successful_angles in {jitter_path}")
	entry = successful[0]
	rpy = entry.get("jittered_local_rpy_deg")
	if not isinstance(rpy, list) or len(rpy) != 3:
		raise ValueError(f"Invalid jittered_local_rpy_deg in {jitter_path}: {entry}")
	jitter_deg = entry.get("jitter_deg")
	return np.asarray(rpy, dtype=np.float64), (float(jitter_deg) if jitter_deg is not None else None)


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


def _compute_grasp_pose_from_object(
	stage: Usd.Stage,
	object_prim_path: str,
	object_key: str,
	local_rpy_override_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	obj_prim = stage.GetPrimAtPath(object_prim_path)
	if not obj_prim.IsValid():
		raise RuntimeError(f"Invalid object prim: {object_prim_path}")
	obj_m = omni.usd.get_world_transform_matrix(obj_prim)
	obj_q = np.array(_matrix_quat_wxyz(obj_m), dtype=np.float64)
	obj_q /= np.linalg.norm(obj_q)
	grasp_local_offset_m, grasp_local_rpy_deg = _select_predefined_grasp_pose(object_key)
	if local_rpy_override_deg is not None:
		grasp_local_rpy_deg = np.asarray(local_rpy_override_deg, dtype=np.float64)
	local_offset = Gf.Vec3d(*grasp_local_offset_m)
	grasp_world_pos = np.array(obj_m.Transform(local_offset), dtype=np.float64)
	grasp_local_q = np.array(euler_angles_to_quat(np.deg2rad(grasp_local_rpy_deg)), dtype=np.float64)
	grasp_local_q /= np.linalg.norm(grasp_local_q)
	grasp_world_q = _quat_mul_wxyz(obj_q, grasp_local_q)
	grasp_world_q /= np.linalg.norm(grasp_world_q)
	return grasp_world_pos, grasp_world_q


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


def _ensure_ros2_bridge_enabled() -> None:
	app = omni.kit.app.get_app()
	ext_manager = app.get_extension_manager()
	if not ext_manager.is_extension_enabled(ROS2_BRIDGE_EXTENSION):
		if hasattr(ext_manager, "set_extension_enabled_immediate"):
			ext_manager.set_extension_enabled_immediate(ROS2_BRIDGE_EXTENSION, True)
		else:
			ext_manager.set_extension_enabled(ROS2_BRIDGE_EXTENSION, True)


def _wait_for_ros2_bridge_ready(timeout_steps: int = 240) -> None:
	"""Wait until the ROS2 bridge extension is fully enabled and OmniGraph node types are registered."""
	app = omni.kit.app.get_app()
	ext_manager = app.get_extension_manager()
	for _ in range(timeout_steps):
		if ext_manager.is_extension_enabled(ROS2_BRIDGE_EXTENSION):
			return
		simulation_app.update()
	raise RuntimeError(f"ROS2 bridge extension did not become ready: {ROS2_BRIDGE_EXTENSION}")


def _ensure_ros2_action_graph(robot_prim_path: str) -> None:
	"""Create a minimal ROS2 bridge graph for MoveIt2: clock and joint_states."""
	_ensure_ros2_bridge_enabled()
	_wait_for_ros2_bridge_ready()

	stage = omni.usd.get_context().get_stage()
	if stage is not None:
		existing_graph = stage.GetPrimAtPath(ROS2_GRAPH_PATH)
		if existing_graph.IsValid():
			stage.RemovePrim(ROS2_GRAPH_PATH)
		simulation_app.update()

	graph_specs = {
		"graph_path": ROS2_GRAPH_PATH,
		"evaluator_name": "execution",
	}

	edit_ops = {
		og.Controller.Keys.CREATE_NODES: [
			("tick", "omni.graph.action.OnPlaybackTick"),
			("read_sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
			("context", "isaacsim.ros2.bridge.ROS2Context"),
			("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
			("joint_state", "isaacsim.ros2.bridge.ROS2PublishJointState"),
		],
		og.Controller.Keys.CONNECT: [
			("tick.outputs:tick", "clock.inputs:execIn"),
			("tick.outputs:tick", "joint_state.inputs:execIn"),
			("context.outputs:context", "clock.inputs:context"),
			("context.outputs:context", "joint_state.inputs:context"),
			("read_sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
			("read_sim_time.outputs:simulationTime", "joint_state.inputs:timeStamp"),
		],
		og.Controller.Keys.SET_VALUES: [
			("context.inputs:useDomainIDEnvVar", True),
			("context.inputs:domain_id", 0),
			("read_sim_time.inputs:resetOnStop", True),
			("clock.inputs:topicName", "clock"),
			("clock.inputs:nodeNamespace", ""),
			("joint_state.inputs:topicName", "joint_states"),
			("joint_state.inputs:nodeNamespace", ""),
			("joint_state.inputs:targetPrim", robot_prim_path),
		],
	}

	try:
		og.Controller.edit(graph_specs, edit_ops)
	except Exception as exc:
		# One more wait-and-retry in case the extension finished registering on the next update.
		simulation_app.update()
		try:
			og.Controller.edit(graph_specs, edit_ops)
		except Exception as retry_exc:
			raise RuntimeError(f"Failed to create ROS2 bridge graph: {retry_exc}") from retry_exc
	carb.log_info(f"[ROS2Bridge] Enabled {ROS2_BRIDGE_EXTENSION} and created {ROS2_GRAPH_PATH}")


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


def _open_space_destinations() -> list[np.ndarray]:
	dests = []
	for x in OPEN_SPACE_DESTINATION_X_CANDIDATES:
		for y in OPEN_SPACE_DESTINATION_Y_CANDIDATES:
			for z in OPEN_SPACE_DESTINATION_Z_CANDIDATES:
				dests.append(np.array([float(x), float(y), float(z)], dtype=np.float64))
	return dests


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


def _get_current_curobo_joint_state(
	motion_gen: MotionGen,
	robot: SingleArticulation,
) -> tuple[CuroboJointState | None, str | None]:
	sim_js = robot.get_joints_state()
	if sim_js is None:
		return None, "robot joint state is None"
	if np.any(np.isnan(sim_js.positions)):
		return None, "NaN in robot joint positions"
	tensor_args = motion_gen.tensor_args
	cu_js = CuroboJointState(
		position=tensor_args.to_device(sim_js.positions),
		velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
		acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
		jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
		joint_names=robot.dof_names,
	)
	return cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names), None


def _set_planning_world_for_grasp(
	motion_gen: MotionGen,
	stage: Usd.Stage,
	robot_prim_path: str,
	ignore_paths_grasp: list[str],
) -> None:
	motion_gen.detach_object_from_robot()
	world_cfg = _build_curobo_world(stage, robot_prim_path, ignore_paths_grasp)
	motion_gen.update_world(world_cfg)


def _resolve_world_object_name(motion_gen: MotionGen, target_prim_path: str) -> str | None:
	world_model = getattr(motion_gen, "world_model", None)
	if world_model is None:
		return None
	obstacles = getattr(world_model, "objects", []) or []
	for obstacle in obstacles:
		name = getattr(obstacle, "name", None)
		if not name:
			continue
		if name == target_prim_path or name.startswith(target_prim_path + "/") or target_prim_path in name:
			return name
	return None


def _attach_target_for_planning(
	motion_gen: MotionGen,
	stage: Usd.Stage,
	robot: SingleArticulation,
	robot_prim_path: str,
	ignore_paths_with_target: list[str],
	target_prim_path: str,
) -> tuple[bool, str | None]:
	world_cfg = _build_curobo_world(stage, robot_prim_path, ignore_paths_with_target)
	motion_gen.update_world(world_cfg)
	cu_js, state_err = _get_current_curobo_joint_state(motion_gen, robot)
	if cu_js is None:
		return False, f"failed to read current state for target attachment: {state_err}"
	resolved_name = _resolve_world_object_name(motion_gen, target_prim_path)
	if resolved_name is None:
		return False, f"failed to resolve CuRobo world object name for {target_prim_path}"
	try:
		attached = motion_gen.attach_objects_to_robot(
			cu_js.unsqueeze(0),
			[resolved_name],
			surface_sphere_radius=0.005,
		)
	except Exception as exc:
		return False, f"failed to attach target collision object '{resolved_name}': {exc}"
	if not attached:
		return False, f"failed to attach target collision object: {resolved_name}"
	return True, None


def _check_joint_state_validity(motion_gen: MotionGen, joint_state: CuroboJointState) -> tuple[bool, str | None]:
	try:
		valid, status = motion_gen.check_start_state(joint_state)
		status_str = str(status) if status is not None else None
		return bool(valid), status_str
	except Exception as e:
		carb.log_warn(f"Error checking joint state validity: {e}")
		return True, None


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

	start_valid, start_status = _check_joint_state_validity(motion_gen, cu_js)
	goal_valid, goal_status = _check_joint_state_validity(motion_gen, goal_state)
	if (not start_valid) or (not goal_valid):
		invalid_reason_parts = []
		if not start_valid:
			invalid_reason_parts.append(f"start={start_status}")
		if not goal_valid:
			invalid_reason_parts.append(f"goal={goal_status}")
		return False, None, None, "; ".join(invalid_reason_parts)

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


def _wait_for_moveit_confirmation(dest: np.ndarray) -> tuple[bool, str | None]:
	print("\n[MoveItGate] External pymoveit2 planning/execution을 실행하세요.", flush=True)
	print(f"[MoveItGate] Destination candidate: {dest.tolist()}", flush=True)
	print("[MoveItGate] 결과 입력: y(의도 위치 도달 확인) / n(실패) / q(종료)", flush=True)
	while True:
		ans = input("[MoveItGate] 입력 > ").strip().lower()
		if ans == "y":
			return True, None
		if ans == "n":
			return False, "MoveIt planning rejected by user"
		if ans == "q":
			return False, "User requested stop"
		print("[MoveItGate] y / n / q 중 하나를 입력하세요.", flush=True)


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

	jitter_local_rpy_deg = None
	jitter_deg_used = None
	if ARGS.jitter:
		jitter_local_rpy_deg, jitter_deg_used = _load_jitter_pose_from_scene(scene_json_path)

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

	_ensure_ros2_action_graph(ARGS.robot_prim_path)

	world = World(stage_units_in_meters=1.0)
	robot = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
	world.reset()
	world.play()
	_step(world, 30)

	ik_solver = KinematicsSolver(robot, end_effector_frame_name=ARGS.ee_frame)
	controller = robot.get_articulation_controller()
	initial_joint_positions = np.array(robot.get_joint_positions(), dtype=np.float64, copy=True)

	ignore_paths_common = [
		ARGS.robot_prim_path,
		ARGS.viewport1_camera,
		"/background",
		"/env/small_KLT",
		"/World/defaultGroundPlane",
		"/physicsScene",
		"/cameras",
	]
	ignore_paths_grasp = ignore_paths_common + [target_prim_path]
	ignore_paths_with_target = ignore_paths_common
	motion_gen = _create_motion_gen(stage, ARGS.robot_prim_path, ignore_paths_grasp)

	all_prim_paths = [obj["prim_path"] for obj in all_objects_data]
	initial_objects_snapshot = _snapshot_objects(stage, all_prim_paths)
	open_space_destinations = _open_space_destinations()

	grasp_pos = np.zeros(3, dtype=np.float64)
	grasp_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
	sim_result = []

	for dest_idx, open_space_destination in enumerate(open_space_destinations):
		msg = f"[MoveItGate] Destination {dest_idx + 1}/{len(open_space_destinations)} target={open_space_destination.tolist()}"
		carb.log_info(msg)
		print(msg, flush=True)

		controller.apply_action(ArticulationAction(joint_positions=initial_joint_positions))
		_step(world, RESET_FRAMES)
		_reload_objects_from_usda(stage, objects_usda, object_root, all_objects_data)
		_step(world, 30)
		if ARGS.include_target:
			_set_planning_world_for_grasp(motion_gen, stage, ARGS.robot_prim_path, ignore_paths_grasp)

		trial_ik_success = False
		trial_move_success = False
		trial_objects_after = _snapshot_objects(stage, all_prim_paths)

		target_prim = stage.GetPrimAtPath(target_prim_path)
		if not target_prim.IsValid():
			sim_result.append({
				"destination": open_space_destination.tolist(),
				"ik_success": False,
				"move_success": False,
				"failure_reason": "Target prim not found",
				"moveit_gate_accepted": False,
				"objects_after": trial_objects_after,
			})
			continue

		grasp_pos, grasp_quat = _compute_grasp_pose_from_object(
			stage,
			target_prim_path,
			grasp_object_key,
			local_rpy_override_deg=jitter_local_rpy_deg,
		)
		_open_gripper(robot, controller)
		_step(world, GRIPPER_CLOSE_FRAMES + 1)

		grasp_ok, _, grasp_reason = _move_pose_with_curobo(
			world=world,
			motion_gen=motion_gen,
			robot=robot,
			ik_solver=ik_solver,
			controller=controller,
			target_position=grasp_pos,
			target_orientation_wxyz=grasp_quat,
		)
		if not grasp_ok:
			sim_result.append({
				"destination": open_space_destination.tolist(),
				"ik_success": False,
				"move_success": False,
				"failure_reason": grasp_reason,
				"moveit_gate_accepted": False,
				"objects_after": _snapshot_objects(stage, all_prim_paths),
			})
			continue

		_close_gripper(robot, controller)
		_step(world, GRIPPER_CLOSE_FRAMES + 1)

		target_prim = stage.GetPrimAtPath(target_prim_path)
		if target_prim.IsValid():
			_set_ccd_for_prim(target_prim, False)
			_set_rigidbody_kinematic(target_prim, True)
		for obj_data in all_objects_data:
			if obj_data.get("prim_path") == target_prim_path:
				continue
			obj_prim = stage.GetPrimAtPath(obj_data.get("prim_path", ""))
			_set_ccd_for_prim(obj_prim, True)

		ee_pos_init, ee_quat_init = _get_end_effector_pose_wxyz(ik_solver)
		obj_pos_init, obj_quat_init = _get_prim_pose_wxyz(stage, target_prim_path)
		obj_in_ee = np.linalg.inv(_pose_to_matrix44(ee_pos_init, ee_quat_init)) @ _pose_to_matrix44(obj_pos_init, obj_quat_init)

		if ARGS.include_target:
			attach_ok, attach_reason = _attach_target_for_planning(
				motion_gen=motion_gen,
				stage=stage,
				robot=robot,
				robot_prim_path=ARGS.robot_prim_path,
				ignore_paths_with_target=ignore_paths_with_target,
				target_prim_path=target_prim_path,
			)
			if not attach_ok:
				sim_result.append({
					"destination": open_space_destination.tolist(),
					"ik_success": False,
					"move_success": False,
					"failure_reason": attach_reason,
					"moveit_gate_accepted": False,
					"objects_after": _snapshot_objects(stage, all_prim_paths),
				})
				continue

		moveit_ok, moveit_reason = _wait_for_moveit_confirmation(open_space_destination)
		if not moveit_ok:
			sim_result.append({
				"destination": open_space_destination.tolist(),
				"ik_success": False,
				"move_success": False,
				"failure_reason": moveit_reason,
				"moveit_gate_accepted": False,
				"objects_after": _snapshot_objects(stage, all_prim_paths),
			})
			if moveit_reason == "User requested stop":
				break
			continue

		trial_ik_success = True
		ee_pos_now, ee_quat_now = _get_end_effector_pose_wxyz(ik_solver)
		pos_error = float(np.linalg.norm(ee_pos_now - open_space_destination))
		reached_destination = pos_error <= MOVE_SUCCESS_POS_TOL_M
		tracking_pos_err, tracking_rot_err_deg, tracking_reason = _evaluate_target_tracking(
			stage=stage,
			target_prim_path=target_prim_path,
			ee_pos_wxyz=(ee_pos_now, ee_quat_now),
			obj_in_ee=obj_in_ee,
		)
		tracking_ok = (
			tracking_reason is None
			and tracking_pos_err is not None
			and tracking_rot_err_deg is not None
			and tracking_pos_err <= TRACKING_POS_TOL_M
			and tracking_rot_err_deg <= TRACKING_ROT_TOL_DEG
		)
		trial_move_success = bool(reached_destination and tracking_ok)
		trial_objects_after = _snapshot_objects(stage, all_prim_paths)

		failure_reason = None
		if not reached_destination:
			failure_reason = f"Destination tolerance miss ({pos_error:.4f} m)"
		elif not tracking_ok:
			if tracking_reason is not None:
				failure_reason = tracking_reason
			else:
				failure_reason = (
					f"Tracking miss (pos={tracking_pos_err:.4f} m, rot={tracking_rot_err_deg:.2f} deg)"
				)

		sim_result.append({
			"destination": open_space_destination.tolist(),
			"ik_success": trial_ik_success,
			"move_success": trial_move_success,
			"failure_reason": failure_reason,
			"moveit_gate_accepted": True,
			"ee_position_error_to_destination_m": pos_error,
			"destination_reached": reached_destination,
			"tracking_position_error_m": tracking_pos_err,
			"tracking_rotation_error_deg": tracking_rot_err_deg,
			"tracking_success": tracking_ok,
			"objects_after": trial_objects_after,
		})

	if ARGS.output_json:
		output_path = _resolve_repo_path(ARGS.output_json)
	else:
		output_path = _default_output_path(ARGS)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output_data = {
		"scene_json": str(scene_json_path),
		"target_object": target_object,
		"target_object_prim_path": target_prim_path,
		"jitter_enabled": bool(ARGS.jitter),
		"jitter_deg_used": jitter_deg_used,
		"jitter_local_rpy_deg": jitter_local_rpy_deg.tolist() if jitter_local_rpy_deg is not None else None,
		"grasp_position": grasp_pos.tolist(),
		"grasp_orientation_wxyz": grasp_quat.tolist(),
		"initial_objects": initial_objects_snapshot,
		"sim_result": sim_result,
	}
	output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
	print(f"[DONE] Saved experiment result: {output_path}", flush=True)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(f"[ERROR] experiment_rrt_moveit failed: {exc}", flush=True)
		traceback.print_exc()
		raise
	finally:
		simulation_app.close()
