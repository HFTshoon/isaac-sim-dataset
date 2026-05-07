#!/usr/bin/env python3
"""Load FetchBench scenes into Isaac Sim.

Usage example (from Isaac Sim Python):
	./python.sh load_scene.py \
		--asset-root /path/to/FetchBench-asset \
		--corl-root /path/to/FetchBench-CORL2024 \
		--scene-config benchmark_eval/RigidObjDesk_0 \
		--task-index 0

This script reads:
1) scene yaml from FetchBench-CORL2024 (benchmark selection),
2) asset/task configs from FetchBench-asset/Task,
3) then imports scene/object URDF assets into Isaac Sim and applies
   actor poses from task_init_state.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from legacy.grasp import run_ik_planned_grasp_demo
from legacy.utils_data import (
    extract_import_path,
    read_scene_yaml,
    remap_asset_root,
    resolve_scene_config,
    sanitize_prim_name,
    to_pose,
)
from legacy.utils_math import quat_multiply_xyzw, quat_normalize_xyzw
from legacy.utils_sim import (
    count_descendants,
    find_articulation_root_prim,
    find_robot_hand_prim,
    get_prim_world_pose,
    hide_collision_geometry,
    set_prim_pose,
)
from legacy.utils_urdf import get_urdf_robot_name, top_level_path

def _enable_extension(ext_name: str) -> None:
	import omni.kit.app

	mgr = omni.kit.app.get_app().get_extension_manager()
	try:
		if not mgr.is_extension_enabled(ext_name):
			mgr.set_extension_enabled_immediate(ext_name, True)
	except Exception:
		# Some Isaac Sim distributions do not ship all extension names.
		pass


def _find_urdf_module_and_enable_extension() -> Optional[Any]:
	# This project uses a fixed URDF importer backend.
	_enable_extension("isaacsim.asset.importer.urdf")
	try:
		from isaacsim.asset.importer.urdf import _urdf

		return _urdf
	except Exception:
		return None


def _set_import_config_option(import_config: Any, key: str, value: Any) -> None:
	"""Set URDF ImportConfig option robustly across bindings/versions."""
	setter = f"set_{key}"
	if hasattr(import_config, setter):
		try:
			getattr(import_config, setter)(value)
			return
		except Exception:
			pass
	if hasattr(import_config, key):
		try:
			setattr(import_config, key, value)
		except Exception:
			pass


def _import_urdf(
	stage,
	urdf_file: str,
	destination_path: str,
	fix_base: bool,
):
	# Preferred path: use the native URDF importer extension if available.
	urdf_module = _find_urdf_module_and_enable_extension()
	if urdf_module is not None:
		import omni.kit.commands

		import_config = urdf_module.ImportConfig()
		for key, value in [
			("merge_fixed_joints", False),
			("replace_cylinders_with_capsules", False),
			("convex_decomp", False),
			("self_collision", False),
			("import_inertia_tensor", True),
			("fix_base", fix_base),
			("density", 1000.0),
			("distance_scale", 1.0),
			("make_default_prim", True),
			("parse_mimic", True),
			("collision_from_visuals", False),
		]:
			_set_import_config_option(import_config, key, value)

		try:
			from pxr import Sdf, Usd, UsdGeom

			tmp_dir = os.path.join(tempfile.gettempdir(), "fetchbench_native_urdf")
			os.makedirs(tmp_dir, exist_ok=True)
			usd_file = os.path.join(
				tmp_dir,
				f"{sanitize_prim_name(os.path.basename(urdf_file))}_{uuid.uuid4().hex}.usd",
			)

			actual_path = omni.kit.commands.execute(
				"URDFParseAndImportFile",
				urdf_path=urdf_file,
				import_config=import_config,
				dest_path=usd_file,
			)
			actual_path = extract_import_path(actual_path, "")

			exported_stage = Usd.Stage.Open(usd_file)
			if exported_stage is None:
				raise RuntimeError(f"Failed to open exported URDF USD: {usd_file}")

			exported_roots = [str(p.GetPath()) for p in exported_stage.GetPseudoRoot().GetChildren()]
			candidate_paths: List[str] = []
			robot_name = get_urdf_robot_name(urdf_file)
			if robot_name:
				candidate_paths.append(f"/{robot_name}")
			if actual_path:
				candidate_paths.append(top_level_path(actual_path))
			candidate_paths.extend(exported_roots)

			source_root = ""
			for cand in candidate_paths:
				if not cand:
					continue
				prim = exported_stage.GetPrimAtPath(cand)
				if prim.IsValid():
					source_root = cand
					break

			if source_root:
				if stage.GetPrimAtPath(destination_path).IsValid():
					stage.RemovePrim(destination_path)
				UsdGeom.Xform.Define(stage, destination_path)
				dst_prim = stage.GetPrimAtPath(destination_path)
				dst_prim.GetReferences().AddReference(Sdf.Reference(usd_file, source_root))
				return destination_path
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[WARN] isaacsim.asset.importer.urdf failed for {urdf_file}: {exc}")

	raise RuntimeError(f"Failed to import URDF with native importer: {urdf_file}")


def _debug_dump_scene(
	stage,
	robot_prim_path: str,
	objects_root: str,
	target_object_prim_path: str,
	output_path: str,
) -> None:
	"""Dump visual/collision pose, robot joint positions, EE pose to a JSON file."""
	import json as _json
	from pxr import Usd, UsdGeom, UsdPhysics

	def _world_pose_dict(prim_path: str) -> dict:
		try:
			pos, quat = get_prim_world_pose(stage, prim_path)
			return {"pos_xyz": pos.tolist(), "quat_xyzw": quat.tolist()}
		except Exception as e:
			return {"error": str(e)}

	def _prim_type(prim) -> str:
		return prim.GetTypeName() or "unknown"

	# --- 1. Objects: visual vs collision pose ---
	objects_info = {}
	obj_root = stage.GetPrimAtPath(objects_root)
	if obj_root.IsValid():
		for top_prim in obj_root.GetChildren():
			obj_path = str(top_prim.GetPath())
			entry: dict = {
				"world_pose": _world_pose_dict(obj_path),
				"type": _prim_type(top_prim),
				"visual_meshes": [],
				"collision_prims": [],
			}
			for prim in Usd.PrimRange(top_prim):
				p_path = str(prim.GetPath())
				if prim.IsA(UsdGeom.Mesh):
					is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
					parent = prim.GetParent()
					while not is_collision and parent and parent.IsValid():
						is_collision = parent.HasAPI(UsdPhysics.CollisionAPI)
						parent = parent.GetParent()
					pose_d = _world_pose_dict(p_path)
					if is_collision:
						entry["collision_prims"].append({"path": p_path, **pose_d})
					else:
						entry["visual_meshes"].append({"path": p_path, **pose_d})
			objects_info[obj_path] = entry

	# --- 2. Robot: joint positions + EE pose ---
	robot_info: dict = {}
	try:
		from isaacsim.core.api import World
		from isaacsim.core.api.robots import Robot

		robot_art_root = find_articulation_root_prim(stage, robot_prim_path)
		if robot_art_root:
			world = World(stage_units_in_meters=1.0)
			robot_obj = world.scene.add(Robot(prim_path=robot_art_root, name="debug_franka"))
			world.reset()
			joint_pos = robot_obj.get_joint_positions()
			dof_names = list(robot_obj.dof_names) if robot_obj.dof_names is not None else []
			robot_info["articulation_root"] = robot_art_root
			robot_info["joint_positions"] = dict(zip(dof_names, [float(v) for v in (joint_pos if joint_pos is not None else [])]))
	except Exception as e:
		robot_info["robot_init_error"] = str(e)

	hand_prim_path = find_robot_hand_prim(stage, robot_prim_path)
	robot_info["hand_prim_path"] = hand_prim_path
	if hand_prim_path:
		robot_info["hand_world_pose"] = _world_pose_dict(hand_prim_path)

	# --- 3. Target object summary ---
	target_info = {
		"prim_path": target_object_prim_path,
		"world_pose": _world_pose_dict(target_object_prim_path),
	}

	dump = {
		"objects": objects_info,
		"robot": robot_info,
		"target_object": target_info,
	}

	with open(output_path, "w") as f:
		_json.dump(dump, f, indent=2)
	print(f"[DEBUG] Scene dump written to: {output_path}")


def _set_view_camera_default(task_camera_pose: np.ndarray) -> None:
	try:
		from omni.isaac.core.utils.viewports import set_camera_view
	except Exception:
		return

	# camera pose format: [eye_x, eye_y, eye_z, lookat_x, lookat_y, lookat_z]
	if task_camera_pose.shape[0] == 0:
		return

	cam0 = task_camera_pose[0]
	eye = [float(cam0[0]), float(cam0[1]), float(cam0[2])]
	target = [float(cam0[3]), float(cam0[4]), float(cam0[5])]
	set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


def load_fetchbench_scene(args: argparse.Namespace) -> Dict[str, Any]:
	scene_yaml = resolve_scene_config(args.scene_config, args.corl_root)
	scene_cfg = read_scene_yaml(scene_yaml)

	scene_list = scene_cfg["scene_list"]
	if args.scene_index < 0 or args.scene_index >= len(scene_list):
		raise IndexError(f"scene_index {args.scene_index} out of range for scene_list len {len(scene_list)}")

	task_rel = scene_list[args.scene_index]
	task_dir = os.path.join(args.asset_root, "Task", task_rel)
	if not os.path.isdir(task_dir):
		raise FileNotFoundError(f"Task directory not found: {task_dir}")

	with open(os.path.join(task_dir, "asset_config.json"), "r", encoding="utf-8") as f:
		asset_config = json.load(f)
	task_npz = np.load(os.path.join(task_dir, "task_config.npz"), allow_pickle=True)

	num_tasks = int(task_npz["task_init_state"].shape[0])
	if args.task_index < 0 or args.task_index >= num_tasks:
		raise IndexError(f"task_index {args.task_index} out of range [0, {num_tasks - 1}]")

	actor_states = task_npz["task_init_state"][args.task_index]
	task_obj_index = int(task_npz["task_obj_index"][args.task_index])
	task_obj_label = str(task_npz["task_obj_label"][args.task_index])
	task_camera_pose = task_npz["task_camera_pose"][args.task_index]

	# Isaac Sim app and runtime modules
	try:
		from isaacsim import SimulationApp  # Isaac Sim 4.x
	except ImportError:
		from omni.isaac.kit import SimulationApp  # Older Isaac Sim

	sim_app = SimulationApp({"headless": args.headless})

	_enable_extension("isaacsim.asset.importer.urdf")
	_enable_extension("omni.usd")

	import omni.usd
	from pxr import UsdGeom, UsdLux

	ctx = omni.usd.get_context()
	ctx.new_stage()
	stage = ctx.get_stage()

	UsdGeom.Xform.Define(stage, "/World")
	UsdGeom.Xform.Define(stage, "/World/FetchBench")
	UsdGeom.Xform.Define(stage, "/World/FetchBench/Robot")
	UsdGeom.Xform.Define(stage, "/World/FetchBench/Objects")

	dome = UsdLux.DomeLight.Define(stage, "/World/DefaultDomeLight")
	dome.CreateIntensityAttr(500.0)

	# Actor order in task_init_state: [robot, table, scene, *objects]
	robot_cfg_entry = dict(asset_config["robot_config"])
	robot_cfg_entry["asset_root"] = remap_asset_root(
		robot_cfg_entry["asset_root"], args.asset_root, args.corl_root
	)
	robot_urdf = os.path.join(robot_cfg_entry["asset_root"], robot_cfg_entry["urdf_file"])

	print(f"\n=== Loading FetchBench Robot into Isaac Sim : {robot_urdf} ===")
	robot_prim_path = _import_urdf(
		stage=stage,
		urdf_file=robot_urdf,
		destination_path="/World/FetchBench/Robot",
		fix_base=True,
	)
	robot_pos, robot_quat = to_pose(actor_states[0])
	print(f"[INFO] Robot base pose xyz=({robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f})") # approx (-0.6, 0.0, 0.4)
	set_prim_pose(stage, robot_prim_path, robot_pos, robot_quat)
	if not args.show_collision_geometry:
		hidden_robot = hide_collision_geometry(stage, robot_prim_path)
		if hidden_robot > 0:
			print(f"[INFO] Hidden {hidden_robot} collision prims under robot root")

	scene_cfg_entry = dict(asset_config["scene_config"])
	scene_cfg_entry["asset_root"] = remap_asset_root(
		scene_cfg_entry["asset_root"], args.asset_root, args.corl_root
	)
	scene_urdf = os.path.join(scene_cfg_entry["asset_root"], scene_cfg_entry["urdf_file"])

	print(f"\n=== Loading FetchBench Scene into Isaac Sim : {scene_urdf} ===")
	scene_prim_path = _import_urdf(
		stage=stage,
		urdf_file=scene_urdf,
		destination_path="/World/FetchBench/Scene",
		fix_base=True,
	)
	scene_pos, scene_quat = to_pose(actor_states[2])
	set_prim_pose(stage, scene_prim_path, scene_pos, scene_quat)
	if not args.show_collision_geometry:
		hidden_scene = hide_collision_geometry(stage, scene_prim_path)
		if hidden_scene > 0:
			print(f"[INFO] Hidden {hidden_scene} collision prims under scene root")

	object_cfgs = list(asset_config.get("object_config", []))
	if args.max_objects >= 0:
		object_cfgs = object_cfgs[: args.max_objects]
	expected_states = 3 + len(object_cfgs)
	if actor_states.shape[0] < expected_states:
		raise RuntimeError(
			f"Actor state count {actor_states.shape[0]} smaller than expected {expected_states}"
		)

	loaded_objects = 0
	skipped_floor_near_objects = 0
	loaded_object_prim_paths: List[str] = []
	loaded_object_prim_paths_by_index: Dict[int, str] = {}
	for i, obj_cfg in enumerate(object_cfgs):
		obj_cfg = dict(obj_cfg)
		obj_cfg["asset_root"] = remap_asset_root(obj_cfg["asset_root"], args.asset_root, args.corl_root)
		obj_pos, obj_quat = to_pose(actor_states[3 + i])
		if float(obj_pos[0]) - float(robot_pos[0]) > 1.0:
			skipped_floor_near_objects += 1
			print(f"[INFO] Skipping object {i}: xyz=({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
			continue
		else:
			print(f"[INFO] Importing object {i} at xyz=({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")

		obj_urdf = os.path.join(obj_cfg["asset_root"], obj_cfg["urdf_file"])
		obj_name = sanitize_prim_name(obj_cfg.get("name", f"obj_{i}"))
		obj_prim_dest = f"/World/FetchBench/Objects/{obj_name}"

		try:
			print(f"[INFO] Importing object {i}: {obj_name} from {obj_urdf}")
			obj_prim_path = _import_urdf(
				stage=stage,
				urdf_file=obj_urdf,
				destination_path=obj_prim_dest,
				fix_base=False,
			)
			set_prim_pose(stage, obj_prim_path, obj_pos, obj_quat)
			child_count = count_descendants(stage, obj_prim_path)
			print(f"[INFO] {obj_name} descendants: {child_count}")
			mesh_prim = stage.GetPrimAtPath(f"{obj_prim_path}/mesh")
			if mesh_prim.IsValid():
				mesh_child_count = count_descendants(stage, f"{obj_prim_path}/mesh")
				print(f"[INFO] {obj_name}/mesh descendants: {mesh_child_count}")
			if not args.show_collision_geometry:
				hidden_obj = hide_collision_geometry(stage, obj_prim_path)
				if hidden_obj > 0:
					print(f"[INFO] Hidden {hidden_obj} collision prims under {obj_name}")
			loaded_objects += 1
			loaded_object_prim_paths.append(obj_prim_path)
			loaded_object_prim_paths_by_index[i] = obj_prim_path
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[WARN] Failed to import object {obj_name} from {obj_urdf}: {exc}")

	if asset_config.get("combo_config"):
		print(
			"[WARN] combo_config is not imported by this loader. "
			"Benchmark RigidObj scenes usually do not require it."
		)

	_set_view_camera_default(task_camera_pose)

	print("\n=== FetchBench Scene Loaded In Isaac Sim ===")
	print(f"scene_yaml: {scene_yaml}")
	print(f"task_dir: {task_dir}")
	print(f"scene_index: {args.scene_index}")
	print(f"task_index: {args.task_index}")
	print(f"scene_name: {scene_cfg.get('name', 'unknown')}")
	print(f"robot_prim_path: {robot_prim_path}")
	print(f"robot_base_pos: {robot_pos.tolist()}")
	print(f"robot_base_quat_xyzw: {robot_quat.tolist()}")
	print(f"objects_loaded: {loaded_objects}/{len(object_cfgs)}")
	print(f"target_object_index: {task_obj_index}")
	print(f"target_object_label: {task_obj_label}")

	target_demo_prim_path = None
	if task_obj_index in loaded_object_prim_paths_by_index:
		target_demo_prim_path = loaded_object_prim_paths_by_index[task_obj_index]
	elif loaded_object_prim_paths:
		target_demo_prim_path = loaded_object_prim_paths[0]

	if getattr(args, "debug_dump", False):
		dump_path = getattr(args, "debug_dump_path", "/tmp/fetchbench_debug.json")
		_debug_dump_scene(
			stage=stage,
			robot_prim_path=robot_prim_path,
			objects_root="/World/FetchBench/Objects",
			target_object_prim_path=target_demo_prim_path or "",
			output_path=dump_path,
		)

	if args.run_grasp_demo:
		if target_demo_prim_path is None:
			print("[WARN] Grasp demo requested but no object was loaded.")
		else:
			try:
				run_ik_planned_grasp_demo(
					stage=stage,
					robot_prim_path=robot_prim_path,
					target_object_prim_path=target_demo_prim_path,
					headless=args.headless,
					close_frames=args.grasp_close_frames,
					lift_frames=args.grasp_lift_frames,
					lift_height=args.grasp_lift_height,
					attach_offset=args.grasp_attach_offset,
					pregrasp_offset=args.grasp_pregrasp_offset,
					grasp_offset=args.grasp_final_offset,
					rrt_max_iterations=args.grasp_rrt_max_iterations,
					settle_frames=args.grasp_settle_frames,
					stop_after_stage=args.grasp_stop_after_stage,
					planner_backend=args.grasp_planner_backend,
					curobo_scene_config=args.grasp_curobo_scene_config,
					curobo_task_name=args.grasp_curobo_task_name,
					curobo_conda_env=args.grasp_curobo_conda_env,
					curobo_corl_root=args.grasp_curobo_corl_root,
				)
			except Exception as exc:  # pylint: disable=broad-except
				print(f"[WARN] IK+RRT grasp demo failed: {exc}")

	# Let Isaac Sim process updates and display the stage.
	if args.headless:
		frames = max(1, int(args.run_frames))
		for _ in range(frames):
			sim_app.update()
		sim_app.close()
	else:
		# Keep the GUI window open until the user closes it.
		while sim_app.is_running():
			sim_app.update()

	return {
		"scene_yaml": scene_yaml,
		"task_dir": task_dir,
		"num_tasks": num_tasks,
		"task_index": args.task_index,
		"target_object_index": task_obj_index,
		"target_object_label": task_obj_label,
		"target_demo_prim_path": target_demo_prim_path,
		"objects_loaded": loaded_objects,
		"objects_skipped_near_floor": skipped_floor_near_objects,
		"objects_total": len(object_cfgs),
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Load FetchBench scene/task into Isaac Sim")
	parser.add_argument(
		"--asset-root",
		type=str,
		default="/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-asset",
		help="Path to FetchBench-asset directory",
	)
	parser.add_argument(
		"--corl-root",
		type=str,
		default="/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024",
		help="Path to FetchBench-CORL2024 directory",
	)
	parser.add_argument(
		"--scene-config",
		type=str,
		default="benchmark_eval/RigidObjDesk_0",
		help=(
			"Scene yaml path or shorthand (e.g., benchmark_eval/RigidObjDesk_0) "
			"from FetchBench-CORL2024/InfiniGym/isaacgymenvs/config/scene"
		),
	)
	parser.add_argument("--scene-index", type=int, default=0, help="Index in scene_list")
	parser.add_argument("--task-index", type=int, default=0, help="Index in task_config.npz")
	parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
	parser.add_argument(
		"--run-frames",
		type=int,
		default=120,
		help="Number of Isaac Sim update frames to run after import",
	)
	parser.add_argument(
		"--max-objects",
		type=int,
		default=-1,
		help="Maximum number of objects to import from object_config (-1 imports all)",
	)
	parser.add_argument(
		"--show-collision-geometry",
		action="store_true",
		help="Keep collision meshes visible in the viewport",
	)
	parser.add_argument(
		"--debug-dump",
		action="store_true",
		help="Dump visual/collision prim poses and robot joint positions to JSON for debugging",
	)
	parser.add_argument(
		"--debug-dump-path",
		type=str,
		default="/isaac-sim/fetchbench_debug.json",
		help="Output path for debug dump JSON (default: /isaac-sim/fetchbench_debug.json)",
	)
	parser.add_argument(
		"--run-grasp-demo",
		action="store_true",
		help="Run grasp demo with IK + RRT planning (pregrasp -> grasp -> lift)",
	)
	parser.add_argument(
		"--grasp-close-frames",
		type=int,
		default=60,
		help="Number of hold frames after gripper close",
	)
	parser.add_argument(
		"--grasp-lift-frames",
		type=int,
		default=120,
		help="Number of hold frames after lift planning",
	)
	parser.add_argument(
		"--grasp-lift-height",
		type=float,
		default=0.15,
		help="Lift height in meters",
	)
	parser.add_argument(
		"--grasp-attach-offset",
		type=float,
		default=0.06,
		help="Attached object's z offset below end-effector while carrying",
	)
	parser.add_argument(
		"--grasp-pregrasp-offset",
		type=float,
		default=0.12,
		help="Pregrasp z offset above target object",
	)
	parser.add_argument(
		"--grasp-final-offset",
		type=float,
		default=0.02,
		help="Final grasp z offset above target object",
	)
	parser.add_argument(
		"--grasp-rrt-max-iterations",
		type=int,
		default=5000,
		help="Maximum RRT iterations per stage",
	)
	parser.add_argument(
		"--grasp-settle-frames",
		type=int,
		default=20,
		help="Settle frames after each IK/RRT stage",
	)
	parser.add_argument(
		"--grasp-stop-after-stage",
		type=str,
		choices=["pregrasp", "grasp", "lift"],
		default="lift",
		help="Stop grasp demo after this stage (default: lift for full demo)",
	)
	parser.add_argument(
		"--grasp-planner-backend",
		type=str,
		choices=["lula", "curobo", "curobo_external"],
		default="lula",
		help="Planner backend for grasp demo: lula, curobo (in-stage), or curobo_external (FetchBench eval.py)",
	)
	parser.add_argument(
		"--grasp-curobo-scene-config",
		type=str,
		default="benchmark_eval/RigidObjDesk_0",
		help="Scene config for external CuRobo backend, e.g. benchmark_eval/RigidObjDesk_0",
	)
	parser.add_argument(
		"--grasp-curobo-task-name",
		type=str,
		default="FetchMeshCurobo",
		help="Hydra task name for external CuRobo backend",
	)
	parser.add_argument(
		"--grasp-curobo-conda-env",
		type=str,
		default="FetchBench",
		help="Conda env name for external CuRobo backend",
	)
	parser.add_argument(
		"--grasp-curobo-corl-root",
		type=str,
		default="/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024",
		help="FetchBench-CORL2024 root path for external CuRobo backend",
	)
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	args.asset_root = os.path.abspath(args.asset_root)
	args.corl_root = os.path.abspath(args.corl_root)

	info = load_fetchbench_scene(args)
	print("Load complete.")
	print(json.dumps(info, indent=2))


if __name__ == "__main__":
	main()
