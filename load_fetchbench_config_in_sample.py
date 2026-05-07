#!/usr/bin/env python3
"""Add FetchBench scene prim to an existing USD stage.

This script reads FetchBench scene/task config like load_fetchbench.py, but only
imports the scene (no robot) to /env/env by default.

The scene transform is computed from actor_states with robot-relative alignment:
T_sample_scene = T_sample_robot * inv(T_fb_robot) * T_fb_scene

This keeps the FetchBench scene in the correct pose relative to the robot that
already exists in sample.usda.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils_data import (
	extract_import_path,
	read_scene_yaml,
	remap_asset_root,
	resolve_scene_config,
	sanitize_prim_name,
	to_pose,
)
from utils_sim import get_prim_world_pose, set_prim_pose


def _enable_extension(ext_name: str) -> None:
	import omni.kit.app

	mgr = omni.kit.app.get_app().get_extension_manager()
	try:
		if not mgr.is_extension_enabled(ext_name):
			mgr.set_extension_enabled_immediate(ext_name, True)
	except Exception:
		pass


def _find_urdf_module_and_enable_extension() -> Optional[Any]:
	_enable_extension("isaacsim.asset.importer.urdf")
	try:
		from isaacsim.asset.importer.urdf import _urdf

		return _urdf
	except Exception:
		return None


def _set_import_config_option(import_config: Any, key: str, value: Any) -> None:
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


def _import_urdf(stage, urdf_file: str, destination_path: str, fix_base: bool) -> str:
	urdf_module = _find_urdf_module_and_enable_extension()
	if urdf_module is None:
		raise RuntimeError("URDF importer module is unavailable.")

	import omni.kit.commands
	from pxr import Sdf, Usd, UsdGeom

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
	if actual_path:
		candidate_paths.append(actual_path if actual_path.startswith("/") else f"/{actual_path}")
	candidate_paths.extend(exported_roots)

	source_root = ""
	for cand in candidate_paths:
		prim = exported_stage.GetPrimAtPath(cand)
		if prim.IsValid():
			source_root = cand
			break

	if not source_root:
		raise RuntimeError(f"Unable to find a valid root in exported URDF USD: {usd_file}")

	if stage.GetPrimAtPath(destination_path).IsValid():
		stage.RemovePrim(destination_path)
	UsdGeom.Xform.Define(stage, destination_path)
	dst_prim = stage.GetPrimAtPath(destination_path)
	dst_prim.GetReferences().AddReference(Sdf.Reference(usd_file, source_root))
	return destination_path


def _quat_xyzw_to_rot_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
	x, y, z, w = [float(v) for v in quat_xyzw]
	n = math.sqrt(x * x + y * y + z * z + w * w)
	if n <= 1e-12:
		x, y, z, w = 0.0, 0.0, 0.0, 1.0
	else:
		x, y, z, w = x / n, y / n, z / n, w / n

	return np.array(
		[
			[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
			[2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
			[2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
		],
		dtype=np.float64,
	)


def _rot_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
	tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
	if tr > 0.0:
		s = math.sqrt(tr + 1.0) * 2.0
		w = 0.25 * s
		x = (rot[2, 1] - rot[1, 2]) / s
		y = (rot[0, 2] - rot[2, 0]) / s
		z = (rot[1, 0] - rot[0, 1]) / s
	elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
		s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
		w = (rot[2, 1] - rot[1, 2]) / s
		x = 0.25 * s
		y = (rot[0, 1] + rot[1, 0]) / s
		z = (rot[0, 2] + rot[2, 0]) / s
	elif rot[1, 1] > rot[2, 2]:
		s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
		w = (rot[0, 2] - rot[2, 0]) / s
		x = (rot[0, 1] + rot[1, 0]) / s
		y = 0.25 * s
		z = (rot[1, 2] + rot[2, 1]) / s
	else:
		s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
		w = (rot[1, 0] - rot[0, 1]) / s
		x = (rot[0, 2] + rot[2, 0]) / s
		y = (rot[1, 2] + rot[2, 1]) / s
		z = 0.25 * s

	quat = np.array([x, y, z, w], dtype=np.float64)
	n = np.linalg.norm(quat)
	if n <= 1e-12:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
	return quat / n


def _pose_to_matrix(pos_xyz: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
	mat = np.eye(4, dtype=np.float64)
	mat[:3, :3] = _quat_xyzw_to_rot_matrix(quat_xyzw)
	mat[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
	return mat


def _matrix_to_pose(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	pos = np.asarray(mat[:3, 3], dtype=np.float64)
	quat = _rot_matrix_to_quat_xyzw(np.asarray(mat[:3, :3], dtype=np.float64))
	return pos, quat


def _load_fetchbench_task(args: argparse.Namespace) -> Dict[str, Any]:
	scene_yaml = resolve_scene_config(args.scene_config, args.corl_root)
	scene_cfg = read_scene_yaml(scene_yaml)

	scene_list = scene_cfg["scene_list"]
	if args.scene_index < 0 or args.scene_index >= len(scene_list):
		raise IndexError(f"scene_index {args.scene_index} out of range [0, {len(scene_list) - 1}]")

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
	if actor_states.shape[0] < 3:
		raise RuntimeError(f"Unexpected actor_states shape: {actor_states.shape}")

	return {
		"scene_yaml": scene_yaml,
		"task_dir": task_dir,
		"asset_config": asset_config,
		"actor_states": actor_states,
	}


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Add FetchBench scene to /env/env in a USD stage")
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
		required=True,
		help="FetchBench scene config, e.g. benchmark_eval/RigidObjDesk_0",
	)
	parser.add_argument("--scene-index", type=int, default=0, help="Index in scene_list")
	parser.add_argument("--task-index", type=int, default=0, help="Index in task_config.npz")
	parser.add_argument(
		"--sample-usda",
		type=str,
		default="/isaac-sim/corl2025/sample.usda",
		help="Input USD stage path that already contains the robot",
	)
	parser.add_argument(
		"--output-usda",
		type=str,
		default="",
		help="Output USD path (default: overwrite --sample-usda)",
	)
	parser.add_argument(
		"--robot-prim-path",
		type=str,
		default="/Franka",
		help="Robot prim path in sample stage used for relative alignment",
	)
	parser.add_argument(
		"--scene-prim-path",
		type=str,
		default="/env/env",
		help="Destination prim path for imported scene",
	)
	parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
	parser.add_argument(
		"--run-frames",
		type=int,
		default=120,
		help="Number of update frames to run in headless mode before exit",
	)
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	args.asset_root = os.path.abspath(args.asset_root)
	args.corl_root = os.path.abspath(args.corl_root)
	args.sample_usda = os.path.abspath(args.sample_usda)
	out_usda = os.path.abspath(args.output_usda) if args.output_usda else args.sample_usda

	if not os.path.isfile(args.sample_usda):
		raise FileNotFoundError(f"sample USD not found: {args.sample_usda}")

	task_info = _load_fetchbench_task(args)
	asset_config = task_info["asset_config"]
	actor_states = task_info["actor_states"]

	scene_cfg_entry = dict(asset_config["scene_config"])
	scene_cfg_entry["asset_root"] = remap_asset_root(
		scene_cfg_entry["asset_root"], args.asset_root, args.corl_root
	)
	scene_urdf = os.path.join(scene_cfg_entry["asset_root"], scene_cfg_entry["urdf_file"])
	if not os.path.isfile(scene_urdf):
		raise FileNotFoundError(f"Scene URDF not found: {scene_urdf}")

	try:
		from isaacsim import SimulationApp
	except ImportError:
		from omni.isaac.kit import SimulationApp

	sim_app = SimulationApp({"headless": args.headless})
	try:
		_enable_extension("isaacsim.asset.importer.urdf")
		_enable_extension("omni.usd")

		import omni.usd
		from pxr import UsdGeom

		ctx = omni.usd.get_context()
		ok = ctx.open_stage(args.sample_usda)
		if not ok:
			raise RuntimeError(f"Failed to open stage: {args.sample_usda}")

		for _ in range(5):
			sim_app.update()

		stage = ctx.get_stage()
		if stage is None:
			raise RuntimeError("USD stage is unavailable after open_stage")

		if not stage.GetPrimAtPath("/env").IsValid():
			UsdGeom.Xform.Define(stage, "/env")

		scene_prim_path = _import_urdf(
			stage=stage,
			urdf_file=scene_urdf,
			destination_path=args.scene_prim_path,
			fix_base=True,
		)

		fb_robot_pos, fb_robot_quat = to_pose(actor_states[0])
		fb_scene_pos, fb_scene_quat = to_pose(actor_states[2])

		sample_robot_pos, sample_robot_quat = get_prim_world_pose(stage, args.robot_prim_path)

		t_fb_robot = _pose_to_matrix(fb_robot_pos, fb_robot_quat)
		t_fb_scene = _pose_to_matrix(fb_scene_pos, fb_scene_quat)
		t_sample_robot = _pose_to_matrix(sample_robot_pos, sample_robot_quat)

		t_robot_to_scene = np.linalg.inv(t_fb_robot) @ t_fb_scene
		t_sample_scene = t_sample_robot @ t_robot_to_scene

		sample_scene_pos, sample_scene_quat = _matrix_to_pose(t_sample_scene)
		set_prim_pose(stage, scene_prim_path, sample_scene_pos, sample_scene_quat)

		for _ in range(3):
			sim_app.update()

		if out_usda != args.sample_usda:
			ok = ctx.save_as_stage(out_usda)
		else:
			ok = ctx.save_stage()

		if not ok:
			raise RuntimeError(f"Failed to save stage: {out_usda}")

		result = {
			"scene_yaml": task_info["scene_yaml"],
			"task_dir": task_info["task_dir"],
			"scene_urdf": scene_urdf,
			"scene_prim_path": scene_prim_path,
			"robot_prim_path": args.robot_prim_path,
			"fb_robot_pos": np.asarray(fb_robot_pos).tolist(),
			"fb_scene_pos": np.asarray(fb_scene_pos).tolist(),
			"sample_robot_pos": np.asarray(sample_robot_pos).tolist(),
			"sample_scene_pos": np.asarray(sample_scene_pos).tolist(),
			"output_usda": out_usda,
		}
		print(json.dumps(result, indent=2))

		if args.headless:
			for _ in range(max(1, int(args.run_frames))):
				sim_app.update()
		else:
			while sim_app.is_running():
				sim_app.update()
	finally:
		sim_app.close()


if __name__ == "__main__":
	main()
