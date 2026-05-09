import argparse
import pathlib

import numpy as np
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Solve IK for the Franka in 01_base.usda.")
	parser.add_argument(
		"--scene-usd",
		type=str,
		default=str(pathlib.Path(__file__).with_name("01_base.usda")),
		help="USD scene to open.",
	)
	parser.add_argument(
		"--robot-prim-path",
		type=str,
		default="/Franka",
		help="Prim path of the Franka articulation root in the stage.",
	)
	parser.add_argument(
		"--ee-frame",
		type=str,
		default="right_gripper",
		help="End-effector frame used by the Lula Franka solver.",
	)
	parser.add_argument(
		"--pos",
		type=float,
		nargs=3,
		default=[0.45, 0.0, 0.35],
		metavar=("X", "Y", "Z"),
		help="Target grasp position in world coordinates, meters.",
	)
	parser.add_argument(
		"--quat",
		type=float,
		nargs=4,
		default=None,
		metavar=("W", "X", "Y", "Z"),
		help="Target grasp orientation quaternion in Isaac/Usd convention (w x y z).",
	)
	parser.add_argument(
		"--euler",
		type=float,
		nargs=3,
		default=None,
		metavar=("ROLL", "PITCH", "YAW"),
		help="Target grasp orientation as Euler angles in radians. Ignored when --quat is given.",
	)
	parser.add_argument(
		"--headless",
		action="store_true",
		help="Run without the Isaac Sim viewport.",
	)
	parser.add_argument(
		"--direction",
		type=float,
		nargs=3,
		default=[0.0, 0.0, 1.0],
		metavar=("DX", "DY", "DZ"),
		help="Cartesian motion direction in world coordinates.",
	)
	parser.add_argument(
		"--distance",
		type=float,
		default=0.15,
		help="Cartesian travel distance in meters along --direction.",
	)
	parser.add_argument(
		"--waypoints",
		type=int,
		default=30,
		help="Number of linear Cartesian waypoints, including the final target.",
	)
	parser.add_argument(
		"--steps-per-waypoint",
		type=int,
		default=8,
		help="Simulation steps to apply after each waypoint action.",
	)
	parser.add_argument(
		"--viewport1-camera",
		type=str,
		default="/cameras/sceneCamera",
		help="Camera prim path to set on Viewport 1 (window name: Viewport).",
	)
	return parser.parse_args()


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver


def _resolve_target_orientation() -> np.ndarray:
	if ARGS.quat is not None:
		quat = np.asarray(ARGS.quat, dtype=np.float64)
	elif ARGS.euler is not None:
		quat = np.asarray(euler_angles_to_quat(np.asarray(ARGS.euler, dtype=np.float64)), dtype=np.float64)
	else:
		quat = np.asarray(euler_angles_to_quat(np.array([-np.pi, 0.0, np.pi])), dtype=np.float64)
	norm = np.linalg.norm(quat)
	if norm == 0.0:
		raise ValueError("Target quaternion must be non-zero.")
	return quat / norm


def _resolve_linear_direction() -> np.ndarray:
	direction = np.asarray(ARGS.direction, dtype=np.float64)
	norm = np.linalg.norm(direction)
	if norm == 0.0:
		raise ValueError("--direction must be a non-zero vector.")
	return direction / norm


def _step_sim(world: World, num_steps: int) -> None:
	for _ in range(num_steps):
		world.step(render=not ARGS.headless)


def _apply_gui_layout_and_camera(usd_context) -> None:
	if ARGS.headless:
		return

	try:
		from omni.kit.viewport.utility import get_active_viewport_window

		viewport_window = get_active_viewport_window(window_name="Viewport")
		if viewport_window is None:
			carb.log_warn("Viewport window named 'Viewport' was not found.")
			return

		stage = usd_context.get_stage()
		camera_prim = stage.GetPrimAtPath(ARGS.viewport1_camera) if stage is not None else None
		if camera_prim is None or not camera_prim.IsValid():
			carb.log_warn(f"Camera prim not found, keeping current camera: {ARGS.viewport1_camera}")
			return

		viewport_window.viewport_api.camera_path = ARGS.viewport1_camera
		carb.log_info(f"Viewport 1 camera set to: {ARGS.viewport1_camera}")
	except Exception as exc:
		carb.log_warn(f"Failed to set Viewport 1 camera: {exc}")


def _apply_cartesian_line(
	world: World,
	ik_solver: KinematicsSolver,
	articulation_controller,
	start_position: np.ndarray,
	target_orientation: np.ndarray,
	direction: np.ndarray,
	distance: float,
	waypoints: int,
	steps_per_waypoint: int,
) -> np.ndarray:
	final_position = np.asarray(start_position, dtype=np.float64) + direction * distance
	for index, alpha in enumerate(np.linspace(0.0, 1.0, waypoints + 1)[1:], start=1):
		waypoint_position = start_position + alpha * direction * distance
		action, success = ik_solver.compute_inverse_kinematics(
			target_position=waypoint_position,
			target_orientation=target_orientation,
		)
		if not success:
			raise RuntimeError(
				f"Linear waypoint IK failed at step {index}/{waypoints}. "
				f"waypoint={np.round(waypoint_position, 4)}"
			)
		articulation_controller.apply_action(action)
		_step_sim(world, steps_per_waypoint)
	return final_position


def main() -> None:
	scene_usd = pathlib.Path(ARGS.scene_usd).resolve()
	if not scene_usd.is_file():
		raise FileNotFoundError(f"Scene USD not found: {scene_usd}")

	usd_context = omni.usd.get_context()
	if not usd_context.open_stage(str(scene_usd)):
		raise RuntimeError(f"Failed to open stage: {scene_usd}")

	while usd_context.get_stage_loading_status()[2] > 0:
		simulation_app.update()

	_apply_gui_layout_and_camera(usd_context)

	world = World(stage_units_in_meters=1.0)
	franka = world.scene.add(SingleArticulation(prim_path=ARGS.robot_prim_path, name="franka"))
	world.reset()
	world.play()

	target_position = np.asarray(ARGS.pos, dtype=np.float64)
	target_orientation = _resolve_target_orientation()
	linear_direction = _resolve_linear_direction()
	ik_solver = KinematicsSolver(franka, end_effector_frame_name=ARGS.ee_frame)
	articulation_controller = franka.get_articulation_controller()

	ee_position_before, _ = ik_solver.compute_end_effector_pose(position_only=True)
	carb.log_info(
		f"IK target for {ARGS.robot_prim_path}: pos={np.round(target_position, 4)}, quat={np.round(target_orientation, 4)}"
	)
	carb.log_info(f"Current EE position before IK: {np.round(ee_position_before, 4)}")

	action, success = ik_solver.compute_inverse_kinematics(
		target_position=target_position,
		target_orientation=target_orientation,
	)
	if not success:
		raise RuntimeError(
			"IK did not converge. Check that the grasp pose is reachable and that the orientation matches the gripper frame."
		)

	articulation_controller.apply_action(action)
	_step_sim(world, 120)

	line_target_position = _apply_cartesian_line(
		world=world,
		ik_solver=ik_solver,
		articulation_controller=articulation_controller,
		start_position=target_position,
		target_orientation=target_orientation,
		direction=linear_direction,
		distance=ARGS.distance,
		waypoints=ARGS.waypoints,
		steps_per_waypoint=ARGS.steps_per_waypoint,
	)

	ee_position_after, _ = ik_solver.compute_end_effector_pose(position_only=True)
	carb.log_info(f"EE position after IK: {np.round(ee_position_after, 4)}")
	carb.log_info(
		f"Linear target: start={np.round(target_position, 4)}, end={np.round(line_target_position, 4)}, direction={np.round(linear_direction, 4)}"
	)
	carb.log_info(f"Position error: {np.round(np.linalg.norm(ee_position_after - line_target_position), 6)}")

	while simulation_app.is_running() and not ARGS.headless:
		world.step(render=True)


if __name__ == "__main__":
	try:
		main()
	finally:
		simulation_app.close()
