"""IK+RRT planned grasp demo execution."""

from __future__ import annotations

import os
import subprocess
import importlib
from typing import Any, Dict, List

import numpy as np

from utils_math import quat_multiply_xyzw, quat_normalize_xyzw
from utils_sim import (
    aabbs_overlap,
    find_articulation_root_prim,
    find_robot_hand_prim,
    get_prim_world_aabb,
    get_prim_world_bbox_center,
    get_prim_world_pose,
    set_prim_pose,
    world_pos_to_frame_pos,
)


# ============================================================================
# Helper functions that operate on context dictionary
# ============================================================================
def _configure_rrt_obstacles(ctx: Dict[str, Any], stage_name: str) -> None:
    """Configure RRT obstacles based on stage."""
    obstacle_enabled_state = ctx["obstacle_enabled_state"]
    for object_path, obstacle in ctx["obstacle_cuboids"].items():
        desired_enabled = not (
            object_path == ctx["target_object_prim_path"] and stage_name in {"grasp", "lift"}
        )
        current_enabled = obstacle_enabled_state.get(object_path, True)
        if desired_enabled == current_enabled:
            continue

        try:
            if desired_enabled:
                ctx["rrt"].enable_obstacle(obstacle)
            else:
                ctx["rrt"].disable_obstacle(obstacle)
            obstacle_enabled_state[object_path] = desired_enabled
        except Exception as exc:
            msg = str(exc)
            if "already-enabled obstacle" in msg or "already-disabled obstacle" in msg:
                obstacle_enabled_state[object_path] = desired_enabled
                continue
            raise
    ctx["rrt"].update_world()


def _stage_collision_object_paths(ctx: Dict[str, Any], stage_name: str) -> List[str]:
    """Get collision object paths relevant for the given stage."""
    if stage_name in {"grasp", "lift"}:
        return [path for path in ctx["object_prim_paths"] if path != ctx["target_object_prim_path"]]
    return list(ctx["object_prim_paths"])


def _robot_object_collision(ctx: Dict[str, Any], stage: Any, stage_name: str) -> bool:
    """Check if robot collides with objects at the current state."""
    collision_prim_paths = ctx["robot_collision_prim_paths"]
    for object_path in _stage_collision_object_paths(ctx, stage_name):
        try:
            obj_bb_min, obj_bb_max = get_prim_world_aabb(stage, object_path)
        except Exception:
            continue
        for robot_collision_path in collision_prim_paths:
            try:
                robot_bb_min, robot_bb_max = get_prim_world_aabb(stage, robot_collision_path)
            except Exception:
                continue
            if aabbs_overlap(robot_bb_min, robot_bb_max, obj_bb_min, obj_bb_max):
                print(
                    f"[WARN] Collision overlap detected: robot={robot_collision_path}, object={object_path}"
                )
                return True
    return False


def _ik_action_in_collision(
    ctx: Dict[str, Any], stage: Any, ik_action: Any, stage_name: str
) -> bool:
    """Check if IK action results in collision."""
    robot = ctx["robot"]
    world = ctx["world"]
    current_joint_positions = robot.get_joint_positions()
    if current_joint_positions is None:
        return False
    current_joint_positions = np.array(current_joint_positions, dtype=np.float64, copy=True)
    _apply_ik_action_safe(ctx, ik_action)
    world.step(render=False)
    in_collision = _robot_object_collision(ctx, stage, stage_name)
    robot.set_joint_positions(current_joint_positions)
    world.step(render=False)
    return in_collision


def _actions_in_collision(
    ctx: Dict[str, Any], stage: Any, actions: List[Any], stage_name: str
) -> bool:
    """Check if action sequence collides with objects."""
    robot = ctx["robot"]
    world = ctx["world"]
    articulation_controller = ctx["articulation_controller"]
    current_joint_positions = robot.get_joint_positions()
    if current_joint_positions is None:
        return False
    current_joint_positions = np.array(current_joint_positions, dtype=np.float64, copy=True)
    for action in actions:
        articulation_controller.apply_action(action)
        world.step(render=False)
        if _robot_object_collision(ctx, stage, stage_name):
            robot.set_joint_positions(current_joint_positions)
            world.step(render=False)
            return True
    robot.set_joint_positions(current_joint_positions)
    world.step(render=False)
    return False


def _attach_object_to_hand(ctx: Dict[str, Any], stage: Any) -> None:
    """Snap object to hand only near grasp contact to avoid teleport artifacts."""
    hand_prim_path = ctx["hand_prim_path"]
    target_object_prim_path = ctx["target_object_prim_path"]
    attach_offset = ctx["attach_offset"]
    obj_quat = ctx["obj_quat"]
    if hand_prim_path is None:
        return
    hand_pos, _ = get_prim_world_pose(stage, hand_prim_path)
    obj_pos_now, _ = get_prim_world_pose(stage, target_object_prim_path)
    dist = float(np.linalg.norm(hand_pos - obj_pos_now))
    if dist < attach_offset * 1.5:
        set_prim_pose(
            stage,
            target_object_prim_path,
            hand_pos + np.array([0.0, 0.0, -attach_offset], dtype=np.float64),
            obj_quat,
        )


def _step_once(ctx: Dict[str, Any], stage: Any, attach_object: bool = False) -> None:
    """Execute one world step with optional object attachment."""
    world = ctx["world"]
    headless = ctx["headless"]
    if attach_object:
        _attach_object_to_hand(ctx, stage)
    world.step(render=not headless)


def _apply_gripper(ctx: Dict[str, Any], width: float) -> None:
    """Apply gripper action to set finger width."""
    from isaacsim.core.utils.types import ArticulationAction

    robot = ctx["robot"]
    articulation_controller = ctx["articulation_controller"]
    joint_pos = robot.get_joint_positions()
    if joint_pos is None:
        return
    joint_pos = np.array(joint_pos, copy=True)
    dof_names = list(robot.dof_names)
    finger_ids = [
        i
        for i, name in enumerate(dof_names)
        if "finger" in name.lower() or ("gripper" in name.lower() and "joint" in name.lower())
    ]
    for idx in finger_ids:
        joint_pos[idx] = float(width)
    articulation_controller.apply_action(ArticulationAction(joint_positions=joint_pos))


def _apply_ik_action_safe(ctx: Dict[str, Any], ik_action: Any) -> None:
    """Apply IK action robustly when IK returns only arm joints (7-DOF) for a 9-DOF robot."""
    from isaacsim.core.utils.types import ArticulationAction

    robot = ctx["robot"]
    articulation_controller = ctx["articulation_controller"]
    if ik_action is None:
        return

    ik_j = getattr(ik_action, "joint_positions", None)
    if ik_j is None:
        articulation_controller.apply_action(ik_action)
        return

    ik_j = np.asarray(ik_j, dtype=np.float64).reshape(-1)
    cur_j = robot.get_joint_positions()
    if cur_j is None:
        articulation_controller.apply_action(ik_action)
        return

    cur_j = np.asarray(cur_j, dtype=np.float64).reshape(-1)
    if ik_j.size == cur_j.size:
        articulation_controller.apply_action(ArticulationAction(joint_positions=ik_j))
        return

    if ik_j.size == 7 and cur_j.size >= 7:
        full = np.array(cur_j, copy=True)
        full[:7] = ik_j
        articulation_controller.apply_action(ArticulationAction(joint_positions=full))
        return

    full = np.array(cur_j, copy=True)
    n = min(full.size, ik_j.size)
    full[:n] = ik_j[:n]
    articulation_controller.apply_action(ArticulationAction(joint_positions=full))


def _execute_rrt_to(
    ctx: Dict[str, Any],
    stage: Any,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    stage_name: str,
    attach_object: bool = False,
) -> None:
    """Execute RRT planning and action execution to target pose."""
    from isaacsim.robot_motion.motion_generation import ArticulationTrajectory

    robot_art_root = ctx["robot_art_root"]
    use_base_relative_target_pos = ctx["use_base_relative_target_pos"]
    kin_solver = ctx["kin_solver"]
    rrt = ctx["rrt"]
    rrt_max_iterations = ctx["rrt_max_iterations"]
    planner_visualizer = ctx["planner_visualizer"]
    traj_gen = ctx["traj_gen"]
    physics_dt = ctx["physics_dt"]
    settle_frames = ctx["settle_frames"]
    robot = ctx["robot"]
    articulation_controller = ctx["articulation_controller"]

    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64)
    target_quat_solver = target_quat_xyzw
    target_pos_solver = np.array(target_pos, dtype=np.float64)
    if use_base_relative_target_pos:
        target_pos_solver = world_pos_to_frame_pos(stage, robot_art_root, target_pos)

    print(
        f"[INFO] Stage: {stage_name} -> IK solving "
        f"(target_pos_world={np.round(target_pos, 3)}, target_pos_solver={np.round(target_pos_solver, 3)}, "
        f"target_quat_xyzw={np.round(target_quat_xyzw, 3)}, "
        f"target_quat_solver={np.round(target_quat_solver, 3)})"
    )
    ik_action, ik_succ = kin_solver.compute_inverse_kinematics(
        target_position=target_pos_solver,
        target_orientation=target_quat_solver,
    )
    if ik_succ and _ik_action_in_collision(ctx, stage, ik_action, stage_name):
        print(f"[WARN] IK solution collides with objects at stage={stage_name}, rejecting IK candidate")
        ik_succ = False
    if not ik_succ:
        print(f"[WARN] IK failed at stage={stage_name}, trying RRT directly")
    else:
        print(f"[INFO] Stage: {stage_name} -> IK solved")

    print(f"[INFO] Stage: {stage_name} -> RRT planning (max_iter={rrt_max_iterations})")
    rrt.set_end_effector_target(target_pos_solver, target_quat_solver)
    _configure_rrt_obstacles(ctx, stage_name)
    rrt.set_max_iterations(rrt_max_iterations)
    active_joints = planner_visualizer.get_active_joints_subset()
    start_pos = active_joints.get_joint_positions()
    rrt_plan = rrt.compute_path(start_pos, np.array([]))

    if rrt_plan is None or len(rrt_plan) <= 1:
        if ik_succ:
            print(f"[WARN] RRT failed at stage={stage_name}, fallback to IK action")
            _apply_ik_action_safe(ctx, ik_action)
            for _ in range(settle_frames):
                _step_once(ctx, stage, attach_object=attach_object)
            return
        raise RuntimeError(f"RRT and IK both failed at stage={stage_name}")

    interpolated = planner_visualizer.interpolate_path(rrt_plan, 0.01)
    trajectory = traj_gen.compute_c_space_trajectory(interpolated)
    actions = ArticulationTrajectory(robot, trajectory, physics_dt).get_action_sequence()
    if _actions_in_collision(ctx, stage, actions, stage_name):
        raise RuntimeError(f"RRT path collides with objects at stage={stage_name}")
    print(f"[INFO] Stage: {stage_name} -> executing {len(actions)} actions")
    for action in actions:
        articulation_controller.apply_action(action)
        _step_once(ctx, stage, attach_object=attach_object)
    for _ in range(settle_frames):
        _step_once(ctx, stage, attach_object=attach_object)
    print(f"[INFO] Stage: {stage_name} -> done")


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
    if quat_xyzw.size != 4:
        raise ValueError(f"Expected quaternion of size 4, got {quat_xyzw.size}")
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _build_curobo_world_from_stage(ctx: Dict[str, Any], stage: Any, stage_name: str) -> Any:
    """Build a CuRobo WorldConfig from current stage object AABBs."""
    curobo_geom_types = importlib.import_module("curobo.geom.types")
    Cuboid = curobo_geom_types.Cuboid
    WorldConfig = curobo_geom_types.WorldConfig

    cuboids: List[Any] = []
    for idx, object_path in enumerate(_stage_collision_object_paths(ctx, stage_name)):
        try:
            bb_min, bb_max = get_prim_world_aabb(stage, object_path)
        except Exception:
            continue
        dims = np.maximum(bb_max - bb_min, np.array([0.02, 0.02, 0.02], dtype=np.float64))
        center = 0.5 * (bb_min + bb_max)
        cuboids.append(
            Cuboid(
                name=f"obj_{idx}",
                pose=[*center.tolist(), 1.0, 0.0, 0.0, 0.0],
                dims=dims.tolist(),
            )
        )
    return WorldConfig(cuboid=cuboids)


def _execute_curobo_to(
    ctx: Dict[str, Any],
    stage: Any,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    stage_name: str,
    attach_object: bool = False,
) -> None:
    """Plan and execute motion using CuRobo in-process backend."""
    from isaacsim.core.utils.types import ArticulationAction

    curobo_types_math = importlib.import_module("curobo.types.math")
    curobo_types_robot = importlib.import_module("curobo.types.robot")
    Pose = curobo_types_math.Pose
    JointState = curobo_types_robot.JointState

    robot = ctx["robot"]
    articulation_controller = ctx["articulation_controller"]
    settle_frames = ctx["settle_frames"]
    tensor_args = ctx["curobo_tensor_args"]
    motion_gen = ctx["curobo_motion_gen"]
    plan_cfg = ctx["curobo_plan_cfg"]
    robot_art_root = ctx["robot_art_root"]
    use_base_relative_target_pos = ctx["use_base_relative_target_pos"]
    curobo_joint_names = ctx["curobo_joint_names"]

    world_cfg = _build_curobo_world_from_stage(ctx, stage, stage_name)
    motion_gen.update_world(world_cfg)

    current_joint_positions = robot.get_joint_positions()
    if current_joint_positions is None:
        raise RuntimeError("Failed to read current robot joints for CuRobo planning")
    current_joint_positions = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)
    arm_dofs = min(7, current_joint_positions.size)
    q_start = tensor_args.to_device(current_joint_positions[:arm_dofs].reshape(1, arm_dofs))

    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64)
    target_pos_solver = np.array(target_pos, dtype=np.float64)
    if use_base_relative_target_pos:
        target_pos_solver = world_pos_to_frame_pos(stage, robot_art_root, target_pos)
    target_quat_wxyz = _quat_xyzw_to_wxyz(target_quat_xyzw)

    print(
        f"[INFO] Stage: {stage_name} -> CuRobo planning "
        f"(target_pos_world={np.round(target_pos, 3)}, target_pos_solver={np.round(target_pos_solver, 3)}, "
        f"target_quat_xyzw={np.round(target_quat_xyzw, 3)})"
    )

    goal_pose = Pose(
        position=tensor_args.to_device(target_pos_solver.reshape(1, 3)),
        quaternion=tensor_args.to_device(target_quat_wxyz.reshape(1, 4)),
    )
    start_state = JointState(position=q_start, joint_names=curobo_joint_names)
    result = motion_gen.plan_goalset(start_state, goal_pose, plan_cfg)

    success = bool(result.success.item()) if hasattr(result.success, "item") else bool(result.success)
    if not success:
        raise RuntimeError(f"CuRobo planning failed at stage={stage_name}")

    traj = result.get_interpolated_plan()
    traj_pos = np.asarray(traj.position.detach().cpu().numpy(), dtype=np.float64)
    if traj_pos.ndim == 3:
        traj_pos = traj_pos[0]
    if traj_pos.ndim != 2 or traj_pos.shape[0] <= 0:
        raise RuntimeError(f"CuRobo trajectory is empty at stage={stage_name}")

    actions: List[Any] = []
    for row in traj_pos:
        full = np.array(current_joint_positions, copy=True)
        n = min(arm_dofs, row.shape[0], full.shape[0])
        full[:n] = row[:n]
        actions.append(ArticulationAction(joint_positions=full))

    if _actions_in_collision(ctx, stage, actions, stage_name):
        raise RuntimeError(f"CuRobo path collides with objects at stage={stage_name}")

    print(f"[INFO] Stage: {stage_name} -> executing {len(actions)} CuRobo actions")
    for action in actions:
        articulation_controller.apply_action(action)
        _step_once(ctx, stage, attach_object=attach_object)
    for _ in range(settle_frames):
        _step_once(ctx, stage, attach_object=attach_object)
    print(f"[INFO] Stage: {stage_name} -> done")


# ============================================================================
# Main grasp demo function
# ============================================================================
def run_ik_planned_grasp_demo(
    stage,
    robot_prim_path: str,
    target_object_prim_path: str,
    headless: bool,
    close_frames: int,
    lift_frames: int,
    lift_height: float,
    attach_offset: float,
    pregrasp_offset: float,
    grasp_offset: float,
    rrt_max_iterations: int,
    settle_frames: int,
    stop_after_stage: str,
    planner_backend: str = "lula",
    curobo_scene_config: str = "benchmark_eval/RigidObjDesk_0",
    curobo_task_name: str = "FetchMeshCurobo",
    curobo_conda_env: str = "FetchBench",
    curobo_corl_root: str = "/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024",
) -> None:
    """Execute IK+RRT planned grasp demo with collision awareness."""
    planner_backend = str(planner_backend).strip().lower()
    if planner_backend == "curobo_external":
        infinigym_dir = os.path.join(curobo_corl_root, "InfiniGym")
        if not os.path.isdir(infinigym_dir):
            raise RuntimeError(f"CuRobo InfiniGym directory not found: {infinigym_dir}")
        cmd = [
            "conda",
            "run",
            "-n",
            curobo_conda_env,
            "python",
            "isaacgymenvs/eval.py",
            f"task={curobo_task_name}",
            f"scene={curobo_scene_config}",
            "scene.num_tasks=1",
        ]
        print("[INFO] Running external FetchBench CuRobo evaluator:")
        print(f"[INFO] cwd={infinigym_dir}")
        print(f"[INFO] cmd={' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=infinigym_dir, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"External CuRobo eval failed with code {proc.returncode}")
        print("[INFO] External CuRobo eval finished")
        return

    if planner_backend not in {"lula", "curobo"}:
        raise ValueError(
            f"Unsupported planner_backend={planner_backend}. "
            "Use 'lula', 'curobo', or 'curobo_external'."
        )

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.api.robots import Robot
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
    from isaacsim.robot_motion.motion_generation import interface_config_loader
    from isaacsim.robot_motion.motion_generation.lula import RRT
    from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
    from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import (
        LulaCSpaceTrajectoryGenerator,
    )
    from isaacsim.robot_motion.motion_generation.path_planner_visualizer import PathPlannerVisualizer
    from pxr import Usd, UsdGeom, UsdPhysics

    # Normalize and validate inputs
    close_frames = max(1, int(close_frames))
    lift_frames = max(1, int(lift_frames))
    lift_height = float(lift_height)
    attach_offset = float(attach_offset)
    pregrasp_offset = float(pregrasp_offset)
    grasp_offset = float(grasp_offset)
    rrt_max_iterations = max(100, int(rrt_max_iterations))
    settle_frames = max(1, int(settle_frames))
    stop_after_stage = str(stop_after_stage).lower().strip()
    if stop_after_stage not in {"pregrasp", "grasp", "lift"}:
        raise ValueError(f"Invalid stop_after_stage: {stop_after_stage}")
    use_base_relative_target_pos = True

    # Franka home configuration
    franka_home_joints = np.array(
        [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04], dtype=np.float64
    )
    physics_dt = 1.0 / 60.0

    # Find robot and hand primitives
    robot_art_root = find_articulation_root_prim(stage, robot_prim_path)
    hand_prim_path = find_robot_hand_prim(stage, robot_prim_path)
    obj_pos0, obj_quat = get_prim_world_pose(stage, target_object_prim_path)

    # Create simulation world and robot
    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Robot(prim_path=robot_art_root, name="fetchbench_franka"))
    world.reset()

    # Initialize robot to home pose
    articulation_controller = robot.get_articulation_controller()
    ready_joint_positions = robot.get_joint_positions()
    if ready_joint_positions is not None:
        robot.set_joint_positions(franka_home_joints)
    else:
        print("[WARN] Could not read robot joint positions before ready-pose apply")

    # Settle world after home pose
    for _ in range(30):
        world.step(render=not headless)


    # Configure IK solver
    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    kin_base = LulaKinematicsSolver(**kinematics_config)
    ee_frame = os.path.basename(hand_prim_path) if hand_prim_path else "right_gripper"
    try:
        kin_solver = ArticulationKinematicsSolver(robot, kin_base, ee_frame)
        print(f"[INFO] IK end-effector frame: {ee_frame}")
    except Exception as exc:
        print(f"[WARN] Failed IK solver with ee_frame={ee_frame}, fallback to right_gripper: {exc}")
        kin_solver = ArticulationKinematicsSolver(robot, kin_base, "right_gripper")

    # Configure RRT path planner
    rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
    rrt = RRT(**rrt_config)
    traj_gen = LulaCSpaceTrajectoryGenerator(rrt_config["robot_description_path"], rrt_config["urdf_path"])
    planner_visualizer = PathPlannerVisualizer(robot, rrt)

    # Register scene objects as planner obstacles
    object_root = stage.GetPrimAtPath("/World/FetchBench/Objects")
    object_prim_paths = [str(child.GetPath()) for child in object_root.GetChildren()] if object_root.IsValid() else []
    obstacle_cuboids = {}
    for index, object_path in enumerate(object_prim_paths):
        try:
            bb_min, bb_max = get_prim_world_aabb(stage, object_path)
            scale = np.maximum(bb_max - bb_min, np.array([0.02, 0.02, 0.02], dtype=np.float64))
            center = 0.5 * (bb_min + bb_max)
            cuboid = world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/FetchBench/PlannerObstacles/object_obstacle_{index}",
                    name=f"object_obstacle_{index}",
                    position=center,
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    size=1.0,
                    scale=scale,
                    visible=False,
                )
            )
            rrt.add_cuboid(cuboid, static=True)
            obstacle_cuboids[object_path] = cuboid
        except Exception as exc:
            print(f"[WARN] Failed to register planner obstacle for {object_path}: {exc}")

    # Detect robot collision primitives
    robot_collision_prim_paths: List[str] = []
    robot_root_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_root_prim.IsValid():
        for prim in Usd.PrimRange(robot_root_prim):
            has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
            parent = prim.GetParent()
            while (
                not has_collision_api
                and parent
                and parent.IsValid()
                and parent != robot_root_prim.GetParent()
            ):
                has_collision_api = parent.HasAPI(UsdPhysics.CollisionAPI)
                parent = parent.GetParent()
            if not has_collision_api:
                continue
            if not prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
                continue
            robot_collision_prim_paths.append(str(prim.GetPath()))
    if not robot_collision_prim_paths:
        robot_collision_prim_paths = [robot_prim_path]

    # Build context dictionary for helper functions
    ctx: Dict[str, Any] = {
        "robot": robot,
        "world": world,
        "articulation_controller": articulation_controller,
        "headless": headless,
        "robot_art_root": robot_art_root,
        "hand_prim_path": hand_prim_path,
        "target_object_prim_path": target_object_prim_path,
        "attach_offset": attach_offset,
        "obj_quat": obj_quat,
        "use_base_relative_target_pos": use_base_relative_target_pos,
        "kin_solver": kin_solver,
        "rrt": rrt,
        "rrt_max_iterations": rrt_max_iterations,
        "planner_visualizer": planner_visualizer,
        "traj_gen": traj_gen,
        "physics_dt": physics_dt,
        "settle_frames": settle_frames,
        "object_prim_paths": object_prim_paths,
        "obstacle_cuboids": obstacle_cuboids,
        "obstacle_enabled_state": {path: True for path in obstacle_cuboids},
        "robot_collision_prim_paths": robot_collision_prim_paths,
    }

    if planner_backend == "curobo":
        curobo_geom_sdf_world = importlib.import_module("curobo.geom.sdf.world")
        curobo_types_base = importlib.import_module("curobo.types.base")
        curobo_types_robot = importlib.import_module("curobo.types.robot")
        curobo_util_file = importlib.import_module("curobo.util_file")
        curobo_motion_gen = importlib.import_module("curobo.wrap.reacher.motion_gen")
        CollisionCheckerType = curobo_geom_sdf_world.CollisionCheckerType
        TensorDeviceType = curobo_types_base.TensorDeviceType
        RobotConfig = curobo_types_robot.RobotConfig
        get_robot_configs_path = curobo_util_file.get_robot_configs_path
        join_path = curobo_util_file.join_path
        load_yaml = curobo_util_file.load_yaml
        MotionGen = curobo_motion_gen.MotionGen
        MotionGenConfig = curobo_motion_gen.MotionGenConfig
        MotionGenPlanConfig = curobo_motion_gen.MotionGenPlanConfig

        tensor_args = TensorDeviceType()
        robot_cfg_file = join_path(get_robot_configs_path(), "franka.yml")
        robot_cfg = RobotConfig.from_dict(load_yaml(robot_cfg_file)["robot_cfg"])
        world_cfg = _build_curobo_world_from_stage(ctx, stage, stage_name="pregrasp")
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=False,
            interpolation_dt=0.02,
            interpolation_steps=500,
            num_ik_seeds=16,
            num_trajopt_seeds=8,
            num_graph_seeds=4,
            self_collision_check=True,
            self_collision_opt=True,
        )
        motion_gen = MotionGen(motion_gen_cfg)
        motion_gen.reset()
        plan_cfg = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            max_attempts=50,
            timeout=20.0,
            enable_finetune_trajopt=False,
        )
        ctx["curobo_tensor_args"] = tensor_args
        ctx["curobo_motion_gen"] = motion_gen
        ctx["curobo_plan_cfg"] = plan_cfg
        ctx["curobo_joint_names"] = list(robot.dof_names[:7])
        print("[INFO] CuRobo in-process backend initialized")

    print(
        "[INFO] Running IK+RRT grasp demo: "
        f"robot_art_root={robot_art_root}, hand={hand_prim_path}, target={target_object_prim_path}"
    )

    # Open hand and settle
    _apply_gripper(ctx, width=0.04)
    for _ in range(settle_frames):
        _step_once(ctx, stage, attach_object=False)

    # Compute target positions
    obj_pos0, obj_quat = get_prim_world_pose(stage, target_object_prim_path)
    try:
        obj_target_pos = get_prim_world_bbox_center(stage, target_object_prim_path)
    except Exception as exc:
        print(f"[WARN] Failed to compute target bbox center, fallback to root pose: {exc}")
        obj_target_pos = np.array(obj_pos0, dtype=np.float64)

    pregrasp_pos = obj_target_pos + np.array([0.0, 0.0, pregrasp_offset], dtype=np.float64)
    grasp_pos = obj_target_pos + np.array([0.0, 0.0, grasp_offset], dtype=np.float64)
    lift_pos = obj_target_pos + np.array([0.0, 0.0, grasp_offset + lift_height], dtype=np.float64)

    print(f"[INFO] Target object root pos={np.round(obj_pos0, 3)}")
    print(f"[INFO] Target object bbox center={np.round(obj_target_pos, 3)}")
    print(
        f"[INFO] Pre-grasp pos={np.round(pregrasp_pos, 3)}, grasp pos={np.round(grasp_pos, 3)}, lift pos={np.round(lift_pos, 3)}"
    )

    # Determine grasp orientation with collision-aware IK
    grasp_quat_xyzw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if hand_prim_path:
        hand_pos_now, hand_quat_now = get_prim_world_pose(stage, hand_prim_path)
        print(f"[INFO] Hand pose at ready: pos={np.round(hand_pos_now, 3)}, quat={np.round(hand_quat_now, 3)}")

        candidates = {
            "fixed_down": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "flip_y": quat_normalize_xyzw(
                quat_multiply_xyzw(hand_quat_now, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            ),
            "flip_x": quat_normalize_xyzw(
                quat_multiply_xyzw(hand_quat_now, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            ),
            "flip_z": quat_normalize_xyzw(
                quat_multiply_xyzw(hand_quat_now, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64))
            ),
            "hand_ready": quat_normalize_xyzw(hand_quat_now),
        }

        preferred_order = ["fixed_down", "flip_y", "flip_x", "flip_z", "hand_ready"]
        selected_label = None
        for label in preferred_order:
            cand_quat = candidates[label]
            cand_pos = np.array(pregrasp_pos, dtype=np.float64)
            if use_base_relative_target_pos:
                cand_pos = world_pos_to_frame_pos(stage, robot_art_root, pregrasp_pos)
            ik_action, ik_succ = kin_solver.compute_inverse_kinematics(
                target_position=cand_pos,
                target_orientation=cand_quat,
            )
            # Reject IK solution if it causes collision
            if ik_succ and _ik_action_in_collision(ctx, stage, ik_action, "pregrasp"):
                ik_succ = False
            if ik_succ:
                print(f"[DEBUG] Orientation candidate {label}: IK success, quat={np.round(cand_quat, 3)}")
                selected_label = label
                grasp_quat_xyzw = cand_quat
                break
            print(f"[DEBUG] Orientation candidate {label}: IK fail")

        if selected_label is None:
            selected_label = "fixed_down"
            grasp_quat_xyzw = candidates[selected_label]
            print("[WARN] All orientation candidates failed IK at pregrasp, fallback to fixed_down")

        print(f"[INFO] Selected grasp orientation: {selected_label}")
    print(f"[INFO] Using grasp quat(xyzw)={np.round(grasp_quat_xyzw, 3)}")

    # Execute pregrasp stage
    if planner_backend == "curobo":
        _execute_curobo_to(
            ctx, stage, pregrasp_pos, grasp_quat_xyzw, stage_name="pregrasp", attach_object=False
        )
    else:
        _execute_rrt_to(ctx, stage, pregrasp_pos, grasp_quat_xyzw, stage_name="pregrasp", attach_object=False)
    if stop_after_stage == "pregrasp":
        print("[INFO] stop_after_stage=pregrasp, stopping demo here")
        return

    # Execute grasp approach stage
    if planner_backend == "curobo":
        _execute_curobo_to(ctx, stage, grasp_pos, grasp_quat_xyzw, stage_name="grasp", attach_object=False)
    else:
        _execute_rrt_to(ctx, stage, grasp_pos, grasp_quat_xyzw, stage_name="grasp", attach_object=False)
    if stop_after_stage == "grasp":
        print("[INFO] stop_after_stage=grasp, stopping demo here")
        return

    # Close gripper and attach object
    _apply_gripper(ctx, width=0.0)
    for _ in range(close_frames):
        _step_once(ctx, stage, attach_object=True)

    # Execute lift stage
    if planner_backend == "curobo":
        _execute_curobo_to(ctx, stage, lift_pos, grasp_quat_xyzw, stage_name="lift", attach_object=True)
    else:
        _execute_rrt_to(ctx, stage, lift_pos, grasp_quat_xyzw, stage_name="lift", attach_object=True)
    for _ in range(lift_frames):
        _step_once(ctx, stage, attach_object=True)
