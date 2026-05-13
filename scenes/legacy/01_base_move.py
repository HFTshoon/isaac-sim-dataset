import asyncio
import inspect
import json
import math
import os
import numpy as np

import omni
import omni.usd
import omni.kit.app

from pxr import Usd, UsdPhysics, Sdf

from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim


# ============================================================
# User config
# ============================================================

SCENE_USD = "/isaac-sim/corl2025/scenes/01_base.usda"

DATASET_ROOT = "/isaac-sim/corl2025/dataset/01"

# scene_000, scene_001, ... 를 자동 순회.
# 특정 scene만 돌리고 싶으면 예: ["scene_003"]
SCENE_NAME_LIST = None

OBJECT_ROOT = "/objects"

# 하나만 테스트할 때.
# None이면 /objects 아래 첫 번째 rigid object를 선택.
TARGET_OBJECT_PATH = None
# 예:
# TARGET_OBJECT_PATH = "/objects/obj_025_mug"

# True면 모든 object × 모든 direction에 대해 반복 실험.
# 처음엔 False로 두고 한 object만 확인하는 걸 추천.
RUN_ALL_OBJECTS = True

# Fibonacci upper hemisphere 방향 개수.
NUM_DIRECTIONS = 64

# 테스트할 direction index들.
# RUN_ALL_OBJECTS=False일 때도 여러 방향을 넣으면 선택 object를 여러 방향으로 테스트함.
DIRECTION_INDICES = list(range(NUM_DIRECTIONS)) # [0]
# 예: DIRECTION_INDICES = list(range(NUM_DIRECTIONS))

# 이동 거리, meter 단위.
MOVE_DISTANCE = 0.15

# 물체를 한 번에 teleport하지 않고, MOVE_STEPS번 나눠서 이동.
# 밀림/접촉 효과를 보고 싶으면 20~80 정도 권장.
MOVE_STEPS = 80

# 초기 안정화 step 수.
INITIAL_SETTLE_STEPS = 60

# trial마다 baseline pose로 복구한 뒤 안정화 step 수.
RESET_SETTLE_STEPS = 10

# 이동 완료 후 다른 물체가 settle하도록 기다리는 step 수.
POST_SETTLE_STEPS = 120

# 변화량 threshold. 이것보다 작으면 출력/저장에서 moved=False.
TRANSLATION_EPS = 1e-4       # meter
ROTATION_EPS_DEG = 0.05      # degree

JSON_FILENAME = "move_effect.json"


# ============================================================
# Utility functions
# ============================================================

def to_numpy(x):
    """torch / warp / numpy / list 등을 numpy array로 변환."""
    if isinstance(x, np.ndarray):
        return x.copy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().copy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy().copy()
    if hasattr(x, "numpy"):
        return x.numpy().copy()
    return np.array(x)


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_angle_deg(q0, q1):
    """
    q format: wxyz.
    두 quaternion 사이의 최소 회전각 degree.
    q와 -q는 같은 orientation이므로 abs(dot)을 사용.
    """
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def fibonacci_upper_hemisphere(n):
    """
    z > 0인 Fibonacci sphere 방향들.
    반환 shape: (n, 3)
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []

    for i in range(n):
        # z가 0~1 사이가 되도록 upper hemisphere만 샘플링
        z = (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = i * golden_angle

        x = math.cos(theta) * r
        y = math.sin(theta) * r

        v = np.array([x, y, z], dtype=np.float64)
        v = v / np.linalg.norm(v)
        dirs.append(v)

    return np.asarray(dirs, dtype=np.float64)


def find_physics_scene_path(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim.GetPath().pathString
    return "/physicsScene"


def has_rigid_body_api(prim):
    if not prim or not prim.IsValid():
        return False
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return True
    attr = prim.GetAttribute("physics:rigidBodyEnabled")
    return attr.IsValid()


def find_object_paths(stage, object_root=OBJECT_ROOT):
    root = stage.GetPrimAtPath(object_root)
    if not root or not root.IsValid():
        raise RuntimeError(f"{object_root} prim이 없습니다.")

    paths = []

    # 보통 /objects 바로 아래가 obj_xxx rigid body임.
    for child in root.GetChildren():
        if child.IsActive() and has_rigid_body_api(child):
            paths.append(child.GetPath().pathString)

    # 혹시 direct child가 아니면 하위 traverse로 fallback.
    if not paths:
        for prim in Usd.PrimRange(root):
            if prim == root:
                continue
            if prim.IsActive() and has_rigid_body_api(prim):
                paths.append(prim.GetPath().pathString)

    paths = sorted(set(paths))

    if not paths:
        raise RuntimeError(f"{object_root} 아래에서 PhysicsRigidBodyAPI가 붙은 object를 찾지 못했습니다.")

    return paths


def replace_objects_from_usd(stage, objects_usd, object_root=OBJECT_ROOT):
    """
    현재 stage의 /objects를 지우고 objects_usd 안의 object prim들을
    /objects 아래에 reference로 붙임.

    objects_usd 안에 /objects가 있으면 그 children을 가져오고,
    없으면 defaultPrim의 children을 가져옴.
    """
    if objects_usd is None:
        return

    print(f"[INFO] Replacing {object_root} using: {objects_usd}")

    if stage.GetPrimAtPath(object_root).IsValid():
        stage.RemovePrim(Sdf.Path(object_root))

    dst_root = stage.DefinePrim(object_root, "Scope")

    src_stage = Usd.Stage.Open(objects_usd)
    if src_stage is None:
        raise RuntimeError(f"objects-only USD를 열 수 없습니다: {objects_usd}")

    src_root = src_stage.GetPrimAtPath(object_root)

    if not src_root or not src_root.IsValid():
        default_prim = src_stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            src_root = default_prim
        else:
            raise RuntimeError(
                f"{objects_usd} 안에 {object_root}도 없고 defaultPrim도 없습니다. "
                f"OBJECTS_ONLY_USD를 None으로 두거나, 파일 구조를 확인하세요."
            )

    src_children = [p for p in src_root.GetChildren() if p.IsActive()]
    if not src_children:
        raise RuntimeError(f"{objects_usd} 안의 {src_root.GetPath()} 아래에 active child가 없습니다.")

    for src_child in src_children:
        name = src_child.GetName()
        dst_path = f"{object_root}/{name}"
        dst_prim = stage.DefinePrim(dst_path, src_child.GetTypeName() or "Xform")
        dst_prim.GetReferences().AddReference(objects_usd, src_child.GetPath().pathString)

    print(f"[INFO] Referenced {len(src_children)} object prims into {object_root}")


def discover_scene_names(dataset_root):
    if SCENE_NAME_LIST is not None:
        return list(SCENE_NAME_LIST)

    names = []
    if not os.path.isdir(dataset_root):
        raise RuntimeError(f"DATASET_ROOT does not exist: {dataset_root}")

    for name in os.listdir(dataset_root):
        full = os.path.join(dataset_root, name)
        if os.path.isdir(full) and name.startswith("scene_"):
            names.append(name)

    def _scene_key(n):
        try:
            return int(n.split("_", 1)[1])
        except Exception:
            return 10**9

    names = sorted(names, key=_scene_key)
    if not names:
        raise RuntimeError(f"No scene_* directories found under {dataset_root}")
    return names


def build_scene_io_paths(dataset_root, scene_name):
    scene_dir = os.path.join(dataset_root, scene_name)
    objects_only_usd = os.path.join(scene_dir, "isaac_objects.usda")
    json_out = os.path.join(scene_dir, JSON_FILENAME)
    return scene_dir, objects_only_usd, json_out


async def step_world(world, n):
    app = omni.kit.app.get_app()
    for _ in range(n):
        stepped = False

        # Isaac Sim 버전에 따라 step_async가 coroutine이거나 None을 반환할 수 있음.
        if hasattr(world, "step_async"):
            try:
                ret = world.step_async()
                if inspect.isawaitable(ret):
                    await ret
                    stepped = True
                else:
                    # non-awaitable 경로(None 포함): 다음 프레임까지 진행
                    await app.next_update_async()
                    stepped = True
            except Exception:
                stepped = False

        if stepped:
            continue

        # fallback: 동기 step 지원 시 사용
        if hasattr(world, "step"):
            world.step(render=False)
            await app.next_update_async()
            continue

        raise RuntimeError("World stepping API를 찾지 못했습니다 (step_async/step 모두 없음).")


def get_poses(view):
    pos, quat = view.get_world_poses()
    pos = to_numpy(pos).astype(np.float64)
    quat = to_numpy(quat).astype(np.float64)

    # quaternion normalize
    for i in range(quat.shape[0]):
        quat[i] = normalize_quat_wxyz(quat[i])

    return pos, quat


def set_all_poses(view, positions, orientations):
    view.set_world_poses(
        positions=np.asarray(positions, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.float64),
    )


def set_one_pose(view, index, position, orientation):
    view.set_world_poses(
        positions=np.asarray([position], dtype=np.float64),
        orientations=np.asarray([orientation], dtype=np.float64),
        indices=np.asarray([index], dtype=np.int32),
    )


def zero_velocities(view, count, indices=None):
    """
    trial 사이 velocity가 남아서 결과가 섞이지 않도록 velocity를 0으로 만듦.
    Isaac Sim 버전에 따라 method 이름이 조금 다를 수 있어서 안전하게 try.
    """
    if indices is None:
        lin = np.zeros((count, 3), dtype=np.float64)
        ang = np.zeros((count, 3), dtype=np.float64)
        idx = None
    else:
        lin = np.zeros((len(indices), 3), dtype=np.float64)
        ang = np.zeros((len(indices), 3), dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int32)

    if hasattr(view, "set_linear_velocities"):
        if idx is None:
            view.set_linear_velocities(lin)
        else:
            view.set_linear_velocities(lin, indices=idx)

    if hasattr(view, "set_angular_velocities"):
        if idx is None:
            view.set_angular_velocities(ang)
        else:
            view.set_angular_velocities(ang, indices=idx)


# ============================================================
# Main experiment
# ============================================================

async def run_single_scene_experiment(scene_name):
    scene_dir, objects_only_usd, json_out = build_scene_io_paths(DATASET_ROOT, scene_name)
    if not os.path.isfile(objects_only_usd):
        raise RuntimeError(f"objects-only USD not found: {objects_only_usd}")

    # 기존 World instance 정리
    if World.instance():
        World.instance().clear_instance()

    print("\n" + "#" * 100)
    print(f"[SCENE] {scene_name}")
    print(f"[SCENE] objects USD: {objects_only_usd}")
    print("#" * 100)

    print(f"[INFO] Opening stage: {SCENE_USD}")
    success, error = await omni.usd.get_context().open_stage_async(SCENE_USD)
    if not success:
        raise RuntimeError(f"Failed to open stage: {error}")

    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Stage가 없습니다.")

    # 필요 시 /objects 교체
    replace_objects_from_usd(stage, objects_only_usd, OBJECT_ROOT)

    for _ in range(5):
        await app.next_update_async()

    object_paths = find_object_paths(stage, OBJECT_ROOT)
    print(f"[INFO] Found {len(object_paths)} rigid objects:")
    for i, p in enumerate(object_paths):
        print(f"  [{i:02d}] {p}")

    if TARGET_OBJECT_PATH is not None and TARGET_OBJECT_PATH not in object_paths:
        raise RuntimeError(
            f"TARGET_OBJECT_PATH가 object list에 없습니다: {TARGET_OBJECT_PATH}\n"
            f"가능한 값:\n" + "\n".join(object_paths)
        )

    physics_scene_path = find_physics_scene_path(stage)
    print(f"[INFO] PhysicsScene: {physics_scene_path}")

    world = World(
        stage_units_in_meters=1.0,
        physics_prim_path=physics_scene_path,
    )
    await world.initialize_simulation_context_async()

    # object_paths list를 직접 넣어 view 내부 index를 고정
    rigid_view = RigidPrim(
        prim_paths_expr=object_paths,
        name="move_effect_objects_view",
    )
    world.scene.add(rigid_view)

    await world.reset_async()
    await world.play_async()

    print("[INFO] Settling initial scene...")
    await step_world(world, INITIAL_SETTLE_STEPS)

    base_pos, base_quat = get_poses(rigid_view)
    n_obj = len(object_paths)

    dirs = fibonacci_upper_hemisphere(NUM_DIRECTIONS)

    if RUN_ALL_OBJECTS:
        target_indices = list(range(n_obj))
    else:
        if TARGET_OBJECT_PATH is None:
            target_indices = [0]
        else:
            target_indices = [object_paths.index(TARGET_OBJECT_PATH)]

    trial_groups = []

    for target_idx in target_indices:
        target_path = object_paths[target_idx]

        for dir_idx in DIRECTION_INDICES:
            if dir_idx < 0 or dir_idx >= NUM_DIRECTIONS:
                raise ValueError(f"direction index out of range: {dir_idx}")

            direction = dirs[dir_idx]
            assert direction[2] > 0.0

            print("")
            print("=" * 80)
            print(f"[TRIAL] target={target_path}")
            print(f"[TRIAL] dir_idx={dir_idx}, dir={direction.tolist()}, distance={MOVE_DISTANCE}")
            print("=" * 80)

            # trial baseline으로 복구
            set_all_poses(rigid_view, base_pos, base_quat)
            zero_velocities(rigid_view, n_obj)
            await step_world(world, RESET_SETTLE_STEPS)

            before_pos, before_quat = get_poses(rigid_view)

            # 선택 object를 direction 방향으로 조금씩 이동
            start_pos = before_pos[target_idx].copy()
            start_quat = before_quat[target_idx].copy()

            for step in range(1, MOVE_STEPS + 1):
                alpha = step / float(MOVE_STEPS)
                new_pos = start_pos + direction * MOVE_DISTANCE * alpha

                set_one_pose(
                    rigid_view,
                    target_idx,
                    new_pos,
                    start_quat,
                )

                # target은 강제로 움직이는 물체처럼 다루기 위해 velocity를 0으로 유지
                zero_velocities(rigid_view, n_obj, indices=[target_idx])
                await step_world(world, 1)

            await step_world(world, POST_SETTLE_STEPS)

            after_pos, after_quat = get_poses(rigid_view)

            # 다른 object 변화량 계산
            trial_rows = []
            moved_objects = []

            for obj_idx, obj_path in enumerate(object_paths):
                if obj_idx == target_idx:
                    continue

                dpos_vec = after_pos[obj_idx] - before_pos[obj_idx]
                dpos = float(np.linalg.norm(dpos_vec))
                drot = float(quat_angle_deg(before_quat[obj_idx], after_quat[obj_idx]))

                moved = (dpos > TRANSLATION_EPS) or (drot > ROTATION_EPS_DEG)

                row = {
                    "target_path": target_path,
                    "direction_index": dir_idx,
                    "direction_x": float(direction[0]),
                    "direction_y": float(direction[1]),
                    "direction_z": float(direction[2]),
                    "move_distance_m": float(MOVE_DISTANCE),
                    "object_path": obj_path,
                    "translation_delta_m": dpos,
                    "rotation_delta_deg": drot,
                    "delta_x": float(dpos_vec[0]),
                    "delta_y": float(dpos_vec[1]),
                    "delta_z": float(dpos_vec[2]),
                    "moved": bool(moved),
                }

                trial_rows.append(row)

            # 큰 변화 순으로 console 출력
            trial_rows = sorted(
                trial_rows,
                key=lambda r: (r["translation_delta_m"], r["rotation_delta_deg"]),
                reverse=True,
            )

            # print("[RESULT] affected objects, sorted by translation_delta:")
            # for r in trial_rows:
            #     if r["moved"]:
            #         print(
            #             f"  {r['object_path']}: "
            #             f"trans={r['translation_delta_m']:.6f} m, "
            #             f"rot={r['rotation_delta_deg']:.3f} deg, "
            #             f"dxyz=({r['delta_x']:.6f}, {r['delta_y']:.6f}, {r['delta_z']:.6f})"
            #         )

            # if not any(r["moved"] for r in trial_rows):
            #     print("  No other object moved above threshold.")

            trial_groups.append({
                "scene_usd": SCENE_USD,
                "target_path": target_path,
                "direction_index": dir_idx,
                "direction": [
                    float(direction[0]),
                    float(direction[1]),
                    float(direction[2]),
                ],
                "move_distance_m": float(MOVE_DISTANCE),
                "num_affected_objects": len([r for r in trial_rows if r["moved"]]),
                "all_object_changes": trial_rows,
            })

    # JSON 저장
    os.makedirs(os.path.dirname(json_out), exist_ok=True)

    payload = {
        "scene_name": scene_name,
        "scene_dir": scene_dir,
        "scene_usd": SCENE_USD,
        "objects_only_usd": objects_only_usd,
        "object_root": OBJECT_ROOT,
        "target_object_path": TARGET_OBJECT_PATH,
        "run_all_objects": RUN_ALL_OBJECTS,
        "num_directions": NUM_DIRECTIONS,
        "move_distance_m": MOVE_DISTANCE,
        "move_steps": MOVE_STEPS,
        "initial_settle_steps": INITIAL_SETTLE_STEPS,
        "reset_settle_steps": RESET_SETTLE_STEPS,
        "post_settle_steps": POST_SETTLE_STEPS,
        "translation_eps_m": TRANSLATION_EPS,
        "rotation_eps_deg": ROTATION_EPS_DEG,
        "num_trials": len(target_indices) * len(DIRECTION_INDICES),
        "num_trial_groups": len(trial_groups),
        "num_rows_flat": sum(len(g["all_object_changes"]) for g in trial_groups),
        "trial_groups": trial_groups,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("")
    print(f"[DONE] Trials: {len(target_indices) * len(DIRECTION_INDICES)}")
    print(f"[DONE] Trial groups saved: {len(trial_groups)}")
    print(f"[DONE] JSON saved to: {json_out}")

    world.pause()


async def run_move_effect_experiment():
    scene_names = discover_scene_names(DATASET_ROOT)
    print(f"[INFO] Found {len(scene_names)} scene directories in {DATASET_ROOT}")

    completed = 0
    failed = []

    for scene_name in scene_names:
        try:
            await run_single_scene_experiment(scene_name)
            completed += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] scene failed: {scene_name} -> {exc}")
            failed.append(scene_name)

    print("\n" + "=" * 100)
    print(f"[SUMMARY] completed={completed}, failed={len(failed)}")
    if failed:
        print("[SUMMARY] failed scenes:")
        for s in failed:
            print(f"  - {s}")
    print("=" * 100)


asyncio.ensure_future(run_move_effect_experiment())