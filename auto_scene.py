import asyncio
import json
import math
import os
import random
import re
from pathlib import Path

import omni.usd
import omni.timeline
import omni.kit.app
import omni.client

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf


# =========================
# 설정값
# =========================

OBJECT_ROOT = "/objects"
KLT_PRIM_PATH = "/env/small_KLT"

NUM_OBJECTS = 15
RANDOM_SEED = None  # 재현하고 싶으면 예: 1234

CLEAR_EXISTING_OBJECTS = True

# KLT 기준으로 물체를 띄울 위치.
# KLT가 존재하면 KLT world position + offset 기준.
# KLT가 없으면 FALLBACK_DROP_CENTER 사용.
DROP_CENTER_OFFSET = (0.0, 0.0, 0.45)
FALLBACK_DROP_CENTER = (0.45, 0.0, 0.85)

# 흩뿌리는 범위. KLT 안에 넣고 싶으면 x/y 범위를 작게.
SPAWN_XY_RANGE = 0.16     # +/- 16cm
SPAWN_Z_MIN = 0.05
SPAWN_Z_MAX = 0.45

# 안정화 판정
SETTLE_MAX_SECONDS_1 = 25.0
SETTLE_MAX_SECONDS_2 = 20.0
SETTLE_STABLE_WINDOW_FRAMES = 90
SETTLE_POS_EPS = 0.0008          # meter/frame
SETTLE_ROT_EPS_RAD = 0.004       # rad/frame

# 스폰한 물체에 부여할 물리 속성
MESH_COLLISION_APPROX = "convexHull"

# 저장 시 workspace 직사각형 밖으로 벗어난 물체는 dataset에서 제외
SAVE_X_MIN = 0.0
SAVE_X_MAX = 0.9
SAVE_Y_MIN = -0.7
SAVE_Y_MAX = 0.7

# 최종 dataset 저장 위치
DATASET_ROOT = "/isaac-sim/corl2025/dataset"
JSON_FILENAME = "isaac_objects_for_moveit.json"
USDA_FILENAME = "isaac_objects.usda"
OBJ_DIR = "/isaac-sim/corl2025/obj"

# 가능하면 Physics asset 폴더를 우선 사용
YCB_SUBDIR_CANDIDATES = [
    "/Isaac/Props/YCB/Axis_Aligned",
]


# =========================
# 유틸
# =========================

def get_isaac_assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path
        return get_assets_root_path()
    except Exception:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        return get_assets_root_path()


def join_url(a, b):
    return a.rstrip("/") + "/" + b.lstrip("/")


def list_usd_files(folder_url):
    result, entries = omni.client.list(folder_url)
    if result != omni.client.Result.OK:
        return []

    usd_files = []
    for e in entries:
        name = e.relative_path
        if name.startswith("."):
            continue
        if name.lower().endswith((".usd", ".usda", ".usdc")):
            usd_files.append(join_url(folder_url, name))
    return sorted(usd_files)


def find_ycb_assets():
    root = get_isaac_assets_root()
    if root is None:
        raise RuntimeError("Isaac Sim assets root를 찾지 못했습니다.")

    tried = []
    for subdir in YCB_SUBDIR_CANDIDATES:
        folder = join_url(root, subdir)
        tried.append(folder)
        files = list_usd_files(folder)
        if files:
            print(f"[YCB] using folder: {folder}")
            print(f"[YCB] found {len(files)} usd files")
            return files, folder

    raise RuntimeError(
        "YCB USD 파일을 찾지 못했습니다.\n"
        "시도한 경로:\n" + "\n".join(tried) + "\n\n"
        "Content Browser에서 실제 YCB 폴더 경로를 확인한 뒤 "
        "YCB_SUBDIR_CANDIDATES를 수정하세요."
    )


def sanitize_name(s):
    s = Path(s).stem
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    if not s:
        s = "object"
    if s[0].isdigit():
        s = "obj_" + s
    return s


def get_next_dataset_index(dataset_root):
    root = Path(dataset_root)
    if not root.exists():
        return 0

    max_index = -1
    for child in root.iterdir():
        if not child.is_dir():
            continue

        match = re.fullmatch(r"scene_(\d+)", child.name)
        if match is None:
            continue

        max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def create_dataset_scene_paths(dataset_root):
    dataset_index = get_next_dataset_index(dataset_root)
    scene_dir = Path(dataset_root) / f"scene_{dataset_index:03d}"
    scene_dir.mkdir(parents=True, exist_ok=False)
    return dataset_index, scene_dir, scene_dir / JSON_FILENAME, scene_dir / USDA_FILENAME


def ensure_xform(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    return prim


def clear_children(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    children = list(root.GetChildren())
    for child in children:
        stage.RemovePrim(child.GetPath())


def ensure_physics_scene(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
    print("[Physics] created /physicsScene")
    return scene


def ensure_startup_state(stage):
    """
    Script 시작 시 초기 상태를 강제:
    1) /objects 하위 prim 모두 삭제
    2) small_KLT를 active + visible 상태로 복구
    """
    objects_root = ensure_xform(stage, OBJECT_ROOT)
    children = list(objects_root.GetChildren())
    if children:
        for child in children:
            stage.RemovePrim(child.GetPath())
        print(f"[Init] cleared {len(children)} prims under {OBJECT_ROOT}")
    else:
        print(f"[Init] {OBJECT_ROOT} already empty")

    klt = stage.GetPrimAtPath(KLT_PRIM_PATH)
    if not klt.IsValid():
        print(f"[Init][Warn] KLT prim not found: {KLT_PRIM_PATH}")
        return

    # Restore active state without touching payloads (stage.Load blocks on remote assets).
    if not klt.IsActive():
        klt.SetActive(True)

    # Restore visibility on root and all descendants.
    for p in Usd.PrimRange(klt):
        imageable = UsdGeom.Imageable(p)
        if imageable:
            imageable.MakeVisible()

        # Re-enable any colliders that were disabled during removal.
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(True)

    print(f"[Init] active + visible + colliders enabled: {KLT_PRIM_PATH}")


def set_prim_transform(prim, translate, rotate_xyz_deg):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    t_op = xform.AddTranslateOp()
    r_op = xform.AddRotateXYZOp()

    t_op.Set(Gf.Vec3d(*translate))
    r_op.Set(Gf.Vec3f(*rotate_xyz_deg))


def matrix_translation(m):
    t = m.ExtractTranslation()
    return [float(t[0]), float(t[1]), float(t[2])]


def matrix_quat_xyzw(m):
    q = m.ExtractRotation().GetQuat()
    imag = q.GetImaginary()
    return [float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]


def matrix_quat_wxyz(m):
    q = m.ExtractRotation().GetQuat()
    imag = q.GetImaginary()
    return [float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]


def quat_angle_rad(q1, q2):
    # q = [w, x, y, z]
    n1 = math.sqrt(sum(v * v for v in q1))
    n2 = math.sqrt(sum(v * v for v in q2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    q1 = [v / n1 for v in q1]
    q2 = [v / n2 for v in q2]

    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def pose_snapshot(stage, prim_paths):
    snap = {}
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue

        m = omni.usd.get_world_transform_matrix(prim)
        snap[path] = {
            "t": matrix_translation(m),
            "q": matrix_quat_wxyz(m),
        }
    return snap


def pose_delta(prev, curr):
    max_pos = 0.0
    max_rot = 0.0

    for path, c in curr.items():
        if path not in prev:
            continue

        p = prev[path]
        dt = math.sqrt(sum((c["t"][i] - p["t"][i]) ** 2 for i in range(3)))
        dr = quat_angle_rad(p["q"], c["q"])

        max_pos = max(max_pos, dt)
        max_rot = max(max_rot, dr)

    return max_pos, max_rot


async def wait_frames(n):
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


async def reset_scene_to_initial_state():
    """
    타임라인을 정지하고 시간을 0으로 되돌려
    시뮬레이션 상태(로봇 포함)를 초기 상태로 리셋.
    """
    timeline = omni.timeline.get_timeline_interface()
    was_playing = False
    if hasattr(timeline, "is_playing"):
        try:
            was_playing = bool(timeline.is_playing())
        except Exception:
            was_playing = False

    timeline.stop()

    # Let the stop propagate before mutating the stage. When edits race with an
    # in-flight physics step, Isaac Sim can appear to freeze.
    if was_playing:
        for _ in range(10):
            await wait_frames(1)
            if hasattr(timeline, "is_playing"):
                try:
                    if not timeline.is_playing():
                        break
                except Exception:
                    break

    # Rewind to authored initial state.
    try:
        timeline.set_current_time(0.0)
    except Exception:
        pass

    await wait_frames(2)
    print("[Init] timeline stopped and rewound to t=0")


async def wait_until_settled(stage, prim_paths, label, max_seconds):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()

    timeline.play()
    await wait_frames(10)

    fps_assume = 60.0
    max_frames = int(max_seconds * fps_assume)

    stable_count = 0
    prev = pose_snapshot(stage, prim_paths)

    print(f"[Settle] waiting: {label}")

    for frame in range(max_frames):
        await app.next_update_async()

        curr = pose_snapshot(stage, prim_paths)
        max_pos, max_rot = pose_delta(prev, curr)
        prev = curr

        if max_pos < SETTLE_POS_EPS and max_rot < SETTLE_ROT_EPS_RAD:
            stable_count += 1
        else:
            stable_count = 0

        if frame % 120 == 0:
            print(
                f"[Settle:{label}] frame={frame}, "
                f"max_pos_delta={max_pos:.6f}, "
                f"max_rot_delta={max_rot:.6f}, "
                f"stable_count={stable_count}"
            )

        if stable_count >= SETTLE_STABLE_WINDOW_FRAMES:
            print(f"[Settle] stable: {label}")
            return True

    print(f"[Settle] timeout but continuing: {label}")
    return False


def get_drop_center(stage):
    klt = stage.GetPrimAtPath(KLT_PRIM_PATH)
    if klt.IsValid():
        m = omni.usd.get_world_transform_matrix(klt)
        t = matrix_translation(m)
        return (
            t[0] + DROP_CENTER_OFFSET[0],
            t[1] + DROP_CENTER_OFFSET[1],
            t[2] + DROP_CENTER_OFFSET[2],
        )

    print(f"[Warn] KLT prim not found: {KLT_PRIM_PATH}")
    print(f"[Warn] using fallback drop center: {FALLBACK_DROP_CENTER}")
    return FALLBACK_DROP_CENTER


def apply_physics_to_object(object_prim):
    """
    object 루트에는 rigid body,
    하위 Mesh prim에는 collision API를 적용.
    질량은 명시하지 않고 physics 엔진 자동 계산을 사용.
    """
    if not object_prim.IsValid():
        return 0

    if not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(object_prim)
    rigid_api = UsdPhysics.RigidBodyAPI(object_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)

    collider_count = 0
    for prim in Usd.PrimRange(object_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.CreateCollisionEnabledAttr(True)

        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_collision_api.CreateApproximationAttr(MESH_COLLISION_APPROX)

        collider_count += 1

    return collider_count


async def spawn_ycb_objects(stage, asset_paths):
    ensure_xform(stage, OBJECT_ROOT)

    if CLEAR_EXISTING_OBJECTS:
        clear_children(stage, OBJECT_ROOT)

    center = get_drop_center(stage)
    selected = random.sample(asset_paths, NUM_OBJECTS)

    spawned_paths = []
    spawned_assets = {}

    for i, asset_path in enumerate(selected):
        base_name = sanitize_name(asset_path)
        # USD prim identifiers cannot start with a digit.
        prim_name = f"obj_{i:02d}_{base_name}"
        prim_path = f"{OBJECT_ROOT}/{prim_name}"

        print(f"[Spawn] {prim_path} from {asset_path}")
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(asset_path)

        # Let USD composition catch up before applying transforms and physics.
        await wait_frames(1)

        x = center[0] + random.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE)
        y = center[1] + random.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE)
        z = center[2] + random.uniform(SPAWN_Z_MIN, SPAWN_Z_MAX)

        # 랜덤 회전. 너무 격하게 튀면 X/Y는 줄이고 Z만 랜덤으로 두세요.
        rx = random.uniform(-25.0, 25.0)
        ry = random.uniform(-25.0, 25.0)
        rz = random.uniform(0.0, 360.0)

        set_prim_transform(
            prim,
            translate=(x, y, z),
            rotate_xyz_deg=(rx, ry, rz),
        )

        collider_count = apply_physics_to_object(prim)

        spawned_paths.append(prim_path)
        spawned_assets[prim_path] = asset_path

        print(f"[Spawn] {prim_path}")
        print(f"        asset={asset_path}")
        print(f"        xyz=({x:.3f}, {y:.3f}, {z:.3f}) rpy=({rx:.1f}, {ry:.1f}, {rz:.1f})")
        print(f"        colliders={collider_count} mass=auto")

        # Spread heavy stage edits over multiple frames to avoid UI stalls.
        await wait_frames(1)

    return spawned_paths, spawned_assets


def remove_klt(stage):
    prim = stage.GetPrimAtPath(KLT_PRIM_PATH)
    if not prim.IsValid():
        print(f"[KLT] not found: {KLT_PRIM_PATH}")
        return "not_found"

    # Never unload payloads — remote asset fetches block the main thread.
    # Instead: hide the mesh and disable all colliders so objects fall through,
    # while keeping the prim active so it can be restored cheaply next run.
    for p in Usd.PrimRange(prim):
        imageable = UsdGeom.Imageable(p)
        if imageable:
            imageable.MakeInvisible()

        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)

    print(f"[KLT] hidden + colliders disabled: {KLT_PRIM_PATH}")
    return "hidden"


def should_save_scene(stage, object_paths):
    for path in object_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[Save] cancel scene save: invalid prim {path}")
            return False

        m = omni.usd.get_world_transform_matrix(prim)
        position = matrix_translation(m)

        if not (
            SAVE_X_MIN <= position[0] <= SAVE_X_MAX and
            SAVE_Y_MIN <= position[1] <= SAVE_Y_MAX
        ):
            print(
                f"[Save] cancel scene save: {path} outside bounds "
                f"x={position[0]:.3f}, y={position[1]:.3f} "
                f"x[{SAVE_X_MIN:.3f}, {SAVE_X_MAX:.3f}] "
                f"y[{SAVE_Y_MIN:.3f}, {SAVE_Y_MAX:.3f}]"
            )
            return False

    print(f"[Save] all {len(object_paths)} objects are within save bounds")
    return True


def cleanup_dataset_scene_dir(scene_dir):
    scene_dir = Path(scene_dir)
    if scene_dir.exists():
        try:
            scene_dir.rmdir()
            print(f"[Save] removed empty dataset dir {scene_dir}")
        except OSError:
            print(f"[Save][Warn] dataset dir not empty, left in place: {scene_dir}")


def save_objects_json(stage, object_paths, spawned_assets, save_path):
    objects = []

    for path in object_paths:
        prim = stage.GetPrimAtPath(path)
        
        if not prim.IsValid():
            continue

        m = omni.usd.get_world_transform_matrix(prim)

        mesh_name = prim.GetName()
        mesh_name = "0" + mesh_name[1:]
        mesh_path = os.path.join(OBJ_DIR, f"{mesh_name}.obj")

        objects.append({
            "id": prim.GetName(),
            "prim_path": path,
            "usd_asset_path": spawned_assets.get(path),
            "frame_id": "world",
            "translation_xyz": matrix_translation(m),
            "rotation_xyzw": matrix_quat_xyzw(m),
            "moveit_mesh_path": mesh_path,
            "note": "Fill moveit_mesh_path later, e.g. package://my_scene_meshes/meshes/ycb/object.obj",
        })

    data = {
        "object_root": OBJECT_ROOT,
        "klt_prim_path": KLT_PRIM_PATH,
        "num_objects": len(objects),
        "objects": objects,
    }

    Path(save_path).write_text(json.dumps(data, indent=2))
    print(f"[Save] wrote {len(objects)} objects to {save_path}")


def save_objects_usda(stage, object_paths, save_path):
    objects_prim = stage.GetPrimAtPath(OBJECT_ROOT)
    if not objects_prim.IsValid():
        raise RuntimeError(f"OBJECT_ROOT not found: {OBJECT_ROOT}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    dst_stage = Usd.Stage.CreateNew(save_path)
    dst_stage.SetDefaultPrim(dst_stage.DefinePrim(OBJECT_ROOT, "Xform"))

    src_layer = stage.GetRootLayer()
    dst_layer = dst_stage.GetRootLayer()

    for path in object_paths:
        Sdf.CopySpec(src_layer, path, dst_layer, path)

    dst_layer.Save()
    print(f"[Save] wrote {len(object_paths)} objects to {save_path}")


async def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()

    dataset_index, scene_dir, json_save_path, usda_save_path = create_dataset_scene_paths(DATASET_ROOT)
    print(f"[Save] dataset_index={dataset_index:03d} dir={scene_dir}")

    await reset_scene_to_initial_state()
    ensure_startup_state(stage)
    ensure_physics_scene(stage)

    asset_paths, ycb_folder = find_ycb_assets()

    if len(asset_paths) < NUM_OBJECTS:
        raise RuntimeError(f"YCB asset 개수가 부족합니다: {len(asset_paths)} < {NUM_OBJECTS}")

    print("[Phase] spawning objects")
    object_paths, spawned_assets = await spawn_ycb_objects(stage, asset_paths)

    # reference asset compose/load 여유
    print("[Phase] waiting for post-spawn composition")
    await wait_frames(30)

    # 1차 안정화: KLT 안에서 뭉치게 하기
    print("[Phase] entering first settle")
    await wait_until_settled(
        stage,
        object_paths,
        label="with small_KLT",
        max_seconds=SETTLE_MAX_SECONDS_1,
    )

    # KLT 제거
    remove_result = remove_klt(stage)
    print(f"[KLT] remove result: {remove_result}")

    # payload unload/deactivate가 physics에 반영되도록 몇 프레임 넘김
    await wait_frames(30)

    # 2차 안정화: KLT 없어진 뒤 최종 clutter
    await wait_until_settled(
        stage,
        object_paths,
        label="after removing small_KLT",
        max_seconds=SETTLE_MAX_SECONDS_2,
    )

    # 멈춘 상태로 두기
    omni.timeline.get_timeline_interface().pause()

    if not should_save_scene(stage, object_paths):
        cleanup_dataset_scene_dir(scene_dir)
        print("\n[SKIP SAVE]")
        print("Scene was not saved because at least one object is outside bounds.")
        return

    save_objects_json(stage, object_paths, spawned_assets, str(json_save_path))
    save_objects_usda(stage, object_paths, str(usda_save_path))

    print("\n[DONE]")
    print(f"Final object JSON: {json_save_path}")
    print(f"Final object USDA: {usda_save_path}")
    print("GUI에서 scene도 저장하려면 지금 상태에서 File -> Save As 하세요.")


asyncio.ensure_future(main())
