import asyncio
import json
import math
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.usd

from pxr import Sdf, Usd, UsdGeom

ASSET_ROOT = "/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-asset"
CORL_ROOT = "/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024"
SCENE_CONFIGS = [
    "benchmark_eval/RigidObjDesk_0",
    "benchmark_eval/RigidObjCellShelfDesk_0",
    "benchmark_eval/RigidObjDeskWall_0",
    "benchmark_eval/RigidObjDoubleDoorCabinet_0",
    "benchmark_eval/RigidObjDrawer_0",
    "benchmark_eval/RigidObjDrawerShelf_0",
    "benchmark_eval/RigidObjEketShelf_0",
    "benchmark_eval/RigidObjLargeShelf_0",
    "benchmark_eval/RigidObjLargeShelfDesk_0",
    "benchmark_eval/RigidObjLayerShelf_0",
    "benchmark_eval/RigidObjRoundTable_0",
    "benchmark_eval/RigidObjSingleDoorCabinetDesk_0",
    "benchmark_eval/RigidObjTriangleShelfDesk_0",
]
CONFIG_INDEX=12
SCENE_CONFIG = SCENE_CONFIGS[CONFIG_INDEX]
SCENE_INDEX = 0
TASK_INDEX = 0

# If set, open this stage path first. Keep empty to use already-open stage.
OPEN_STAGE_PATH = ""

# If set, save to a new path. Keep empty to save current stage in place.
SAVE_STAGE_AS = ""

ROBOT_PRIM_PATH = "/Franka"
SCENE_PRIM_PATH = "/env/env"

# Safety switches for Script Editor stability.
PREFER_DATASET_SCENE_USD = False
ALLOW_URDF_FALLBACK =True

# ---------------------------------------------------------------------------
# Inlined from utils_data.py
# ---------------------------------------------------------------------------

def read_scene_yaml(scene_yaml_path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(scene_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid yaml format: {scene_yaml_path}")
        return data
    except Exception:
        pass

    import re as _re
    result: Dict[str, Any] = {"scene_list": []}
    in_scene_list = False
    with open(scene_yaml_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("name:"):
                result["name"] = line.split(":", 1)[1].strip()
                in_scene_list = False
                continue
            if line.startswith("num_tasks:"):
                result["num_tasks"] = int(line.split(":", 1)[1].strip())
                in_scene_list = False
                continue
            if line.startswith("scene_list:"):
                in_scene_list = True
                continue
            if in_scene_list and line.startswith("-"):
                result.setdefault("scene_list", []).append(line[1:].strip())
                continue
            in_scene_list = False
    if not result.get("scene_list"):
        raise ValueError(f"No scene_list found in yaml: {scene_yaml_path}")
    return result


def resolve_scene_config(scene_config: str, corl_root: str) -> str:
    if os.path.isfile(scene_config):
        return os.path.abspath(scene_config)
    base = os.path.join(corl_root, "InfiniGym", "isaacgymenvs", "config", "scene")
    candidates: List[str] = []
    candidates.append(os.path.join(base, scene_config))
    if not scene_config.endswith(".yaml"):
        candidates.append(os.path.join(base, f"{scene_config}.yaml"))
    if "/" not in scene_config and not scene_config.startswith("benchmark_"):
        candidates.append(os.path.join(base, "benchmark_eval", f"{scene_config}.yaml"))
        candidates.append(os.path.join(base, "benchmark_train", f"{scene_config}.yaml"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"Unable to resolve scene config '{scene_config}'. Checked under: {base}"
    )


def remap_asset_root(raw_root: str, asset_root: str, corl_root: str) -> str:
    if os.path.exists(raw_root):
        return raw_root
    import re as _re
    for token in ["benchmark_scenes", "benchmark_objects", "Task", "objects", "scenes", "combos"]:
        marker = f"/{token}/"
        idx = raw_root.find(marker)
        if idx != -1:
            rel = raw_root[idx + 1:]
            cand = os.path.join(asset_root, rel)
            if os.path.exists(cand):
                return cand
    if "IsaacGymEnvs/assets" in raw_root:
        ig_assets = os.path.join(corl_root, "InfiniGym", "assets")
        if os.path.exists(ig_assets):
            return ig_assets
    return raw_root


def sanitize_prim_name(name: str) -> str:
    import re as _re
    cleaned = _re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not cleaned:
        cleaned = "Prim"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def to_pose(state13: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(state13, dtype=np.float64)
    if arr.shape[0] < 7:
        raise ValueError(f"Expected state with >=7 values, got shape {arr.shape}")
    pos = arr[:3]
    quat_xyzw = arr[3:7]
    return pos, quat_xyzw


def extract_import_path(exec_result: Any, fallback_path: str) -> str:
    if isinstance(exec_result, str):
        return exec_result
    if isinstance(exec_result, tuple):
        for item in exec_result:
            if isinstance(item, str) and item.startswith("/"):
                return item
    return fallback_path


# ---------------------------------------------------------------------------
# Inlined from utils_sim.py
# ---------------------------------------------------------------------------

def set_prim_pose(stage, prim_path: str, pos_xyz: Sequence[float], quat_xyzw: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom as _UsdGeom
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for pose set: {prim_path}")
    xform = _UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    mat = Gf.Matrix4d()
    mat.SetTranslateOnly(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
    mat.SetRotateOnly(Gf.Rotation(Gf.Quatd(float(w), float(x), float(y), float(z))))
    xform.AddTransformOp().Set(mat)


def get_prim_world_pose(stage, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for world pose query: {prim_path}")
    xf = _UsdGeom.Xformable(prim)
    mat = xf.ComputeLocalToWorldTransform(_Usd.TimeCode.Default())
    tr = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    pos = np.array([float(tr[0]), float(tr[1]), float(tr[2])], dtype=np.float64)
    quat_xyzw = np.array([float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())], dtype=np.float64)
    return pos, quat_xyzw


# =========================
# Settings
# =========================
def _enable_extension(ext_name: str) -> None:
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


def _reference_stage_prim(stage, source_usd: str, source_prim_path: str, destination_path: str) -> str:
    if not os.path.isfile(source_usd):
        raise FileNotFoundError(f"Source USD not found: {source_usd}")

    src_stage = Usd.Stage.Open(source_usd)
    if src_stage is None:
        raise RuntimeError(f"Failed to open source USD: {source_usd}")

    src_prim = src_stage.GetPrimAtPath(source_prim_path)
    if not src_prim.IsValid():
        raise RuntimeError(
            f"Source prim path is invalid: {source_prim_path} in {source_usd}"
        )

    if stage.GetPrimAtPath(destination_path).IsValid():
        stage.RemovePrim(destination_path)
    UsdGeom.Xform.Define(stage, destination_path)
    dst_prim = stage.GetPrimAtPath(destination_path)
    dst_prim.GetReferences().AddReference(Sdf.Reference(source_usd, source_prim_path))
    return destination_path


def _candidate_dataset_roots() -> List[str]:
    roots: List[str] = []

    # 1) Next to currently opened stage (best match for Script Editor usage).
    try:
        ctx = omni.usd.get_context()
        stage_url = ctx.get_stage_url()
        if stage_url:
            stage_dir = os.path.dirname(stage_url)
            if stage_dir:
                roots.append(os.path.abspath(stage_dir))
    except Exception:
        pass

    # 2) Current working directory where Isaac Sim was launched.
    roots.append(os.path.abspath(os.getcwd()))

    # 3) Script location fallback.
    roots.append(os.path.abspath(os.path.dirname(__file__)))

    # De-duplicate while preserving order.
    uniq: List[str] = []
    for r in roots:
        if r and r not in uniq:
            uniq.append(r)
    return uniq


def _try_import_scene_from_dataset_cache(stage, destination_path: str, scene_index: int) -> Optional[str]:
    file_candidates: List[str] = []
    for root in _candidate_dataset_roots():
        dataset_dir = os.path.join(root, "dataset", f"scene_{scene_index:03d}")
        file_candidates.extend(
            [
                os.path.join(dataset_dir, "isaac_objects.usda"),
                os.path.join(dataset_dir, "isaac_objects.usd"),
                os.path.join(dataset_dir, "isaac_objects.usdc"),
            ]
        )

    existing = [p for p in file_candidates if os.path.isfile(p)]
    if not existing:
        print(
            f"[INFO] No dataset cache file found for scene_{scene_index:03d}. "
            f"searched_roots={_candidate_dataset_roots()}"
        )
        return None

    source_usd = existing[0]
    src_stage = Usd.Stage.Open(source_usd)
    if src_stage is None:
        return None

    # Common source paths for scene-only prim in cached stages.
    source_prim_candidates = [
        "/objects",
        "/env/env",
        "/World/FetchBench/Scene",
        "/World/Scene",
        "/env",
    ]
    source_prim_path = ""
    for cand in source_prim_candidates:
        if src_stage.GetPrimAtPath(cand).IsValid():
            source_prim_path = cand
            break

    if not source_prim_path:
        print(f"[INFO] Dataset cache found but no known scene prim in: {source_usd}")
        return None

    print(f"[INFO] Using dataset cache file: {source_usd}, source_prim={source_prim_path}")
    return _reference_stage_prim(stage, source_usd, source_prim_path, destination_path)


def _import_scene_asset(stage, scene_urdf: str, destination_path: str, scene_index: int) -> str:
    if PREFER_DATASET_SCENE_USD:
        cached_path = _try_import_scene_from_dataset_cache(stage, destination_path, scene_index)
        if cached_path:
            print(
                f"[INFO] Imported scene from cached dataset USD for scene_{scene_index:03d} "
                f"-> {destination_path}"
            )
            return cached_path

    if not ALLOW_URDF_FALLBACK:
        raise RuntimeError(
            "Scene USD cache not found and URDF fallback is disabled. "
            "Set ALLOW_URDF_FALLBACK = True to force URDF import (can crash in Script Editor)."
        )

    print("[WARN] Falling back to URDF importer. This path may be unstable in Script Editor.")
    return _import_urdf(
        stage=stage,
        urdf_file=scene_urdf,
        destination_path=destination_path,
        fix_base=True,
    )


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


def _load_fetchbench_task() -> Dict[str, Any]:
    scene_yaml = resolve_scene_config(SCENE_CONFIG, CORL_ROOT)
    scene_cfg = read_scene_yaml(scene_yaml)

    scene_list = scene_cfg["scene_list"]
    if SCENE_INDEX < 0 or SCENE_INDEX >= len(scene_list):
        raise IndexError(f"scene_index {SCENE_INDEX} out of range [0, {len(scene_list) - 1}]")

    task_rel = scene_list[SCENE_INDEX]
    task_dir = os.path.join(ASSET_ROOT, "Task", task_rel)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    with open(os.path.join(task_dir, "asset_config.json"), "r", encoding="utf-8") as f:
        asset_config = json.load(f)

    task_npz = np.load(os.path.join(task_dir, "task_config.npz"), allow_pickle=True)
    num_tasks = int(task_npz["task_init_state"].shape[0])
    if TASK_INDEX < 0 or TASK_INDEX >= num_tasks:
        raise IndexError(f"task_index {TASK_INDEX} out of range [0, {num_tasks - 1}]")

    actor_states = task_npz["task_init_state"][TASK_INDEX]
    if actor_states.shape[0] < 3:
        raise RuntimeError(f"Unexpected actor_states shape: {actor_states.shape}")

    return {
        "scene_yaml": scene_yaml,
        "task_dir": task_dir,
        "asset_config": asset_config,
        "actor_states": actor_states,
    }


async def _wait_frames(n: int) -> None:
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


async def main() -> None:
    _enable_extension("isaacsim.asset.importer.urdf")
    _enable_extension("omni.usd")

    ctx = omni.usd.get_context()

    if OPEN_STAGE_PATH:
        stage_path = os.path.abspath(OPEN_STAGE_PATH)
        if not os.path.isfile(stage_path):
            raise FileNotFoundError(f"stage file not found: {stage_path}")
        ok = ctx.open_stage(stage_path)
        if not ok:
            raise RuntimeError(f"Failed to open stage: {stage_path}")
        await _wait_frames(5)

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No opened stage in Isaac Sim. Open a USD stage first.")

    task_info = _load_fetchbench_task()
    asset_config = task_info["asset_config"]
    actor_states = task_info["actor_states"]

    scene_cfg_entry = dict(asset_config["scene_config"])
    scene_cfg_entry["asset_root"] = remap_asset_root(
        scene_cfg_entry["asset_root"], ASSET_ROOT, CORL_ROOT
    )
    scene_urdf = os.path.join(scene_cfg_entry["asset_root"], scene_cfg_entry["urdf_file"])
    if not os.path.isfile(scene_urdf):
        raise FileNotFoundError(f"Scene URDF not found: {scene_urdf}")

    if not stage.GetPrimAtPath("/env").IsValid():
        UsdGeom.Xform.Define(stage, "/env")

    scene_prim_path = _import_scene_asset(
        stage=stage,
        scene_urdf=scene_urdf,
        destination_path=SCENE_PRIM_PATH,
        scene_index=SCENE_INDEX,
    )

    fb_robot_pos, fb_robot_quat = to_pose(actor_states[0])
    fb_scene_pos, fb_scene_quat = to_pose(actor_states[2])
    sample_robot_pos, sample_robot_quat = get_prim_world_pose(stage, ROBOT_PRIM_PATH)

    t_fb_robot = _pose_to_matrix(fb_robot_pos, fb_robot_quat)
    t_fb_scene = _pose_to_matrix(fb_scene_pos, fb_scene_quat)
    t_sample_robot = _pose_to_matrix(sample_robot_pos, sample_robot_quat)

    t_robot_to_scene = np.linalg.inv(t_fb_robot) @ t_fb_scene
    t_sample_scene = t_sample_robot @ t_robot_to_scene

    sample_scene_pos, sample_scene_quat = _matrix_to_pose(t_sample_scene)
    set_prim_pose(stage, scene_prim_path, sample_scene_pos, sample_scene_quat)

    await _wait_frames(3)

    if SAVE_STAGE_AS:
        save_path = os.path.abspath(SAVE_STAGE_AS)
        ok = ctx.save_as_stage(save_path)
        if not ok:
            raise RuntimeError(f"Failed to save stage: {save_path}")
    else:
        ok = ctx.save_stage()
        if not ok:
            raise RuntimeError("Failed to save current stage")

    result = {
        "scene_yaml": task_info["scene_yaml"],
        "task_dir": task_info["task_dir"],
        "scene_urdf": scene_urdf,
        "scene_prim_path": scene_prim_path,
        "robot_prim_path": ROBOT_PRIM_PATH,
        "fb_robot_pos": np.asarray(fb_robot_pos).tolist(),
        "fb_scene_pos": np.asarray(fb_scene_pos).tolist(),
        "sample_robot_pos": np.asarray(sample_robot_pos).tolist(),
        "sample_scene_pos": np.asarray(sample_scene_pos).tolist(),
        "save_stage_as": SAVE_STAGE_AS,
    }
    print(json.dumps(result, indent=2))


async def _entrypoint() -> None:
    try:
        await main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] add_fetchbench_config failed: {exc}")


asyncio.ensure_future(_entrypoint())























