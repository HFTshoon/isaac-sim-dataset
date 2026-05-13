import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def read_scene_yaml(scene_yaml_path: str) -> Dict[str, Any]:
    """Read scene yaml with a PyYAML fallback-free parser for simple files."""
    try:
        import yaml  # type: ignore

        with open(scene_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid yaml format: {scene_yaml_path}")
        return data
    except Exception:
        pass

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
    """Resolve scene config input to an existing yaml path."""
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
        f"Unable to resolve scene config '{scene_config}'. "
        f"Checked under: {base}"
    )


def remap_asset_root(raw_root: str, asset_root: str, corl_root: str) -> str:
    """Map absolute paths embedded in FetchBench configs to local workspace paths."""
    if os.path.exists(raw_root):
        return raw_root

    for token in ["benchmark_scenes", "benchmark_objects", "Task", "objects", "scenes", "combos"]:
        marker = f"/{token}/"
        idx = raw_root.find(marker)
        if idx != -1:
            rel = raw_root[idx + 1 :]
            cand = os.path.join(asset_root, rel)
            if os.path.exists(cand):
                return cand

    if "IsaacGymEnvs/assets" in raw_root:
        ig_assets = os.path.join(corl_root, "InfiniGym", "assets")
        if os.path.exists(ig_assets):
            return ig_assets

    return raw_root


def sanitize_prim_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
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
