import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCENE_INFO_PATH = Path(__file__).with_name("scene_info.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_FILENAME = "isaac_objects_for_moveit.json"
DEFAULT_FRANKA_CAMERA_PRIM_PATH = "/Franka/panda_hand/FrankaCamera"
BEV_CAMERA_PRIM_PATH = "/World/BEVCamera"
BEV_CAMERA_HEIGHT_M = 0.40
CENTER_TOL_FRAC = 1.0 / 6.0
MAX_OBJECT_CLASSES = 15
SETTLE_FRAMES = 8


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
		raise KeyError(f"Unknown scene '{scene_key}'. Available: {available}")
	return scene_info_all[scene_key]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Capture one BEV RGB/depth/class view.")
	parser.add_argument("--scene", type=str, default=None, help="Scene key from scene_info.json, e.g. 01.")
	parser.add_argument("--scene-num", type=str, default=None, help="Scene index in dataset root, e.g. 000.")
	parser.add_argument("--scene-json", type=str, default=None, help="Explicit path to isaac_objects_for_moveit.json.")
	parser.add_argument("--base-usd", type=str, default=None)
	parser.add_argument("--camera-prim-path", type=str, default=DEFAULT_FRANKA_CAMERA_PRIM_PATH)
	parser.add_argument("--bev-camera-prim-path", type=str, default=BEV_CAMERA_PRIM_PATH)
	parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
	parser.add_argument("--headless", action="store_true")
	return parser.parse_args()


ARGS = _parse_args()
scene_info = _load_scene_info(ARGS.scene) if ARGS.scene else None

if ARGS.scene_json:
	scene_json_path = _resolve_repo_path(ARGS.scene_json)
elif scene_info is not None and ARGS.scene_num is not None:
	try:
		scene_num = int(ARGS.scene_num)
	except ValueError as exc:
		raise ValueError(f"Invalid --scene-num: {ARGS.scene_num}") from exc
	dataset_root = _resolve_repo_path(str(scene_info.get("dataset_root")))
	scene_json_path = dataset_root / f"scene_{scene_num:03d}" / JSON_FILENAME
else:
	scene_json_path = None

if scene_info is not None and ARGS.base_usd is None:
	ARGS.base_usd = str(_resolve_repo_path(str(scene_info.get("scene_usd"))))

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.usd
import PIL.Image
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom


def _step(world: World, n: int) -> None:
	for _ in range(n):
		world.step(render=True)


def _save_rgb(rgba: np.ndarray, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if rgba is None:
		carb.log_warn(f"RGB data is None, skipping save: {path}")
		return
	arr = np.asarray(rgba)
	if arr.ndim == 3 and arr.shape[2] == 4:
		img = PIL.Image.fromarray(arr[:, :, :3].astype(np.uint8), "RGB")
	elif arr.ndim == 3 and arr.shape[2] == 3:
		img = PIL.Image.fromarray(arr.astype(np.uint8), "RGB")
	else:
		carb.log_warn(f"Unexpected RGB shape {arr.shape}, skipping: {path}")
		return
	img.save(str(path))


def _save_depth(depth: np.ndarray, path: Path, expected_hw: tuple[int, int] | None = None) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if depth is None:
		carb.log_warn(f"Depth data is None, skipping save: {path}")
		return

	arr = np.asarray(depth)
	if arr.size == 0:
		carb.log_warn(f"Depth data is empty, skipping save: {path}")
		return

	if arr.ndim == 3 and arr.shape[2] == 1:
		arr = arr[:, :, 0]
	elif arr.ndim == 1 and expected_hw is not None and arr.size == expected_hw[0] * expected_hw[1]:
		arr = arr.reshape(expected_hw)

	if arr.ndim != 2:
		carb.log_warn(f"Unexpected depth shape {arr.shape}, skipping: {path}")
		return

	arr = np.asarray(arr, dtype=np.float32)
	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
	arr_mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
	img = PIL.Image.fromarray(arr_mm)
	img.save(str(path))


def _set_semantic_class_label(prim: Usd.Prim, class_label: str) -> None:
	if not prim.IsValid():
		return
	sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
	sem_api.CreateSemanticTypeAttr().Set("class")
	sem_api.CreateSemanticDataAttr().Set(class_label)


def _build_object_class_mapping(
	all_objects_data: list[dict],
	target_prim_path: str,
	max_classes: int = MAX_OBJECT_CLASSES,
) -> tuple[dict[str, tuple[int, str]], list[dict], dict[str, int]]:
	prim_paths = []
	for obj in all_objects_data:
		p = obj.get("prim_path")
		if p and p not in prim_paths:
			prim_paths.append(p)

	ordered_paths = []
	if target_prim_path in prim_paths:
		ordered_paths.append(target_prim_path)
	for p in prim_paths:
		if p != target_prim_path:
			ordered_paths.append(p)
	ordered_paths = ordered_paths[:max_classes]

	prim_to_class = {}
	class_entries = []
	label_to_class_id = {}
	for class_id, prim_path in enumerate(ordered_paths, start=1):
		class_label = f"class_{class_id:02d}"
		prim_to_class[prim_path] = (class_id, class_label)
		label_to_class_id[class_label] = class_id
		class_entries.append(
			{
				"class_id": class_id,
				"prim_path": prim_path,
				"label": class_label,
				"is_target": prim_path == target_prim_path,
			}
		)
	return prim_to_class, class_entries, label_to_class_id


def _apply_object_class_semantics(stage: Usd.Stage, prim_to_class: dict[str, tuple[int, str]]) -> None:
	for prim_path, (_, class_label) in prim_to_class.items():
		root = stage.GetPrimAtPath(prim_path)
		if not root.IsValid():
			continue
		for prim in Usd.PrimRange(root):
			_set_semantic_class_label(prim, class_label)


def _extract_semantic_payload(semantic_frame) -> tuple[np.ndarray | None, dict | None]:
	if semantic_frame is None:
		return None, None
	if isinstance(semantic_frame, dict):
		data = semantic_frame.get("data")
		info = semantic_frame.get("info")
		return (np.asarray(data) if data is not None else None), (info if isinstance(info, dict) else None)
	return np.asarray(semantic_frame), None


def _collect_labels_from_info_entry(entry) -> list[str]:
	labels = []
	if isinstance(entry, str):
		labels.append(entry)
	elif isinstance(entry, dict):
		for value in entry.values():
			if isinstance(value, str):
				labels.append(value)
			elif isinstance(value, (list, tuple)):
				labels.extend([v for v in value if isinstance(v, str)])
	elif isinstance(entry, (list, tuple)):
		labels.extend([v for v in entry if isinstance(v, str)])
	return labels


def _build_semantic_id_to_class_map(info: dict | None, label_to_class_id: dict[str, int]) -> dict[int, int]:
	sid_to_class = {}
	if not info:
		return sid_to_class

	id_maps = []
	for key in ["idToLabels", "idToSemantics"]:
		value = info.get(key)
		if isinstance(value, dict):
			id_maps.append(value)

	for id_map in id_maps:
		for sid_key, payload in id_map.items():
			try:
				sid = int(sid_key)
			except Exception:
				continue
			class_id = 0
			for label in _collect_labels_from_info_entry(payload):
				if label in label_to_class_id:
					class_id = int(label_to_class_id[label])
					break
			sid_to_class[sid] = class_id
	return sid_to_class


def _semantic_to_class_mask(
	semantic_data: np.ndarray | None,
	expected_hw: tuple[int, int],
	sid_to_class: dict[int, int],
) -> np.ndarray | None:
	if semantic_data is None:
		return None
	arr = np.asarray(semantic_data)
	if arr.size == 0:
		return None

	if arr.ndim == 3 and arr.shape[2] == 1:
		arr = arr[:, :, 0]
	elif arr.ndim == 1 and arr.size == expected_hw[0] * expected_hw[1]:
		arr = arr.reshape(expected_hw)

	if arr.ndim != 2:
		return None

	arr = np.asarray(arr, dtype=np.uint32)
	class_mask = np.zeros(arr.shape, dtype=np.uint8)
	if not sid_to_class:
		return class_mask

	flat = arr.reshape(-1)
	out = np.zeros(flat.shape, dtype=np.uint8)
	unique_ids = np.unique(flat)
	for sid in unique_ids:
		class_id = sid_to_class.get(int(sid), 0)
		if class_id > 0:
			out[flat == sid] = np.uint8(class_id)
	class_mask[:, :] = out.reshape(arr.shape)
	return class_mask


def _save_class_mask(class_mask: np.ndarray, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if class_mask is None:
		carb.log_warn(f"Class mask is None, skipping save: {path}")
		return
	arr = np.asarray(class_mask, dtype=np.uint8)
	if arr.ndim != 2:
		carb.log_warn(f"Unexpected class mask shape {arr.shape}, skipping: {path}")
		return
	img = PIL.Image.fromarray(arr)
	img.save(str(path))


def _build_intrinsics_payload(camera: Camera, stage: Usd.Stage, camera_prim_path: str) -> dict:
	width, height = [int(v) for v in camera.get_resolution()]

	K = None
	try:
		K_raw = camera.get_intrinsics_matrix()
		K = np.asarray(K_raw, dtype=np.float64)
	except Exception:
		focal_length = float(camera.get_focal_length())
		horizontal_aperture = float(camera.get_horizontal_aperture())
		vertical_aperture = float(camera.get_vertical_aperture())
		fx = width * focal_length / horizontal_aperture if abs(horizontal_aperture) > 1e-12 else 0.0
		fy = height * focal_length / vertical_aperture if abs(vertical_aperture) > 1e-12 else 0.0
		cx = width * 0.5
		cy = height * 0.5
		K = np.array(
			[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
			dtype=np.float64,
		)

	fx = float(K[0, 0])
	fy = float(K[1, 1])
	cx = float(K[0, 2])
	cy = float(K[1, 2])

	projection = None
	camera_prim = stage.GetPrimAtPath(camera_prim_path)
	if camera_prim.IsValid():
		proj_attr = camera_prim.GetAttribute("cameraProjectionType")
		if proj_attr and proj_attr.IsValid():
			projection = proj_attr.Get()

	return {
		"camera_prim_path": camera_prim_path,
		"projection": projection,
		"width": width,
		"height": height,
		"fx": fx,
		"fy": fy,
		"cx": cx,
		"cy": cy,
		"K": K.tolist(),
	}


def _capture_rgbd_semantic_with_retry(camera: Camera, world: World, max_retry_frames: int = 24):
	for _ in range(max_retry_frames + 1):
		rgba = camera.get_rgba()
		depth = camera.get_depth()
		semantic_frame = camera.get_current_frame().get("semantic_segmentation")
		semantic_data, semantic_info = _extract_semantic_payload(semantic_frame)
		if rgba is not None and depth is not None and semantic_data is not None:
			if np.asarray(rgba).size > 0 and np.asarray(depth).size > 0 and np.asarray(semantic_data).size > 0:
				return rgba, depth, semantic_data, semantic_info
		world.step(render=True)
	return None, None, None, None


def _get_prim_world_pose(stage: Usd.Stage, prim_path: str):
	prim = stage.GetPrimAtPath(prim_path)
	if not prim.IsValid():
		return None, None
	m = omni.usd.get_world_transform_matrix(prim)
	t = m.ExtractTranslation()
	q = m.ExtractRotation().GetQuat()
	im = q.GetImaginary()
	pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
	quat_xyzw = np.array([float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())], dtype=np.float64)
	return pos, quat_xyzw


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
	x, y, z, w = [float(v) for v in q_xyzw]
	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z
	return np.array([
		[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
		[2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
		[2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
	], dtype=np.float64)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
	tr = float(np.trace(R))
	if tr > 0.0:
		S = np.sqrt(tr + 1.0) * 2.0
		w = 0.25 * S
		x = (R[2, 1] - R[1, 2]) / S
		y = (R[0, 2] - R[2, 0]) / S
		z = (R[1, 0] - R[0, 1]) / S
	elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
		S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
		w = (R[2, 1] - R[1, 2]) / S
		x = 0.25 * S
		y = (R[0, 1] + R[1, 0]) / S
		z = (R[0, 2] + R[2, 0]) / S
	elif R[1, 1] > R[2, 2]:
		S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
		w = (R[0, 2] - R[2, 0]) / S
		x = (R[0, 1] + R[1, 0]) / S
		y = 0.25 * S
		z = (R[1, 2] + R[2, 1]) / S
	else:
		S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
		w = (R[1, 0] - R[0, 1]) / S
		x = (R[0, 2] + R[2, 0]) / S
		y = (R[1, 2] + R[2, 1]) / S
		z = 0.25 * S
	q = np.array([x, y, z, w], dtype=np.float64)
	n = np.linalg.norm(q)
	if n < 1e-12:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
	return q / n


def _build_c2w(position, quat_xyzw) -> list[list[float]]:
	R = _quat_xyzw_to_rotmat(np.asarray(quat_xyzw, dtype=np.float64))
	mat = np.eye(4, dtype=np.float64)
	mat[:3, :3] = R
	mat[:3, 3] = np.asarray(position, dtype=np.float64)
	return mat.tolist()


def _look_at_quat_xyzw(cam_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
	forward = target_pos - cam_pos
	fn = np.linalg.norm(forward)
	if fn < 1e-8:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
	forward = forward / fn

	up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
	z_axis = -forward
	x_axis = np.cross(up_hint, z_axis)
	xn = np.linalg.norm(x_axis)
	if xn < 1e-8:
		up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
		x_axis = np.cross(up_hint, z_axis)
		xn = np.linalg.norm(x_axis)
		if xn < 1e-8:
			return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
	x_axis = x_axis / xn
	y_axis = np.cross(z_axis, x_axis)
	R = np.stack([x_axis, y_axis, z_axis], axis=1)
	return _rotmat_to_quat_xyzw(R)


def _set_camera_world_pose(stage: Usd.Stage, camera_prim_path: str, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
	prim = stage.GetPrimAtPath(camera_prim_path)
	if not prim.IsValid():
		raise RuntimeError(f"Camera prim not found: {camera_prim_path}")
	xf = UsdGeom.Xformable(prim)
	translate_op = None
	orient_op = None
	for op in xf.GetOrderedXformOps():
		op_name = op.GetOpName()
		if op_name.endswith("xformOp:translate"):
			translate_op = op
		elif op_name.endswith("xformOp:orient"):
			orient_op = op

	if translate_op is None:
		translate_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
	if orient_op is None:
		orient_op = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

	translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
	orient_op.Set(Gf.Quatd(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))


def _project_world_to_image(world_pt: np.ndarray, cam_pos: np.ndarray, cam_quat_xyzw: np.ndarray, K: np.ndarray):
	R = _quat_xyzw_to_rotmat(cam_quat_xyzw)
	pc = R.T @ (world_pt - cam_pos)
	# USD camera looks along local -Z.
	if pc[2] >= -1e-8:
		return None
	x = pc[0] / (-pc[2])
	y = pc[1] / (-pc[2])
	u = float(K[0, 0] * x + K[0, 2])
	v = float(K[1, 1] * y + K[1, 2])
	return np.array([u, v], dtype=np.float64)


def _ensure_camera_prim_like_source(stage: Usd.Stage, src_camera_path: str, dst_camera_path: str) -> None:
	src = stage.GetPrimAtPath(src_camera_path)
	if not src.IsValid() or src.GetTypeName() != "Camera":
		raise RuntimeError(f"Source camera prim invalid: {src_camera_path}")

	dst = stage.GetPrimAtPath(dst_camera_path)
	if not dst.IsValid():
		UsdGeom.Camera.Define(stage, dst_camera_path)
		dst = stage.GetPrimAtPath(dst_camera_path)

	for attr_name in [
		"projection",
		"cameraProjectionType",
		"focalLength",
		"horizontalAperture",
		"verticalAperture",
		"horizontalApertureOffset",
		"verticalApertureOffset",
		"clippingRange",
		"clippingPlanes",
		"fStop",
		"focusDistance",
	]:
		src_attr = src.GetAttribute(attr_name)
		if not src_attr or not src_attr.IsValid():
			continue
		value = src_attr.Get()
		if value is None:
			continue
		dst_attr = dst.GetAttribute(attr_name)
		if not dst_attr or not dst_attr.IsValid():
			dst_attr = dst.CreateAttribute(attr_name, src_attr.GetTypeName())
		dst_attr.Set(value)


def main() -> None:
	if scene_json_path is None:
		raise ValueError("Provide --scene and --scene-num, or an explicit --scene-json path.")
	if not scene_json_path.is_file():
		raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

	with scene_json_path.open(encoding="utf-8") as f:
		scene_data = json.load(f)
	target_prim_path = scene_data.get("target_object_prim_path", "")
	all_objects_data = scene_data.get("objects", [])

	if target_prim_path == "":
		raise ValueError("target_object_prim_path is missing in scene json.")

	if ARGS.base_usd is None:
		raise ValueError("Could not determine base USD. Pass --scene or --base-usd.")
	base_usd = Path(ARGS.base_usd).resolve()
	if not base_usd.is_file():
		raise FileNotFoundError(f"Base USD not found: {base_usd}")

	scene_dir = scene_json_path.parent
	bev_dir = scene_dir / "bev"
	bev_dir.mkdir(parents=True, exist_ok=True)
	rgb_path = bev_dir / "rgb_bev.png"
	depth_path = bev_dir / "depth_bev.png"
	class_path = bev_dir / "class_bev.png"
	pose_json_path = bev_dir / "pose.json"
	intrinsics_json_path = bev_dir / "intrinsics.json"

	usd_context = omni.usd.get_context()
	if not usd_context.open_stage(str(base_usd)):
		raise RuntimeError(f"Failed to open stage: {base_usd}")
	while usd_context.get_stage_loading_status()[2] > 0:
		simulation_app.update()
	stage = usd_context.get_stage()
	if stage is None:
		raise RuntimeError("USD stage unavailable.")

	_ensure_camera_prim_like_source(stage, ARGS.camera_prim_path, ARGS.bev_camera_prim_path)

	cam_w, cam_h = (1280, 720)
	if ARGS.resolution is not None:
		cam_w, cam_h = int(ARGS.resolution[0]), int(ARGS.resolution[1])

	world = World(stage_units_in_meters=1.0)
	camera = Camera(
		prim_path=ARGS.bev_camera_prim_path,
		name="bev_camera",
		resolution=(cam_w, cam_h),
		render_product_path=None,
	)
	world.scene.add(camera)

	prim_to_class, class_entries, label_to_class_id = _build_object_class_mapping(
		all_objects_data=all_objects_data,
		target_prim_path=target_prim_path,
		max_classes=MAX_OBJECT_CLASSES,
	)
	_apply_object_class_semantics(stage, prim_to_class)

	world.reset()
	camera.initialize()
	camera.add_distance_to_image_plane_to_frame()
	camera.add_semantic_segmentation_to_frame({"colorize": False})
	world.play()
	_step(world, 10)

	target_pos, _ = _get_prim_world_pose(stage, target_prim_path)
	if target_pos is None:
		raise RuntimeError(f"Target prim pose unavailable: {target_prim_path}")

	cam_pos = target_pos.copy()
	cam_pos[2] = target_pos[2] + BEV_CAMERA_HEIGHT_M
	cam_quat = _look_at_quat_xyzw(cam_pos, target_pos)

	intr = _build_intrinsics_payload(camera, stage, ARGS.bev_camera_prim_path)
	K = np.asarray(intr["K"], dtype=np.float64)
	center = np.array([intr["width"] * 0.5, intr["height"] * 0.5], dtype=np.float64)
	tol_px = np.array([intr["width"] * CENTER_TOL_FRAC, intr["height"] * CENTER_TOL_FRAC], dtype=np.float64)

	for _ in range(2):
		_set_camera_world_pose(stage, ARGS.bev_camera_prim_path, cam_pos, cam_quat)
		_step(world, SETTLE_FRAMES)
		pix = _project_world_to_image(target_pos, cam_pos, cam_quat, K)
		if pix is None:
			break
		delta = pix - center
		if abs(delta[0]) <= tol_px[0] and abs(delta[1]) <= tol_px[1]:
			break
		# Top-down linearized correction in world XY.
		cam_pos[0] += (delta[0] / max(K[0, 0], 1e-9)) * BEV_CAMERA_HEIGHT_M
		cam_pos[1] += (delta[1] / max(K[1, 1], 1e-9)) * BEV_CAMERA_HEIGHT_M
		cam_quat = _look_at_quat_xyzw(cam_pos, target_pos)

	_set_camera_world_pose(stage, ARGS.bev_camera_prim_path, cam_pos, cam_quat)
	_step(world, SETTLE_FRAMES)

	rgba, depth, semantic_data, semantic_info = _capture_rgbd_semantic_with_retry(camera, world)
	if rgba is None or depth is None or semantic_data is None:
		raise RuntimeError("Failed to capture BEV rgb/depth/semantic frames.")

	sid_to_class = _build_semantic_id_to_class_map(semantic_info, label_to_class_id)
	class_mask = _semantic_to_class_mask(semantic_data, expected_hw=(cam_h, cam_w), sid_to_class=sid_to_class)
	if class_mask is None:
		raise RuntimeError("Failed to build class mask from semantic frame.")

	_save_rgb(rgba, rgb_path)
	_save_depth(depth, depth_path, expected_hw=(cam_h, cam_w))
	_save_class_mask(class_mask, class_path)

	actual_cam_pos, actual_cam_quat_xyzw = _get_prim_world_pose(stage, ARGS.bev_camera_prim_path)
	if actual_cam_pos is None or actual_cam_quat_xyzw is None:
		actual_cam_pos = cam_pos
		actual_cam_quat_xyzw = cam_quat

	cam_matrix = _build_c2w(actual_cam_pos.tolist(), actual_cam_quat_xyzw.tolist())

	with intrinsics_json_path.open("w", encoding="utf-8") as f:
		json.dump(intr, f, indent=2)

	pose_data = {
		"reached_indices": [0],
		"num_reached": 1,
		"class_mapping": class_entries,
		"target_class_id": 1,
		"poses": [
			{
				"index": 0,
				"cam_position": actual_cam_pos.tolist(),
				"cam_quaternion_xyzw": actual_cam_quat_xyzw.tolist(),
				"cam_matrix": cam_matrix,
				"ee_position": None,
				"ee_quaternion_xyzw": None,
				"joint_angles": [],
			}
		],
	}
	with pose_json_path.open("w", encoding="utf-8") as f:
		json.dump(pose_data, f, indent=2)

	print(f"[BEV] Saved RGB   -> {rgb_path}", flush=True)
	print(f"[BEV] Saved Depth -> {depth_path}", flush=True)
	print(f"[BEV] Saved Class -> {class_path}", flush=True)
	print(f"[BEV] Saved Intrinsics -> {intrinsics_json_path}", flush=True)
	print(f"[BEV] Saved Pose -> {pose_json_path}", flush=True)

	simulation_app.close()


if __name__ == "__main__":
	main()
