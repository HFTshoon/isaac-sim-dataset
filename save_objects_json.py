import json
import math
import os
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom, Gf


# ====== 여기만 필요에 맞게 수정 ======
OBJECT_ROOT = "/objects"          # 예: "/World/objects", "/Objects", "/World/Objects"
SAVE_PATH = "/isaac-sim/corl2025/isaac_objects_for_moveit.json"
OBJ_DIR = "/isaac-sim/corl2025/obj"

# MoveIt planning frame과 Isaac world가 다르면 기준 prim을 넣으세요.
# 예: REFERENCE_FRAME_PRIM = "/Franka" 또는 "/Franka/panda_link0"
# 그냥 Isaac world 기준으로 저장하려면 None.
REFERENCE_FRAME_PRIM = None
# ===================================


def matrix_to_list(m):
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]


def vec3_to_list(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def get_quat_xyzw_from_matrix(m):
    """
    ROS geometry_msgs/Pose orientation 순서: x, y, z, w
    """
    rot = m.ExtractRotation()
    q = rot.GetQuat()
    imag = q.GetImaginary()
    return [
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(q.GetReal()),
    ]


def get_translation_from_matrix(m):
    t = m.ExtractTranslation()
    return vec3_to_list(t)


def get_scale_approx_from_matrix(m):
    """
    world matrix에서 대략적인 scale 추정.
    non-uniform scale이 없으면 대부분 [1,1,1].
    """
    sx = math.sqrt(float(m[0][0])**2 + float(m[0][1])**2 + float(m[0][2])**2)
    sy = math.sqrt(float(m[1][0])**2 + float(m[1][1])**2 + float(m[1][2])**2)
    sz = math.sqrt(float(m[2][0])**2 + float(m[2][1])**2 + float(m[2][2])**2)
    return [sx, sy, sz]


def unique_keep_order(items):
    out = []
    seen = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def listop_items(listop):
    """
    USD references/payload listOp에서 assetPath들을 최대한 robust하게 추출.
    """
    items = []
    for method_name in [
        "GetExplicitItems",
        "GetAddedOrExplicitItems",
        "GetPrependedItems",
        "GetAppendedItems",
    ]:
        if hasattr(listop, method_name):
            try:
                items.extend(getattr(listop, method_name)())
            except Exception:
                pass
    return items


def collect_asset_paths(root_prim):
    """
    object prim과 그 하위 prim에서 reference/payload/assetInfo 경로를 수집.
    YCB는 보통 상위 Xform prim에 USD reference가 걸려 있습니다.
    """
    asset_paths = []

    for p in Usd.PrimRange(root_prim):
        # omni.usd convenience API
        try:
            url = omni.usd.get_url_from_prim(p)
            if url:
                asset_paths.append(str(url))
        except Exception:
            pass

        # references / payloads metadata
        for key in ["references", "payload"]:
            try:
                meta = p.GetMetadata(key)
                if meta:
                    for item in listop_items(meta):
                        asset_path = getattr(item, "assetPath", None)
                        if asset_path:
                            asset_paths.append(str(asset_path))
            except Exception:
                pass

        # assetInfo identifier
        try:
            asset_info = p.GetAssetInfo()
            identifier = asset_info.get("identifier") if asset_info else None
            if identifier:
                asset_paths.append(str(identifier))
        except Exception:
            pass

    return unique_keep_order(asset_paths)


def collect_mesh_prims(root_prim):
    mesh_prims = []
    for p in Usd.PrimRange(root_prim):
        if p.IsA(UsdGeom.Mesh):
            mesh_prims.append(str(p.GetPath()))
    return mesh_prims


def classify_assets(asset_paths):
    usd_exts = {".usd", ".usda", ".usdc"}

    usd_assets = []

    for p in asset_paths:
        suffix = Path(p).suffix.lower()
        if suffix in usd_exts:
            usd_assets.append(p)

    return usd_assets


ctx = omni.usd.get_context()
stage = ctx.get_stage()

root = stage.GetPrimAtPath(OBJECT_ROOT)
if not root.IsValid():
    raise RuntimeError(f"OBJECT_ROOT not found: {OBJECT_ROOT}")

# 기준 프레임 설정
if REFERENCE_FRAME_PRIM:
    ref_prim = stage.GetPrimAtPath(REFERENCE_FRAME_PRIM)
    if not ref_prim.IsValid():
        raise RuntimeError(f"REFERENCE_FRAME_PRIM not found: {REFERENCE_FRAME_PRIM}")
    T_world_ref = omni.usd.get_world_transform_matrix(ref_prim)
    T_ref_world = T_world_ref.GetInverse()
else:
    ref_prim = None
    T_ref_world = None

objects = []

# /objects 바로 아래 child들을 object 하나로 취급
for obj_prim in root.GetChildren():
    if not obj_prim.IsValid() or not obj_prim.IsActive():
        continue

    T_world_obj = omni.usd.get_world_transform_matrix(obj_prim)

    if REFERENCE_FRAME_PRIM:
        # USD/Gf는 row-vector convention이라 object->reference는 obj_world * world_ref
        T_ref_obj = T_world_obj * T_ref_world
        pose_matrix = T_ref_obj
        frame_id = REFERENCE_FRAME_PRIM
    else:
        pose_matrix = T_world_obj
        frame_id = "world"

    asset_paths = collect_asset_paths(obj_prim)
    usd_assets = classify_assets(asset_paths)
    mesh_prims = collect_mesh_prims(obj_prim)

    mesh_name = obj_prim.GetName()
    mesh_name = "0" + mesh_name[1:]
    mesh_path = os.path.join(OBJ_DIR, f"{mesh_name}.obj")
    
    item = {
        "id": obj_prim.GetName(),
        "prim_path": str(obj_prim.GetPath()),

        # MoveIt에서 header.frame_id로 맞춰야 하는 기준
        "frame_id": frame_id,

        # Isaac/Omniverse asset 경로들
        "asset_paths": asset_paths,
        "usd_asset_path": usd_assets[0] if usd_assets else None,

        # MoveIt에서 바로 쓰기 좋은 mesh 후보.
        # 비어 있으면 USD를 STL/DAE/OBJ로 변환해서 직접 채워야 함.
        "moveit_mesh_path": mesh_path,

        # 참고용: 실제 composed USD 안의 Mesh prim들
        "mesh_prims": mesh_prims,

        # ROS geometry_msgs/Pose에 바로 넣기 좋은 형식
        "translation_xyz": get_translation_from_matrix(pose_matrix),
        "rotation_xyzw": get_quat_xyzw_from_matrix(pose_matrix),

        # scale이 들어간 물체면 MoveIt mesh scale에 반영 필요
        "scale_xyz_approx": get_scale_approx_from_matrix(pose_matrix),

        # 디버깅/정확 재현용
        "transform_matrix": matrix_to_list(pose_matrix),
    }

    objects.append(item)

data = {
    "object_root": OBJECT_ROOT,
    "reference_frame_prim": REFERENCE_FRAME_PRIM,
    "note": "translation_xyz is in meters. rotation_xyzw is ROS quaternion order x,y,z,w.",
    "objects": objects,
}

Path(SAVE_PATH).write_text(json.dumps(data, indent=2))
print(f"Saved {len(objects)} objects to {SAVE_PATH}")

