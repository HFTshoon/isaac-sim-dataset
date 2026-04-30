import xml.etree.ElementTree as ET
from typing import Optional


def top_level_path(path: str) -> str:
    if not path.startswith("/"):
        return ""
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    return f"/{parts[0]}"


def get_urdf_robot_name(urdf_file: str) -> Optional[str]:
    try:
        root = ET.parse(urdf_file).getroot()
        if root.tag == "robot":
            name = root.attrib.get("name", "").strip()
            if name:
                return name
    except Exception:
        pass
    return None
