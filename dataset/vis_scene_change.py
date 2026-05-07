import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CMAP = "plasma"


def direction_to_lon_lat_deg(direction):
	x, y, z = direction
	lon = math.degrees(math.atan2(y, x))
	lat = math.degrees(math.asin(max(-1.0, min(1.0, z))))
	return lon, lat


def metric_from_group(group, metric):
	moved = group.get("moved_objects")
	if moved is None:
		moved = [r for r in group.get("all_object_changes", []) if r.get("moved", False)]

	all_changes = group.get("all_object_changes", [])

	if metric == "affected_count":
		return float(len(moved))
	if metric == "sum_translation":
		return float(sum(r.get("translation_delta_m", 0.0) for r in moved))
	if metric == "max_translation":
		if not moved:
			return 0.0
		return float(max(r.get("translation_delta_m", 0.0) for r in moved))
	if metric == "sum_rotation":
		return float(sum(r.get("rotation_delta_deg", 0.0) for r in moved))
	if metric == "composite":
		# translation(m) + 0.01 * rotation(deg) 를 전체 object에 대해 합산
		return float(
			sum(
				r.get("translation_delta_m", 0.0) + 0.01 * r.get("rotation_delta_deg", 0.0)
				for r in all_changes
			)
		)

	raise ValueError(f"Unsupported metric: {metric}")


def load_trial_groups(data, target_path=None):
	groups = data.get("trial_groups", [])
	if target_path:
		groups = [g for g in groups if g.get("target_path") == target_path]

	if not groups:
		raise RuntimeError("No trial_groups found for given target_path.")

	return groups


def sanitize_target_name(target_path):
	if not target_path:
		return "unknown_target"
	name = target_path.split("/")[-1] or target_path
	name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
	return name


def discover_scene_dirs(dataset_root):
	root = Path(dataset_root)
	if not root.exists():
		raise RuntimeError(f"Dataset root not found: {root}")

	scene_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("scene_")]

	def _scene_key(p):
		m = re.match(r"scene_(\d+)$", p.name)
		if m:
			return int(m.group(1))
		return 10**9

	return sorted(scene_dirs, key=_scene_key)


def plot_directional_scene_change(groups, metric, title, out_png):
	dirs = []
	lons = []
	lats = []
	vals = []
	dir_indices = []
	affected_counts = []

	for g in groups:
		direction = g.get("direction")
		if direction is None:
			direction = [g.get("direction_x"), g.get("direction_y"), g.get("direction_z")]

		if direction is None or len(direction) != 3:
			continue

		x, y, z = [float(v) for v in direction]
		n = math.sqrt(x * x + y * y + z * z)
		if n < 1e-12:
			continue
		x, y, z = x / n, y / n, z / n

		lon, lat = direction_to_lon_lat_deg((x, y, z))
		value = metric_from_group(g, metric)

		dirs.append((x, y, z))
		lons.append(lon)
		lats.append(lat)
		vals.append(value)
		dir_indices.append(g.get("direction_index", -1))
		affected_counts.append(g.get("num_affected_objects", 0))

	if not vals:
		raise RuntimeError("No valid trial direction data to visualize.")

	vals = np.asarray(vals, dtype=np.float64)
	lons = np.asarray(lons, dtype=np.float64)
	lats = np.asarray(lats, dtype=np.float64)
	dirs = np.asarray(dirs, dtype=np.float64)
	affected_counts = np.asarray(affected_counts, dtype=np.float64)

	# Colormap range policy: min=0, max=max(observed, 0.1)
	vmin = 0.0
	vmax = max(float(np.max(vals)), 0.1)

	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
	ax_ll = fig.add_subplot(gs[0, 0])
	ax_xy = fig.add_subplot(gs[0, 1])
	fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.10, wspace=0.28)

	marker_sizes = 60.0 + 12.0 * np.sqrt(np.maximum(affected_counts, 0.0))

	sc1 = ax_ll.scatter(
		lons,
		lats,
		c=vals,
		s=marker_sizes,
		cmap=CMAP,
		vmin=vmin,
		vmax=vmax,
		edgecolors="black",
		linewidths=0.4,
	)
	for i, idx in enumerate(dir_indices):
		ax_ll.text(lons[i], lats[i], str(idx), fontsize=8, ha="center", va="center")

	ax_ll.set_title("Direction (Longitude/Latitude)")
	ax_ll.set_xlabel("Longitude (deg)")
	ax_ll.set_ylabel("Latitude (deg)")
	ax_ll.set_xlim(-180, 180)
	ax_ll.set_ylim(0, 90)
	ax_ll.grid(True, alpha=0.3)

	sc2 = ax_xy.scatter(
		dirs[:, 0],
		dirs[:, 1],
		c=vals,
		s=marker_sizes,
		cmap=CMAP,
		vmin=vmin,
		vmax=vmax,
		edgecolors="black",
		linewidths=0.4,
	)
	ax_xy.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1.0))
	ax_xy.set_aspect("equal", "box")
	ax_xy.set_xlim(-1.05, 1.05)
	ax_xy.set_ylim(-1.05, 1.05)
	ax_xy.set_xlabel("x")
	ax_xy.set_ylabel("y")
	ax_xy.set_title("Upper Hemisphere Projection (z > 0)")
	ax_xy.grid(True, alpha=0.3)

	# Put colorbar in the gap between plots, shifted left to avoid overlapping ax_xy.
	cax = fig.add_axes([0.492, 0.13, 0.016, 0.70])
	cbar = fig.colorbar(sc1, cax=cax)
	cbar.set_label(metric)

	fig.suptitle(title)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def render_one_scene(input_json_path, output_png_path, target_path, all_targets, metric):
	in_path = Path(input_json_path)
	out_path = Path(output_png_path)

	with in_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	# Always save visualizations under a dedicated directory.
	out_dir = out_path.parent / "move_effect_vis"
	out_dir.mkdir(parents=True, exist_ok=True)
	base_stem = out_path.stem
	out_suffix = out_path.suffix or ".png"

	groups = load_trial_groups(data, target_path=target_path)

	# If target-path is not provided, default to all targets (one image per object).
	render_all_targets = all_targets or (target_path is None)

	if render_all_targets:
		targets = sorted({g.get("target_path") for g in groups if g.get("target_path")})
		if not targets:
			raise RuntimeError("No target_path values found in trial_groups.")

		saved_paths = []
		for target in targets:
			t_groups = [g for g in groups if g.get("target_path") == target]
			target_slug = sanitize_target_name(target)
			target_out = out_dir / f"{base_stem}_{target_slug}{out_suffix}"

			title = (
				f"Scene Change by Move Direction\n"
				f"metric={metric}, target={target}, n={len(t_groups)}"
			)
			plot_directional_scene_change(t_groups, metric=metric, title=title, out_png=target_out)
			saved_paths.append(target_out)

		print(f"[DONE] {in_path.parent.name}: saved {len(saved_paths)} visualizations")
		return saved_paths

	target_label = target_path
	title = (
		f"Scene Change by Move Direction\n"
		f"metric={metric}, target={target_label}, n={len(groups)}"
	)
	if target_label:
		single_name = f"{base_stem}_{sanitize_target_name(target_label)}{out_suffix}"
	else:
		single_name = f"{base_stem}{out_suffix}"
	single_out = out_dir / single_name
	plot_directional_scene_change(groups, metric=metric, title=title, out_png=single_out)

	print(f"[DONE] {in_path.parent.name}: saved visualization: {single_out}")
	return [single_out]


def main():
	parser = argparse.ArgumentParser(description="Visualize directional scene change from move_effect.json")
	parser.add_argument(
		"--input",
		default="/isaac-sim/corl2025/dataset/01/scene_000/move_effect.json",
		help="Path template for move_effect.json (dataset root is inferred as parent of scene_000)",
	)
	parser.add_argument(
		"--output",
		default="/isaac-sim/corl2025/dataset/01/scene_000/move_effect_vis.png",
		help="Output PNG template name (saved per scene under scene_xxx/move_effect_vis)",
	)
	parser.add_argument(
		"--dataset-root",
		default=None,
		help="Dataset root containing scene_* directories. If omitted, inferred from --input.",
	)
	parser.add_argument(
		"--target-path",
		default=None,
		help="Optional target object path filter (e.g. /objects/obj_025_mug)",
	)
	parser.add_argument(
		"--all-targets",
		action="store_true",
		help="If set, save one visualization image per target object.",
	)
	parser.add_argument(
		"--metric",
		default="sum_translation",
		choices=["affected_count", "sum_translation", "max_translation", "sum_rotation", "composite"],
		help="Metric for colormap",
	)
	args = parser.parse_args()

	in_template = Path(args.input)
	out_template = Path(args.output)

	if args.dataset_root is not None:
		dataset_root = Path(args.dataset_root)
	else:
		# e.g. .../dataset/01/scene_000/move_effect.json -> .../dataset/01
		dataset_root = in_template.parent.parent

	scene_dirs = discover_scene_dirs(dataset_root)
	if not scene_dirs:
		raise RuntimeError(f"No scene_* directories found in {dataset_root}")

	print(f"[INFO] Found {len(scene_dirs)} scenes in {dataset_root}")
	saved_total = 0
	skipped = []
	failed = []

	for scene_dir in scene_dirs:
		in_path = scene_dir / in_template.name
		out_path = scene_dir / out_template.name

		if not in_path.exists():
			print(f"[SKIP] {scene_dir.name}: missing {in_path.name}")
			skipped.append(scene_dir.name)
			continue

		try:
			saved = render_one_scene(
				input_json_path=in_path,
				output_png_path=out_path,
				target_path=args.target_path,
				all_targets=args.all_targets,
				metric=args.metric,
			)
			saved_total += len(saved)
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[ERROR] {scene_dir.name}: {exc}")
			failed.append(scene_dir.name)

	print("\n[SUMMARY]")
	print(f"  scenes_total={len(scene_dirs)}")
	print(f"  scenes_skipped_missing_json={len(skipped)}")
	print(f"  scenes_failed={len(failed)}")
	print(f"  images_saved={saved_total}")


if __name__ == "__main__":
	main()
