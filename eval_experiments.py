import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = REPO_ROOT / "experiment"
OUTPUT_PATH = EXPERIMENT_ROOT / "eval.json"

METHODS = [
	"curobo",
	"curobo_jit",
	"curobo_withobj",
	"curobo_withobj_jit",
	"heuristic_approach",
	"heuristic_vertical",
    "heuristic_approach_jit",
    "heuristic_vertical_jit",
    "rrt",
    "rrt_jit",
    "rrt_withobj",
    "rrt_withobj_jit",
]

DATASETS = ["01_robot", "02_robot"]

# Practical thresholds for counting an object as affected.
# Position: 5 mm, Rotation: 1 deg.
POS_EPS_M = 5e-3
ROT_EPS_DEG = 1.0


def _quat_to_wxyz(q):
	# Stored as xyzw in experiment outputs.
	return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _quat_angle_deg_xyzw(q1, q2):
	a = _quat_to_wxyz(q1)
	b = _quat_to_wxyz(q2)
	dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
	dot = max(-1.0, min(1.0, abs(dot)))
	return math.degrees(2.0 * math.acos(dot))


def _evaluate_one_trial(initial_objects, objects_after, success):
	if not success:
		return {
			"success": 0,
			"affected_object_cnt": 0,
			"diff_sum_pos_m": 0.0,
			"diff_max_pos_m": 0.0,
			"diff_sum_rot_deg": 0.0,
			"diff_max_rot_deg": 0.0,
		}

	init_map = {obj.get("prim_path"): obj for obj in initial_objects if obj.get("prim_path")}
	affected = 0
	sum_pos = 0.0
	max_pos = 0.0
	sum_rot = 0.0
	max_rot = 0.0

	for after in objects_after:
		prim_path = after.get("prim_path")
		if not prim_path or prim_path not in init_map:
			continue
		before = init_map[prim_path]
		p0 = before.get("translation_xyz", [0.0, 0.0, 0.0])
		p1 = after.get("translation_xyz", [0.0, 0.0, 0.0])
		q0 = before.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
		q1 = after.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])

		pos_diff = math.dist([float(p0[0]), float(p0[1]), float(p0[2])], [float(p1[0]), float(p1[1]), float(p1[2])])
		rot_diff = _quat_angle_deg_xyzw(q0, q1)

		sum_pos += pos_diff
		max_pos = max(max_pos, pos_diff)
		sum_rot += rot_diff
		max_rot = max(max_rot, rot_diff)

		if pos_diff > POS_EPS_M or rot_diff > ROT_EPS_DEG:
			affected += 1

	return {
		"success": 1,
		"affected_object_cnt": int(affected),
		"diff_sum_pos_m": float(sum_pos),
		"diff_max_pos_m": float(max_pos),
		"diff_sum_rot_deg": float(sum_rot),
		"diff_max_rot_deg": float(max_rot),
	}


def _trial_metrics_from_scene(scene_data):
	initial_objects = scene_data.get("initial_objects", [])
	sim_result = scene_data.get("sim_result", [])
	trial_metrics = []
	for trial in sim_result:
		success = bool(trial.get("move_success", False))
		trial_metrics.append(_evaluate_one_trial(initial_objects, trial.get("objects_after", []), success=success))
	return trial_metrics


def _aggregate_trial_metrics(trial_metrics, num_scenes):
	n = len(trial_metrics)
	if n == 0:
		return {
			"num_scenes": int(num_scenes),
			"num_trials": 0,
			"success_rate_pct": 0.0,
			"avg_affected_object_cnt": 0.0,
			"avg_diff_sum_pos_m": 0.0,
			"avg_diff_max_pos_m": 0.0,
			"avg_diff_sum_rot_deg": 0.0,
			"avg_diff_max_rot_deg": 0.0,
		}

	success_sum = sum(m["success"] for m in trial_metrics)
	return {
		"num_scenes": int(num_scenes),
		"num_trials": n,
		"success_rate_pct": 100.0 * success_sum / n,
		"avg_affected_object_cnt": sum(m["affected_object_cnt"] for m in trial_metrics) / n,
		"avg_diff_sum_pos_m": sum(m["diff_sum_pos_m"] for m in trial_metrics) / n,
		"avg_diff_max_pos_m": sum(m["diff_max_pos_m"] for m in trial_metrics) / n,
		"avg_diff_sum_rot_deg": sum(m["diff_sum_rot_deg"] for m in trial_metrics) / n,
		"avg_diff_max_rot_deg": sum(m["diff_max_rot_deg"] for m in trial_metrics) / n,
	}


def _evaluate_method_dataset(method, dataset):
	folder = EXPERIMENT_ROOT / method / dataset
	scene_files = sorted(folder.glob("scene_*.json")) if folder.is_dir() else []
	trial_metrics = []

	for path in scene_files:
		try:
			scene_data = json.loads(path.read_text(encoding="utf-8"))
			trial_metrics.extend(_trial_metrics_from_scene(scene_data))
		except Exception:
			# Skip unreadable files but keep evaluation robust.
			continue

	return {
		"method": method,
		"dataset": dataset,
		**_aggregate_trial_metrics(trial_metrics, num_scenes=len(scene_files)),
	}


def _print_table(rows):
	headers = [
		"method",
		"dataset",
		"num",
		"succ%",
		"aff_cnt",
		"sum_pos",
		"max_pos",
		"sum_rot",
		"max_rot",
	]
	print("\n=== Experiment Evaluation ===")
	print(
		f"{headers[0]:<22} {headers[1]:<10} {headers[2]:>4} {headers[3]:>8} {headers[4]:>8} "
		f"{headers[5]:>10} {headers[6]:>10} {headers[7]:>10} {headers[8]:>10}"
	)
	print("-" * 102)
	for r in rows:
		print(
			f"{r['method']:<22} {r['dataset']:<10} {r['num_trials']:>4d} "
			f"{r['success_rate_pct']:>8.2f} {r['avg_affected_object_cnt']:>8.3f} "
			f"{r['avg_diff_sum_pos_m']:>10.6f} {r['avg_diff_max_pos_m']:>10.6f} "
			f"{r['avg_diff_sum_rot_deg']:>10.4f} {r['avg_diff_max_rot_deg']:>10.4f}"
		)


def _combine_metrics(base_result, jit_result):
	"""Combine metrics from base and jittered variants into an _all aggregate."""
	if base_result["num_trials"] == 0 and jit_result["num_trials"] == 0:
		return {
			"num_scenes": 0,
			"num_trials": 0,
			"success_rate_pct": 0.0,
			"avg_affected_object_cnt": 0.0,
			"avg_diff_sum_pos_m": 0.0,
			"avg_diff_max_pos_m": 0.0,
			"avg_diff_sum_rot_deg": 0.0,
			"avg_diff_max_rot_deg": 0.0,
		}
	
	total_trials = base_result["num_trials"] + jit_result["num_trials"]
	base_successes = int(base_result["success_rate_pct"] * base_result["num_trials"] / 100.0)
	jit_successes = int(jit_result["success_rate_pct"] * jit_result["num_trials"] / 100.0)
	
	return {
		"num_scenes": base_result["num_scenes"] + jit_result["num_scenes"],
		"num_trials": total_trials,
		"success_rate_pct": 100.0 * (base_successes + jit_successes) / total_trials if total_trials > 0 else 0.0,
		"avg_affected_object_cnt": (
			(base_result["avg_affected_object_cnt"] * base_result["num_trials"] + 
			 jit_result["avg_affected_object_cnt"] * jit_result["num_trials"]) / total_trials
			if total_trials > 0 else 0.0
		),
		"avg_diff_sum_pos_m": (
			(base_result["avg_diff_sum_pos_m"] * base_result["num_trials"] + 
			 jit_result["avg_diff_sum_pos_m"] * jit_result["num_trials"]) / total_trials
			if total_trials > 0 else 0.0
		),
		"avg_diff_max_pos_m": max(base_result["avg_diff_max_pos_m"], jit_result["avg_diff_max_pos_m"]),
		"avg_diff_sum_rot_deg": (
			(base_result["avg_diff_sum_rot_deg"] * base_result["num_trials"] + 
			 jit_result["avg_diff_sum_rot_deg"] * jit_result["num_trials"]) / total_trials
			if total_trials > 0 else 0.0
		),
		"avg_diff_max_rot_deg": max(base_result["avg_diff_max_rot_deg"], jit_result["avg_diff_max_rot_deg"]),
	}


def main():
	rows = []
	results_by_key = {}  # Store results by (method, dataset) for easy lookup
	
	for method in METHODS:
		for dataset in DATASETS:
			result = _evaluate_method_dataset(method, dataset)
			rows.append(result)
			results_by_key[(method, dataset)] = result

	# Add _all aggregations (combining base + jit variants)
	base_methods_with_jit = [
		("curobo", "curobo_jit"),
		("curobo_withobj", "curobo_withobj_jit"),
		("heuristic_approach", "heuristic_approach_jit"),
		("heuristic_vertical", "heuristic_vertical_jit"),
		("rrt", "rrt_jit"),
		("rrt_withobj", "rrt_withobj_jit"),
	]
	
	for dataset in DATASETS:
		for base_method, jit_method in base_methods_with_jit:
			base_result = results_by_key.get((base_method, dataset))
			jit_result = results_by_key.get((jit_method, dataset))
			if base_result and jit_result:
				combined_metrics = _combine_metrics(base_result, jit_result)
				all_result = {
					"method": f"{base_method}_all",
					"dataset": dataset,
					**combined_metrics,
				}
				rows.append(all_result)

	_print_table(rows)

	out = {
		"experiment_root": str(EXPERIMENT_ROOT),
		"methods": METHODS,
		"datasets": DATASETS,
		"notes": {
			"destination_handling": "Aggregated over all destination trials (typically 5 scenes x 8 destinations = 40 trials).",
			"zero_on_failure": "If scene success is 0, all other metrics for that scene are set to 0.",
			"rotation_unit": "degrees",
			"position_unit": "meters",
			"all_aggregation": "Combines base method and _jit variant results.",
		},
		"results": rows,
	}
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
	print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
