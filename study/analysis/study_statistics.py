"""Prespecified paired-case and root-level summary functions."""
import math

def exact_sign(differences):
    positive = sum(x > 0 for x in differences)
    negative = sum(x < 0 for x in differences)
    n = positive + negative
    if n == 0:
        return {"positive": 0, "negative": 0, "zero": len(differences), "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return {"positive": positive, "negative": negative,
            "zero": len(differences) - n, "p": min(1.0, 2 * tail / 2**n)}

def paired_cases(initial, final, expected=32):
    a = {case["seed"]: case for case in initial["cases"]}
    b = {case["seed"]: case for case in final["cases"]}
    if len(a) != expected or len(b) != expected or set(a) != set(b):
        raise ValueError("complete paired panel required")
    keys = ["initial_state_sha256", "initial_model_xml_sha256", "compiled_limits"]
    for seed, x in a.items():
        y = b[seed]
        if not all(x[key] == y[key] for key in keys):
            raise ValueError("paired initial state/model/limits identity")
        if type(x["success"]) is not bool or type(y["success"]) is not bool:
            raise ValueError("native boolean success required")
    old = sum(x["success"] for x in a.values())
    child = sum(y["success"] for y in b.values())
    loss = sum(a[s]["success"] and not b[s]["success"] for s in a)
    gains = sum(not a[s]["success"] and b[s]["success"] for s in a)
    if old + gains - loss != child:
        raise ValueError("paired count reconciliation")
    return {"initial_successes": old, "final_successes": child, "cases": expected,
            "gains": gains, "losses": loss,
            "old_success_retention": None if old == 0 else (old - loss) / old}

def measured_component(measurements):
    counts = {}
    for measurement in measurements:
        for key, value in measurement["counts"].items():
            counts[key] = counts.get(key, 0) + value
    return {"counts": counts, "worker_seconds_sum_estimate": sum(x["elapsed_seconds"] for x in measurements)}

def behavior_control(left, right):
    keys = ["seed", "success", "steps", "max_success_hold", "stop_reason",
            "initial_state_sha256", "initial_model_xml_sha256", "compiled_limits"]
    a = {x["seed"]: x for x in left["cases"]}
    b = {x["seed"]: x for x in right["cases"]}
    return {"paired_cases": len(a), "same_recorded_behavior": set(a) == set(b)
            and all(all(a[seed][k] == b[seed][k] for k in keys) for seed in a),
            "compared_fields": keys, "complete_action_trajectories_compared": False}

def summarize(p, report, boards, sealed, native_state=None, job_measurements=None):
    rows, roots = {}, {}
    native_state, job_measurements = native_state or {}, job_measurements or {}
    controls = []
    for identity, trajectory in p["plan"]["trajectories"].items():
        row = {"tasks": {}, "complete": False}
        rows[identity] = row
        for task in ["Can", "Square"]:
            name = p["schedule"][identity]["final"][task]
            if name in boards and "initial_" + task in boards:
                row["tasks"][task] = paired_cases(boards["initial_" + task], boards[name])
        row["complete"] = len(row["tasks"]) == 2
        row["missing_tasks"] = {}
        for task in ["Can", "Square"]:
            if task not in row["tasks"]:
                name = p["schedule"][identity]["final"][task]
                state = native_state.get("jobs", {}).get(name)
                row["missing_tasks"][task] = {"job": name,
                    "reason": ("initial_reference_missing_or_incomplete" if name in boards and "initial_" + task not in boards
                               else "recorded_job_failed_or_incomplete" if state is not None
                               else "not_started_before_whole_stage_stopped"),
                    "missing_initial_job": "initial_" + task if "initial_" + task not in boards else None,
                    "job_terminal": state, "whole_stage_error": report.get("error")}
        first_task = trajectory["requests"][0]
        pre_name = p["schedule"][identity]["pre"]
        row["first_request_task_development_changes"] = []
        previous = boards.get(pre_name)
        for index, post in enumerate(p["schedule"][identity]["post"]):
            current = boards.get(post[first_task])
            if current is not None and previous is not None:
                row["first_request_task_development_changes"].append({"stage": index + 1,
                    "relative_to_previous_observation": paired_cases(previous, current, expected=4)})
            previous = current
        nodes = [stage["physical_training_node"] for stage in trajectory["stages"]]
        if sealed is not None and all(n in sealed["node_measurements"] for n in nodes):
            measurements = [sealed["node_measurements"][n] for n in nodes]
            unique_final = {sealed["actual_states"][route["state"]]["sha256"]:
                            sealed["actual_states"][route["state"]]["bytes"]
                            for route in trajectory["stages"][-1]["routes_after"].values()}
            row["logical_training_cost"] = {
                "training_operations": measured_component(measurements),
                "optimizer_attempts": sum(x["counts"]["optimizer_attempts"] for x in measurements),
                "node_worker_seconds_sum_estimate": sum(x["elapsed_seconds"] for x in measurements),
                "scope": "measured physical node costs reused as a logical-path estimate; excludes root construction and all observation overhead; not independently measured path wall time",
                "active_endpoint_unique_checkpoints": len(unique_final),
                "active_endpoint_checkpoint_bytes": sum(unique_final.values()),
                "intermediate_training_checkpoint_bytes_sum": sum(x["checkpoint_bytes"] for x in measurements),
                "resident_memory_measured": False}
        evaluation_names = [pre_name] + [name for event in p["schedule"][identity]["post"] for name in event.values()] + list(p["schedule"][identity]["final"].values())
        row["logical_other_cost_components"] = {}
        for component, names in {
            "root_initialization": ["init_" + trajectory["initial_state"]],
            "path_observation": evaluation_names,
            "common_research_baseline": ["initial_Can", "initial_Square"]}.items():
            available = [job_measurements[name] for name in names if name in job_measurements]
            row["logical_other_cost_components"][component] = {
                **measured_component(available), "measured_jobs": len(available), "required_jobs": len(names),
                "scope": "sum of measured job components, reused as estimate; common baseline physically run once for whole study"}
    differences = {"ER_minus_FT": [], "early_minus_delayed": []}
    direct, second_tasks = [], []
    all_complete = report["status"] == "completed" and all(row["complete"] for row in rows.values())
    for seed in p["lineage_seeds"]:
        root_rows = [identity for identity in rows if identity.startswith(str(seed) + "/")]
        complete = len(root_rows) == 8 and all(rows[identity]["complete"] for identity in root_rows)
        root_record = {"complete": complete, "paths": {identity: {
            "complete": rows[identity]["complete"], "missing_tasks": rows[identity]["missing_tasks"]} for identity in root_rows}}
        roots[str(seed)] = root_record
        if not complete:
            continue
        scores = {}
        for policy in p["study_spec"]["policies"]:
            paths = [identity for identity in root_rows if identity.endswith("/" + policy)]
            scores[policy] = sum(rows[identity]["tasks"][task]["final_successes"]
                                 for identity in paths for task in ["Can", "Square"])
        root_record["aggregate_successes_of_128"] = scores
        differences["ER_minus_FT"].append(scores["shared_ER"] - scores["shared_FT"])
        differences["early_minus_delayed"].append(scores["early_branch"] - scores["delayed_branch"])
        for stream, requests in p["study_spec"]["streams"].items():
            prefix = str(seed) + "/" + stream + "/"
            for specialist in ["early_branch", "delayed_branch"]:
                direct.append({"seed": seed, "stream": stream, "comparison": "shared_ER-minus-" + specialist,
                    "tasks": {task: {"success_difference": rows[prefix + "shared_ER"]["tasks"][task]["final_successes"] - rows[prefix + specialist]["tasks"][task]["final_successes"],
                        "ER_losses": rows[prefix + "shared_ER"]["tasks"][task]["losses"],
                        "specialist_losses": rows[prefix + specialist]["tasks"][task]["losses"]} for task in ["Can", "Square"]},
                    "successes_and_training_costs": {policy: {
                        "tasks": rows[prefix + policy]["tasks"],
                        "training": rows[prefix + policy].get("logical_training_cost"),
                        "other_components": rows[prefix + policy]["logical_other_cost_components"]}
                        for policy in ["shared_ER", specialist]}, "inferential_test": False})
            second = requests[1]
            first = requests[0]
            first_early = p["schedule"][prefix + "early_branch"]["final"][first]
            first_delayed = p["schedule"][prefix + "delayed_branch"]["final"][first]
            controls.append({"kind": "first_request_early_delayed", "seed": seed, "stream": stream,
                "task": first, "early": rows[prefix + "early_branch"]["tasks"][first],
                "delayed": rows[prefix + "delayed_branch"]["tasks"][first],
                **behavior_control(boards[first_early], boards[first_delayed])})
            second_tasks.append({"seed": seed, "stream": stream, "task": second,
                "early_minus_delayed_successes": rows[prefix + "early_branch"]["tasks"][second]["final_successes"] - rows[prefix + "delayed_branch"]["tasks"][second]["final_successes"],
                "first_task_is_construction_control": first})
        streams = list(p["study_spec"]["streams"])
        er_names = [str(seed) + "/" + stream + "/shared_ER" for stream in streams]
        for index in range(3):
            for task in ["Can", "Square"]:
                a, b = [p["schedule"][identity]["post"][index][task] for identity in er_names]
                if a in boards and b in boards:
                    controls.append({"kind": "ER_stream_semantic_control", "seed": seed,
                        "stage": index + 1, "task": task, **behavior_control(boards[a], boards[b])})
    controls_passed = len(controls) == 48 and all(x["same_recorded_behavior"] for x in controls)
    tests = {}
    if all_complete and controls_passed:
        if not all(len(values) == 6 for values in differences.values()):
            raise ValueError("six complete root pairs required")
        tests = {key: {**exact_sign(values), "root_differences_in_successes_of_128": values,
                       "root_labels": p["lineage_seeds"],
                       "difference_range_in_successes_of_128": [min(values), max(values)],
                       "mean_rate_difference": sum(values) / (6 * 128)} for key, values in differences.items()}
        ordered = sorted(tests, key=lambda name: tests[name]["p"])
        running = 0.0
        for index, name in enumerate(ordered):
            running = max(running, min(1.0, (2 - index) * tests[name]["p"]))
            tests[name]["Holm_adjusted_p"] = running
    return {"kind": "bounded-system-study-summary", "all_prespecified_paths_complete": all_complete,
            "path_results": rows, "root_completion": roots, "primary_tests": tests,
            "construction_controls": controls, "recorded_construction_controls_passed": controls_passed,
            "interpretation_rule": "behavior control mismatch prevents primary-test interpretation; no automatic rerun or seed exclusion",
            "missing_policy": "missing is unmeasured, never zero; no primary tests unless whole six-root design completed",
            "root_directional_tests_are_not_equivalence_tests": True,
            "direct_ER_specialist_descriptive": direct, "second_request_task_descriptive": second_tasks,
            "whole_experiment_physical_counts": report.get("recorded_counts"),
            "reserved_including_unknown_work": report.get("reserved_counts"),
            "unknown_work": report.get("unknown_work"),
            "audit_scope": "case summaries and supplied completion metadata; not independent recomputation of neural updates, raw attempt journals, optimizer states or the physical success predicate"}
