"""Actual native scene boards with paired initial states and explicit routing."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import math
from pathlib import Path
import random
import sys
import time

from common import Job, read, sha
from contracts import canonical, require, verify_checkpoint_metadata
from learner import TASK_TAGS, construct, load_checkpoint, runtime


def paired_reference(record: dict, protocol_sha: str, expected_job: str,
                     task: str, split: str, seeds: list[int]) -> dict:
    require(record["status"] == "completed" and record["protocol_sha256"] == protocol_sha
            and record["job"] == expected_job, "completed matching baseline job wrapper")
    prior = record["board"]
    require(prior["status"] == "completed" and prior["protocol_sha256"] == protocol_sha
            and prior["task"] == task and prior["split"] == split, "paired baseline board")
    rows = prior["cases"]
    require(len(rows) == len(seeds) and len({item["seed"] for item in rows}) == len(rows)
            and {item["seed"] for item in rows} == set(seeds), "complete unique paired cases")
    return {item["seed"]: item for item in rows}


def evaluate(job: Job) -> None:
    _, np, torch = runtime(job, False)
    sys.path.append(job.p["evaluation_runtime"]["robosuite_site_packages"])
    import h5py
    import mujoco
    import robosuite as suite
    from decoder import install_argmax
    from native_guard import guard_decision, inspect_actuation
    for helper in ["decoder", "native_guard"]:
        module = importlib.import_module(helper)
        require(Path(module.__file__).resolve() == Path(__file__).with_name(helper + ".py").resolve(), "actual native helper " + helper)
    require(suite.__version__ == "1.4.1" and mujoco.__version__ == "2.3.2"
            and Path(suite.__file__).resolve() == Path(job.p["evaluation_runtime"]["robosuite_site_packages"]) / "robosuite/__init__.py",
            "native simulator identity")
    for item in job.p["native_source_files"]:
        require(sha(Path(item["path"])) == item["sha256"], "native task/control source")
    task = job.definition["task"]
    execution = job.p["native_tasks"][task]
    if task == "ToolHang":
        meta = execution["env_metadata"]
    else:
        path = job.registered_input("metadata_" + task)
        with h5py.File(path, "r") as stream:
            from contracts import strict_json
            meta = strict_json(stream["data"].attrs["env_args"])
    require(meta["env_version"] == "1.4.1" and meta["env_name"] == execution["environment_name"], "native metadata")
    references = None
    if job.definition["pairing_reference"] is not None:
        path = job.root / job.definition["pairing_reference"]
        references = paired_reference(read(path), job.digest, path.parent.name, task,
                                      job.definition["split"], job.definition["seeds"])
    models, identities, traces = {}, {}, {}
    for route_task, ref in job.definition["routes"].items():
        item = read(job.root / ref)
        identity = (item["path"], item["sha256"], item["profile"])
        if identity not in identities:
            saved = load_checkpoint(job, item, torch, "cpu", "CPU_checkpoint_load_attempts")
            counter_scope = verify_checkpoint_metadata(saved, item, job.p["external_references"])
            job.value.setdefault("checkpoint_metadata_scope", {})[ref] = counter_scope
            model = construct(job, item["profile"], torch, "cpu")
            model.nets.load_state_dict(saved["weights"], strict=True)
            model.set_eval()
            trace = {"mode": "argmax"}
            install_argmax(model, trace)
            native_forward = model.nets["policy"].forward_train

            def counted(*args, _native=native_forward, **kwargs):
                job.charge(deployment_policy_forward_attempts=1)
                return _native(*args, **kwargs)

            model.nets["policy"].forward_train = counted
            identities[identity] = (model, trace)
        models[route_task], traces[route_task] = identities[identity]
    require(task in models and len(identities) <= 2, "actual bounded task dispatcher")
    model, trace = models[task], traces[task]
    profile = read(job.root / job.definition["routes"][task])["profile"]
    shapes = job.p["profiles"][profile]["policy_shapes"]
    raw_shapes = {key: (width,) for key, width in shapes.items() if key != "task_id"}
    result = {"status": "running", "protocol_sha256": job.digest, "task": task,
              "split": job.definition["split"], "cases": [], "routes": job.definition["routes"],
              "loaded_policy_count": len(identities), "high_level_planner_evaluated": False}
    job.value["board"] = result
    job.persist()
    for seed in job.definition["seeds"]:
        job.monitor()
        require(type(seed) is int and 0 <= seed < 2**32, "native NumPy seed range")
        case = {"seed": seed, "success": False, "steps": 0, "max_success_hold": 0,
                "grasp_steps": 0, "clipped_components": 0, "legacy_20N_steps": 0,
                "stop_reason": "not_started", "selected_task_route": task}
        result["cases"].append(case)
        case_start = time.monotonic()
        job.value["active_case_deadline_monotonic"] = case_start + execution["case_seconds"]
        job.persist()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(700000 + 2 * seed)
        job.charge(env_constructors=1)
        env = suite.make(meta["env_name"], **meta["env_kwargs"])
        primary = None
        try:
            native_source = Path(importlib.import_module(type(env).__module__).__file__).resolve()
            require(sha(native_source) == execution["environment_source_sha256"], "actual task class source")
            np.random.seed(seed)
            job.charge(explicit_resets=1)
            obs = env.reset()
            model.reset()
            native = env.sim.model._model
            limits = {key: getattr(native, key).tolist() for key in
                      ["actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange"]}
            limits["actuator_names"] = [mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(native.nu)]
            require(native.nu == 9 and native.opt.timestep == execution["timestep"], "native Panda controls")
            if task == "ToolHang":
                components = execution["object_component_names"]
                grasp_objects = [env.frame, env.tool]
            else:
                require(limits == execution["compiled_limits"] and env.single_object_mode == 2, "original compiled controls")
                if task == "Can":
                    obj = env.objects[env.object_id]
                    require(obj.name.lower() == "can", "Can object selected")
                    components = list(env.object_id_to_sensors[env.object_id])
                else:
                    require(task == "Square" and env.nut_id == 0, "Square object selected")
                    obj = env.nuts[env.nut_id]
                    components = list(env.nut_id_to_sensors[env.nut_id])
                grasp_objects = [obj]
            lower, upper = env.action_spec
            require(lower.shape == upper.shape == (7,) and np.isfinite(lower).all()
                    and np.isfinite(upper).all() and (lower <= upper).all(), "native action bounds")
            case["compiled_limits"] = limits
            case["initial_model_xml_sha256"] = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
            hold = 0
            with gzip.open(job.out / ("seed" + str(seed) + ".jsonl.gz"), "wt") as stream:
                def emit(row):
                    # Nonfinite outputs stop the board; their repr is preserved without emitting invalid JSON.
                    stream.write(canonical(row).decode() + "\n")
                    stream.flush()

                def measure(reward=None):
                    state = env.sim.get_state().flatten()
                    force = np.asarray(env.robots[0].ee_force)
                    mapped = {key: np.asarray(obs["object-state" if key == "object" else key]) for key in raw_shapes}
                    require(all(value.shape == raw_shapes[key] for key, value in mapped.items()), "raw native observation widths")
                    packed = np.concatenate([np.asarray(obs[key]).reshape(-1) for key in components])
                    require(np.array_equal(mapped["object"], packed, equal_nan=True), "native object order")
                    finite = bool(all(np.isfinite(value).all() for value in [state, force, *mapped.values()])
                                  and (reward is None or math.isfinite(float(reward))))
                    force_norm = float(np.linalg.norm(force))
                    workspace = True if task == "ToolHang" else bool(all(lo <= value <= hi for value, (lo, hi) in
                                                                                     zip(mapped["robot0_eef_pos"], execution["workspace_xyz"])))
                    data = env.sim.data._data
                    actuation = inspect_actuation(data.ctrl.tolist(), data.actuator_force.tolist(), limits, 1e-9)
                    decision = guard_decision(finite, workspace, force_norm, actuation)
                    if not finite:
                        raise ValueError("nonfinite native state/observation/reward/force: " + repr({"state": state, "force": force, "mapped": mapped, "reward": reward}))
                    return mapped, {"state": state.tolist(), "native_observation": {k: v.tolist() for k, v in mapped.items()},
                                    "force_vector_N": force.tolist(), "force_norm_N": force_norm,
                                    "workspace_ok": workspace, "guard": decision["stop"],
                                    "guard_decision": decision, "actuation": actuation}

                mapped, measured = measure()
                native_success = bool(env._check_success())
                case["initial_state_sha256"] = hashlib.sha256(np.asarray(measured["state"], dtype=np.float64).tobytes()).hexdigest()
                require(not native_success, "initial success is invalid; no resampling")
                if references is not None:
                    reference = references[seed]
                    require(all(case[key] == reference[key] for key in
                                ["initial_state_sha256", "initial_model_xml_sha256", "compiled_limits"]), "paired native state/model/controls")
                emit({"event": "initial", "native_success": native_success, **measured})
                if measured["guard"]:
                    case["stop_reason"] = "initial_guard"
                else:
                    for step in range(1, execution["horizon"] + 1):
                        job.monitor()
                        require(time.monotonic() < job.value["active_case_deadline_monotonic"], "native case deadline")
                        tensors = {key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in mapped.items()}
                        if "task_id" in shapes:
                            tensors["task_id"] = torch.tensor(TASK_TAGS[task], dtype=torch.float32).unsqueeze(0)
                        job.charge(policy_inference_attempts=1)
                        with torch.no_grad():
                            raw = models[task].get_action(tensors).squeeze(0).numpy()
                        require(raw.shape == (7,) and np.isfinite(raw).all(), "finite native action: " + repr(raw))
                        action = np.clip(raw, lower, upper)
                        clipped = int(np.count_nonzero(raw != action))
                        case["clipped_components"] += clipped
                        job.charge(action_attempts=1)
                        emit({"event": "attempt", "step": step, "raw_action": raw.tolist(), "action": action.tolist(), "clipped_components": clipped, "decoder": trace})
                        obs, reward, done, _ = env.step(action)
                        job.charge(action_returns=1)
                        mapped, measured = measure(reward)
                        native_success = bool(env._check_success())
                        if task == "ToolHang":
                            require(native_success == bool(env._check_frame_assembled() and env._check_tool_on_frame()), "compound native success")
                        grasp = any(bool(env._check_grasp(env.robots[0].gripper, obj.contact_geoms)) for obj in grasp_objects)
                        hold = hold + 1 if native_success and not measured["guard"] else 0
                        case.update(steps=step, max_success_hold=max(case["max_success_hold"], hold))
                        case["grasp_steps"] += int(grasp)
                        case["legacy_20N_steps"] += int(measured["guard_decision"]["legacy_20N_exceeded"])
                        emit({"event": "result", "step": step, "native_success": native_success,
                              "reward": float(reward), "done": bool(done), "grasp": grasp,
                              "success_hold": hold, **measured})
                        if measured["guard"]:
                            case["stop_reason"] = "guard"
                            break
                        if hold >= 5:
                            case.update(success=True, stop_reason="success_hold")
                            break
                    else:
                        case["stop_reason"] = "horizon"
        except BaseException as error:
            primary = error
            case["primary_error"] = repr(error)
            raise
        finally:
            try:
                env.close()
                case["environment_closed"] = True
            except BaseException as error:
                case["close_error"] = repr(error)
                if primary is None:
                    raise
            finally:
                job.persist()
        case["telemetry_sha256"] = sha(job.out / ("seed" + str(seed) + ".jsonl.gz"))
        job.value.pop("active_case_deadline_monotonic", None)
        job.persist()
    result["status"] = "completed"
    result["successes"] = sum(case["success"] for case in result["cases"])
    hashes = [case["initial_state_sha256"] for case in result["cases"]]
    result["unique_initial_states"] = len(set(hashes))
    result["initial_state_aliases_present"] = len(set(hashes)) != len(hashes)
    require(job.counts.get("deployment_policy_forward_attempts", 0) <=
            2 * job.counts.get("policy_inference_attempts", 0), "at most two native forwards per action attempt")
    job.finish(board=result)
