"""Native fixed learning operators and exact complete-parent sampling."""
from __future__ import annotations
import copy,gzip,importlib,math,random,sys
from pathlib import Path
from types import MethodType
from common import Job,read,sha,write,runtime,load_checkpoint
from contracts import canonical,digest,require,same_binding,validate_update
import expression

RAW = {"robot0_eef_pos": (0, 3), "robot0_eef_quat": (3, 7), "robot0_gripper_qpos": (7, 9)}

TASK_TAGS = {"Square": (1., 0.), "Can": (0., 1.)}

def construct(job: Job, profile: str, torch, device: str):
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    import robomimic.utils.obs_utils as ObsUtils
    info = job.p["profiles"][profile]
    config = config_factory("bc", dic=info["config"])
    ObsUtils.initialize_obs_utils_with_config(config)
    job.charge(**{"GPU_model_construction_attempts" if device.startswith("cuda") else "CPU_model_construction_attempts": 1})
    order = info["config"]["observation"]["modalities"]["obs"]["low_dim"]
    require(set(order) == set(info["policy_shapes"]), "exact native ordered observation keys")
    model = algo_factory("bc", config, {key: (info["policy_shapes"][key],) for key in order}, 7, torch.device(device))
    require(type(model).__name__ == "BC_RNN_GMM" and not model.nets["policy"].use_tanh, "native model family")
    parameters = list(model.nets.parameters())
    job.value.setdefault("constructed_model_parameters", []).append({"profile": profile, "device": device,
        "registered_parameter_elements": sum(value.numel() for value in parameters),
        "registered_parameter_logical_bytes": sum(value.numel() * value.element_size() for value in parameters),
        "scope": "registered parameter enumeration, not unique storage, resident memory or checkpoint bytes"})
    return model

class Pack:
    def __init__(self, job: Job, task: str, np):
        self.task, self.info, self.np = task, job.p["data"][task], np
        path = job.registered_input("data_" + task)
        with np.load(path, allow_pickle=False) as values:
            self.obs = values["observations"].copy()
            self.actions = values["actions"].copy()
            self.ids = values["demo_ids"].copy()
            self.offsets = values["episode_offsets"].copy()
        self.n = self.info["rows"]
        width = 53 if task == "ToolHang" else 23
        require(self.obs.shape == (self.n, width) and self.actions.shape == (self.n, 7)
                and self.obs.dtype == self.actions.dtype == np.float64 and
                np.isfinite(self.obs).all() and np.isfinite(self.actions).all()
                and np.max(np.abs(self.actions)) <= 1., "finite native train data and command bounds")
        require(self.ids.tolist() == self.info["ids"] and len(self.ids) == self.info["episodes"]
                and len(set(self.ids.tolist())) == len(self.ids), "exact learning IDs")
        require(self.offsets.shape == (len(self.ids) + 1,) and np.issubdtype(self.offsets.dtype, np.integer)
                and self.offsets[0] == 0 and self.offsets[-1] == self.n and
                (np.diff(self.offsets) > 0).all(), "episode partition")
        self.ends = np.repeat(self.offsets[1:] - 1, np.diff(self.offsets))
        self.cache = {}

    def batch(self, starts, torch, device):
        np = self.np
        require(np.issubdtype(starts.dtype, np.integer) and ((starts >= 0) & (starts < self.n)).all(), "start indices")
        index = np.minimum(starts[:, None] + np.arange(10), self.ends[starts, None])
        x = torch.as_tensor(self.obs[index], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[index], dtype=torch.float32, device=device)
        spans = {**RAW, "object": (9, self.obs.shape[-1])}
        obs = {key: x[..., lo:hi] for key, (lo, hi) in spans.items()}
        if self.task in TASK_TAGS:
            obs["task_id"] = actions.new_tensor(TASK_TAGS[self.task]).expand(len(starts), 10, 2)
        return {"obs": obs, "actions": actions}, index

    def targets(self, index, radius: float, torch, device):
        import counterfactual_target
        require(Path(counterfactual_target.__file__).resolve() == Path(__file__).with_name("counterfactual_target.py").resolve(), "actual target sampler")
        sample_arm_targets = counterfactual_target.sample_arm_targets
        for row in self.np.unique(index).tolist():
            key = (radius, row)
            if key not in self.cache:
                self.cache[key] = sample_arm_targets(self.actions[row].tolist(), radius, 5, 29,
                                                     self.info["sha256"] + ":" + str(row))
        return torch.tensor([[self.cache[(radius, int(row))] for row in seq] for seq in index],
                            device=device, dtype=torch.float32)

class Samplers:
    """Source state keeps the exact native sampler representation."""

    def __init__(self, packs: dict, core, seed=29):
        self.items = {task: core.Sampler(pack.n, seed) for task, pack in packs.items()}

    def next(self, task: str, amount: int):
        item = self.items[task]
        if item.cursor + amount > item.n:
            item.permutation = item.generator.permutation(item.n)
            item.cursor = 0
            item.passes += 1
        out = item.permutation[item.cursor:item.cursor + amount].copy()
        item.cursor += amount
        return out

    def state(self):
        if set(self.items) == {"ToolHang"}:
            return self.items["ToolHang"].state()
        return {"kind": "pilot-Can-Square", "items": {key: value.state() for key, value in self.items.items()}}

    def restore(self, saved):
        if set(self.items) == {"ToolHang"}:
            self.items["ToolHang"].restore(saved)
        else:
            require(saved["kind"] == "pilot-Can-Square" and set(saved["items"]) == set(self.items), "sampler pack set")
            for key, item in self.items.items():
                item.restore(saved["items"][key])

def features(obs, torch):
    template = obs["robot0_eef_pos"][..., 0]
    require(template.shape == (100, 10), "native batch shape")
    raw = [obs[key] for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]]
    require(all(value.dtype == torch.float32 and value.device == template.device for value in raw)
            and bool(torch.isfinite(torch.cat(raw, dim=-1)).all()), "finite native expression inputs")
    z = {"eef_" + name: obs["robot0_eef_pos"][..., i] for i, name in enumerate(("x", "y", "z"))}
    z.update({"quat_" + str(i): obs["robot0_eef_quat"][..., i] for i in range(4)})
    z.update({"gripper_" + str(i): obs["robot0_gripper_qpos"][..., i] for i in range(2)})
    width = obs["object"].shape[-1]
    require(width in {14, 44}, "original raw object dimensions")
    for i in range(44):
        z["object_" + str(i)] = obs["object"][..., i] if i < width else torch.zeros_like(template)
        z["object_valid_" + str(i)] = torch.ones_like(template) if i < width else torch.zeros_like(template)
    return z

def evaluate_expression(source: str, z: dict, torch, limits: tuple):
    template = z["eef_x"]
    # features() validated raw inputs once; all zero/one masks are constructed here.
    # Retained z tensors stay alive, so their IDs cannot alias transient intermediates.
    known_input_ids = {id(value) for value in z.values()}

    def check(value):
        require((type(value) is float and math.isfinite(value)) or
                (type(value) is torch.Tensor and value.shape == template.shape and value.device == template.device
                 and value.dtype == torch.float32 and
                 (id(value) in known_input_ids or bool(torch.isfinite(value).all()))), "finite expression intermediate")
        return value

    operations = {"positive": lambda a: a, "negative": lambda a: -a, "add": lambda a, b: a + b,
                  "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
                  "abs": abs, "square": lambda a: a * a,
                  "clamp": lambda a, lo, hi: torch.clamp(a, lo, hi) if type(a) is torch.Tensor else min(hi, max(lo, a))}
    raw = expression.evaluate(source, z, operations, check)
    value = torch.full_like(template, raw) if type(raw) is float else raw
    return value.detach().clamp(*limits)

class NativeLoss:
    def __init__(self, job: Job, model, U: dict, space: str, torch):
        validate_update(U, space)
        self.job, self.model, self.U, self.space, self.torch = job, model, U, space, torch
        self.original_forward = model._forward_training
        self.original_losses = model._compute_losses
        self.original_backward = torch.autograd.backward
        self.optimizer = model.optimizers["policy"]
        self.original_step = self.optimizer.step
        self.targets = None
        self.last = None
        self.policy_forward = model.nets["policy"].forward_train
        adapter = self

        def forward(_model, batch):
            job.charge(training_policy_forward_attempts=1)
            dist = adapter.policy_forward(obs_dict=batch["obs"], goal_dict=batch.get("goal_obs"))
            require(tuple(dist.batch_shape) == (100, 10) and tuple(dist.event_shape) == (7,), "native GMM axes")
            job.charge(training_log_prob_calls=1,
                       training_density_positions=5000 if space == "neighborhood" else 1000)
            if space == "neighborhood":
                require(adapter.targets is not None and adapter.targets.shape == (100, 10, 5, 7)
                        and torch.equal(adapter.targets[..., 0, :], batch["actions"]), "included exact anchor")
                logp = dist.log_prob(adapter.targets.movedim(2, 0)).movedim(0, -1)
            else:
                logp = dist.log_prob(batch["actions"]).unsqueeze(-1)
            require(bool(torch.isfinite(logp).all()), "finite native log densities")
            return {"log_probs": logp[..., 0], "target_log_probs": logp}

        def losses(_model, predictions, batch):
            job.charge(scalar_objective_attempts=2)
            result = adapter.original_losses(predictions, batch)
            reference_nll = float(result["action_loss"].detach())
            z = features(batch["obs"], torch)
            weights = evaluate_expression(U["weight_source"], z, torch, (.25, 4.))
            mix = evaluate_expression(U["mix_source"], z, torch, (0., 1.))
            if space == "anchor":
                require(not bool(mix.any()), "zero mix in assigned anchor space")
            logp = predictions["target_log_probs"]
            # Probabilities depend on the current policy; do not detach them.
            finite_target_loss = -(torch.softmax(logp, dim=-1) * logp).sum(dim=-1)
            composite = (weights * ((1 - mix) * (-predictions["log_probs"]) + mix * finite_target_loss)).sum() / weights.sum()
            require(composite.ndim == 0 and composite.requires_grad and bool(torch.isfinite(composite)), "finite scalar loss")
            result["action_loss"] = composite
            adapter.last = {"loss": float(composite.detach()), "anchor_NLL": reference_nll,
                            "mix_positive_positions": int((mix > 0).sum()), "mix_mean": float(mix.mean()),
                            "weight_min": float(weights.min()), "weight_max": float(weights.max())}
            return result

        def backward(*args, **kwargs):
            job.charge(training_gradient_attempts=1)
            return adapter.original_backward(*args, **kwargs)

        def step(_optimizer, *args, **kwargs):
            grads = [p.grad for p in model.nets["policy"].parameters() if p.grad is not None]
            require(grads and all(bool(torch.isfinite(g).all()) for g in grads), "finite pre-mutation gradients")
            job.charge(optimizer_entries=1)
            value = adapter.original_step(*args, **kwargs)
            job.charge(optimizer_returns=1)
            return value

        model._forward_training = MethodType(forward, model)
        model._compute_losses = MethodType(losses, model)
        self.optimizer.step = MethodType(step, self.optimizer)
        torch.autograd.backward = backward

    def close(self):
        self.model._forward_training = self.original_forward
        self.model._compute_losses = self.original_losses
        self.optimizer.step = self.original_step
        self.torch.autograd.backward = self.original_backward

def save_state(job: Job, name: str, model, sampler, counters, binding, core, torch):
    job.charge(state_saves=1)
    path = job.out / (name + ".pt")
    require(not path.exists(), "new checkpoint only")
    core.save(path, core.state(model, sampler, counters, binding))
    require(path.stat().st_size <= 33554432, "new checkpoint bytes")
    if model.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(0)
        job.value["max_allocated_cuda_bytes"] = peak
        require(peak <= 805306368, "allocated GPU peak")
    info = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "binding": binding, "counters": copy.deepcopy(counters), "profile": job.definition["profile"]}
    write(path.with_suffix(".json"), info)
    return info

def initialize_target(job: Job) -> None:
    core, np, torch = runtime(job, True)
    packs = {task: Pack(job, task, np) for task in ["Can", "Square"]}
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    model = construct(job, "target", torch, "cuda:0")
    if job.definition["id"] == "target_S0":
        saved = load_checkpoint(job, job.p["target_parent_input"], torch, "cpu", "training_checkpoint_load_attempts")
        require(saved["schema"] == 1 and same_binding(saved["binding"], job.p["target_original_binding"])
                and saved["counters"]["updates"] == 0, "original Square compatible parent")
        model.nets.load_state_dict(saved["weights"], strict=True)
    else:
        require(job.definition["id"] == "target_fresh_Can_plus_original_Square", "fixed initialization ID")
    require(all(not opt.state for opt in model.optimizers.values()), "fresh target optimizer")
    sampler = Samplers(packs, core)
    counters = {"updates": 0, "sampled_windows": 0, "epochs": 0}
    info = save_state(job, "S0", model, sampler, counters, job.p["target_stage_binding"], core, torch)
    job.finish(checkpoints={"0": info}, initialization="weights only / fresh complete stage; not source-state transfer")

def train(job: Job) -> None:
    core, np, torch = runtime(job, True)
    profile = job.definition["profile"]
    tasks = ["ToolHang"] if profile == "source" else ["Can", "Square"]
    packs = {task: Pack(job, task, np) for task in tasks}
    model = construct(job, profile, torch, "cuda:0")
    sampler = Samplers(packs, core)
    start = read(job.root / job.definition["start_checkpoint_reference"])
    saved = load_checkpoint(job, start, torch, "cuda:0", "training_checkpoint_load_attempts")
    expected_binding = job.p["source_state_binding"] if profile == "source" else job.p["target_stage_binding"]
    require(same_binding(saved["binding"], expected_binding), "complete canonical starting binding")
    binding = copy.deepcopy(saved["binding"])
    counters = core.restore(model, sampler, saved, binding)
    require(core.equal(saved, core.state(model, sampler, counters, binding)) > 0, "complete starting state restored")
    del saved
    initial_update = counters["updates"]
    U = read(job.root / job.definition["U_reference"])
    space = job.definition["space"]
    adapter = NativeLoss(job, model, U, space, torch)
    checkpoint_records = {}
    try:
        with gzip.open(job.out / "training.jsonl.gz", "wt") as stream:
            for local in range(1, job.definition["updates"] + 1):
                model.set_train()
                job.charge(optimizer_attempts=1, sampled_windows=100, supervised_positions=1000)
                amounts = {"ToolHang": 100} if profile == "source" else \
                    ({"Can": 50, "Square": 50} if job.definition.get("replay") else {"Can": 100})
                batches, target_batches, starts_record = [], [], {}
                for task, amount in amounts.items():
                    starts = sampler.next(task, amount)
                    batch, index = packs[task].batch(starts, torch, "cuda:0")
                    batches.append(batch)
                    starts_record[task] = starts.tolist()
                    job.charge(**{"windows_" + task: amount})
                    if space == "neighborhood":
                        target_batches.append(packs[task].targets(index, U["radius"], torch, "cuda:0"))
                batch = {"actions": torch.cat([b["actions"] for b in batches]),
                         "obs": {key: torch.cat([b["obs"][key] for b in batches]) for key in batches[0]["obs"]}}
                adapter.targets = torch.cat(target_batches) if target_batches else None
                info = model.train_on_batch(model.process_batch_for_training(batch), epoch=counters["updates"] // 100 + 1)
                require(adapter.last is not None and math.isfinite(float(info["losses"]["action_loss"])), "actual native loss used")
                counters["updates"] += 1
                counters["sampled_windows"] += 100
                if counters["updates"] % 100 == 0:
                    model.on_epoch_end(counters["updates"] // 100)
                    counters["epochs"] += 1
                job.charge(update_returns=1)
                stream.write(canonical({"local_update": local, "native_update": counters["updates"],
                                        "starts": starts_record, **adapter.last}).decode() + "\n")
                if local % 100 == 0:
                    stream.flush()
                    job.value["last_training"] = {"local_update": local, **adapter.last}
                    job.persist()
                if local in job.definition["save_updates"]:
                    checkpoint_records[str(local)] = save_state(job, "step" + str(local), model, sampler,
                                                                counters, binding, core, torch)
        require(counters["updates"] == initial_update + job.definition["updates"], "complete fixed update stage")
        require(all(job.counts.get(key, 0) == job.definition["updates"] for key in
                    ["optimizer_attempts", "optimizer_entries", "optimizer_returns", "update_returns",
                     "training_policy_forward_attempts", "training_log_prob_calls", "training_gradient_attempts"]), "actual per-update native operation counts")
        require(set(checkpoint_records) == {str(x) for x in job.definition["save_updates"]}, "all fixed checkpoints saved")
        job.finish(checkpoints=checkpoint_records, last_training=adapter.last)
    finally:
        adapter.close()
