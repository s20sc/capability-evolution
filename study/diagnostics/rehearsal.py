"""Frozen-parent distribution rehearsal; no imports or scientific work at import."""


def reserve_samples(status: dict, count: int, ceiling: int) -> None:
    """Reserve possible draws before a fallible call; never claim these all returned."""
    used = status.get('teacher_sample_vectors_reserved', 0)
    if type(count) is not int or count <= 0 or used + count > ceiling:
        raise ValueError('teacher sample reservation budget')
    status['teacher_sample_vectors_reserved'] = used + count


def teacher_targets(teacher, obs: dict, seed: int, torch, cuda_devices: list[int]):
    if teacher.training or teacher.low_noise_eval is not False:
        raise ValueError('use frozen learned distribution, not deployment low-noise scale')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('teacher must be frozen')
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        dist = teacher.forward_train(obs_dict=obs)
        actions = dist.sample()
        logp = dist.log_prob(actions)
    # Do not clip Gaussian samples: that would change the cloning objective.
    return actions.detach(), logp.detach()


def fit_probe(model, teacher, old, new, np, torch, device, seed: int) -> dict:
    """Fixed training-data checks, not validation or checkpoint selection."""
    was_training = model.nets.training
    flag = model.nets['policy'].low_noise_eval
    model.set_eval()
    model.nets['policy'].low_noise_eval = False
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            target = new.batch(np.arange(311, 480), device)
            td = model.nets['policy'].forward_train(obs_dict=target['obs'])
            nll = -td.log_prob(target['actions']).mean()
            fixed = old.batch(np.arange(100) * 240, device)
            pd = teacher.forward_train(obs_dict=fixed['obs'])
            qd = model.nets['policy'].forward_train(obs_dict=fixed['obs'])
            torch.manual_seed(seed)
            actions = pd.sample((4,))
            values = pd.log_prob(actions) - qd.log_prob(actions)
            if not torch.isfinite(nll) or not torch.isfinite(values).all():
                raise ValueError('nonfinite fixed probe')
            return {'target_window_nll': float(nll), 'old_learned_distribution_kl_mc': float(values.mean()),
                    'old_windows': 100, 'target_windows': 169, 'teacher_sample_vectors': 4000,
                    'scope': 'training inputs, raw learned densities; no retention guarantee; MC KL can be negative'}
    finally:
        model.nets['policy'].low_noise_eval = flag
        if was_training:
            model.set_train()
        else:
            model.set_eval()
