# Original scientific source excerpts

Literal lines from original files; indentation retained. These are excerpts, not a standalone runnable entry. `p`, `episodes`, `save`, `before_forward` and the model/runtime context are supplied by the original wrapper.

## driver.py:88-110

Original whole-file SHA-256: `68f7f8a5292accb50ed7ca9bc1a99ee7d09e3187b379691913be51d3606f8105`

```python
        if cfg['algo']['rnn']['open_loop'] or cfg['algo']['rnn']['horizon'] != 10:
            raise RuntimeError('unsupported recurrent semantics')
        tensors = [[{k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                     for k, v in obs.items()} for obs in ep['inputs']] for _, ep in episodes]
        means_by_role = {}
        for role in ['parent', 'child']:
            saved = torch.load(p['checkpoints'][role]['path'], map_location='cpu', weights_only=False)
            if saved['schema'] != 1 or saved['normalization'] is not None or saved['binding']['config'] != cfg:
                raise ValueError('checkpoint config/normalization')
            if saved['counters']['updates'] != p['checkpoints'][role]['updates']:
                raise ValueError('checkpoint update identity')
            config = config_factory('bc', dic=cfg)
            ObsUtils.initialize_obs_utils_with_config(config)
            result['models_attempted'] += 1; save()
            model = algo_factory('bc', config, {k: (v,) for k,v in SHAPES.items()}, 7, torch.device('cpu'))
            result['models_returned'] += 1; save()
            model.nets.load_state_dict(saved['weights'], strict=True)
            model.set_eval()
            policy = model.nets['policy']
            if policy.use_tanh or model._rnn_is_open_loop or model._rnn_horizon != 10:
                raise RuntimeError('unsupported distribution/recurrent semantics')
            if not policy.low_noise_eval:
                raise RuntimeError('expected original low-noise evaluation')
```

## driver.py:112-146

Original whole-file SHA-256: `68f7f8a5292accb50ed7ca9bc1a99ee7d09e3187b379691913be51d3606f8105`

```python
            def distribution(obs: dict):
                before_forward()
                if model._rnn_hidden_state is None or model._rnn_counter % model._rnn_horizon == 0:
                    model._rnn_hidden_state = policy.get_rnn_init_state(batch_size=1, device=model.device)
                model._rnn_counter += 1
                dist, model._rnn_hidden_state = policy.forward_train_step(obs, rnn_state=model._rnn_hidden_state)
                result['forward_returned'] += 1
                return dist

            with torch.no_grad():
                # Cross horizon 10: same seed, same inputs, same official action sampler.
                fixture = tensors[0][:12]
                if len(fixture) != 12:
                    raise ValueError('fixture needs twelve observations')
                model.reset(); torch.manual_seed(20261027)
                expected = []
                for obs in fixture:
                    before_forward(); expected.append(model.get_action(obs))
                    result['forward_returned'] += 1
                model.reset(); torch.manual_seed(20261027)
                for obs, reference in zip(fixture, expected):
                    if not torch.equal(distribution(obs).sample(), reference):
                        raise ValueError('official action path mismatch across horizon')
                result.setdefault('official_path_fixture', {})[role] = '12 exact actions including reset boundary'
                role_means = []
                for episode in tensors:
                    model.reset()
                    values = []
                    for obs in episode:
                        mean = distribution(obs).mean
                        if mean.shape != (1, 7) or not torch.isfinite(mean).all():
                            raise ValueError('finite analytic mixture mean')
                        values.append(mean.squeeze(0).tolist())
                    role_means.append(values)
                means_by_role[role] = role_means
```

## driver.py:151-161

Original whole-file SHA-256: `68f7f8a5292accb50ed7ca9bc1a99ee7d09e3187b379691913be51d3606f8105`

```python
        for i, (item, ep) in enumerate(episodes):
            paired = zip(means_by_role['parent'][i], means_by_role['child'][i])
            drift = [math.sqrt(sum(((a-b)/2.)**2 for a,b in zip(left,right))/7.) for left,right in paired]
            result['rows'].append({'seed': item['seed'], 'steps': ep['steps'],
                                   'tail20_drift': top_tail_mean(drift),
                                   'per_step_drift': drift, 'supervision_schema_valid': True})
        ordered = sorted(result['rows'], key=lambda row: (-row['tail20_drift'], row['seed']))
        result['drift_selected'] = [row['seed'] for row in ordered[:3]]
        result['loss_selected'] = p['loss_selected']
        result['same_selection'] = set(result['drift_selected']) == set(p['loss_selected'])
        result['mean_output_warning'] = 'Analytic GMM mean is a scoring proxy, not a deployed action; cancellation is possible. Not full RIFAR.'
```

## log_inputs.py:75-79

Original whole-file SHA-256: `3b7485aeb917e3378f228e496f2c90612190528e9b59585847ff370c64d4c443`

```python
def top_tail_mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('drift values')
    k = math.ceil(len(values) / 5)
    return sum(sorted(values, reverse=True)[:k]) / k
```
