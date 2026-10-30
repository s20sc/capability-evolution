"""Fixed GMM readout intervention. No scientific imports at module import time."""
from __future__ import annotations
from types import MethodType


def install_argmax(model, trace: dict, audit: dict | None = None):
    policy=model.nets['policy']
    if policy.use_tanh or not policy.low_noise_eval or policy.training:
        raise ValueError('fixed non-tanh low-noise eval GMM required')
    original=policy.forward_step

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        sequence={k:v.unsqueeze(1) for k,v in obs_dict.items()}
        distribution,state=self.forward_train(obs_dict=sequence,goal_dict=goal_dict,
                                               rnn_init_state=rnn_state,return_state=True)
        if type(distribution).__name__!='MixtureSameFamily':
            raise ValueError('unexpected fixed policy distribution')
        means=distribution.component_distribution.mean
        probabilities=distribution.mixture_distribution.probs
        if tuple(means.shape)!=(1,1,5,7) or tuple(probabilities.shape)!=(1,1,5):
            raise ValueError('fixed single-step GMM shapes changed')
        index=probabilities.argmax(dim=-1)
        selector=index.unsqueeze(-1).unsqueeze(-1).expand(1,1,1,7)
        action=means.gather(2,selector).squeeze(2)[:,0]
        trace.update(mode='argmax',component_index=int(index.item()),
                     max_component_probability=float(probabilities.max().item()))
        if audit is not None:
            audit.update(means=means.detach().clone(),probabilities=probabilities.detach().clone())
        return action,state

    policy.forward_step=MethodType(forward_step,policy)
    return original
