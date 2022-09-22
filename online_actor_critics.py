# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import hydra
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils_hyp import UpdatableModule

LN4 = np.log(4)
LNROOT2 = np.log(np.sqrt(2))


def run_layers(inputs, layers):
    out = inputs
    for layer in layers:
        out = layer.forward(out)
    return out


class ActorCritic(nn.Module):
    def __init__(self, modules, **model_kwargs):
        super(ActorCritic, self).__init__()
        if '_target_' in modules:
            modules = hydra.utils.call(modules, **model_kwargs)
        else:
            modules = modules
            print('WARNING: model_kwargs being ignored:')
            for kw, value in model_kwargs:
                print('{}: {}'.format(kw, value))
        self.auxiliary_dims = model_kwargs.get('auxiliary_dims',
                                               modules.get('auxiliary_dims', None))
        if self.auxiliary_dims:
            assert isinstance(self.auxiliary_dims, int)
            assert self.auxiliary_dims > 0
        shared_modules = modules.get('shared_modules', [])
        actor_modules = modules.get('actor_modules', [])
        critic_modules = modules.get('critic_modules', [])
        if len(actor_modules) == 0:
            self.fully_shared = True
            self.sm = nn.Sequential(*shared_modules)  # Shared trunk
        else:
            self.fully_shared = False
            self.sm = nn.Sequential(*shared_modules)  # Shared trunk
            self.am = nn.Sequential(*actor_modules)  # Actor head
            self.cm = nn.Sequential(*critic_modules)  # Critic head

        self.modules_to_update = []

        for mod in shared_modules + actor_modules + critic_modules:
            if isinstance(mod, UpdatableModule):
                self.modules_to_update.append(mod)
                if 'representation_metrics' in model_kwargs:
                    mod.setup_metric(metrics=model_kwargs['representation_metrics'])

    def get_all_outputs(self, inputs):
        shared = self.sm(inputs)
        if self.fully_shared:
            out_dims = shared.shape[-1]
            if self.auxiliary_dims:
                act, aux, value = th.split(shared, [out_dims - 1 - self.auxiliary_dims, self.auxiliary_dims, 1], dim=-1)
            else:
                act, value = th.split(shared, [out_dims - 1, 1], dim=-1)
                aux = None
        else:
            act = self.am(shared)
            value = self.cm(shared)
            if self.auxiliary_dims:
                out_dims = act.shape[-1]
                act, aux = th.split(act, [out_dims - self.auxiliary_dims, self.auxiliary_dims], dim=-1)
            else:
                aux = None
        return act, aux, value

    def forward(self, inputs):
        act, aux, value = self.get_all_outputs(inputs)
        return act, value

    def get_actor_outputs(self, inputs):
        shared = self.sm(inputs)
        if self.fully_shared:
            act = shared[..., :-1]
        else:
            act = self.am(shared)
        if self.auxiliary_dims:
            out_dims = act.shape[-1]
            act, aux = th.split(act, [out_dims - self.auxiliary_dims, self.auxiliary_dims], dim=-1)
        else:
            aux = None
        return act, aux

    def get_action(self, inputs):
        act, aux = self.get_actor_outputs(inputs)
        return act

    def get_value(self, inputs):
        shared = self.sm(inputs)
        if self.fully_shared:
            return shared[..., -1]
        return self.cm(shared).squeeze(-1)

    def update(self, current_steps):
        for mod in self.modules_to_update:
            mod.update(current_steps=current_steps)

class DiscreteActorCritic(ActorCritic):
    """Actor and critic neural networks for discrete action spaces."""

    def __init__(self, modules, **model_kwargs):
        super(DiscreteActorCritic, self).__init__(modules, **model_kwargs)
        self.is_discrete = True

    def forward(self, inputs):
        act_logits, value = super().forward(inputs)
        act_logprobs = F.log_softmax(act_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        act = th.multinomial(act_probs, num_samples=1)
        return act, act_probs, act_logprobs, value

    def get_action_logprob_entropy_value(self, inputs):
        act, act_probs, act_logprobs, value = self.forward(inputs)
        logprob = th.gather(act_logprobs, dim=-1, index=act)
        entropy = (-1 * act_probs * act_logprobs).sum(-1)
        return act.squeeze(-1), logprob.squeeze(-1), entropy, value.squeeze(-1)

    def get_logprob_entropy_value(self, obs, act):
        act_logits, value = super().forward(obs)
        act_logprobs = F.log_softmax(act_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        logprob = th.gather(act_logprobs, dim=-1, index=act.unsqueeze(-1))
        entropy = (-1 * act_probs * act_logprobs).sum(-1)
        return logprob.squeeze(-1), entropy, value.squeeze(-1)

    def get_action_logprob_value(self, inputs):
        act, act_probs, act_logprobs, value = self.forward(inputs)
        logprob = th.gather(act_logprobs, dim=-1, index=act)
        return act.squeeze(-1), logprob.squeeze(-1), value.squeeze(-1)

    def get_prob_logprob(self, inputs):
        act_logits = super().get_action(inputs)
        act_logprobs = F.log_softmax(act_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        return act_probs, act_logprobs

    def get_prob_logprob_value(self, inputs):
        act_logits, value = super().forward(inputs)
        act_logprobs = F.log_softmax(act_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        return act_probs, act_logprobs, value

    def get_logprob_aux_value(self, inputs, value_inputs=None):
        if value_inputs is None:
            act_logits, aux, value = super().get_all_outputs(inputs)
        else:
            act_logits, aux = super().get_actor_outputs(inputs)
            value = super().get_value(value_inputs)
        act_logprobs = F.log_softmax(act_logits, dim=-1)
        return act_logprobs, aux.squeeze(-1), value.squeeze(-1)

    def get_action(self, inputs, det=False):
        act_logits = super().get_action(inputs)
        if det:
            act = act_logits.argmax(dim=-1)
        else:
            act_probs = F.softmax(act_logits, dim=-1)
            act = th.multinomial(act_probs, num_samples=1).squeeze(-1)
        return act


class ContinuousActorCritic(ActorCritic):
    """Actor and critic neural networks for continuous action spaces."""

    def __init__(self, modules, act_dims, scalar_std=False, separate_std=False,
                 log_std_range=None, squash_action=True):

        super(ContinuousActorCritic, self).__init__(modules)

        self.act_dims = act_dims
        if scalar_std:
            self.std_dims = 1
            self.std_reps = self.act_dims
        else:
            self.std_dims = self.act_dims
            self.std_reps = 1

        self.separate_std = separate_std
        if separate_std:
            # 1 x (act_dims or 1)
            self.log_std = nn.Parameter(th.zeros(1, self.std_dims,
                                                 dtype=th.float32))

        if log_std_range is not None:
            assert log_std_range[1] >= log_std_range[0]
            self.bound_log_std = True
            self.min_log_std = log_std_range[0]
            self.range_log_std = log_std_range[1] - log_std_range[0]
        else:
            self.bound_log_std = False

        self.log_prob_offset = act_dims / 2 * np.log(np.pi * 2)
        self.squash_action = squash_action
        self.is_discrete = False

    def _get_action_parameters(self, act_logits):
        if self.separate_std:
            act_mean = act_logits
            log_std = self.log_std
        else:
            act_mean, log_std = th.split(
                act_logits, [self.act_dims, self.std_dims], dim=-1)
        if self.bound_log_std:
            log_std = self.min_log_std + (th.tanh(log_std)
                                          + 1) / 2 * self.range_log_std
        return act_mean, log_std

    def _get_action_probs(self, act_mean, log_std):
        raw_noise = th.randn_like(act_mean, device=act_mean.device)
        std = th.exp(log_std)
        noise = std * raw_noise
        act = act_mean + noise
        logprob = (-1 / 2 * raw_noise.pow(2).sum(-1) -
                   (log_std.sum(-1) * self.std_reps) - self.log_prob_offset)
        if self.squash_action:
            squash_features = -2 * act
            squash_correction = (LN4 + squash_features - 2 *
                                 F.softplus(squash_features)).sum(-1)
            logprob -= squash_correction
            act = th.tanh(act)
        return act, logprob

    def _get_action(self, act_mean, log_std, mean=False):
        if mean:
            act = act_mean
        else:
            raw_noise = th.randn_like(act_mean, device=act_mean.device)
            std = th.exp(log_std)
            noise = std * raw_noise
            act = act_mean + noise
        if self.squash_action:
            act = th.tanh(act)
        return act

    def forward(self, inputs):
        act_logits, value = super().forward(inputs)
        act_mean, log_std = self._get_action_parameters(act_logits)
        act, logprob = self._get_action_probs(act_mean, log_std)
        return act, logprob, value

    def get_action_logprob_entropy_value(self, inputs):
        act_logits, value = super().forward(inputs)
        act_mean, log_std = self._get_action_parameters(act_logits)
        act, logprob = self._get_action_probs(act_mean, log_std)
        entropy = log_std + LNROOT2 + 1 / 2
        return act, logprob, entropy, value

    def get_logprob_entropy_value(self, obs, act):
        act_logits, value = super().forward(obs)
        act_mean, log_std = self._get_action_parameters(act_logits)

        if self.squash_action:
            raw_act = th.atanh(act)
            squash_features = -2 * raw_act
            squash_correction = (LN4 + squash_features - 2 *
                                 F.softplus(squash_features)).sum(-1)
        else:
            raw_act = act
            squash_correction = 0.0

        std = th.exp(log_std)
        raw_noise = (raw_act - act_mean) / std

        logprob = (-1 / 2 * raw_noise.pow(2).sum(-1) -
                   (log_std.sum(-1) * self.std_reps) - self.log_prob_offset)

        logprob -= squash_correction

        entropy = log_std + LNROOT2 + 1 / 2

        return logprob, entropy, value

    def get_action_logprob_value(self, inputs):
        return self.forward(inputs)

    def get_action(self, inputs, mean=False):
        act_mean, log_std = self._get_action_parameters(
            self.am(self.sm(inputs)))
        return self._get_action(act_mean, log_std, mean)
