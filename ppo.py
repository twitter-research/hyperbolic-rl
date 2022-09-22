# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import hydra.utils
import torch
import torch as th
from torch import nn
from utils import to_numpy

from logging_metrics import AverageMetrics

class PPO():
    def __init__(self,
                 device,
                 actor_critic_model,
                 optimizer_fn=th.optim.Adam,
                 optimizer_kwargs={'lr': 5e-4},
                 clipping=0.2,
                 value_clipping=None,
                 ent_coeff=0.0,
                 val_coeff=0.5,
                 max_gradient_norm=None,
                 predictions_logging=False,
                 optimizer_scheduler=None,):

        self.device = device
        self.ac_model = actor_critic_model.to(device=device)
        self.optimizer = optimizer_fn(self.ac_model.parameters(),
                                      **optimizer_kwargs)

        if optimizer_scheduler is not None:
            self.scheduler = hydra.utils.instantiate(optimizer_scheduler, optimizer=self.optimizer)
        else:
            self.scheduler = None

        self.clipping = clipping
        self.value_clipping = value_clipping

        self.ent_coeff = ent_coeff
        self.val_coeff = val_coeff

        self.max_gradient_norm = max_gradient_norm
        self.predictions_logging = predictions_logging
        self.metrics = AverageMetrics('actor_loss',
                                      'entropy_loss',
                                      'value_loss', )
        if predictions_logging:
            self.metrics.add('mean_logprobs', 'predicted_values', 'discounted_returns')
        self.debug_metrics = AverageMetrics('average_returns',
                                            'std_returns',
                                            'std_advantages')

    def save(self, dir, step, preprocessor=None):
        checkpoint = '{}/checkpoint-{}.pt'.format(dir, step)
        agent_params = self.ac_model.state_dict()
        optimizer_params = self.optimizer.state_dict()
        if preprocessor is not None:
            th.save({'agent': agent_params,
                     'optimizer': optimizer_params,
                     'preprocessor': preprocessor}, checkpoint)
        else:
            th.save({'agent': agent_params,
                     'optimizer': optimizer_params, }, checkpoint)
        return checkpoint

    def load(self, path):
        params = th.load(path)
        self.ac_model.load_state_dict(params['agent'])
        self.optimizer.load_state_dict(params['optimizer'])
        if 'preprocessor' in params:
            return params['preprocessor']

    def get_action_logprob_value(self, obs):
        with th.no_grad():
            act, logprob, value = self.ac_model.get_action_logprob_value(obs)
            return act, logprob, value

    def get_value(self, obs):
        with th.no_grad():
            return self.ac_model.get_value(obs)

    def act(self, obs, det=False):
        with th.no_grad():
            return self.ac_model.get_action(obs, det=det).cpu().numpy()

    def get_losses(self, obs, act, old_logprobs, old_values, returns,
                   advantages):
        metrics_dict = dict()
        logprobs, entropies, values = self.ac_model.get_logprob_entropy_value(
            obs=obs, act=act)
        with th.no_grad():
            adv_std, adv_mean = th.std_mean(advantages)
            norm_advantages = (advantages - adv_mean) / (adv_std + 1e-7)
        ratio = th.exp(logprobs - old_logprobs)
        clipped_ratio = th.clamp(ratio, min=1 - self.clipping,
                                 max=1 + self.clipping)
        pessimistic_adv = th.minimum(ratio * norm_advantages,
                                     clipped_ratio * norm_advantages)
        actor_loss = -1 * pessimistic_adv.mean()

        entropy_loss = -1 * entropies.mean()

        value_losses = (values - returns).pow(2)
        if self.value_clipping is not None:
            clipped_values = old_values + th.clamp(values - old_values,
                                                   min=-self.value_clipping,
                                                   max=self.value_clipping)
            clipped_value_losses = (clipped_values - returns).pow(2)
            value_losses = th.maximum(value_losses, clipped_value_losses)

        value_loss = 0.5 * value_losses.mean()
        if self.predictions_logging:
            metrics_dict['mean_logprobs'] = logprobs.mean()
            metrics_dict['predicted_values'] = values.mean()
            metrics_dict['discounted_returns'] = returns.mean()
        return actor_loss, entropy_loss, value_loss, metrics_dict

    def learn(self, buffer, epochs, batch_size, current_steps=None):
        if current_steps is not None:
            self.ac_model.update(current_steps=current_steps)
        for e in range(epochs):
            for batch in buffer.get_batches(batch_size):
                actor_loss, entropy_loss, value_loss, metrics_dict = self.get_losses(*batch)
                total_loss = (actor_loss + self.ent_coeff * entropy_loss +
                              self.val_coeff * value_loss)
                self.ac_model.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.max_gradient_norm is not None:
                    nn.utils.clip_grad_norm_(self.ac_model.parameters(),
                                             self.max_gradient_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                metrics_dict = {k: to_numpy(t) for k, t in metrics_dict.items()}
                self.metrics.update(actor_loss=to_numpy(actor_loss),
                                    entropy_loss=to_numpy(entropy_loss),
                                    value_loss=to_numpy(value_loss),
                                    **metrics_dict)


    def return_and_reset_metrics(self, ):
        metrics = self.metrics.get()
        self.metrics.reset()
        return metrics


class ModularPPO(PPO):
    def __init__(self,
                 device,
                 actor_critic_model,
                 optimizer,
                 clipping=0.2,
                 value_clipping=None,
                 ent_coeff=0.0,
                 val_coeff=0.5,
                 max_gradient_norm=None,
                 predictions_logging=False,
                 optimizer_scheduler=None,):

        self.metrics = AverageMetrics('actor_loss',
                                      'entropy_loss',
                                      'value_loss',)
        if predictions_logging:
            self.metrics.add('mean_logprobs',
                             'predicted_values',
                             'discounted_returns',)

        self.device = device
        self.ac_model = hydra.utils.instantiate(actor_critic_model, _recursive_=False).to(device=device)
        self.optimizer = hydra.utils.instantiate(optimizer, params=self.ac_model.parameters())
        if optimizer_scheduler is not None:
            self.scheduler = hydra.utils.instantiate(optimizer_scheduler, optimizer=self.optimizer)
        else:
            self.scheduler = None
        self.clipping = clipping
        self.value_clipping = value_clipping

        self.ent_coeff = ent_coeff
        self.val_coeff = val_coeff

        self.max_gradient_norm = max_gradient_norm

        self.predictions_logging = predictions_logging
