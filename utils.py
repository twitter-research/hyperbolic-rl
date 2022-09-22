# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def dequantize(ims):
    return (ims / 127.5 - 1)


def log_with_suffix(returns, suffix='', log=True):
    results = {'n_test{}'.format(suffix): len(returns),
               'mean_returns{}'.format(suffix): np.mean(returns),
               'std_returns{}'.format(suffix): np.std(returns),
               'max_returns{}'.format(suffix): np.max(returns),
               'min_returns{}'.format(suffix): np.min(returns)}
    if log:
        print(suffix)
        print('N Collected episodes: {}'.format(results['n_test{}'.format(suffix)]))
        print('Mean returns: {}'.format(results['mean_returns{}'.format(suffix)]))
        print('Std. returns: {}'.format(results['std_returns{}'.format(suffix)]))
        print('Max returns: {}'.format(results['max_returns{}'.format(suffix)]))
        print('Min returns: {}'.format(results['min_returns{}'.format(suffix)]))
    return results


def log_returns_stats(returns, log=True):
    if isinstance(returns, dict):
        results = dict()
        for setting, setting_returns in returns.items():
            results.update(log_with_suffix(returns=setting_returns, suffix=setting, log=log))
    else:
        results = log_with_suffix(returns=returns, suffix='', log=log)
    return results


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def final_weight_init(m):
    nn.init.orthogonal_(m.weight.data, gain=0.01)
    if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.0)


class RunningStats:
    def __init__(self, device, shape, epsilon=1e-4):
        self._shape = list(shape)
        self._running_mean = th.zeros(self._shape, dtype=th.float32, device=device)
        self._running_var = th.ones(self._shape, dtype=th.float32, device=device)
        self._count = torch.tensor(epsilon, dtype=th.float32, device=device)

    def update(self, samples):
        samples = th.reshape(samples, [-1, *self._shape])

        n_samples = samples.shape[0]

        samples_mean = th.sum(samples, dim=0) / n_samples
        samples_var = th.sum(th.square(samples - samples_mean), dim=0) / (n_samples)  # -1 - omit for edge cases for now

        mean_delta = samples_mean - self._running_mean
        new_count = self._count + n_samples
        self._running_mean = self._running_mean + mean_delta * n_samples / new_count

        m_a = self._running_var * self._count
        m_b = samples_var * n_samples
        M2 = m_a + m_b + th.square(mean_delta) * self._count * n_samples / new_count
        self._running_var = M2 / new_count

        self._count = new_count

    def get(self):
        return self._running_mean, self._running_var


class Preprocessor(nn.Module):
    def __init__(self,
                 device,
                 reward_pre='normalization_backwards',
                 observation_pre='normalization',
                 obs_dims=None,
                 observation_clip=10,
                 reward_clip=10,
                 gamma=0.999):
        super(Preprocessor, self).__init__()
        self.device = device
        self._reward_pre = reward_pre
        if reward_pre == 'normalization_backwards':
            self._reward_stats = RunningStats(device=device, shape=[])
            self._current_returns = 0.0
            self._last_nonterminals = 0.0
            self._gamma = gamma
        elif reward_pre == 'normalization':
            self._reward_stats = RunningStats(device=device, shape=[])
            self._current_returns = 0.0
        self._reward_clip = reward_clip

        self._observation_pre = observation_pre
        if observation_pre == 'normalization':
            self._obs_stats = RunningStats(device=device,
                                           shape=obs_dims)
        self._obs_clip = observation_clip

    def preprocess_obs(self, observation):
        observation = th.tensor(observation, device=self.device, dtype=th.float32)
        if self._observation_pre == 'normalization':
            mean, var = self._obs_stats.get()
            observation = (observation - mean) / th.sqrt(var + 1e-7)
        elif self._observation_pre == 'dequantization':
            observation = dequantize(observation)
        return th.clip(observation, min=-1 * self._obs_clip,
                       max=self._obs_clip)

    def preprocess_rew(self, reward):
        if self._reward_pre is not None:
            mean, var = self._reward_stats.get()
            reward = reward / (th.sqrt(var + 1e-7).cpu().numpy())
        return np.clip(reward, a_min=-1 * self._reward_clip,
                       a_max=self._reward_clip)

    def update(self, observation, rewards, nonterminals):
        self._update_obs(th.tensor(observation, device=self.device))
        self._update_rew(th.tensor(rewards, device=self.device))
        self._last_nonterminals = th.tensor(nonterminals, device=self.device)

    def _update_obs(self, observation):
        if self._observation_pre == 'normalization':
            self._obs_stats.update(observation)

    def _update_rew(self, rewards):
        if self._reward_pre == 'normalization_backwards':
            self._current_returns = (rewards +
                                     self._gamma * self._current_returns * self._last_nonterminals)
            self._reward_stats.update(self._current_returns)
        elif self._reward_pre == 'normalization':
            self._reward_stats.update(rewards)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if self.training:
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, 'replicate')
            eps = 1.0 / (h + 2 * self.pad)
            arange = torch.linspace(-1.0 + eps,
                                    1.0 - eps,
                                    h + 2 * self.pad,
                                    device=x.device,
                                    dtype=x.dtype)[:h]
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

            shift = torch.randint(0,
                                  2 * self.pad + 1,
                                  size=(n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)
            shift *= 2.0 / (h + 2 * self.pad)
            grid = base_grid + shift
            return F.grid_sample(x,
                                 grid,
                                 padding_mode='zeros',
                                 align_corners=False)
        else:
            return x
