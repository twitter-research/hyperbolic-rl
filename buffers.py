# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch as th

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


class RolloutBuffer():
    def __init__(self, device, buffer_size, obs_dims, act_dims, use_gae,
                 preprocessor, n_envs=1, gamma=0.95, lambd=0.99,):
        self.buffer_size = buffer_size
        self.obs_dims = list(obs_dims)
        self.act_dims = list(act_dims)

        self.use_gae = use_gae

        self.preprocessor = preprocessor
        self.device = device
        self.n_envs = n_envs

        self.total_size = buffer_size * n_envs

        self.gamma = gamma  # discount
        self.lambd = lambd  # lambda is reserved, gae coefficient

        self.last_state=None
        self.obs_dtype = np.uint8


        self.reset()

    def reset(self, ):
        self.observations = np.zeros(
            [self.buffer_size, self.n_envs, *self.obs_dims], dtype=self.obs_dtype)
        self.actions = np.zeros([self.buffer_size, self.n_envs, *self.act_dims],
                                dtype=np.float32)
        self.rewards = np.zeros([self.buffer_size, self.n_envs],
                                dtype=np.float32)
        self.nonterminals = np.zeros([self.buffer_size, self.n_envs],
                                     dtype=np.float32)
        self.logprobs = np.zeros([self.buffer_size, self.n_envs],
                                 dtype=np.float32)
        self.values = np.zeros([self.buffer_size, self.n_envs],
                               dtype=np.float32)
        self.advantages = np.zeros([self.buffer_size, self.n_envs],
                                   dtype=np.float32)

        self.index = 0
        self.full = False
        self.flattened_data = False

    def add(self, observation, action, reward, nonterminal, logprob, value):

        self.preprocessor.update(observation, reward, nonterminal)

        self.observations[self.index] = np.array(observation).copy()
        self.actions[self.index] = action.cpu().numpy()
        self.rewards[self.index] = np.array(reward).copy()
        self.nonterminals[self.index] = np.array(nonterminal).copy()
        self.logprobs[self.index] = logprob.cpu().numpy()
        self.values[self.index] = value.cpu().numpy()

        self.index += 1
        if self.index == self.buffer_size:
            self.full = True

    def process_returns_and_advantages(self, last_value, last_state=None):  # , last_nonterminal):
        next_values = last_value.cpu().numpy()
        current_advantages = np.zeros([self.n_envs, ])
        preprocessed_rewards = self.preprocessor.preprocess_rew(self.rewards)
        for i in reversed(range(self.buffer_size)):
            current_values = self.values[i]
            #current_rewards = self.rewards[i]
            current_rewards = preprocessed_rewards[i]
            nonterminals = self.nonterminals[i]
            current_advantages = (current_rewards - current_values + self.gamma * nonterminals * (self.lambd * current_advantages + next_values))

            self.advantages[i] = current_advantages
            next_values = current_values
            # next_nonterminals = self.nonterminals[i]
        self.returns = self.advantages + self.values  # can compute in torch?
        self.last_state = last_state


    def flatten_data(self, ):
        self.observations = self.observations.swapaxes(0, 1).reshape(
            [self.total_size, *self.obs_dims])
        self.actions = self.actions.swapaxes(0, 1).reshape(
            [self.total_size, *self.act_dims])
        self.logprobs = self.logprobs.swapaxes(0, 1).reshape(
            [self.total_size])
        self.values = self.values.swapaxes(0, 1).reshape(
            [self.total_size])
        self.returns = self.returns.swapaxes(0, 1).reshape(
            [self.total_size])
        self.advantages = self.advantages.swapaxes(0, 1).reshape(
            [self.total_size])

        self.flattened_data = True

    def get_batches(self, batch_size):
        if not self.flattened_data:
            self.flatten_data()
        permuted_indices = np.random.permutation(self.total_size)
        current_idx = 0
        while current_idx < self.total_size:
            start_idx = current_idx
            current_idx += batch_size
            yield self.get_data_indices(permuted_indices[start_idx:current_idx])

    def get_random_batch(self, batch_size):
        if not self.flattened_data:
            self.flatten_data()
        indices = np.random.randint(low=0, high=self.total_size, size=batch_size)
        return self.get_data_indices(indices)

    def get_data_indices(self, indices):
        observations = self.preprocessor.preprocess_obs(self.observations[indices])
        actions = th.tensor(self.actions[indices], device=self.device).to(dtype=th.int64)
        logprobs = th.tensor(self.logprobs[indices], device=self.device)
        values = th.tensor(self.values[indices], device=self.device)
        returns = th.tensor(self.returns[indices], device=self.device)
        advantages = th.tensor(self.advantages[indices], device=self.device)
        return observations, actions, logprobs, values, returns, advantages

    def get_average_reward(self, ):
        return np.mean(self.rewards)

