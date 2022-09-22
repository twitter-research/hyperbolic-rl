# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import hydra
import gym3

class Tester():
    def __init__(self, make_eval_env, preprocessor, max_timesteps=1e6):
        self.make_eval_env = make_eval_env
        self.preprocessor = preprocessor
        self.max_timesteps = max_timesteps

    def evaluate(self, ppo, min_traj):
        self.env = self.make_eval_env()
        rew, obs, first = self.env.observe()
        last_returns = 0
        collected_returns = []
        i = 0
        while len(collected_returns) < min_traj and i < self.max_timesteps:
            obs = self.preprocessor.preprocess_obs(
                (obs['rgb'].transpose(0, 3, 1, 2)).astype(np.float32))
            act = ppo.act(obs, det=True)
            self.env.act(act)
            rew, obs, first = self.env.observe()
            last_returns = rew + last_returns
            collected_returns += last_returns[np.where(first)].tolist()
            last_returns = last_returns * (1 - first)
            i += 1
        return collected_returns

class ModularTester():
    def __init__(self, eval_env_cfg, preprocessor, max_eval_timesteps=1e6,
                 min_eval_episodes=5, train_env_cfg=None):
        self.env_cfg = eval_env_cfg
        self.preprocessor = preprocessor
        self.max_timesteps = max_eval_timesteps
        self.min_episodes = min_eval_episodes
        self.train_env_cfg = train_env_cfg
        if train_env_cfg is not None:
            self.train_env_cfg['num'] = self.env_cfg['num']

    def evaluate_from_config(self, cfg, agent, **kwargs):
        collected_returns = []
        while len(collected_returns) < self.min_episodes:
            self.env = hydra.utils.instantiate(cfg)
            rew, obs, first = self.env.observe()
            last_returns = 0
            collected_returns = []
            notdones = np.ones([cfg.num])
            i = 0
            while np.any(notdones) and i < self.max_timesteps:
                obs = self.preprocessor.preprocess_obs(
                    (obs['rgb'].transpose(0, 3, 1, 2)).astype(np.float32))
                act = agent.act(obs, **kwargs)
                self.env.act(act)
                rew, obs, first = self.env.observe()
                last_returns = rew*notdones + last_returns
                notdones = notdones * (1 - first)
                i += 1
            collected_returns += last_returns[np.where((1-notdones))].tolist()
        return collected_returns

    def visualize_from_config(self, cfg, agent, num_episodes=1, **kwargs):
        collected_returns = []
        while len(collected_returns) < num_episodes:
            env = hydra.utils.instantiate(cfg, num=1, render_mode="rgb_array")
            env = gym3.ViewerWrapper(env, info_key='rgb')
            rew, obs, first = env.observe()
            last_returns = 0
            collected_returns = []
            notdones = np.ones([1])
            i = 0
            while np.any(notdones) and i < self.max_timesteps:
                obs = self.preprocessor.preprocess_obs(
                    (obs['rgb'].transpose(0, 3, 1, 2)).astype(np.float32))
                act = agent.act(obs, **kwargs)
                env.act(act)
                rew, obs, first = env.observe()
                last_returns = rew*notdones + last_returns
                notdones = notdones * (1 - first)
                i += 1
            collected_returns += last_returns[np.where((1-notdones))].tolist()
        if num_episodes > 0:
            renderer = env._renderer
            renderer._glfw.destroy_window(renderer._window)
        return collected_returns




    def visualize(self, agent, num_episodes=1, test=True, **kwargs):
        if not test:
            assert self.train_env_cfg is not None
            returns = self.visualize_from_config(self.train_env_cfg, agent, num_episodes=num_episodes, **kwargs)
        else:
            returns = self.visualize_from_config(self.env_cfg, agent, num_episodes=num_episodes, **kwargs)
        return returns

    def run_record_metrics(self, cfg, agent, num_episodes=1, metric=None, visualize=False, **kwargs):
        collected_returns = []
        collected_observation = []
        collected_actions = []
        collected_first = []
        collected_rew = []
        if metric is not None:
            collected_metrics = {m: [] for m in metric.logged_metrics}
        else:
            collected_metrics = {}
        collected_returns = []
        while len(collected_returns) < num_episodes:
            env = hydra.utils.instantiate(cfg, num=1, render_mode="rgb_array")
            if visualize:
                env = gym3.ViewerWrapper(env, info_key='rgb')
            rew, obs, first = env.observe()
            last_returns = 0
            notdones = np.ones([1])
            i = 0
            while np.any(notdones) and i < self.max_timesteps:
                collected_observation.append(obs)
                collected_first.append(first)
                collected_rew.append(rew)
                obs = self.preprocessor.preprocess_obs(
                    (obs['rgb'].transpose(0, 3, 1, 2)).astype(np.float32))
                act = agent.act(obs, **kwargs)
                collected_actions.append(act)
                if metric is not None:
                   for cm, cv in metric.get().items():
                       collected_metrics[cm].append(cv)
                env.act(act)
                rew, obs, first = env.observe()
                last_returns = rew*notdones + last_returns
                notdones = notdones * (1 - first)
                i += 1
            collected_returns += last_returns[np.where((1-notdones))].tolist()
        if num_episodes > 0 and visualize:
            renderer = env._renderer
            renderer._glfw.destroy_window(renderer._window)
        return_dict = {'returns': collected_returns,
                       'observations': collected_observation,
                       'actions': collected_actions,
                       'firsts': collected_first,
                       'rewards': collected_rew}
        return_dict.update(collected_metrics)
        return return_dict

    def collect_transitions(self, agent, num_transitions=6400, num_envs=64, test=True, **kwargs):
        collected_returns = []
        collected_observation = []
        collected_actions = []
        collected_first = []
        collected_rew = []
        num_steps = int(np.ceil(num_transitions/num_envs))
        if test:
            cfg = self.env_cfg
        else:
            cfg = self.train_env_cfg
        env = hydra.utils.instantiate(cfg, num=num_envs, render_mode="rgb_array")
        rew, obs, first = env.observe()
        last_returns = np.zeros([num_envs])
        for _ in range(num_steps):
            collected_observation.append(obs)
            collected_first.append(first)
            collected_rew.append(rew)
            obs = self.preprocessor.preprocess_obs(
                (obs['rgb'].transpose(0, 3, 1, 2)).astype(np.float32))
            act = agent.act(obs, **kwargs)
            collected_actions.append(act)

            env.act(act)
            rew, obs, first = env.observe()
            last_returns = rew + last_returns
            collected_returns += last_returns[np.where(first)].tolist()
            last_returns = last_returns * (1-first)

        def make_flattened_array(arr):
            if isinstance(arr, list):
                arr = np.array(arr)
            shape = arr.shape
            return arr.reshape(num_transitions, *shape[2:])

        observations = make_flattened_array([o['rgb'] for o in collected_observation])
        actions = make_flattened_array(collected_actions)
        firsts = make_flattened_array(collected_first)
        rewards = make_flattened_array(collected_rew)
        return_dict = {'returns': collected_returns,
                       'observations': observations,
                       'actions': actions,
                       'firsts': firsts,
                       'rewards': rewards}
        return return_dict

    def record_metrics(self, agent, num_episodes=1, metric=None, visualize=False, test=True, **kwargs):
        if not test:
            assert self.train_env_cfg is not None
            stats = self.run_record_metrics(
                self.train_env_cfg, agent, num_episodes=num_episodes,
                metric=metric, visualize=visualize, **kwargs)
        else:
            stats = self.run_record_metrics(
                self.train_env_cfg, agent, num_episodes=num_episodes,
                metric=metric, visualize=visualize, **kwargs)
        return stats

    def evaluate(self, agent, **kwargs):
        test_returns = self.evaluate_from_config(self.env_cfg, agent, **kwargs)
        if self.train_env_cfg is not None:
            train_returns = self.evaluate_from_config(self.train_env_cfg, agent, **kwargs)
            return {'': test_returns, '_train': train_returns}
        return test_returns