# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import time
import torch
import numpy as np
from pathlib import Path

import logger
from utils import log_returns_stats
from logging_metrics import AverageMetrics

def log_dict(dictionary):
    for k, v in dictionary.items():
        logger.log_key_val(k, v)

def run_training(agent, buffer, env, tester, preprocessor,
                 total_steps,
                 steps_per_rollout,
                 n_envs,
                 train_epochs_per_iteration,
                 log_frequency,
                 batch_size,
                 save_weights_every=-1,
                 detect_anomaly=False,
                 ):
    if detect_anomaly:
        print('WARNING: running with activated anomaly detection')
        torch.autograd.set_detect_anomaly(True)
    work_dir = Path.cwd()
    print('workspace: {}'.format(work_dir))
    logger.configure_output_dir('data')
    n_rollouts = int(np.ceil(total_steps / (steps_per_rollout * n_envs)))
    print("Total number of rollouts: {}".format(n_rollouts))
    start_training_time = time.time()
    current_steps = 0
    logger.log_key_val('iter', 0)
    logger.log_key_val('frame', current_steps)
    stats = log_returns_stats(tester.evaluate(agent), log=False)
    log_dict(stats)
    metrics = agent.return_and_reset_metrics()
    log_dict(metrics)
    epoch_end_eval_time = time.time()
    evaluation_time = epoch_end_eval_time - start_training_time
    training_loop_metrics = AverageMetrics('training_time', 'evaluation_time')
    training_loop_metrics.update(training_time=0.0, evaluation_time=np.around(evaluation_time, decimals=3))
    current_training_metrics = training_loop_metrics.get()
    training_loop_metrics.reset()
    log_dict(current_training_metrics)
    logger.log_iteration()

    rew, obs, first = env.observe()  # First observation
    for r in range(n_rollouts):
        rollout_start_time = time.time()
        buffer.reset()
        for s in range(steps_per_rollout):
            # convert to channel first
            current_obs = obs['rgb'].transpose(0, 3, 1, 2)
            input_obs = preprocessor.preprocess_obs(
                current_obs.astype(np.float32))
            act, logprob, value = agent.get_action_logprob_value(input_obs)
            env.act(act.cpu().numpy())
            rew, obs, first = env.observe()
            buffer.add(observation=current_obs, action=act, reward=rew,
                       nonterminal=1 - first, logprob=logprob, value=value)
        last_input_obs_pre = obs['rgb'].transpose(0, 3, 1, 2)
        last_input_obs = preprocessor.preprocess_obs(
            last_input_obs_pre.astype(np.float32))

        buffer.process_returns_and_advantages(
            last_value=agent.get_value(last_input_obs), last_state=last_input_obs_pre)

        current_steps = steps_per_rollout * n_envs * (r + 1)

        agent.learn(buffer=buffer, epochs=train_epochs_per_iteration,
                    batch_size=batch_size, current_steps=current_steps)

        rollout_end_training_time = time.time()
        training_time = rollout_end_training_time - rollout_start_time
        training_loop_metrics.update(training_time=np.around(training_time, 3))
        if (r + 1) % log_frequency == 0:
            logger.log_key_val('iter', r + 1)
            logger.log_key_val('frame', current_steps)
            stats = log_returns_stats(tester.evaluate(agent), log=False)
            log_dict(stats)
            metrics = agent.return_and_reset_metrics()
            log_dict(metrics)
            end_eval_time = time.time()
            evaluation_time = end_eval_time - rollout_end_training_time
            training_loop_metrics.update(evaluation_time=np.around(evaluation_time, 3))
            current_training_metrics = training_loop_metrics.get()
            training_loop_metrics.reset()
            log_dict(current_training_metrics)
            logger.log_iteration()


        if save_weights_every > 0:
            if (r + 1) % save_weights_every == 0:
                checkpoint = agent.save(work_dir, step=current_steps, preprocessor=preprocessor)
                print('Saved checkpoint {}'.format(checkpoint))
