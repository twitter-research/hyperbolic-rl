# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import torch as th

import hydra

import logger


def log_dict(dictionary):
    for k, v in dictionary.items():
        logger.log_key_val(k, v)


def make_models(cfg):
    if th.cuda.is_available() and not cfg.disable_cuda:
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'

    print('Running experiment on {}'.format(cfg.device))

    env = hydra.utils.instantiate(cfg.env)
    cfg.obs_dims = env.ob_space['rgb'].shape[::-1]
    cfg.n_actions = env.ac_space.eltype.n
    cfg.act_dims = []

    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    tester = hydra.utils.instantiate(cfg.tester, preprocessor=preprocessor, _recursive_=False)

    buffer = hydra.utils.instantiate(cfg.buffer, preprocessor=preprocessor)

    agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

    return agent, buffer, env, tester, preprocessor


def train(cfg, agent, buffer, env, tester, preprocessor):
    hydra.utils.call(cfg.training_fn, agent=agent, buffer=buffer, env=env,
                     tester=tester, preprocessor=preprocessor)
    return agent, buffer, env, tester, preprocessor

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    agent, buffer, env, tester, preprocessor = make_models(cfg)
    hydra.utils.call(cfg.training_fn, agent=agent, buffer=buffer, env=env,
                     tester=tester, preprocessor=preprocessor) #run_training(agent, buffer, env, tester, preprocessor, cfg)
    return agent, buffer, env, tester, preprocessor


if __name__ == '__main__':
    main()
