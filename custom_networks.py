# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


from torch import nn
import torch
import torch as th
from utils import weight_init, final_weight_init
from torch.nn.utils.parametrizations import spectral_norm
from utils import to_numpy

def apply_sn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m

def apply_sn_all(modules):
    for module in modules:
        module.apply(apply_sn)


def get_cont_mean_norm(input):
    input_shape = input.size()
    rs_input = input.view(input_shape[0], -1)
    return th.norm(rs_input, p=2, dim=-1, keepdim=True).mean()


class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual_block = nn.Sequential(nn.ReLU(),
                                            nn.Conv2d(in_channels=channels,
                                                      out_channels=channels,
                                                      kernel_size=3,
                                                      padding=1, ),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=channels,
                                                      out_channels=channels,
                                                      kernel_size=3,
                                                      padding=1, ))

    def forward(self, inputs):
        return self.residual_block(inputs) + inputs


class ImpalaResidualStack(nn.Module):
    def __init__(self, in_channels, stack_channels):
        super().__init__()
        self.stack = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                             out_channels=stack_channels,
                                             kernel_size=3,
                                             padding=1, ),
                                   nn.MaxPool2d(kernel_size=3, stride=2,
                                                padding=1),
                                   ImpalaResidualBlock(channels=stack_channels),
                                   ImpalaResidualBlock(channels=stack_channels))

    def forward(self, inputs):
        return self.stack(inputs)


def make_impala_modules(obs_dims,
                        n_actions,
                        channels=[16, 32, 32],
                        hidden_units=256, shared_conv_trunk=True,
                        shared_fc_head=True,
                        init=weight_init,
                        final_init=final_weight_init,
                        pre_final_sn=False,
                        auxiliary_dims=None,
                        ):
    in_channels, w, h = obs_dims[-3:]
    number_of_stacks = len(channels)
    flattened_dims = (w * h) // (4 ** number_of_stacks) * channels[-1]
    if auxiliary_dims:
        assert isinstance(auxiliary_dims, int)
        assert auxiliary_dims > 0
        actor_output_dims = n_actions + auxiliary_dims
    else:
        actor_output_dims = n_actions
    shared_modules = []
    actor_modules = []
    critic_modules = []
    if shared_conv_trunk:
        for stack_channels in channels:
            shared_modules.append(ImpalaResidualStack(in_channels,
                                                      stack_channels))
            in_channels = stack_channels
        shared_modules += [nn.Flatten(),
                           nn.ReLU(),]
    else:
        assert not shared_fc_head
        for stack_channels in channels:
            actor_modules.append(ImpalaResidualStack(in_channels,
                                                     stack_channels))
            critic_modules.append(ImpalaResidualStack(in_channels,
                                                      stack_channels))
            in_channels = stack_channels
        actor_modules += [nn.Flatten(),
                           nn.ReLU(),]
        critic_modules += [nn.Flatten(),
                           nn.ReLU(), ]

    if shared_fc_head:
        assert shared_conv_trunk
        shared_modules += [nn.Linear(in_features=flattened_dims,
                                     out_features=hidden_units),
                           nn.ReLU(),
                           nn.Linear(in_features=hidden_units,
                                     out_features=actor_output_dims + 1)]
        if pre_final_sn:
            apply_sn_all(shared_modules[:-1])
        for m in shared_modules[:-1]:
            init(m)
        final_init(shared_modules[-1])

    else:
        actor_modules += [nn.Linear(in_features=flattened_dims,
                                    out_features=hidden_units),
                          nn.ReLU(),
                          nn.Linear(in_features=hidden_units,
                                    out_features=actor_output_dims)]

        critic_modules += [nn.Linear(in_features=flattened_dims,
                                     out_features=hidden_units),
                           nn.ReLU(),
                           nn.Linear(in_features=hidden_units,
                                     out_features=1)]
        if pre_final_sn:
            apply_sn_all(shared_modules)
            apply_sn_all(critic_modules[:-1])
            apply_sn_all(actor_modules[:-1])
        for m in shared_modules:
            init(m)
        for m in actor_modules[:-1]:
            init(m)
        for m in critic_modules[:-1]:
            init(m)
        final_init(actor_modules[-1])
        final_init(critic_modules[-1])
    modules = {}
    modules['shared_modules'] = shared_modules
    modules['actor_modules'] = actor_modules
    modules['critic_modules'] = critic_modules
    modules['auxiliary_dims'] = auxiliary_dims
    return modules
