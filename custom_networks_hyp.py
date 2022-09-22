# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

import utils_hyp
from utils import to_numpy
from custom_networks import ImpalaResidualStack
from utils_hyp import PoincarePlaneDistance, ClipNorm, TemperatureScaling, weight_init_hyp, final_weight_init_hyp


def get_cont_mean_norm(input):
    input_shape = input.size()
    rs_input = input.view(input_shape[0], -1)
    return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

def apply_sn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m


def apply_sn_only_conv(m):
    if isinstance(m, nn.Conv2d):
        return spectral_norm(m)
    else:
        return m


def apply_sn_all_convs(modules):
    for module in modules:
        module.apply(apply_sn_only_conv)

def register_grad_hook(metric, layer, layer_name):
    grad_input_norm_name = 'grad_output_norm_{}'.format(layer_name)
    grad_weight_norm_name = 'grad_weight_norm_{}'.format(layer_name)
    metric.add(grad_input_norm_name)
    dict = {grad_input_norm_name: 1}
    metric.update(**dict)
    def output_gradient_hook(model, input_grad, output_grad):
        grad_input_norm = get_cont_mean_norm(output_grad[0])
        dict = {grad_input_norm_name: to_numpy(grad_input_norm)}
        metric.update(**dict)
    layer.register_backward_hook(output_gradient_hook)
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        metric.add(grad_weight_norm_name)
        dict = {grad_weight_norm_name: 1}
        metric.update(**dict)
        def kernel_gradient_hook(grad):
            grad_norm = get_cont_mean_norm(grad)
            dict = {grad_weight_norm_name: to_numpy(grad_norm)}
            metric.update(**dict)
            return grad
        layer.weight.register_hook(kernel_gradient_hook)
def register_all_layers_grads(modules, metric):
    count_conv = 0
    count_lin = 0
    count_stack = 0
    for m in modules:
        if isinstance(m, nn.Conv2d):
            name = 'conv{}'.format(count_conv)
            register_grad_hook(metric=metric, layer=m, layer_name=name)
            count_conv += 1
        elif isinstance(m, nn.Linear):
            name = 'linear{}'.format(count_lin)
            register_grad_hook(metric=metric, layer=m, layer_name=name)
            count_lin += 1
        elif isinstance(m, ImpalaResidualStack):
            name = 'stack{}'.format(count_stack)
            register_grad_hook(metric=metric, layer=m, layer_name=name)
            count_stack += 1
        elif isinstance(m, nn.Flatten):
            name = 'flatten'
            register_grad_hook(metric=metric, layer=m, layer_name=name)
        else:
            pass

def apply_sn_until_instance(modules, layer_instance):
    reached_instance = False
    application_modules = []
    for module in modules:
        if isinstance(module, layer_instance):
            reached_instance = True
        elif not reached_instance:
            application_modules.append(module)
    for module in application_modules:
        module.apply(apply_sn)


def get_nonlinear_layer(layer_name):
    if layer_name == 'tanh':
        return nn.Tanh()
    elif layer_name == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError


def make_impala_modules_hyp(obs_dims, n_actions, max_euclidean_norm,
                            channels=[16, 32, 32],
                            hidden_units=256,
                            shared_conv_trunk=True,
                            shared_fc_head=True,
                            hyperbolic_critic=True,
                            init=weight_init_hyp,
                            final_init=final_weight_init_hyp,
                            pre_hyp_final_init=False,
                            hyperbolic_layer_index=-1,
                            pre_hyperbolic_relu=True,
                            post_hyperbolic_relu=True,
                            temperature_scaling=False,
                            pre_hyperbolic_sn=False,
                            dimensions_per_space=None,
                            critic_hidden_units=None,
                            magnitude_warmup=None,
                            auxiliary_dims=None,
                            **hyperbolic_layer_kwargs):
    if critic_hidden_units:
        if critic_hidden_units != hidden_units:
            assert shared_fc_head == False
    else:
        critic_hidden_units = hidden_units
    if isinstance(final_init, str):
        if final_init == 'normal':
            final_init = utils_hyp.final_weight_init_hyp
        elif final_init == 'small':
            final_init = utils_hyp.final_weight_init_hyp_small
        else:
            raise NotImplementedError


    in_channels, w, h = obs_dims[-3:]
    number_of_stacks = len(channels)
    flattened_dims = (w * h) // (4 ** number_of_stacks) * channels[-1]
    if not post_hyperbolic_relu:
        raise NotImplementedError
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    if isinstance(critic_hidden_units, int):
        critic_hidden_units = [critic_hidden_units]
    if temperature_scaling:
        actor_output = n_actions + 1
    else:
        actor_output = n_actions
    if auxiliary_dims:
        assert isinstance(auxiliary_dims, int)
        assert auxiliary_dims > 0
        actor_output = actor_output + auxiliary_dims
    num_linear_layers = len(hidden_units) + 1
    hyperbolic_layer_index = hyperbolic_layer_index % num_linear_layers
    shared_modules = []
    actor_modules = []
    critic_modules = []
    if shared_conv_trunk:
        for stack_channels in channels:
            shared_modules.append(ImpalaResidualStack(in_channels,
                                                      stack_channels))
            in_channels = stack_channels
        shared_modules += [nn.Flatten(), ]
        for m in shared_modules:
            init(m)
    else:
        assert not shared_fc_head
        for stack_channels in channels:
            actor_modules.append(ImpalaResidualStack(in_channels,
                                                     stack_channels))
            critic_modules.append(ImpalaResidualStack(in_channels,
                                                      stack_channels))
            in_channels = stack_channels
        actor_modules += [nn.Flatten(), ]
        critic_modules += [nn.Flatten(), ]
    projection = ClipNorm(max_norm=max_euclidean_norm,
                          dimensions_per_space=dimensions_per_space)
    if shared_fc_head:
        assert shared_conv_trunk
        assert hyperbolic_critic
        in_features = flattened_dims
        units_list = hidden_units + [actor_output + 1]
        for i, units in enumerate(units_list):
            if i == hyperbolic_layer_index:
                distance_layer = PoincarePlaneDistance(in_features=in_features,
                                                       num_planes=units,
                                                       dimensions_per_space=dimensions_per_space,
                                                       **hyperbolic_layer_kwargs)
                if pre_hyperbolic_relu:
                    if isinstance(pre_hyperbolic_relu, str):
                        shared_modules.append(get_nonlinear_layer(pre_hyperbolic_relu))
                    else:
                        shared_modules.append(nn.ReLU())
                shared_modules += [projection, distance_layer]
                if magnitude_warmup is not None:
                    shared_modules.insert(-1, magnitude_warmup)
            else:
                shared_modules += [nn.ReLU(),
                                   nn.Linear(in_features=in_features,
                                             out_features=units), ]
            in_features = units
        last_linear = None
        reached_hyp = False
        for m in shared_modules[:-1]:
            if isinstance(m, nn.Linear):
                last_linear = m
            elif isinstance(m, PoincarePlaneDistance):
                reached_hyp = False
            init(m)
        if not reached_hyp:
            assert isinstance(shared_modules[-1], PoincarePlaneDistance)
        final_init(shared_modules[-1])
        if pre_hyp_final_init:
            assert last_linear is not None
            final_init(last_linear)
        if temperature_scaling:
            shared_modules.append(TemperatureScaling(logits_number=n_actions))
    else:
        in_features = flattened_dims
        critic_in_features = flattened_dims
        units_list = hidden_units + [actor_output]
        critic_units_list = critic_hidden_units + [1]
        for i, units in enumerate(units_list):
            if i == hyperbolic_layer_index:
                actor_distance_layer = PoincarePlaneDistance(in_features=in_features,
                                                             num_planes=units,
                                                             dimensions_per_space=dimensions_per_space,
                                                             **hyperbolic_layer_kwargs)
                if pre_hyperbolic_relu:
                    if isinstance(pre_hyperbolic_relu, str):
                        actor_modules.append(get_nonlinear_layer(pre_hyperbolic_relu))
                        critic_modules.append(get_nonlinear_layer(pre_hyperbolic_relu))
                    else:
                        actor_modules.append(nn.ReLU())
                        critic_modules.append(nn.ReLU())
                actor_modules += [projection, actor_distance_layer]
                if magnitude_warmup is not None:
                    actor_modules.insert(-1, magnitude_warmup)
                if hyperbolic_critic:
                    critic_distance_layer = PoincarePlaneDistance(in_features=critic_in_features,
                                                                  num_planes=critic_units_list[i],
                                                                  dimensions_per_space=dimensions_per_space,
                                                                  **hyperbolic_layer_kwargs)
                    critic_modules += [projection, critic_distance_layer]
                    if magnitude_warmup is not None:
                        critic_modules.insert(-1, magnitude_warmup)
                else:
                    critic_modules += [nn.ReLU(), nn.Linear(in_features=critic_in_features, out_features=critic_units_list[i])]
            else:
                actor_modules += [nn.ReLU(),
                                  nn.Linear(in_features=in_features,
                                            out_features=units), ]
                critic_modules += [nn.ReLU(),
                                   nn.Linear(in_features=critic_in_features,
                                             out_features=critic_units_list[i])]
            critic_in_features = critic_units_list[i]
            in_features = units
        for m in shared_modules:
            init(m)
        for m in actor_modules[:-1]:
            init(m)
        for m in critic_modules[:-1]:
            init(m)
        final_init(actor_modules[-1])
        final_init(critic_modules[-1])
        if temperature_scaling:
            actor_modules.append(TemperatureScaling(logits_number=n_actions))

    if isinstance(pre_hyperbolic_sn, str):
        if pre_hyperbolic_sn == 'conv':
            apply_sn_all_convs(shared_modules)
            apply_sn_all_convs(actor_modules)
            if hyperbolic_critic:
                apply_sn_all_convs(critic_modules)
        else:
            raise NotImplementedError
    elif pre_hyperbolic_sn:
        apply_sn_until_instance(shared_modules, PoincarePlaneDistance)
        apply_sn_until_instance(actor_modules, PoincarePlaneDistance)
        if hyperbolic_critic:
            apply_sn_until_instance(critic_modules, PoincarePlaneDistance)
    modules = {}
    modules['shared_modules'] = shared_modules
    modules['actor_modules'] = actor_modules
    modules['critic_modules'] = critic_modules
    modules['auxiliary_dims'] = auxiliary_dims
    return modules
