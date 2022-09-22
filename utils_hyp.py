# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import division

import numpy as np
import torch
from torch import nn
import geoopt

import utils


class UpdatableModule:
    def __init__(self, ):
        pass
    def update(self, current_steps):
        raise NotImplementedError
    def setup_metric(self, metrics):
        raise NotImplementedError


class WarmupScale(nn.Module, UpdatableModule):
    def __init__(self, initial_scale, update_steps=None, fraction=None, total_training_steps=None):
        nn.Module.__init__(self)
        self.initial_scale = initial_scale
        self.current_scale = self.initial_scale
        if update_steps is None:
            assert fraction is not None and total_training_steps is not None
            self.update_steps = int(total_training_steps * fraction)
        else:
            self.update_steps = update_steps
        self.metrics = None

    def update(self, current_steps):
        self.current_scale = np.minimum(current_steps/self.update_steps, 1.0)
        if self.metrics is not None:
            self.metrics.set(warmup_scale=self.current_scale)

    def forward(self, input):
        return input * self.current_scale

    def setup_metric(self, metrics):
        metrics.add('warmup_scale')
        self.metrics = metrics



class ClipNorm(nn.Module):
    def __init__(self, max_norm=15, dimensions_per_space=None):
        super().__init__()
        self.max_norm = max_norm
        self.dimension_per_space = dimensions_per_space

    def get_mean_norm(self, input):
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def forward(self, input):  # input bs x in_feat
        if self.dimension_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimension_per_space,
                                  self.dimension_per_space)
        else:
            rs_input = input
        input_l2 = torch.norm(rs_input, p=2, dim=-1, keepdim=True)
        clipped_input = torch.minimum(self.max_norm / input_l2,
                                      torch.ones_like(input_l2)) * rs_input
        if self.dimension_per_space:
            clipped_input = clipped_input.view(*input_shape)
        return clipped_input



class TemperatureScaling(torch.nn.Module):
    def __init__(self, logits_number, logits_start_index=0):
        super().__init__()
        assert logits_start_index == 0
        self.logits_number = logits_number

    def forward(self, input):
        temp = torch.ones_like(input)
        temp[..., 1:self.logits_number + 1] = torch.exp(input[..., :1])
        output = input[..., 1:] / temp[..., 1:]
        return output


class PoincarePlaneDistance(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            num_planes: int,  # out_features
            c=1.0,
            euclidean_inputs=True,
            rescale_euclidean_norms_gain=None,  # rescale euclidean norms based on the dimensions per space
            signed=True,
            scaled=True,
            squared=False,
            project_input=True,
            normal_std=None,
            dimensions_per_space=None,
            rescale_normal_params=False,
            effective_softmax_rescale=None,
            hyperbolic_representation_metric=None,
    ):
        super().__init__()
        self.euclidean_inputs = euclidean_inputs
        self.rescale_norms_gain = rescale_euclidean_norms_gain
        self.signed = signed
        self.scaled = scaled
        self.squared = squared
        self.project_input = project_input
        self.ball = geoopt.PoincareBall(c=c)
        self.in_features = in_features
        self.num_planes = num_planes
        self.rescale_normal_params = rescale_normal_params

        if effective_softmax_rescale is not None:
            if self.rescale_normal_params:
                self.logits_multiplier = effective_softmax_rescale
            else:
                self.logits_multiplier = effective_softmax_rescale * 2
        else:
            self.logits_multiplier = 1

        if dimensions_per_space is not None:
            assert in_features % dimensions_per_space == 0
            self.dimensions_per_space = dimensions_per_space
            self.num_spaces = in_features // dimensions_per_space
        else:
            self.dimensions_per_space = self.in_features
            self.num_spaces = 1

        self.normals = nn.Parameter(torch.empty((num_planes, self.num_spaces, self.dimensions_per_space)))
        self.bias = geoopt.ManifoldParameter(torch.zeros(num_planes, self.num_spaces, self.dimensions_per_space),
                                             manifold=self.ball)

        self.normal_std = normal_std
        self.reset_parameters()

        self.hyperbolic_representation_metric = hyperbolic_representation_metric
        if self.hyperbolic_representation_metric is not None and self.euclidean_inputs:
            self.hyperbolic_representation_metric.add('hyperbolic_representations')

    def get_mean_norm(self, input):
        if self.dimensions_per_space:
            input_shape = input.size()
            input_batch_dims = input_shape[:-1]
            input_feature_dim = input_shape[-1]
            rs_input = input.view(*input_batch_dims, input_feature_dim // self.dimensions_per_space,
                                  self.dimensions_per_space)
        else:
            rs_input = input
        return torch.norm(rs_input, p=2, dim=-1, keepdim=True).mean()

    def map_to_ball(self, input):  # input bs x in_feat
        if self.rescale_norms_gain:  # make expected tangent vector norm independent of initial dimension (approximately)
            input = self.rescale_norms_gain * input / np.sqrt(self.dimensions_per_space)
        return self.ball.expmap0(input, project=self.project_input)

    def manual_distance(self, points, other_points):
        dist = torch.arccosh(1 + 2 * (points - other_points).pow(2).sum(-1) / (1 - points.pow(2).sum(-1)) / (
                    1 - other_points.pow(2).sum(-1)))
        return dist

    def distance_matrix(self, input, euclidean_inputs=True, cpu=False):
        if euclidean_inputs:
            input = self.map_to_ball(input)
        input_batch_dims = input.size()[:-1]
        input = input.view(*input_batch_dims, self.num_spaces, self.dimensions_per_space)
        if cpu:
            input = input.cpu()
        distances = self.manual_distance(input.unsqueeze(0), input.unsqueeze(1))

        return distances.sum(-1)

    def distance_to_space(self, input, other, euclidean_inputs):
        if euclidean_inputs:
            input = self.map_to_ball(input)
            other = self.map_to_ball(other)
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        other = other.view(-1, self.num_spaces, self.dimensions_per_space)
        summed_dists = self.ball.dist(x=input, y=other).sum(-1)
        return summed_dists.view(input_batch_dims)

    def forward(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        if self.rescale_normal_params:
            conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
            a = self.normals * conformal_factor.unsqueeze(-1)
        else:
            a = self.normals
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=a,
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        if self.rescale_normal_params:
            distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance * self.logits_multiplier

    def forward_rs(self, input):  # input bs x in_feat
        input_batch_dims = input.size()[:-1]
        input = input.view(-1, self.num_spaces, self.dimensions_per_space)
        if self.euclidean_inputs:
            input = self.map_to_ball(input)
            if self.hyperbolic_representation_metric is not None:
                self.hyperbolic_representation_metric.set(hyperbolic_representations=input)
        input_p = input.unsqueeze(-3)  # bs x 1 x num_spaces x dim_per_space
        conformal_factor = 1 - self.bias.pow(2).sum(dim=-1)
        distances = self.ball.dist2plane(x=input_p, p=self.bias, a=self.normals * conformal_factor.unsqueeze(-1),
                                         signed=self.signed, scaled=self.scaled, dim=-1)
        distances = distances * 2 / conformal_factor
        distance = distances.sum(-1)
        distance = distance.view(*input_batch_dims, self.num_planes)
        return distance

    def extra_repr(self):
        return (
            "poincare_dim={num_spaces}x{dimensions_per_space} ({in_features}), "
            "num_planes={num_planes}, "
            .format(**self.__dict__))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
        if self.normal_std:
            nn.init.normal_(self.normals, std=self.normal_std)
        else:
            nn.init.normal_(self.normals, std=1 / np.sqrt(self.in_features))


def weight_init_hyp(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    else:
        utils.weight_init(m)


def final_weight_init_hyp(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        utils.final_weight_init(m=m)


def final_weight_init_hyp_small(m):
    if isinstance(m, PoincarePlaneDistance):
        nn.init.normal_(m.normals.data, 1 / np.sqrt(m.in_features) * 0.01)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        utils.final_weight_init(m=m)
