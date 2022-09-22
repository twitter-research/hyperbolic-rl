# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


class AverageMetrics():
    '''Object keeping running average of relevant metrics to log. '''

    def __init__(self, *args):
        self.metrics = {arg: 0 for arg in args}
        self.latest_metrics = {arg: 0 for arg in args}
        self.samples = {arg: 1e-8 for arg in args}
        self.logged_metrics = [arg for arg in args]

    def reset(self, ):
        for arg in self.metrics:
            self.metrics[arg] = 0
            self.samples[arg] = 1e-8

    def add(self, *args):
        for arg in args:
            if arg not in self.metrics:
                self.logged_metrics.append(arg)
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8
            self.metrics[arg] += val
            self.samples[arg] += 1

    def set(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = val
                self.samples[arg] = 1
            self.metrics[arg] = val
            self.samples[arg] = 1

    def get(self, ):
        for arg, metric_agg in self.metrics.items():
            samples = self.samples[arg]
            if samples >= 1:
                self.latest_metrics[arg] = metric_agg/samples
        return self.latest_metrics
