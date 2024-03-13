# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import torch.nn as nn
from typing import List, Callable, Union


class FCNet(to.nn.Module):
    def __init__(
        self,
        shape: List[int],
        activations: Union[Callable, List[Callable]],
        dropouts: List[bool] = None,
        dropout_rate: float = None,
    ):
        """
        Adjustable fully connected network class.

        :param shape: Network shape, (H0-H1-...-D)
        :param activations: List of layer-specific activations
        :param dropouts: List of dropout booleans.
        :param dropout_rate: global dropout rate
        """
        super().__init__()

        # sanity
        if dropouts is None:
            dropouts = [False for _ in range(len(shape) - 1)]
        else:
            assert dropout_rate is not None

        activations = (
            [activations for _ in range(len(shape) - 1)]
            if isinstance(activations, Callable)
            else activations
        )

        self.shape, self.H0, self.D = shape, shape[0], shape[-1]

        # build fully connected blocks
        self.net = nn.Sequential()
        for i in range(len(shape) - 1):
            self.net.add_module(
                "linear_{}".format(i),
                nn.Linear(in_features=shape[i], out_features=shape[i + 1]),
            )
            dropout = dropouts[i]
            if dropout:
                self.net.add_module(
                    "dropout_layer{}".format(i), nn.Dropout(dropout_rate)
                )
            activation = activations[i]
            self.net.add_module("activation_{}".format(i), activation())

    def forward(self, x):
        return self.net(x).to(dtype=x.dtype)
