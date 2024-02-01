# Copyright (c) 2023, Florian Heinrichs
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Built-in imports
from string import ascii_lowercase
from typing import Callable, Optional, Union

# External imports
import tensorflow as tf

# Internal imports
from basis import define_basis, calculate_linear_combination


class FunctionalDense(tf.keras.layers.Layer):
    """
    Implementation of functional dense layer, according to:
    Heinrichs, F., Heim, M. & Weber, C. (2023). Functional Neural Networks:
    Shift invariant models for functional data with applications to EEG
    classification. Proceedings of the 40th International Conference on Machine
    Learning, 12866-12881.
    URL: https://proceedings.mlr.press/v202/heinrichs23a.html
    """

    def __init__(
            self,
            n_neurons: int,
            basis_options: dict,
            pooling: bool = False,
            activation: Optional[Union[str, Callable]] = None,
            calculate_weights: Callable = calculate_linear_combination,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.basis_options = basis_options
        self.pooling = pooling
        self.activation = tf.keras.activations.get(activation)
        self.calculate_weights = calculate_weights

        n_basis_functions = self.basis_options.get('n_functions', 1)
        resolution = self.basis_options.get('resolution', 1)
        basis_type = self.basis_options.get('basis_type', 'Fourier')

        # basis.shape = resolution.shape + (n_functions,)
        self.basis = define_basis(basis_type, n_basis_functions, resolution)

        self.scalar_weights = None
        self.call_equation = None

    def build(self, input_shape: tuple):
        resolution = self.basis.shape[:-1]

        if input_shape[1:-1] != resolution:
            raise TypeError(f"Shapes of input and basis not compatible: "
                            f"{input_shape[1:-1]=}, {resolution=}")

        n_functions = self.basis.shape[-1]
        n_channels = input_shape[-1]

        self.scalar_weights = self.add_weight(
            name='scalar_weights',
            shape=(n_functions, n_channels, self.n_neurons),
            initializer='random_normal',
            trainable=True
        )

        n_dims = len(resolution)
        indices = ascii_lowercase[:n_dims]

        self.call_equation = f"x{indices}y, {indices}yz -> x{indices}z"

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Calculates activated, weighted sum of inputs. Note that the weights
        have shape:
            - self.scalar_weights.shape = (n_functions, n_channels, n_filters)
            - self.basis.shape = resolution + (n_functions,)
            - weights.shape = resolution + (n_channels, n_filters)

        :param inputs: Inputs of shape:
            (batch_size,) + resolution + (n_channels,)
        :return: Activated, weighted sum of inputs with shape:
            - (batch_size, n_channels) if self.pooling
            - (batch_size,) + resolution + (n_channels,) if not self.pooling
        """
        weights = self.calculate_weights(self.scalar_weights, self.basis)
        outputs = tf.einsum(self.call_equation, inputs, weights)

        if self.pooling:
            axes = list(range(1, len(self.basis.shape)))
            outputs = tf.math.reduce_mean(outputs, axes)

        outputs = self.activation(outputs)

        return outputs
