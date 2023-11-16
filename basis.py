# Copyright (c) 2023, Florian Heinrichs
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Built-in imports
from math import pi
from string import ascii_lowercase
from typing import Union

# External imports
import numpy as np
from scipy.special import legendre
import tensorflow as tf


def define_basis(
        basis_type: str,
        n_functions: int,
        resolution: Union[int, tuple]
) -> tf.Tensor:
    """
    Define basis functions - called when functional layer is initialized. The
    basis functions are defined at discrete time steps. The number of time
    steps is specified by the resolution.
    Note: The resolution might be given as int if the basis functions depend on
    a single variable. If the basis functions depend on multiple variables, a
    tuple is required.

    :param basis_type: Type of basis, given as string. Currently available:
        - 'Fourier'
        - 'Legendre'
    :param n_functions: Number of basis functions.
    :param resolution: Resolution of the basis - the basis functions are
        defined at equidistant points.
    :return: Basis as tf.Tensor with shape (resolution, n_functions).
    """
    if isinstance(resolution, int):
        resolution = (resolution,)

    if basis_type == 'Legendre' and len(resolution) == 1:
        support = np.linspace(-1, 1, resolution[0])
        basis = [legendre(i)(support) for i in range(n_functions)]

    elif basis_type == 'Fourier' and len(resolution) == 1:
        support = tf.linspace(0, 1, resolution[0])
        factor = tf.cast(tf.sqrt(2.0), dtype=tf.float64)
        basis = [factor * tf.math.sin(pi * (i + 1) * support) if i % 2 == 1
                 else factor * tf.math.cos(pi * i * support) if i > 0
                 else tf.ones_like(support)
                 for i in range(n_functions)]

    else:
        raise NotImplementedError(f"Basis type {basis_type} not implemented.")

    basis = tf.stack(basis, axis=-1)
    basis = tf.cast(basis, dtype=tf.float32)

    return basis


def calculate_linear_combination(
        scalar_weights: tf.Tensor,
        basis: tf.Tensor
) -> tf.Tensor:
    """
    Calculate the linear combination of scalar weights with basis functions.

    :param scalar_weights: Scalar weights with shape:
        (n_basis_functions, n_channels, n_filters)
    :param basis: Basis functions with shape:
        resolution + (n_basis_functions,)
    :return: Linear combination of weights with shape:
        resolution + (n_channels, n_filters)
    """
    n_dims = len(basis.shape) - 1
    dim_indices = ascii_lowercase[:n_dims]
    equation = f'xyz, {dim_indices}x -> {dim_indices}yz'

    weights = tf.einsum(equation, scalar_weights, basis)

    return weights
