# Copyright (c) 2023, Florian Heinrichs
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Built-in imports
import random
from typing import Optional

# External imports
import tensorflow as tf

# Internal imports
from convolution import FunctionalConvolution
from dense import FunctionalDense


def setup_model(
        input_shape: tuple,
        filter_options: list,
        layer_options: list,
        loss: str = 'categorical_crossentropy',
        metrics: Optional[list] = None
) -> tf.keras.Model:
    """
    Setup model depending on the inputs shape and prediction aim. After a
    channel-wise normalization, FunctionalConvolution and FunctionalDense
    layers are added (according to the specifications).

    :param input_shape: Shape of input data (excluding batch dimension), e.g.
        - (time, n_channels) for multivariate time series
        - (space1, space2, n_channels) for images
        - (space1, space2, time, n_channels) for videos
    :param filter_options: List of dictionaries specifying the options used to
        create FunctionalConvolution layers. The number of convolutional layers
        equals the length of the list.
    :param layer_options: List of dictionaries specifying the options used to
        create FunctionalDense layers. The number of dense layers equals the
        length of the list.
    :param loss: Loss function to be used during training.
    :param metrics: List of metrics to be used during training.
    :return: Compiled model
    """
    if metrics is None:
        metrics = ['Accuracy']

    inputs = tf.keras.layers.Input(shape=input_shape)

    norm_axes = list(range(len(input_shape) - 1))
    layer = tf.keras.layers.LayerNormalization(
        axis=norm_axes,
        center=False,
        scale=False,
        epsilon=1e-10,
        name='Normalization'
    )(inputs)

    for i, filter_option in enumerate(filter_options):
        layer = FunctionalConvolution(
            **filter_option,
            name=f'FunctionalConvolution_{i}'
        )(layer)

    for i, layer_option in enumerate(layer_options):
        layer = FunctionalDense(
            **layer_option,
            name=f'FunctionalDense_{i}'
        )(layer)

    outputs = layer

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Example_FNN")
    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    return model


def load_data(
        filepath: str,
        ratio: float = 0.8,
        n_labels: int = 1,
        batch_size: int = 32
) -> tuple:
    """
    Load data as tuple of TensorFlow datasets (split into training and test
    data).

    :param filepath: Path to csv file containing the data. The file should
        contain samples as rows and features as columns, where the last column
        corresponds to the label.
    :param ratio: Ratio of train and test data.
    :param n_labels: Number of label columns.
    :param batch_size: Size of batches.
    :return: TensorFlow dataset yielding labeled data of shape (24,), ().
    """
    data = tf.io.read_file(filepath)
    data = tf.strings.split(data, sep='\r\n')[1:-1]
    data = tf.strings.split(data, sep=',').to_tensor()
    data = tf.strings.to_number(data)

    data, label = data[:, :-n_labels], data[:, -n_labels:]
    n_samples = data.shape[0]
    indices = list(range(n_samples))
    break_point = int(ratio * n_samples)

    random.shuffle(indices)

    datasets = []
    for idx in [indices[:break_point], indices[break_point:]]:
        data_idx = tf.gather(data, idx)
        label_idx = tf.gather(label, idx)

        ds = tf.data.Dataset.from_tensor_slices((data_idx, label_idx))
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
        datasets.append(ds)

    train_data, test_data = datasets
    train_data = train_data.repeat(-1)

    return train_data, test_data


def tecator_example():
    """
    Example based on the Tecator dataset:
        http://lib.stat.cmu.edu/datasets/tecator
    """
    filepath = "data/tecator_fat.csv"
    train_data, test_data = load_data(filepath, n_labels=1)

    input_shape = (100, 1)
    filter_options = [
        {'n_filters': 20,
         'basis_options': {'n_functions': 7,
                           'resolution': 25,
                           'basis_type': 'Legendre'},
         'activation': 'elu'},
        {'n_filters': 5,
         'basis_options': {'n_functions': 7,
                           'resolution': 25,
                           'basis_type': 'Legendre'},
         'activation': 'elu'}
    ]
    layer_options = [{
        'n_neurons': 1,
        'basis_options': {'n_functions': 1,
                          'resolution': 52,
                          'basis_type': 'Fourier'},
        'activation': 'elu',
        'pooling': True
    }]
    loss = 'mse'
    metrics = ['mae']

    model = setup_model(
        input_shape,
        filter_options,
        layer_options,
        loss=loss,
        metrics=metrics
    )
    model.summary()

    model.fit(
        train_data,
        epochs=10,
        steps_per_epoch=1000,
        validation_data=test_data,
        verbose=1
    )


def phoneme_example():
    """
    Example based on the Phoneme dataset:
        https://web.stanford.edu/~hastie/ElemStatLearn/
    """
    filepath = "data/phoneme_one_hot.csv"
    train_data, test_data = load_data(filepath, n_labels=5)

    input_shape = (256, 1)
    filter_options = [
        {'n_filters': 20,
         'basis_options': {'n_functions': 8,
                           'resolution': 32,
                           'basis_type': 'Legendre'},
         'activation': 'elu'},
        {'n_filters': 5,
         'basis_options': {'n_functions': 12,
                           'resolution': 64,
                           'basis_type': 'Legendre'},
         'activation': 'elu'}
    ]
    layer_options = [{
        'n_neurons': 5,
        'basis_options': {'n_functions': 4,
                          'resolution': 162,
                          'basis_type': 'Fourier'},
        'activation': 'softmax',
        'pooling': True
    }]

    model = setup_model(input_shape, filter_options, layer_options)
    model.summary()

    model.fit(
        train_data,
        epochs=5,
        steps_per_epoch=500,
        validation_data=test_data,
        verbose=1
    )


def nox_example():
    """
    Example based on the NOx dataset:
        https://fda.readthedocs.io/en/stable/modules/autosummary/skfda.datasets.fetch_nox.html
    """
    filepath = "data/nox_one_hot.csv"
    train_data, test_data = load_data(filepath, n_labels=2)

    input_shape = (24, 1)
    filter_options = []
    layer_options = [{
            'n_neurons': 5,
            'basis_options': {'n_functions': 6,
                              'resolution': 24,
                              'basis_type': 'Legendre'},
            'activation': 'softmax',
            'pooling': False
        }, {
        'n_neurons': 2,
        'basis_options': {'n_functions': 3,
                          'resolution': 24,
                          'basis_type': 'Fourier'},
        'activation': 'softmax',
        'pooling': True
    }]

    model = setup_model(input_shape, filter_options, layer_options)
    model.summary()

    model.fit(
        train_data,
        epochs=10,
        steps_per_epoch=500,
        validation_data=test_data,
        verbose=1
    )


if __name__ == '__main__':
    tecator_example()
    phoneme_example()
    nox_example()
