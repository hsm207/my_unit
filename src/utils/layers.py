"""
This module contains layers that are used repeatedly in the models
"""
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU


class LeakyReLUBNNSConv2d:
    """
    This layer performs a 2D convolution, batch normalization followed by leaky relu activation.

    Note:

        1. The convolution has no activation function.
        2. The batch norm layer has no scale parameter

    :param n_filters: Number of filters for the convolution layer
    :param kernel_size: The kernel size for the convolution layer
    :param stride: The stride size for the convolution layer
    :param padding: The padding type for the convolution layer
    :param data_format: The data_format of the input tensor to be passed to this layer
    """

    def __init__(self, n_filters, kernel_size, stride, padding='valid', data_format='channels_first'):
        bn_axis = 3 if data_format == 'channels_last' else 1
        self.conv = Conv2D(filters=n_filters,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding=padding,
                           activation='linear',
                           kernel_initializer=RandomNormal(mean=0, stddev=0.02),
                           data_format=data_format)
        # TODO: Figure out why can leave out the scale parameter in batch norm if the following layer is a relu
        self.batch_norm = BatchNormalization(axis=bn_axis, scale=False, center=True)
        self.leaky_relu = LeakyReLU(alpha=1e-2)

    def __call__(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)

        return x
