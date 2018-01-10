"""
This module contains layers that are used repeatedly in the models
"""
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation


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


class GaussianVAE2D:
    """
    This layer uses convolutional layers to generate the mean and standard deviation for each of the component of
    the given input tensor.

    The mean and standard deviation for each component is assumed to be independent. The forward pass returns a
    tuple of the form:

        (independent random samples from a normal distribution using the generated mean and sd from the input tensor
        aka the latent variables of the input tensors,
        the respective means generated from the input tensor,
        the respective standard deviations generated from the input tensor)


    :param n_filters: The number of filters to use in the convolutional layer
    :param kernel_size: The kernel size to use in the convolutional layer
    :param stride: The stride to use in the convolutional layer
    :param padding: The padding to use in the convolutional layer
    :param data_format: The data format of the input tensor to the convolutional layer
    """

    def __init__(self, n_filters, kernel_size, stride, padding='valid', data_format='channels_first'):
        param_initialzer = RandomNormal(mean=0, stddev=0.002)
        self.mean = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, padding=padding,
                           activation='linear',
                           data_format=data_format, kernel_initializer=param_initialzer,
                           bias_initializer=param_initialzer,
                           name='conv2d_mean')

        self.standard_deviation = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, padding=padding,
                                         activation='linear',
                                         data_format=data_format, kernel_initializer=param_initialzer,
                                         bias_initializer=param_initialzer,
                                         name='conv2d_sd')

        # We need to pass standard_deviation to soft plus activation layer because standard deviation is > 0
        self.soft_plus = Activation('softplus', name='softplus_sd')

    def __call__(self, x):
        # TODO: Check if forward pass is the same thing as generating a sample and returning the mean and sd
        # mu = self.mean(x)
        #
        # sd = self.standard_deviation(x)
        # sd = self.soft_plus(sd)
        #
        # return mu, sd
        return self._sample(x)

    def _sample(self, x):
        mu = self.mean(x)

        sd = self.standard_deviation(x)
        sd = self.soft_plus(sd)

        noise = tf.random_normal(mu.get_shape())

        samples = mu + sd * noise  # The multiplication is element-wise

        return samples, mu, sd
