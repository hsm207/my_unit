import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout


# Coupled discriminator model for digit classification (Appendix B, Table 6)
class CoDis32x32:

    # TODO: Rewrite this layer into a separate class
    def _conv2d(self, n_filters, kernel_size, max_pool_stride, conv_padding):
        """
        Returns a layer that applies a 2D convolution followed by a max pool operation.

        The stride in the convolution operation is set to (1, 1)

        :param n_filters: Number of filters for the convolution operation
        :param kernel_size: The kernel size of the convolution operation
        :param max_pool_stride: The kernel size AND stride of the max pool operation
        :param conv_padding: The padding to use for the convolution layer
        :return: A function that takes as input a tensor to apply the convolution and max pool operation to.
        """
        df = self.data_format

        def compose_conv2d_with_maxpool(x):
            x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding=conv_padding,
                       data_format=df,
                       activation='relu')(x)

            x = MaxPooling2D(pool_size=max_pool_stride, strides=max_pool_stride,
                             data_format=df, padding='valid')(x)
            return x

        return compose_conv2d_with_maxpool

    def __init__(self, data_format='channels_first'):
        self.data_format = data_format
        # TODO: Rename conv layers starting from 1
        # The unshared convolutional layer for image from domain A
        self.conv0_a = tf.make_template('conv0_a', self._conv2d(64, (5, 5), (2, 2), 'same'))

        # The unshared convolutional layer for image from domain B
        self.conv0_b = tf.make_template('conv0_b', self._conv2d(64, (5, 5), (2, 2), 'same'))

        # The rest of the convolutional layers are shared
        self.conv1 = tf.make_template('conv1', self._conv2d(128, (5, 5), (2, 2), 'same'))
        self.conv2 = tf.make_template('conv2', self._conv2d(256, (5, 5), (2, 2), 'same'))
        self.conv3 = tf.make_template('conv3', self._conv2d(512, (5, 5), (2, 2), 'same'))

        # This conv layer determine if the image is real or fake
        self.conv4 = Conv2D(2, (2, 2), (1, 1), padding='valid', data_format=self.data_format, name='conv4')

        # This conv layer classifies the image (digit 0 to 9)
        self.conv_cl = Conv2D(10, (2, 2), (1, 1), padding='valid', data_format=self.data_format, name='conv_cl')

        self.dropout0 = tf.make_template('dropout0', Dropout(0.1))
        self.dropout1 = tf.make_template('dropout1', Dropout(0.3))
        self.dropout2 = tf.make_template('dropout2', Dropout(0.5))
        self.dropout3 = tf.make_template('dropout3', Dropout(0.5))

    def _forward_core(self, h0):
        """
        Perform the shared convolution layers without dropout

        :param h0: A tensor representing the input (16 x 16, 64 channels) to be convolution. This is the output of passing the
                   svhn/mnist image to their unshared convolution layer
        :return: A tensor representing the output of the convolutions (2 x 2, 512 channels)
        """
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return h3

    def _forward_core_dropout(self, h0):
        """
        Perform the shared convolution layers with dropout
        :param h0: A tensor representing the input (16 x 16, 64 channels) to be convolution. This is the output of passing the
                   svhn/mnist image to their unshared convolution layer
        :return: A tensor representing the output of the convolutions (2 x 2, 512 channels)
        """
        h0_dropout = self.dropout0(h0)
        h1 = self.conv1(h0_dropout)

        h1_dropout = self.dropout1(h1)
        h2 = self.conv2(h1_dropout)

        h2_dropout = self.dropout2(h2)
        h3 = self.conv3(h2_dropout)

        h3_dropout = self.dropout3(h3)

        return h3_dropout

    def __call__(self, image_a, image_b):
        """
        The Discriminator's forward pass logic.

        During training, the Discriminator will extract the features of the given pair of images (
        using the convolutional layers without dropout) and return a two-dimensional "logits" representing
        the prediction whether the given images are real or fake (using the same convolutional layers but with dropout).

        :param image_a: A tensor representing the images from domain A (including batch size)
        :param image_b: A tensor representing the images from domain B (including batch size)
        :return: A tuple of tensors (logits whether images are real or fake, features from image A,
                features from image B)
        """
        # Pass the pair of images to their respective unshared convolutional layer
        h0_a = self.conv0_a(image_a)
        h0_b = self.conv0_b(image_b)

        img_a_features = self._forward_core(h0_a)
        img_b_features = self._forward_core(h0_b)

        h0 = tf.concat([h0_a, h0_b], axis=0)
        h3_dropout = self._forward_core_dropout(h0)
        real_or_fake_logits = self.conv4(h3_dropout)
        real_or_fake_logits = tf.squeeze(real_or_fake_logits)

        return real_or_fake_logits, img_a_features, img_b_features

    def classify_image_a(self, image_a):
        """
        Classifies image from domain A
        :param image_a: A tensor representing images from domain A (including batch size0
        :return: A tensor of shape (batch size, 10) representing the logits for each image
        """
        h0_a = self.conv0_a(image_a)
        h3_a = self._forward_core_dropout(h0_a)
        h4_a = self.conv_cl(h3_a)
        h4_a = tf.squeeze(h4_a)

        return h4_a

    def classify_image_b(self, image_b):
        """
        Classifies image from domain B
        :param image_a: A tensor representing images from domain B (including batch size0
        :return: A tensor of shape (batch size, 10) representing the logits for each image
        """
        h0_b = self.conv0_b(image_b)
        h3_b = self._forward_core_dropout(h0_b)
        h4_b = self.conv_cl(h3_b)
        h4_b = tf.squeeze(h4_b)

        return h4_b
