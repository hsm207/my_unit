import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Activation

from utils.layers import LeakyReLUBNNSConv2d, GaussianVAE2D, LeakyReLUBNNSConvTranspose2d


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
        :param image_b: A tensor representing images from domain B (including batch size0
        :return: A tensor of shape (batch size, 10) representing the logits for each image
        """
        h0_b = self.conv0_b(image_b)
        h3_b = self._forward_core_dropout(h0_b)
        h4_b = self.conv_cl(h3_b)
        h4_b = tf.squeeze(h4_b)

        return h4_b


# Coupled generator model for digit classification (Appendix B, Table 4)
class CoVAE32x32:
    def __init__(self, data_format='channels_first', domain_a_image_channels=3, domain_b_image_channels=1):
        # Encoding layers
        self.g_en_conv0_a = tf.make_template('conv0_a',
                                             LeakyReLUBNNSConv2d(n_filters=64, kernel_size=(5, 5), stride=(2, 2),
                                                                 padding='same', data_format=data_format))
        self.g_en_conv0_b = tf.make_template('conv0_b',
                                             LeakyReLUBNNSConv2d(n_filters=64, kernel_size=(5, 5), stride=(2, 2),
                                                                 padding='same', data_format=data_format))
        self.g_en_conv1 = tf.make_template('conv1',
                                           LeakyReLUBNNSConv2d(n_filters=128, kernel_size=(5, 5), stride=(2, 2),
                                                               padding='same', data_format=data_format))
        self.g_en_conv2 = tf.make_template('conv2',
                                           LeakyReLUBNNSConv2d(n_filters=256, kernel_size=(8, 8), stride=(1, 1),
                                                               padding='valid', data_format=data_format))
        self.g_en_conv3 = tf.make_template('conv3',
                                           LeakyReLUBNNSConv2d(n_filters=512, kernel_size=(1, 1), stride=(1, 1),
                                                               padding='valid', data_format=data_format))

        # Layer to generate the latent variables
        self.g_vae = tf.make_template('vae',
                                      GaussianVAE2D(n_filters=512, kernel_size=(1, 1), stride=(1, 1), padding='valid',
                                                    data_format=data_format))

        # Decoding layers
        # We will reconstruct the images using transposed convolutional layers followed by a tanh activation function
        # because the input image has been normalized such that the value of any pixel is between -1 and 1.
        self.g_de_conv0 = tf.make_template('de_conv0',
                                           LeakyReLUBNNSConvTranspose2d(n_filters=512, kernel_size=(4, 4),
                                                                        stride=(2, 2),
                                                                        padding='valid', data_format=data_format))

        self.g_de_conv1 = tf.make_template('de_conv1',
                                           LeakyReLUBNNSConvTranspose2d(n_filters=256, kernel_size=(4, 4),
                                                                        stride=(2, 2),
                                                                        padding='same', data_format=data_format))

        self.g_de_conv2 = tf.make_template('de_conv2',
                                           LeakyReLUBNNSConvTranspose2d(n_filters=128, kernel_size=(4, 4),
                                                                        stride=(2, 2),
                                                                        padding='same', data_format=data_format))

        self.g_de_conv3_a = tf.make_template('de_conv3_a',
                                             LeakyReLUBNNSConvTranspose2d(n_filters=64, kernel_size=(4, 4),
                                                                          stride=(2, 2),
                                                                          padding='same', data_format=data_format))

        self.g_de_conv3_b = tf.make_template('de_conv3_b',
                                             LeakyReLUBNNSConvTranspose2d(n_filters=64, kernel_size=(4, 4),
                                                                          stride=(2, 2),
                                                                          padding='same', data_format=data_format))

        self.de_conv4_a = tf.make_template('de_conv4_a',
                                           Conv2DTranspose(filters=domain_a_image_channels, kernel_size=(1, 1),
                                                           strides=(1, 1), padding='valid', data_format=data_format))

        self.de_conv4_b = tf.make_template('de_conv4_b',
                                           Conv2DTranspose(filters=domain_b_image_channels, kernel_size=(1, 1),
                                                           strides=(1, 1), padding='valid', data_format=data_format))

        self.tanh_a = tf.make_template('tanh_a', Activation('tanh'))
        self.tanh_b = tf.make_template('tanh_b', Activation('tanh'))

    def __call__(self, image_a, image_b):
        with tf.name_scope('Encoding_Layer'):
            en_h0_a = self.g_en_conv0_a(image_a)
            en_h0_b = self.g_en_conv0_b(image_b)

            en_h0 = tf.concat([en_h0_a, en_h0_b], axis=0)
            en_h1 = self.g_en_conv1(en_h0)
            en_h2 = self.g_en_conv2(en_h1)
            en_h3 = self.g_en_conv3(en_h2)

        with tf.name_scope('Latent_Layer'):
            z, mu, sd = self.g_vae(en_h3)

        with tf.name_scope("Decoding_Layer"):
            de_h0 = self.g_de_conv0(z)
            de_h1 = self.g_de_conv1(de_h0)
            de_h2 = self.g_de_conv2(de_h1)

            de_h3_a = self.g_de_conv3_a(de_h2)
            de_h3_b = self.g_de_conv3_b(de_h2)

            # Since we stacked image_a and image_b together in en_h0, the first batch_size images in de_h4_a
            # will contain the reconstructed images of image_a and the next batch_size images are the translation
            # from image_a to an image in domain b
            de_h4_a = self.de_conv4_a(de_h3_a)
            de_h4_a = self.tanh_a(de_h4_a)

            # Similarly, the first batch_size images in de_h4_b will contain the reconstructed images of image_b and
            # the next batch_size images are the translation from image_b to an image in domain a
            de_h4_b = self.de_conv4_b(de_h3_b)
            de_h4_b = self.tanh_b(de_h4_b)

            img_aa, img_ab = tf.split(de_h4_a, 2, axis=0)
            img_bb, img_ba = tf.split(de_h4_b, 2, axis=0)
            distribution_params = (mu, sd)

        return img_aa, img_ab, img_bb, img_ba, distribution_params
