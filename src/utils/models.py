from itertools import product

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Activation

from utils.layers import LeakyReLUBNNSConv2d, GaussianVAE2D, LeakyReLUBNNSConvTranspose2d
from utils.losses import MSE_Images, Batch_Cross_Entropy, L2_Regularization, KL_Divergence


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
    # TODO: Document this model
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
            # from image_b to an image in domain a
            de_h4_a = self.de_conv4_a(de_h3_a)
            de_h4_a = self.tanh_a(de_h4_a)

            # Similarly, the first batch_size images in de_h4_b will contain the translation from image_a to an image
            # in domain b and the next batch_size images are the reconstructed images of image_b
            de_h4_b = self.de_conv4_b(de_h3_b)
            de_h4_b = self.tanh_b(de_h4_b)

            img_aa, img_ba = tf.split(de_h4_a, 2, axis=0)
            img_ab, img_bb = tf.split(de_h4_b, 2, axis=0)
            distribution_params = (mu, sd)

        return img_aa, img_ab, img_bb, img_ba, distribution_params


# Model to use the unsupervised image-to-image translation framework to do domain adaptation from the SHVN dataset
# to the MNIST dataset.
class UNIT_DA_SHVN_TO_MNIST:
    def __init__(self, svhn_images_channels=3, mnist_image_channels=1, data_format='channels_first', batch_size=64):
        self.channel_axis = 1 if data_format == 'channels_first' else 3

        # Create the discriminator and generator model
        self.gen = tf.make_template('Generator', CoVAE32x32(data_format=data_format,
                                                            domain_a_image_channels=svhn_images_channels,
                                                            domain_b_image_channels=mnist_image_channels))
        dis = CoDis32x32(data_format=data_format)
        self.dis = tf.make_template('', dis, unique_name_='Discriminator')
        self.classify_image_a = tf.make_template('', dis.classify_image_a, unique_name_='Discriminator')
        self.classify_image_b = tf.make_template('', dis.classify_image_b, unique_name_='Discriminator')

        # Create the optimizers for the discriminator and generator
        # TODO: Figure out how to do L2 regularization
        # Note that the decay parameter in pytorch is not learning rate decay!
        self.opt_gen = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
        self.opt_dis = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)

        # Losses

        # We use the MSE loss to compute the loss between a real image and a generated image within the same domain
        # i.e. real image A and reconstructed image A
        # In the Auto-Encoding Variational Bayes paper (arXiv:1312.6114v10), this should be the log likelihood of the
        # image given its latent variable.
        # Since the likelihood function is modelled using a Laplacian distribution, the authors in this paper claim that
        # minimizing the log likelihood is the same as minimizing the distance between the image and the reconstructed
        # image.
        self.ll_loss_criterion = tf.make_template('ll_loss_mse', MSE_Images())

        self.batch_cross_entropy = tf.make_template('batch_cross_entropy', Batch_Cross_Entropy())

        self.l2_reg = tf.make_template('L2_regularization', L2_Regularization())

        self.kl_div = tf.make_template('KL_divergence', KL_Divergence())

        # Create the normalized x and y coordinates feature. This is not part of the preprocessing step
        # because it is the same for all images
        self.xy_normalized = self._create_xy_image(width=32, data_format=data_format, batch_size=batch_size)

        # Metrics stuff
        self.softmax = tf.make_template('softmax', Activation('softmax'))

    def _create_xy_image(self, width=32, data_format='channels_first', batch_size=64):
        """
        Create the normalized x and y coordinates features. See Appendix B, section SVHN->MNIST

        :param width: Height and Width of the x and y plane
        :param data_format: data format of the input image
        :param batch_size: batch size of the inputs to the model
        :return: A tensor with shape (batch_size, 2, height, width) if data_format='channels_first' or
                (batch_size, height, width, 2) if data_format='channels_last'
        """
        # Create the array of coordinates (one channel for x and y)
        coordinates = list(product(range(width), range(width)))
        coordinates = np.array(coordinates).reshape((width, width, 2))

        # Normalize the coordinates
        coordinates = (coordinates - width / 2) / (width / 2)

        # Switch the channels axis if data_format is channels_first
        coordinates = np.transpose(coordinates, [2, 0, 1]) if data_format == 'channels_first' else coordinates

        # Replicate this array batch_size times
        coordinates = np.expand_dims(coordinates, 0)
        coordinates = np.tile(coordinates, (batch_size, 1, 1, 1))

        coordinates = tf.constant(coordinates, tf.float32)

        return coordinates

    def update_discriminator(self, images_a, images_b, labels_a, loss_wt_gan=1.0, loss_wt_class=10.0,
                             loss_wt_feature=1, loss_wt_l2=0.0005):
        # Forward Pass
        # Step 1. Feed the Discriminator with real image pairs
        real_logits, real_img_a_feat, real_img_b_feat = self.dis(images_a, images_b)

        # Step 2. Compute the cross entropy loss on the Discriminator's ability to identify real images
        # Since the images fed into the Discriminator were real images (both in domain a and domain b), the
        # ground truth is 1 (real image)
        # We call this loss the adversarial real loss
        real_labels = tf.ones(real_logits.get_shape()[0], tf.int32)
        ad_real_loss = self.batch_cross_entropy(real_labels, real_logits)

        # Step 3. Feed the Discriminator with fake image pairs
        # We will generate the faka images by feeding the Generator with real images that has been augmented
        # with the normalized x and y coordinates
        img_a_xy = tf.concat([images_a, self.xy_normalized], axis=self.channel_axis, name='img_a_augment')
        img_b_xy = tf.concat([images_b, self.xy_normalized], axis=self.channel_axis, name='img_b_augment')

        fake_img_aa, fake_img_ab, fake_img_bb, fake_img_ba, latent_codes = self.gen(img_a_xy, img_b_xy)

        # Feed the Discriminator with the fake reconstructed images followed by the fake translated images
        fake_recon_logits, fake_img_aa_feat, fake_img_bb_feat = self.dis(fake_img_aa, fake_img_bb)
        fake_trans_logits, fake_img_ba_feat, fake_img_ab_feat = self.dis(fake_img_ba, fake_img_ab)

        # Step 4. Compute the cross entropy loss on the Discriminator's ability to identify fake images
        # Since the images fed into the Discriminator were fake images (both in domain a and domain b), the
        # ground truth is 0 (fake image)
        # We call this loss the adversarial fake loss, which is the average of the adversarial reconstructed loss
        # and the adversarial translation loss
        fake_labels = tf.zeros(fake_recon_logits.get_shape()[0], tf.int32)
        ad_fake_recon_loss = self.batch_cross_entropy(fake_labels, fake_recon_logits)
        ad_fake_trans_loss = self.batch_cross_entropy(fake_labels, fake_trans_logits)
        ad_fake_loss = 0.5 * (ad_fake_recon_loss + ad_fake_trans_loss)

        # Step 5. Compute the L2 distance (not L1, typo in paper) between the features extracted by the highest layer of the discriminators
        # for a pair of generated images. This further encourages the Discriminator to interpret a pair of corresponding
        # images in the same way (see Domain Adaptation section in the paper).
        # Intuition:
        # Suppose we have image a and we feed this to the Generator. Then, we will have the reconstructed version of
        # image a and the translated (and hopefully correct) version of image a. If we feed these two images to the
        # Discriminator, it must produce similar features because it originated from image a so that if it can
        # correctly classify the image in domain a, it should be able to correctly classify the corresponding image
        # in domain b.
        # The ground truth is the 0 feature map because if the extracted features are similar, then their difference
        # should be close to 0.
        zero_feat_map = tf.zeros(fake_img_aa_feat.get_shape(), tf.float32)
        feat_loss_a = self.ll_loss_criterion(zero_feat_map, fake_img_aa_feat - fake_img_ab_feat)
        feat_loss_b = self.ll_loss_criterion(zero_feat_map, fake_img_ba_feat - fake_img_bb_feat)

        # Step 6. Compute the Discriminator's classification loss on domain a
        cls_logits = self.classify_image_a(images_a)
        cls_loss = self.batch_cross_entropy(labels_a, cls_logits)

        # Step 7. Compute the Discriminator's total loss
        total_loss = loss_wt_gan * (ad_real_loss + ad_fake_loss) \
                     + loss_wt_class * cls_loss \
                     + loss_wt_feature * (feat_loss_a + feat_loss_b) \
                     + loss_wt_l2 * self.l2_reg(tf.trainable_variables(scope="Discriminator"))

        # Name the loss for logging purposes
        tf.identity(total_loss, 'loss_discriminator')

        return total_loss

    def update_generator(self, images_a, images_b, loss_wt_gan=1.0, loss_wt_kl=0.0001, loss_wt_ll=0.001,
                         loss_wt_l2=0.0005):
        # Step 1: Augment the real images with the normalized x and y coordinates
        img_a_xy = tf.concat([images_a, self.xy_normalized], axis=self.channel_axis)
        img_b_xy = tf.concat([images_b, self.xy_normalized], axis=self.channel_axis)

        # Step 2: Generate the fake images from the real images (reconstructed and translated)
        fake_img_aa, fake_img_ab, fake_img_bb, fake_img_ba, latent_codes = self.gen(img_a_xy, img_b_xy)

        # Step 3: Pass the fake images to the Discriminator in two sets, the first set is the set of
        # reconstructed images and the other set is the set of translated images
        fake_recon_logits, _, _ = self.dis(fake_img_aa, fake_img_bb)
        fake_trans_logits, _, _ = self.dis(fake_img_ba, fake_img_ab)

        # Step 4: Compute the Generator's loss from trying to fool the Discriminator
        # Since the goal of the Generator is to fool the Discriminator, the ground truth in both sets is 1 i.e.
        # the Discriminator thinks that the images are real
        fake_labels = tf.ones(fake_recon_logits.get_shape()[0], tf.int32)
        ad_fake_recon_loss = self.batch_cross_entropy(fake_labels, fake_recon_logits)
        ad_fake_trans_loss = self.batch_cross_entropy(fake_labels, fake_trans_logits)
        ad_loss = ad_fake_recon_loss + ad_fake_trans_loss

        # Step 5: Compute the losses for the VAE part of the Generator
        # The reconstruction loss for image a
        ll_loss_a = self.ll_loss_criterion(images_a, fake_img_aa)

        # The reconstruction loss for image b
        ll_loss_b = self.ll_loss_criterion(images_b, fake_img_bb)

        ll_loss = ll_loss_a + ll_loss_b

        # Compute the KL divergence
        kl_div = self.kl_div(*latent_codes)

        total_loss = loss_wt_gan * ad_loss \
                     + loss_wt_kl * kl_div \
                     + loss_wt_ll * ll_loss \
                     + loss_wt_l2 * self.l2_reg(tf.trainable_variables(scope='Generator'))

        return total_loss, fake_img_aa, fake_img_ba, fake_img_ab, fake_img_bb

    def get_train_op(self, loss_fn_generator, loss_fn_discriminator):
        train_op_generator = self.opt_gen.minimize(loss_fn_generator, global_step=tf.train.get_or_create_global_step(),
                                                   var_list=tf.trainable_variables(scope='Generator'))

        train_op_discriminator = self.opt_dis.minimize(loss_fn_discriminator, global_step=None,
                                                       var_list=tf.trainable_variables(scope='Discriminator'))

        train_op = tf.group(train_op_discriminator, train_op_generator)

        return train_op

    def get_predictions(self, images_b):
        y_hat_logits = self.classify_image_b(images_b)
        y_hat_probs = self.softmax(y_hat_logits)
        y_hat_class = tf.argmax(y_hat_logits, axis=1)

        predictions = {
            'class': y_hat_class,
            'prob': y_hat_probs
        }

        return predictions

    def get_metrics(self, true_label_b, pred_label_b):
        class_acc = tf.metrics.accuracy(true_label_b, pred_label_b)

        metrics = {
            'classification_accuracy': class_acc
        }

        return metrics
