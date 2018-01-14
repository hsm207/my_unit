import tensorflow as tf


class MSE_Images():
    """
    Computes the mean squared error (MSE) when the target variable is an image
    :param y: A 4-D tensor representing the ground truth where the first dimension is the batch size
    :param y_hat: A 4-D tensor representing the predictions where the first dimension is the batch size
    :return: A scalar representing the MSE between y and y_hat
    """

    def __call__(self, y, y_hat):
        # Get the number of images in the batch (assume both y and y_hat have the same batch size)
        n_y = y.shape[0].value

        sse = tf.reduce_sum(tf.square(y - y_hat))
        mse = sse / n_y

        return mse


class Batch_Cross_Entropy():
    def __call__(self, y, y_hat_logits):
        """
        Comptutes the cross entropy loss for a given batch of prediction logits  and class labels
        :param y: A tensor of ints of shape (batch_size, ) representing the ground truth
        :param y_hat_logits: A tensor of shape (batch_size, number_of_classes) representing the logits of each
                             class prediction
        :return: A scalar representing the cross entropy for the given batch
        """
        n_classes = y_hat_logits.get_shape()[1].value
        y_ohe = tf.one_hot(y, n_classes)
        loss_per_observation = tf.nn.softmax_cross_entropy_with_logits(labels=y_ohe, logits=y_hat_logits)
        mean_loss = tf.reduce_mean(tf.reduce_mean(loss_per_observation))

        return mean_loss


class L2_Regularization():
    def __call__(self, weights):
        """
        Computes the L2 Regularization for a given list of weights
        :param weights: List of tensors representing the weights of a model we want to compute the L2 regularization
        :return: A scalar representing the total L2 regularization of weights
        """
        l2_losses = [tf.nn.l2_loss(wt) for wt in weights]
        total_l2_loss = tf.add_n(l2_losses)

        return total_l2_loss

class KL_Divergence():
    def __call__(self, mu, sd):
        """
        Computes the mean KL divergence term for the Generator's VAE loss

        Let z be the latent variables produced by the Generator.

        This implementation assumes that the posterior distribution of z, p(z|x) follows a
        Gaussian distribution (x is the observed image) and the prior distribution of z, p(z)
        follows a standard normal distribution

        The formula is derived from Appendix B in the Auto-Encoding Variational Bayes paper
        (arXiv:1312.6114v10). We compute Dkl instead of -Dkl because maximizing -Dkl is the
        same as minimizing Dkl.

        Note that each component of z is assumed to be independent of each other.

        :param mu: A tensor of shape (batch_size, 1, 1, dimension_of_latent_variables) representing the mean
                   of each component of z
        :param sd: A tensor of shape (batch_size, 1, 1, dimension_of_latent_variables) representing the
                      standard deviation of each component of z
        :return: A scalar representing mean KL divergence of the given batch
        """

        batch_size = mu.get_shape()[0].value
        mu_2 = tf.square(mu)
        sd_2 = tf.square(sd)
        kl_divergence = 0.5 * tf.reduce_sum(-1 - tf.log(sd_2) + mu_2 + sd_2)/batch_size

        return kl_divergence
