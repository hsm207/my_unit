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
