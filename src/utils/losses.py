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
