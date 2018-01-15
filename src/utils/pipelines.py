import tensorflow as tf

from datasets.dataset_mnist import dataset_mnist32x32_train, dataset_mnist32x32_test
from datasets.dataset_svhn import dataset_svhn_extra


def train_input_fn_builder(subset=(-1, -1), data_format='channels_first', batch_size=64, image_a_shape=(3, 32, 32),
                           image_b_shape=(1, 32, 32), num_epochs=200000):
    # TODO: Document function
    """
    Build a function to feed the estimator data during training
    :param subset: A tuple of the form (number of images to use from domain 1, number of images to use from domain2)
    :param data_format: A string specifying whether the images should be channels_first or channels_last format
    :return: A function that takes no arguments and returns a tuple of dictionaries of the form:
            {image1: images from the svhn domain,
             image2: images from the mnist domain},
            {label1: correponding labels for image1,
             label2: corresponding labels for image2}
    """

    def train_input_fn():
        def set_images_shape(dataset_a, dataset_b):
            (img_a, lab_a), (img_b, lab_b) = dataset_a, dataset_b

            shape_a = (batch_size, *image_a_shape)
            shape_b = (batch_size, *image_b_shape)

            img_a.set_shape(shape_a)
            img_b.set_shape(shape_b)

            return ((img_a, lab_a), (img_b, lab_b))

        def switch_image_channels(dataset_a, dataset_b):
            (image_a, label_a), (image_b, label_b) = dataset_a, dataset_b
            image_a = tf.transpose(image_a, (0, 2, 3, 1))
            image_b = tf.transpose(image_b, (0, 2, 3, 1))

            return ((image_a, label_a), (image_b, label_b))

        def is_full_batch(dataset_a, dataset_b):
            a_batch_size = tf.shape(dataset_a[0])[0]
            b_batch_size = tf.shape(dataset_b[0])[0]

            full_batch = tf.logical_and(tf.equal(a_batch_size, batch_size),
                                        tf.equal(b_batch_size, batch_size))

            return full_batch

        # Note: The format of the images stored in the dataset files is channels_first
        n_domain1, n_domain2 = subset
        # TODO: swithc to full svhn dataset
        ds_domain_1 = dataset_svhn_extra().dataset() \
            .take(n_domain1) \
            .shuffle(buffer_size=200000) \
            .batch(batch_size) \
            .repeat(num_epochs)

        ds_domain_2 = dataset_mnist32x32_train().dataset() \
            .take(n_domain2) \
            .shuffle(buffer_size=120000) \
            .batch(batch_size) \
            .repeat(num_epochs)

        # Train the model only on batches where the number of images in both domains are equal to the batch size
        # On large datasets, the remainder is unlikely to be significant. Furthermore, since we shuffle at
        # every epoch, we still have a chance to train on every possible image pair
        ds = tf.data.Dataset.zip((ds_domain_1, ds_domain_2)) \
            .filter(is_full_batch)

        # Set the shape of the datasets because TensorFlow is unable to infer the shape of svhn and batch size
        # during graph construction
        ds = ds.map(set_images_shape)

        # Adjust the channel axis if data_format is channels_last
        if data_format == 'channels_last':
            ds = ds.map(switch_image_channels)

        (image_domain1, label_domain1), (image_domain2, label_domain2) = ds.make_one_shot_iterator().get_next()
        return {'image1': image_domain1, 'image2': image_domain2}, {'label1': label_domain1, 'label2': label_domain2}

    return train_input_fn


def test_input_fn_builder(subset=-1, data_format='channels_first', batch_size=100, image_shape=(1, 32, 32)):
    # TODO: Document function
    """
    Build a function to feed the estimator data during testing
    :param subset: An integer representing the number of images to use from the MNIST test set
    :param data_format: A string specifying whether the images should be channels_first or channels_last format
    :return: A function that takes no arguments and returns a tuple of dictionaries of the form:
            {'image': images from the MNIST test set},
            {'label': corresponding labels of the images from the MNIST test set}
    """

    def test_input_fn():
        def set_image_shape(image, label):
            shape = (batch_size, *image_shape)

            image.set_shape(shape)

            return image, label

        ds = dataset_mnist32x32_test().dataset() \
            .take(subset) \
            .batch(batch_size) \
            .map(set_image_shape)

        if data_format == 'channels_last':
            ds = ds.map(lambda img, lab: (tf.transpose(img, (0, 2, 3, 1)), lab))

        image, label = ds.make_one_shot_iterator().get_next()
        return {'image': image}, {'label': label}

    return test_input_fn
