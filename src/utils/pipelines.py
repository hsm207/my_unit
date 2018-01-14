import tensorflow as tf

from datasets.dataset_mnist import dataset_mnist32x32_train, dataset_mnist32x32_test
from datasets.dataset_svhn import dataset_svhn_extra


def train_input_fn_builder(subset=(-1, -1)):
    """
    Build a function to feed the estimator data during training

    :param subset: A tuple of the form (number of images to use from domain 1, number of images to use from domain2)
    :return: A function that takes no arguments and returns a tuple of dictionaries of the form:
            {image1: images from the svhn domain,
             image2: images from the mnist domain},
            {label1: correponding labels for image1,
             label2: corresponding labels for image2}
    """

    def train_input_fn():
        n_domain1, n_domain2 = subset
        ds_domain_1 = dataset_svhn_extra().dataset() \
            .take(n_domain1) \
            .shuffle(buffer_size=200000) \
            .batch(64)
        ds_domain_2 = dataset_mnist32x32_train().dataset() \
            .take(n_domain2) \
            .shuffle(buffer_size=120000) \
            .batch(64)

        ds = tf.data.Dataset.zip((ds_domain_1, ds_domain_2))
        (image_domain1, label_domain1), (image_domain2, label_domain2) = ds.make_one_shot_iterator().get_next()
        return {'image1': image_domain1, 'image2': image_domain2}, {'label1': label_domain1, 'label2': label_domain2}

    return train_input_fn()


def test_input_fn_builder(subset=-1):
    """
    Build a function to feed the estimator data during testing
    :param subset: An integer representing the number of images to use from the MNIST test set
    :return: A tuple of dictionaries of the form:
            {'image': images from the MNIST test set},
            {'label': corresponding labels of the images from the MNIST test set}
    """
    def test_input_fn():
        ds = dataset_mnist32x32_test().dataset() \
            .take(subset) \
            .batch(100)

        image, label = ds.make_one_shot_iterator().get_next()
        return {'image': image}, {'label': label}

    return test_input_fn
