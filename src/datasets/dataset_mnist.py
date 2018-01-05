import gzip
import os
import pickle
import urllib

import cv2
import numpy as np
import tensorflow as tf


class dataset_mnist32x32_train:
    def __init__(self):
        self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist32x32.pkl.gz'
        self.filepath = '../datasets/mnist/'
        self.use_inversion = 1
        full_path = os.sep.join([self.filepath, self.filename])
        self._download(full_path, self.url)
        self.images, self.labels = self._load_samples(full_path)
        self.num = self.images.shape[0]

    def __len__(self):
        return self.num

    def _download(self, filename, url):
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if os.path.isfile(filename):
            print(f'{filename} exists')
            return
        print(f'Download {url} to {filename}')
        urllib.request.urlretrieve(url, filename)
        print(f'Finish downloading {filename}')
        print('Resizing images to 32x32')
        self._resize32x32(filename)

    def _resize32x32(self, full_filepath):
        def _resize(data_in):
            num_samples = data_in.shape[0]
            tmp_data_out = np.zeros((num_samples, 1, 32, 32))
            for i in range(num_samples):
                tmp_img = data_in[i, :].reshape(28, 28)
                new_img = cv2.resize(tmp_img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
                tmp_data_out[i, 0, :, :] = new_img
            return tmp_data_out

        with gzip.open(full_filepath, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        # overwrite the downloaded file!
        with gzip.open(full_filepath, 'wb') as f:
            pickle.dump(
                ([_resize(train_set[0]), train_set[1]],
                 [_resize(valid_set[0]), valid_set[1]],
                 [_resize(test_set[0]), test_set[1]]),
                f
            )

    def _load_samples(self, full_filepath):
        with gzip.open(full_filepath, 'rb') as f:
            train_set, valid_set, _ = pickle.load(f, encoding='latin1')
        images = np.concatenate((train_set[0], valid_set[0]), axis=0)
        labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
        images = images.reshape((images.shape[0], 1, 32, 32))
        if self.use_inversion == 1:
            images = np.concatenate((images, 1 - images), axis=0)
            labels = np.concatenate((labels, labels), axis=0)
        images = (images - 0.5) * 2
        return np.float32(images), labels

    def dataset(self):
        images = tf.data.Dataset.from_tensor_slices(self.images)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        return tf.data.Dataset.zip((images, labels))
