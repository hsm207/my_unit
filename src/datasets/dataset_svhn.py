import os
import urllib

import numpy as np
import scipy.io
import tensorflow as tf


class img_to_tf_record_writer:
    # based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos
    # /reading_data/convert_to_records.py
    def __init__(self, images, labels, save_path):
        """

        :param images: A numpy array of images (number of images, channels, height, width) to convert to a tfrecord
        :param labels: A numpy array of labels ((number of images, channels, height, width) to convert to a tfrecord
        :param save_path: A string representing the full path to save the tfrecord
        """
        self.images = images
        self.labels = labels
        self.filename = save_path

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def encode(self):
        if os.path.isfile(self.filename):
            print(f'{self.filename} exists')
            return

        num_examples, depth, rows, cols = self.images.shape

        print(f'Converting to TF Record format')

        with tf.python_io.TFRecordWriter(self.filename) as writer:
            for i in range(num_examples):
                image_raw = self.images[i].tostring()
                label = int(self.labels[i])

                feature_dict = {
                    'height': self._int64_feature(rows),
                    'width': self._int64_feature(cols),
                    'depth': self._int64_feature(depth),
                    'label': self._int64_feature(label),
                    'image_raw': self._bytes_feature(image_raw)
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature_dict)
                )
                writer.write(example.SerializeToString())

        print(f'Finished converting to TF Record format')

    @staticmethod
    def decode(serialized_example):
        # Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/
        # reading_data/fully_connected_reader.py
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            }
        )

        image_shape = tf.stack([features['depth'], features['height'], features['width']])
        image = tf.decode_raw(features['image_raw'], tf.float32)
        image = tf.reshape(image, image_shape)

        label = tf.cast(features['label'], tf.int32)

        return image, label


class dataset_svhn_extra:
    # This dataset consist of 531,131 images of size 32 x 32 with 3 channels.
    def __init__(self):
        # Convert the .mat file into a TF record because the Dataset API cannot create tensors greater than 2 GB.
        # See https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays for more info

        self.url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
        self.filename_mat = 'extra_32x32.mat'
        self.filename_tfrec = 'extra_32x32.tfrecords'
        self.filepath = '../datasets/svhn/'

        full_filepath_mat = os.path.join(self.filepath, self.filename_mat)
        full_filepath_tf_rec = full_filepath_mat.replace('.mat', '.tfrecords')

        # do not download and load the matlab version of svhn if the tf records version exists
        if os.path.isfile(full_filepath_tf_rec):
            return

        self._download(full_filepath_mat, self.url)
        images, labels = self._load_samples(full_filepath_mat)

        img2tf_rec = img_to_tf_record_writer(images, labels, full_filepath_tf_rec)
        img2tf_rec.encode()

    def _download(self, filename, url):
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if os.path.isfile(filename):
            print(f'{filename} exists')
            return
        print(f'Downloading {url} to {filename}')
        urllib.request.urlretrieve(url, filename)
        print(f'Finished downloadiang {filename}')

    def _load_samples(self, file_path):

        print(f'Loading {file_path}')
        mat = scipy.io.loadmat(file_path, squeeze_me=True)
        y = mat['y']
        x = mat['X']

        # set the label for the digit 0 to be 0 instead of 10
        index_digit_0 = np.where(y == 10)
        y[index_digit_0] = 0

        # transpose the image from (height, width, channel, number of images) to (number of images, channel, height, width)
        x = np.transpose(x, (3, 2, 0, 1))
        # normalize the input image...not sure why the authors multiplied by 2 and subtract 1
        x = 2 * np.float32(x / 255.0) - 1

        return x, y

    def dataset(self):
        full_filepath_tfrec = os.path.join(self.filepath, self.filename_tfrec)
        ds = tf.data.TFRecordDataset(full_filepath_tfrec) \
            .map(img_to_tf_record_writer.decode)
        return ds
