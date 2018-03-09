import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(data_set, filename):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]

    filename = os.path.join(FLAGS.directory, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename = os.path.join(FLAGS.directory, filename + '.tfrecords')
    print('Reading', filename)
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # tf.TFRecordReader().read() only accept queue as param
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image.set_shape([28 * 28])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image, label

def main(unused_argv):
    '''
    Generate TFRecord file
    data_set = mnist.read_data_sets(FLAGS.directory,
                                    dtype=tf.uint8,
                                    reshape=False,
                                    validation_size=FLAGS.validation_size)
    convert_to_tfrecord(data_set.train, 'mnist_train')
    convert_to_tfrecord(data_set.validation, 'mnist_validation')
    convert_to_tfrecord(data_set.test, 'mnist_test')
    '''
    with tf.Graph().as_default():

        first_image, first_label = read_and_decode('mnist_train')

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()
        # Run the initialization
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image, label = sess.run([first_image, first_label])
        print(label)
        print(image)

        coord.request_stop()
        coord.join(threads)
        sess.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='C:\\Boyuan\\MyPython\\MNIST_Dataset',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)