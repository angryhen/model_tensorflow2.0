import os
import tensorflow as tf
import numpy as np
import config


def _bytes_feature(img):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))


def _ints64_feature(index):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))


def create_tf_record(src, out):
    writer = tf.io.TFRecordWriter(out)
    dirs = os.listdir(src)
    for index, name in enumerate(dirs):
        class_path = os.path.join(src, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.io.read_file(img_path)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img_raw': _bytes_feature(img),
                        'label': _ints64_feature(index)
                    }))
            writer.write(example.SerializeToString())
    writer.close()


def split_train_val():
    pass


if __name__ == '__main__':
    from config import data_dir, train_tfrecord, valid_tfrecord, test_tfrecord

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # create train/val/test tfrecord
    create_tf_record(train_dir, train_tfrecord)
    create_tf_record(valid_dir, valid_tfrecord)
    create_tf_record(test_dir, test_tfrecord)


