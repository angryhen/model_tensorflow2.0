import os
import tensorflow as tf
import numpy as np

def _bytes_feature(img):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))

def _ints64_feature(index):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))


def create_tf_record(src,out):
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
    from config import src_dataset, train_tfrecord, val_tfrecord, test_tfrecord

    train_dir = os.path.join(src_dataset, 'train')
    val_dir = os.path.join(src_dataset, 'val')
    test_dir = os.path.join(src_dataset, 'test')

    # create train-tfrecord
    create_tf_record(train_dir, train_tfrecord)

    # create val-tfrecord
    create_tf_record(val_dir, val_tfrecord)

    # create test-tfrecord
    create_tf_record(test_dir, test_tfrecord)

