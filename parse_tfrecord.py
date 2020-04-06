import tensorflow as tf
from config import train_tfrecord, val_tfrecord, test_tfrecord, size

# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.image.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    feature_dict['image'] = tf.image.resize( feature_dict['image'], (size, size))
    # norm
    feature_dict['image'] = feature_dict['image'] / 255.0
    return feature_dict['image'], feature_dict['label']

if __name__ == '__main__':
    raw_dataset = tf.data.TFRecordDataset(train_tfrecord)    # 读取 TFRecord 文件
    dataset = raw_dataset.map(_parse_example)