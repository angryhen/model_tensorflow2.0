import tensorflow as tf
from config import train_tfrecord, valid_tfrecord, test_tfrecord, size

# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'img_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['img_raw'] = tf.image.decode_jpeg(feature_dict['img_raw'])    # 解码JPEG图片
    feature_dict['img_raw'] = tf.image.resize( feature_dict['img_raw'], (size, size))
    # norm [0-1]
    feature_dict['img_raw'] = feature_dict['img_raw'] / 255.0
    return feature_dict['img_raw'], feature_dict['label']

def _parse_record(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(_parse_example)
    return dataset

# 展示图片和标签
def _visdom(dataset, n):
    dataset = dataset.shuffle(1000)
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(dataset.take(n)):
        ax = plt.subplot(np.ceil(np.sqrt(n)), np.ceil(np.sqrt(n)), i+1)
        ax.imshow(img.numpy())
        ax.set_title(f'label = {label}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



if __name__ == '__main__':
    raw_dataset = _parse_record(train_tfrecord)    # 读取 TFRecord 文件
    _visdom(raw_dataset, 10)
