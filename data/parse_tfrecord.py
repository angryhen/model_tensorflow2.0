import tensorflow as tf
from config import train_tfrecord, valid_tfrecord, test_tfrecord, size

from albumentations import CLAHE, GaussianBlur, IAASharpen, OpticalDistortion, GridDistortion, IAAPiecewiseAffine
from albumentations import OneOf, Compose, IAAEmboss, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise
from albumentations import RandomBrightnessContrast, RandomGamma, HorizontalFlip, ShiftScaleRotate
from albumentations import Resize, RandomCrop, Normalize, RandomRotate90, Flip, Transpose, MotionBlur, MedianBlur, Blur
import numpy as np


# 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
def _parse_example(example_string):
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'img_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['img_raw'] = tf.image.decode_jpeg(feature_dict['img_raw'])  # 解码JPEG图片
    feature_dict['img_raw'] = tf.image.resize(feature_dict['img_raw'], (size, size))
    # norm [0-1]
    feature_dict['img_raw'] = feature_dict['img_raw'] / 255.0
    return feature_dict['img_raw'], feature_dict['label']


def _parse_record(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(_parse_example)
    return dataset


# TODO: add augment to tf.data.dataset
def augment(img):
    img = np.array(img * 255, dtype=np.uint8)
    print(img.shape)

    generator = Compose([
        Resize(240, 240),
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=75, p=0.5),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        RandomCrop(224, 224),

        HueSaturationValue(p=0.3),
    ], p=1)
    img = generator(image=img)['image']
    return img


# 展示图片和标签
def _visdom(dataset, n):
    dataset = dataset.shuffle(1000)
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(dataset.take(n)):
        ax = plt.subplot(np.ceil(np.sqrt(n)), np.ceil(np.sqrt(n)), i + 1)
        img = img.numpy()

        # aug
        img = augment(img)

        # show
        ax.imshow(img)
        ax.set_title(f'label = {label}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    raw_dataset = _parse_record(train_tfrecord)  # 读取 TFRecord 文件
    _visdom(raw_dataset, 10)  # show image, last param mean show number
