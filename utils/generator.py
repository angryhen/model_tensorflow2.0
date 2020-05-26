import numpy as np
import pandas as pd
import sklearn
import cv2
import os
import albumentations
from albumentations import (Blur, Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
                            HueSaturationValue, Transpose, RandomBrightnessContrast, CLAHE,
                            CoarseDropout, Normalize, ToFloat, OneOf, Compose)
import keras
import tensorflow as tf
import keras.backend as K


class MyGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, labels, root_directory='',
                 batch_size=128, mix=False,
                 shuffle=True, augment=True):
        self.image_filenames = image_filenames
        self.labels = labels
        self.root_directory = root_directory
        self.batch_size = batch_size
        self.is_mix = mix
        self.is_augment = augment
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()
        if self.is_augment:
            self.generator = Compose([Blur(), Flip(), Transpose(), ShiftScaleRotate(),
                                      RandomBrightnessContrast(), HueSaturationValue(),
                                      CLAHE(), GridDistortion(), ElasticTransform(), CoarseDropout(),
                                      ToFloat(max_value=255.0, p=1.0)], p=1.0)
        else:
            self.generator = Compose([ToFloat(max_value=255.0, p=1.0)], p=1.0)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.labels = sklearn.utils.shuffle(self.image_filenames, self.labels)

    def mix_up(self, x, y):
        original_index = np.arange(x.shape[0])
        new_index = np.arange(x.shape[0])
        np.random.shuffle(new_index)
        beta = np.random.beta(0.2, 0.4)
        mix_x = beta * x[original_index] + (1 - beta) * x[new_index]
        mix_y = beta * y[original_index] + (1 - beta) * y[new_index]
        return mix_x, mix_y

    def __getitem__(self, index):
        batch_x = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        new_images = []
        new_labels = []
        for image_name, label in zip(batch_x, batch_y):
            image = cv2.imread(os.path.join(self.root_directory, image_name))
            image = cv2.resize(image, (300, 300))
            img = self.generator(image=image)['image']
            new_images.append(img)
            new_labels.append(label)
        new_images = np.array(new_images)
        new_labels = np.array(new_labels)
        if self.is_mix:
            new_images, new_labels = self.mix_up(new_images, new_labels)
        return new_images, new_labels
