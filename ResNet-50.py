import tensorflow as tf
from tensorflow import keras
import numpy as np
import time


class Block(tf.keras.Model):

    def __init__(self, filters, block_name, downsample=False, stride=1, **kwargs):
        super(Block, self).__init__(**kwargs)

        filter1, filter2, filter3 = filters
        conv_name = 'res' + block_name + '_branch'
        bn_name = 'bn' + block_name + '_branch'
        self.downsample = downsample

        self.conv2a = keras.layers.Conv2D(filters=filter1,
                                          kernel_size=1,
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2a')
        self.bn2a = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')

        self.conv2b = keras.layers.Conv2D(filters=filter2,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2b')
        self.bn2b = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')

        self.conv2c = keras.layers.Conv2D(filters=filter3,
                                          kernel_size=1,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2c')
        self.bn2c = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')

        if self.downsample:
            self.conv_shortcut = keras.layers.Conv2D(filters=filter3,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     name=conv_name + '1')
            self.bn_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name + '1')

    def call(self, inputs, training=False, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs

        x = keras.layers.add([x, shortcut])
        x = tf.nn.relu(x)

        return x


class ResNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = keras.layers.Conv2D(filters=64,
                                         kernel_size=7,
                                         strides=2,
                                         kernel_initializer='glorot_uniform',
                                         name='conv1')
        self.bn_conv1 = keras.layers.BatchNormalization(axis=3, name='bn_conv1')
        self.max_pool = keras.layers.MaxPooling2D((3, 3),
                                                  strides=2,
                                                  padding='same')
        # layer2
        self.res2a = Block(filters=[64, 64, 256], block_name='2a',
                           downsample=True, stride=1)
        self.res2b = Block(filters=[64, 64, 256], block_name='2b')
        self.res2c = Block(filters=[64, 64, 256], block_name='2c')

        # layer3
        self.res3a = Block(filters=[128, 128, 512], block_name='3a',
                           downsample=True, stride=2)
        self.res3b = Block(filters=[128, 128, 512], block_name='3b')
        self.res3c = Block(filters=[128, 128, 512], block_name='3c')
        self.res3d = Block(filters=[128, 128, 512], block_name='3d')

        # layer4
        self.res4a = Block(filters=[256, 256, 1024], block_name='4a',
                           downsample=True, stride=2)
        self.res4b = Block(filters=[256, 256, 1024], block_name='4b')
        self.res4c = Block(filters=[256, 256, 1024], block_name='4c')
        self.res4d = Block(filters=[256, 256, 1024], block_name='4d')
        self.res4e = Block(filters=[256, 256, 1024], block_name='4e')
        self.res4f = Block(filters=[256, 256, 1024], block_name='4f')

        # layer5
        self.res5a = Block(filters=[512, 512, 2048], block_name='5a',
                           downsample=True, stride=2)
        self.res5b = Block(filters=[512, 512, 2048], block_name='5b')
        self.res5c = Block(filters=[512, 512, 2048], block_name='5c')

    def call(self, inputs, training=True, **kwargs):
        start = time.time()
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        # layer1
        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        x = self.res2c(x, training=training)

        x = self.res3a(x, training=training)
        x = self.res3b(x, training=training)
        x = self.res3c(x, training=training)
        x = self.res3d(x, training=training)

        x = self.res4a(x, training=training)
        x = self.res4b(x, training=training)
        x = self.res4c(x, training=training)
        x = self.res4d(x, training=training)
        x = self.res4e(x, training=training)
        x = self.res4f(x, training=training)

        x = self.res5a(x, training=training)
        x = self.res5b(x, training=training)
        x = self.res5c(x, training=training)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(2, activation='softmax', name='fc1000')(x)
        # print('耗时', time.time()-start)
        return x
        
if __name__ == '__main__':
    model = ResNet()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
