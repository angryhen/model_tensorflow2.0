from tensorflow.python import keras


def VGG16(classes=1000):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224,224,3)),
        keras.layers.Conv2D(filters=64, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv1_1'),
        keras.layers.Conv2D(filters=64, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv1_2'),
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1'),

        keras.layers.Conv2D(filters=128, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv2_1'),
        keras.layers.Conv2D(filters=128, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv2_2'),
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2'),

        keras.layers.Conv2D(filters=256, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv3_1'),
        keras.layers.Conv2D(filters=256, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv3_2'),
        keras.layers.Conv2D(filters=256, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv3_3'),
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3'),

        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv4_1'),
        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv4_2'),
        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv4_3'),
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4'),

        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv5_1'),
        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv5_2'),
        keras.layers.Conv2D(filters=512, kernel_size=3,
                            strides=1, padding='same',
                            activation='relu', name='conv5_3'),
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same', name='pool5'),

        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(4096, activation='relu', name='Dense1'),
        keras.layers.Dense(4096, activation='relu', name='Dense2'),
        keras.layers.Dense(classes, activation='softmax', name='softmax')
    ])
    return model

if __name__ == '__main__':
    model = VGG16()
    model.summary()
