import tensorflow as tf


def data_augment(image):
    image = tf.image.random_flip_left_right(image=image)
    image = tf.image.random_brightness(image=image, max_delta=0.3)
    return image
