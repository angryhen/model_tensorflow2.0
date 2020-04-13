import tensorflow as tf
import numpy as np


a = tf.Variable(0)
@tf.function
def add(iter):
    a.assign_add(next(iter))
    tf.print("a", a)

iter = iter([0,1,2,3,])
add(iter)
add(iter)
add(iter)
add(iter)