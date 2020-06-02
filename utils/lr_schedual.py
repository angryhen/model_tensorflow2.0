import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    warmup + cosinedecay
    '''
    def __init__(self, lr, total_steps, warmup_init_rate=0.000001, warmup_steps=4000, min_lr=0.):
        super(CustomSchedule, self).__init__()
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_init_rate = warmup_init_rate
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        if self.total_steps < self.warmup_steps:
            raise ValueError(f'total_step: {self.total_steps} is smaller than warmup_steps: {self.warmup_steps}')
        if self.lr < self.warmup_init_rate:
            raise ValueError(f'init_lr: {self.lr} is smaller than warmup_init_lr: {self.warmup_init_rate}')

    def __call__(self, step):
        learning_rate = self.min_lr + 0.5 * ((self.lr - self.min_lr) *
                                             (1 + tf.math.cos(3.141592653589793 * (step - self.warmup_steps)
                                                         / float(self.total_steps - self.warmup_steps))))
        if self.warmup_steps > 0:
            # linear increase
            slope = (self.lr - self.warmup_init_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_init_rate
            learning_rate = tf.where(step < self.warmup_steps,
                                     warmup_rate, learning_rate)
        return learning_rate


class test(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=10):
        super(test, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model':self.d_model,
            'warmup_steps':self.warmup_steps

        }
        base_config = super(test, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    lr = CustomSchedule(lr=0.001, total_steps=560, warmup_steps=28, min_lr=0.0001)
    plt.plot(lr(tf.range(560, dtype=tf.float32)))
    plt.xlabel('learning rate')
    plt.ylabel('train step')
    plt.show()
