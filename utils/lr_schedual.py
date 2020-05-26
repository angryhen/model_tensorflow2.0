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
                                             (1 + np.cos(np.pi * (step - self.warmup_steps)
                                                         / float(self.total_steps - self.warmup_steps))))
        if self.warmup_steps > 0:
            # linear increase
            slope = (self.lr - self.warmup_init_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_init_rate
            learning_rate = np.where(step < self.warmup_steps,
                                     warmup_rate, learning_rate)
        return learning_rate


if __name__ == '__main__':
    lr = CustomSchedule(lr=0.01, total_steps=30000, warmup_steps=300, min_lr=0.003)
    plt.plot(lr(tf.range(30000, dtype=tf.float32)))
    plt.xlabel('learning rate')
    plt.ylabel('train step')
    plt.show()
