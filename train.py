import numpy as np
import tensorflow as tf

from base_model import select_model
from data.parse_tfrecord import _parse_record
from utils.lr_schedual import CustomSchedule
import config

# set gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu = gpus[0]
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logical_gpu = tf.config.experimental.list_logical_devices('GPU')
        tf.print(f'Physical GPUs: {len(gpus)}, Logical GPU: {len(logical_gpu)}')
    except RuntimeError as e:
        print(e)

def get_datasets():
    train_dataset = _parse_record(config.train_tfrecord)
    valid_dataset = _parse_record(config.valid_tfrecord)

    # get length
    len_train, len_valid = 0, 0
    for _ in train_dataset:
        len_train += 1
    for _ in valid_dataset:
        len_valid += 1

    train_dataset = train_dataset.shuffle().batch(config.batch_size)
    valid_dataset = valid_dataset.batch(config.batch_size)

    return train_dataset, valid_dataset, len_train, len_valid


def main():
    # dataset train/valid
    train_dataset, valid_dataset, l_t, l_v = get_datasets()
    step_per_epoch = np.ceil(l_t/config.BATCH_SIZE)
    total_step = step_per_epoch * config.EPOCHES


    # model
    model = select_model.model()

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()  # [0, 1, 2] -> [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    lr_schedule = CustomSchedule(lr=config.LEARNING_RATE,
                                 total_steps=total_step,
                                 warmup_steps=300,
                                 min_lr=0.003)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=5)



if __name__ == '__main__':
    pass


