import numpy as np
import tensorflow as tf

from base_model import select_model
from data.parse_tfrecord import _parse_record
from utils.lr_schedual import CustomSchedule, test
from utils.train_core import train, train_step, valid_step
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
    print(f'train:{len_train}, valid: {len_valid}')

    train_dataset = train_dataset.shuffle(len_train // 2).batch(config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE)

    return train_dataset, valid_dataset, len_train, len_valid


def main():
    # dataset train/valid
    train_dataset, valid_dataset, l_t, l_v = get_datasets()
    config.step_per_epoch = np.ceil(l_t / config.BATCH_SIZE)
    config.total_step = config.step_per_epoch * config.EPOCHES
    config.warmup_steps = config.step_per_epoch * config.WARMUP_EPOCH

    # model
    model = select_model.model()

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()  # [1, 2] -> [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    lr_schedule = CustomSchedule(lr=config.LEARNING_RATE,
                                 total_steps=config.total_step,
                                 warmup_steps=config.warmup_steps,
                                 min_lr=config.END_LR_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=5)
    summary_writer = tf.summary.create_file_writer('./logs')

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

    valid_loss = tf.keras.metrics.Mean(name='val_loss')
    valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    # train
    # train(train_dataset, valid_dataset, model, config, loss, optimizer,
    #       train_loss, train_acc, valid_loss, valid_acc, summary_writer)

    for epoch in tf.range(1, config.EPOCHES):
        # train
        step = 0
        for x, y in train_dataset:
            step += 1
            train_step(x, y, model, loss, optimizer, train_loss, train_acc)
            tf.print(f'Epoch: {epoch}/ {config.EPOCHES},'
                     f'step: {step} / {config.step_per_epoch},'
                     f'loss: {train_loss.result():.5f},'
                     f'acc: {train_acc.result():.5f}')

        # valid
        for x, y in valid_dataset:
            valid_step(x, y, model, loss, valid_loss, valid_acc)
        tf.print(f'loss: {valid_loss.result():.5f},'
                 f'acc: {valid_acc.result():.5f}')

        # reset metric
        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()


if __name__ == '__main__':
    main()
