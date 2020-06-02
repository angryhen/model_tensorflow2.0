import tensorflow as tf


@tf.function
def train_step(x, y, model, loss_func, optimizer, train_loss, train_acc):
    with tf.GradientTape() as tape:
        pre = model(x, training=True)
        losses = loss_func(y, pre)
    gradients = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(losses)
    train_acc.update_state(y, pre)


@tf.function
def valid_step(x, y, model, loss_func, valid_loss, valid_acc):
    pre = model(x, training=False)
    losses = loss_func(y, pre)

    valid_loss.update_state(losses)
    valid_acc.update_state(y, pre)


def train(train_data, valid_data, model, config, loss_func, optimizer,
          train_loss, train_acc, valid_loss, valid_acc, summary_writer):

    for epoch in tf.range(1, config.EPOCHES):
        # print(optimizer._decayed_lr(tf.float32), optimizer.iterations)
        # train
        step = 0
        for x, y in train_data:
            step += 1
            train_step(x, y, model, loss_func, optimizer, train_loss, train_acc)
            tf.print(f'Epoch: {epoch}/ {config.EPOCHES},'
                     f'step: {step} / {config.step_per_epoch},')
                     # f'loss: {train_loss.result.numpy():.5f},'
                     # f'acc: {train_acc.result.numpy():.5f}')

        # valid
        # for x, y in valid_data:
        #     valid_step(x, y, model, loss_func, valid_loss, valid_acc)

        # reset metric
        # train_loss.reset_states()
        # train_acc.reset_states()
        # valid_loss.reset_states()
        # valid_acc.reset_states()

