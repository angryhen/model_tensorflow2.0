
# prepare datasets
data_dir = '/home/du/Desktop/dataset/ibox/cls'
tfrecord_dir = 'tfrecord/'
train_tfrecord = tfrecord_dir + 'train.tfrecord'
valid_tfrecord= tfrecord_dir + 'valid.tfrecord'
test_tfrecord = tfrecord_dir + 'test.tfrecord'

SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHES = 20
WARMUP_EPOCH = 1