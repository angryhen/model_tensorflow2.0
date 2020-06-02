
# prepare datasets
data_dir = 'dataset'
tfrecord_dir = 'data/tfrecord/'
train_tfrecord = tfrecord_dir + 'train.tfrecord'
valid_tfrecord= tfrecord_dir + 'valid.tfrecord'
test_tfrecord = tfrecord_dir + 'test.tfrecord'

SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
END_LR_RATE = 0.00001
EPOCHES = 100
WARMUP_EPOCH = 1

TRAIN_SET_RATIO = 0.8
TEST_SET_RATIO = 0.0
split_dataset = '/home/du/disk2/Desk/dataset/ibox/cls/ibox_c15'