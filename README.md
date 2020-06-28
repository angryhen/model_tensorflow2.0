# model tensorflow2.x

###  About the tutorial
_____
Can refer to the following URL：
https://www.jianshu.com/u/9f7146c71955



### base_model
________
Various basic models code by tensorflow2.x,now included:
- [x] VGG16,
- [x] ResNet50
- [x] MobileNet_v2
- [x] SE-Resnet

### Update log

- [x] TF-record
- [x] Config-file
- [x] Train
- [ ] Eval
- [ ] Predict
- [ ] Performance

### Prepare Datasets
1. use *split_dataset.py* to split train/valid/test data
```bash
# change path of dataset_dir in config.py's split_dataset
python split_datdaset.py 
```
2. create TF-record
```bash
python create_tfrecord.py
```

### Change param

All parameters are saved in **config.py** , and you can modify it to suit yout training.

### Train

```ba
python train.py
```

### Eval

- [ ] TODO