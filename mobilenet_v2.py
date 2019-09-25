from tensorflow import keras


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2) // divisor*divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _Inverted_res_block(keras.Model):
    def __init__(self, expansion, stride, alpha, filters, block_id, **kwargs):
        super(_Inverted_res_block, self).__init__(**kwargs)

        self.list = [0] + [16] + [24] *2 + [32] * 3 + [64] *4 + [96] * 3 + [160] * 3 + [320]
        self.stride = stride
        self.block_id = block_id
        self.pointwise_conv_filters = int(filters * alpha)
        self.pointwise_filters = _make_divisible(self.pointwise_conv_filters, 8)
        self.prefix = 'block_{}_'.format(block_id)
        self.expansion = expansion
        self.expand_conv = keras.layers.Conv2D(self.expansion * self.list[self.block_id],
                                               kernel_size=1,
                                               padding='same',
                                               use_bias=False,
                                    name=self.prefix+'expand')
        self.expand_bn = keras.layers.BatchNormalization(axis=-1,
                                                         momentum=0.999,
                                                         name=self.prefix+'expand_BN')
        self.expand_relu = keras.layers.ReLU(6., name=self.prefix+'expand_relu')
        self.block_pad = keras.layers.ZeroPadding2D(name=self.prefix+'pad')
        self.block_dw = keras.layers.DepthwiseConv2D(kernel_size=3,
                                     strides=self.stride,
                                     use_bias=False,
                                     padding='same' if self.stride == 1 else 'valid',
                                     name=self.prefix + 'depthwise')
        self.dw_bn = keras.layers.BatchNormalization(axis=-1,
                                                     momentum=0.999,
                                                     name=self.prefix+'dw_bn')
        self.block_relu = keras.layers.ReLU(6., name=self.prefix + 'depthwise_relu')

        self.project_conv = keras.layers.Conv2D(self.pointwise_filters,
                            kernel_size=1,
                            padding='same',
                            use_bias=False,
                            name=self.prefix + 'project')
        self.project_bn = keras.layers.BatchNormalization(axis=-1,
                                        momentum=0.999,
                                        name=self.prefix + 'project_bn')

    def call(self, inputs, **kwargs):
        x = inputs
        if self.block_id:

            x = self.expand_conv(inputs)
            x = self.expand_bn(x)
            x = self.expand_relu(x)
        else:
            self.prefix = 'expanded_conv_'

        # depthwise
        if self.stride == 2:
            x = self.block_pad(x)
        x = self.block_dw(x)
        x = self.dw_bn(x)
        x = self.block_relu(x)

        # project
        x = self.project_conv(x)
        x = self.project_bn(x)
        if inputs[-1] == self.pointwise_filters and self.stride == 1:
            x = keras.layers.Add(name=self.prefix+'add')([inputs, x])
        return x

class Mbnet_v2(keras.Model):
    def __init__(self, alpha=1, **kwargs):
        super(Mbnet_v2, self).__init__(**kwargs)
        self.alpha = alpha
        self.Padding = keras.layers.ZeroPadding2D((1,1), name='conv1_pad')
        self.conv1 = keras.layers.Conv2D(filters=32,
                                         kernel_size=3,
                                         strides=2,
                                         padding='valid',
                                         use_bias=False,
                                         name='conv1')
        self.bn1 = keras.layers.BatchNormalization(axis=-1,
                                                   momentum=0.999,
                                                   name='bn_conv1')
        self.relu1 = keras.layers.ReLU(6., name='conv1_relu')

        self.block0 = _Inverted_res_block(expansion=1, filters=16, stride=1, alpha=alpha, block_id=0)

        self.block1 = _Inverted_res_block(expansion=6, filters=24, stride=2, alpha=alpha, block_id=1)
        self.block2 = _Inverted_res_block(expansion=6, filters=24, stride=1, alpha=alpha, block_id=2)

        self.block3 = _Inverted_res_block(expansion=6, filters=32, stride=2, alpha=alpha, block_id=3)
        self.block4 = _Inverted_res_block(expansion=6, filters=32, stride=1, alpha=alpha, block_id=4)
        self.block5 = _Inverted_res_block(expansion=6, filters=32, stride=1, alpha=alpha, block_id=5)

        self.block6 = _Inverted_res_block(expansion=6, filters=64, stride=2, alpha=alpha, block_id=6)
        self.block7 = _Inverted_res_block(expansion=6, filters=64, stride=1, alpha=alpha, block_id=7)
        self.block8 = _Inverted_res_block(expansion=6, filters=64, stride=1, alpha=alpha, block_id=8)
        self.block9 = _Inverted_res_block(expansion=6, filters=64, stride=1, alpha=alpha, block_id=9)

        self.block10 = _Inverted_res_block(expansion=6, filters=96, stride=1, alpha=alpha, block_id=10)
        self.block11 = _Inverted_res_block(expansion=6, filters=96, stride=1, alpha=alpha, block_id=11)
        self.block12 = _Inverted_res_block(expansion=6, filters=96, stride=1, alpha=alpha, block_id=12)

        self.block13 = _Inverted_res_block(expansion=6, filters=160, stride=2, alpha=alpha, block_id=13)
        self.block14 = _Inverted_res_block(expansion=6, filters=160, stride=1, alpha=alpha, block_id=14)
        self.block15 = _Inverted_res_block(expansion=6, filters=160, stride=1, alpha=alpha, block_id=15)

        self.block16 = _Inverted_res_block(expansion=6, filters=320, stride=1, alpha=alpha, block_id=16)

        self.last_conv = keras.layers.Conv2D(1280,
                                             kernel_size=1,
                                             use_bias=False,
                                             name='last_conv')
        self.last_bn = keras.layers.BatchNormalization(momentum=0.999,
                                                       name='last_bn')
        self.last_relu = keras.layers.ReLU(6., name='last_relu')

        # top
        self.pooling = keras.layers.GlobalAveragePooling2D()
        self.Dense = keras.layers.Dense(2,activation='softmax',
                                        use_bias=True,
                                        name='Logits')


    def call(self, inputs, **kwargs):
        alpha = self.alpha
        x = self.Padding(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block0(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.block16(x)

        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.last_relu(x)

        x = self.pooling(x)
        x = self.Dense(x)
        return x

if __name__ == '__main__':
    model = Mbnet_v2(alpha=1)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()



