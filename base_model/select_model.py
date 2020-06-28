from base_model import ResNet, se_resnet

def model():
    model = se_resnet.resnet50()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    return model