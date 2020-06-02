from tensorflow.keras import losses


# TODO
class FocalLoss(losses.Loss):
    def __init__(self, gama, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gama
        self.alpha = alpha

    def call(self, y_true, y_pred):
        pass