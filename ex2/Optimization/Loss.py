import numpy as np
class CrossEntropyLoss:
    """
       Cross Entropy Loss layer for the Neural-Network, which implements a forward
       and backward pass taking predictions and labels to return CE loss
       """
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        """
               loss = Σ -In(predictions + epsilon)
               """
        self.prediction_tensor = prediction_tensor
        loss = np.sum(-label_tensor * np.log(prediction_tensor + np.finfo(prediction_tensor.dtype).eps))
        return loss

    def backward(self, label_tensor):
        """
                error_tensor = - (labels / predictions + epsilon)
                """
        gradient = -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
        return gradient
