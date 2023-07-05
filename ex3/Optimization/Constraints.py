import numpy as np


class L2_Regularizer:
    """
    L2 regularization
    """

    def __init__(self, alpha):
        """Constructor

        Args:
            alpha (float/tensor): λ (lamda) i.e regularization weight
        """
        self.alpha = alpha

    def calculate_gradient(self, weight_tensor):
        """calculate regularization weights for given weight tensor

        Args:
            weight_tensor (tensor):

        Returns:
            tensor: The regularization gradient for the given weight tensor.
        """

        return self.alpha * weight_tensor

    def norm(self, weight_tensor):
        """calculate L2 norm enhanced loss.

        Args:
            weight_tensor (tensor):

        Returns:
            float: norm enhanced regularization loss
        """
        return self.alpha * np.sum(np.square(weight_tensor))


class L1_Regularizer:
    def __init__(self, alpha):
        """Constructor

        Args:
            alpha (float/tensor): λ (lamda) i.e regularization weight
        """
        self.alpha = alpha

    def calculate_gradient(self, weight_tensor):
        """calculate regularization weights for given weight tensor

        Args:
            weight_tensor (tensor):

        Returns:
            tensor: The regularization gradient for the given weight tensor.
        """
        return self.alpha * np.sign(weight_tensor)

    def norm(self, weight_tensor):
        """calculate L1 norm enhanced loss.

        Args:
            weight_tensor (tensor):

        Returns:
            float: norm enhanced regularization loss
        """

        return self.alpha * np.sum(np.abs(weight_tensor))