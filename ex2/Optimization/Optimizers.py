import numpy as np


class Sgd:
    """
        Stocastic Gradient Descent (SGD), returns next updated tensor values by given learning rate
        y_n+1 = y_n - mu * d/dx.f(n)
         """
    def __init__(self, learning_rate):
        self.learning_rate=learning_rate
    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        weight_tensor += self.v
        return weight_tensor

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.eps = 1e-8
        self.k=0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k+=1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor**2
        vb =self.v/(1-(self.mu**self.k))
        rb= self.r/(1-(self.rho**self.k))

        weight_tensor -= self.learning_rate * (vb / (np.sqrt(rb) + self.eps))
        return weight_tensor
