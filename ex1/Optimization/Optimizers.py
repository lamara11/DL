
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
