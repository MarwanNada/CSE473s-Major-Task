class SGD:
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    Responsible for updating the weights of the network.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Iterates through the network layers and updates weights/biases
        if gradients have been computed.
        """
        # CRITICAL: This loop must be indented INSIDE the step function
        for layer in layers:
            # Only update layers that have learnable parameters (like Dense)
            if hasattr(layer, 'weights'):
                # W = W - eta * dW
                layer.weights -= self.learning_rate * layer.grad_weights
                
                # B = B - eta * dB
                layer.bias -= self.learning_rate * layer.grad_bias