import numpy as np

class Layer:
    """
    Base Layer class that all other layers will inherit from.

    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Exception
        """
        raise NotImplementedError

    def backward(self, output_gradient):
        """
        Exception
        """
        raise NotImplementedError

class Dense(Layer):
    """
    A fully connected (Dense) layer.
    Equation: Y = X.W + B
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize Weights: Random small values 
        self.weights = np.random.randn(input_size, output_size) * 0.1
        
        # Initialize Biases: Zeros
        self.bias = np.zeros((1, output_size))
        # New: Storage for gradients
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input_data):
        """
        Forward pass: Y = XW + B
        """
        self.input = input_data
        # Matrix multiplication of Input and Weights, plus Bias
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient):
        
        # 1. Calculate Gradients
        # dL/dW = X^T . dL/dY
        #weights_gradient = np.dot(self.input.T, output_gradient)
        
        # dL/dB = sum(dL/dY) 
        #bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        # 1. Calculate Gradients
        # We store them in 'self' so the Optimizer can find them later
        self.grad_weights = np.dot(self.input.T, output_gradient)
        self.grad_bias = np.sum(output_gradient, axis=0, keepdims=True)
        # 2. Calculate Input Gradient (to return to previous layer)
        # dL/dX = dL/dY . W^T
        input_gradient = np.dot(output_gradient, self.weights.T)


        return input_gradient