import numpy as np
from lib.layers import Layer

class Activation(Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient):
        """
        The backward pass:
        dL/dX = dL/dY * f'(X)
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))

    def activation(self, x):
        raise NotImplementedError

    def activation_prime(self, x):
        raise NotImplementedError

'''
class Sigmoid(Activation):
   
    def activation(self, x):
        #To avoid overflow/underflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def activation_prime(self, x):
        # f'(x) = f(x) * (1 - f(x))
        s = self.activation(x)
        return s * (1 - s)
'''
class Sigmoid(Activation):
    def activation(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)


class Tanh(Activation):
    
    def activation(self, x):
        return np.tanh(x)

    def activation_prime(self, x):
        # f'(x) = 1 - f(x)^2
        t = np.tanh(x)
        return 1 - t ** 2


class ReLU(Activation):
    
    def activation(self, x):
        return np.maximum(0, x)

    def activation_prime(self, x):
        # f'(x) = 1 if x > 0 else 0
        return (x > 0).astype(float)


class Softmax(Layer):
   
    def forward(self, input_data):
        self.input = input_data
        # Subtract max for numerical stability (prevents exp overflow)
        tmp = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        
        # dL/dX = Y * (dL/dY - sum(dL/dY * Y))
        
        # 1. Calculate the dot product of Gradient and Output for each sample
        dot_product = np.sum(output_gradient * self.output, axis=1, keepdims=True)
        
        # 2. Subtract that scalar from the original gradient
        # 3. Multiply element-wise by the output
        input_gradient = self.output * (output_gradient - dot_product)
        
        return input_gradient