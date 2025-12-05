import numpy as np

class Loss:
    
    def forward(self, y_pred, y_true):
        """
        Exception
        """
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        """
        Exception
        """
        raise NotImplementedError

class MSE(Loss):
    """
    Mean Squared Error (MSE) Loss.
    Formula: L = (1/N) * sum((y_pred - y_true)^2)
    """
    def forward(self, y_pred, y_true):
        # Calculate mean squared error
        return np.mean(np.power(y_pred - y_true, 2))

    def backward(self, y_pred, y_true):
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        # N is the total number of elements in the input arrays
        return 2 * (y_pred - y_true) / y_pred.size