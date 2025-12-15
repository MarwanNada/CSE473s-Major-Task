class Network:
    """
    The Neural Network container.
    It holds a list of layers and manages the training process.
    """
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        #Adds a layer to the network.
       
        self.layers.append(layer)

    def use(self, loss_function, optimizer=None):
        #Sets the loss function to use.

        self.loss_function = loss_function
        self.optimizer = optimizer

    def predict(self, input_data):        
        # We need to loop through layers to get the final output
        # The input to the next layer is the output of the previous one
        
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        return output

    def train(self, x_train, y_train, epochs):
       
        for i in range(epochs):
            # 1. Forward Pass
            output = self.predict(x_train)

            # 2. Compute Loss 
            loss = self.loss_function.forward(output, y_train)

            # 3. Backward Pass
            # calculate the gradient of the loss function
            error_gradient = self.loss_function.backward(output, y_train)

            # We iterate in reverse order
            for layer in reversed(self.layers):
                error_gradient = layer.backward(error_gradient)

            # Optimizer Step (Update Weights)
            if self.optimizer:
                self.optimizer.step(self.layers)

            # Print status every 1000 epochs
            if (i + 1) % 1000 == 0:
                print(f"Epoch {i+1}/{epochs}, Error: {loss:.6f}")