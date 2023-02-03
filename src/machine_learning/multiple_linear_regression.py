import numpy as np
from typing import Dict

class MultipleLinearRegression():

    def __init__(self, xs: list, ys: list, learning_rate: float=0.01, epochs: int=10):
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.random(size=(1, len(xs[0])))
        self.bias = np.random.random()
        
    def fit(self) -> Dict:
        """Function responsible for fitting the model to the data.

        Returns:
            [Dict] information regarding the adjustment.
        """
        history = {"weights": [], "bias": 0, "errors": []}

        for epoch in range(self.epochs):

            error = self.mean_squared_error()

            history["errors"].append([epoch, error]) 

            new_weights = []
            for weight_idx in range(len(self.weights)):
                 new_weights.append(self.weights[weight_idx] - self.learning_rate * self.derivative_ws(weight_idx))
             
            self.bias = self.bias - self.learning_rate * self.derivative_b()
            self.weights = new_weights
        
        history["weights"] = self.weights[0]
        history["bias"] = self.bias

        return history

    def predict(self, xs: list) -> float:
        """This function receveis a value and return the prediction according to the adjusted weights."""
        return np.dot(xs, self.weights[0]) + self.bias

    def mean_squared_error(self) -> float:
        """Calculates the model cost function using euclidian distance as a loss function.

        Returns:
            [Float] the error value.
        """
        mse = sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias))**2 for i in range(len(self.ys))]) / len(self.xs)

        return mse

    def derivative_ws(self, j: int) -> float:
        """Calculates the gradient descent in relation to the linear coefficient (b).

        Args:
            j[int]: he index of the parameter that is to be updated
            
        Returns:
            [float] the gradient descent value.
        """
        return (-2 * sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias))*self.xs[i][j] for i in range(len(self.ys))])) / len(self.ys)

    def derivative_b(self) -> float:
        """Calculates the gradient descent in relation to the slope coefficient (w).
        
        Returns:
            [float] the gradient descent value.
        """
        return (-2 * sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias)) for i in range(len(self.ys))])) / len(self.ys)
