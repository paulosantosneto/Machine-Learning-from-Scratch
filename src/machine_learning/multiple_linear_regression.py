import numpy as np

class MultipleLinearRegression():

    def __init__(self, xs: list, ys: list, learning_rate: float=0.01, epochs: int=10):
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.random(size=(1, len(xs[0])))
        self.bias = np.random.random()
        
    def fit(self):

        history = {"weights": [], "bias": 0, "errors": []}

        for epoch in range(self.epochs):

            error = self.mean_squared_error()

            history["errors"].append(error) 

            new_weights = []
            for weight_idx in range(len(self.weights)):
                 new_weights.append(self.weights[weight_idx] - self.learning_rate * self.derivative_ws(weight_idx))
             
            self.bias = self.bias - self.learning_rate * self.derivative_b()
            self.weights = new_weights
        
        history["weights"] = self.weights[0]
        history["bias"] = self.bias

        return history

    def predict(self, xs: list):

        return np.dot(xs, self.weights[0]) + self.bias

    def mean_squared_error(self):
        
        mse = sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias))**2 for i in range(len(self.ys))]) / len(self.xs)

        return mse

    def derivative_ws(self, j: int):

        return (-2 * sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias))*self.xs[i][j] for i in range(len(self.ys))])) / len(self.ys)

    def derivative_b(self):

        return (-2 * sum([(self.ys[i] - (np.dot(self.xs[i], self.weights[0]) + self.bias)) for i in range(len(self.ys))])) / len(self.ys)
