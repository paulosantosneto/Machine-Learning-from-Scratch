import numpy as np
from typing import Dict

class LinearRegression(object):

    def __init__(self, xs: list, ys: list, learning_rate: float=0.001, epochs: int=100):
        self.xs = np.array(xs, dtype=np.float32)
        self.ys = np.array(ys, dtype=np.float32)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = np.random.random()
        self.b = np.random.random()

        error_message = f"The number of examples needs to be the same. Does not match: ({self.xs.shape[0]},) != ({self.ys.shape[0]},)."
        
        assert self.xs.shape[0] == self.ys.shape[0], error_message
    
    def verbose(self, history: Dict) -> None:

        print(f"Equation: {self.w:.2f}x + {self.b:.2f}")
        print(f"First error: {history['errors'][0]:.3f} | last error: {history['errors'][-1]:.3f}")

    def fit(self, verbose: bool=False) -> Dict:
        """Function responsible for fitting the model to the data.

        Args:
            verbose[Bool]: when true shows tuning information such as error and parameter values.

        Returns:
            [Dict] information regarding the adjustment.
        """
        history = {"coef": self.w, "intercept": self.b, "errors": []}

        for epoch in range(self.epochs):
        
            error = self.mean_squared_error()
            
            history["errors"].append([epoch, error])

            dw = self.derivative_w()
            db = self.derivative_b()
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
        history["coef"] = self.w
        history["intercept"] = self.b

        if verbose: self.verbose(history)

        return history

    def predict(self, value: float) -> float:
        """This function receveis a value and return the prediction according to the adjusted weights."""
        return self.w * value + self.b

    def mean_squared_error(self) -> float:
        """Calculates the model cost function using euclidian distance as a loss function.

        Returns:
            [Float] the error value.
        """
        mse = sum([(y[0] - self.predict(x[0]))**2 for x, y in zip(self.xs, self.ys)]) / self.xs.shape[0]
        return mse

    def derivative_b(self) -> float:
        """Calculates the gradient descent in relation to the linear coefficient (b).

        Returns:
            [float] the gradient descent value.
        """
        return (-2 * sum([(y[0] - self.predict(x[0])) for x, y in zip(self.xs, self.ys)])) / self.xs.shape[0]

    def derivative_w(self) -> float:
        """Calculates the gradient descent in relation to the slope coefficient (w).
        
        Returns:
            [float] the gradient descent value.
        """
        return (-2 * sum([(y[0] - self.predict(x[0])) * x[0] for x, y, in zip(self.xs, self.ys)])) / self.xs.shape[0]
