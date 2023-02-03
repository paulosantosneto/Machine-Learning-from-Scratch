import numpy as np

class PolynomialRegression():

    def __init__(self, xs: list, ys: list, learning_rate: float=0.01, epochs=100):
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.random((1, len(xs[0])))
        self.bias = np.random.random()

    def f(self, xs: list, idx: int):

        return sum([self.weights[m]*xs[idx]**(m+1) for m in range(len(self.weights))]) + self.bias

    def mean_squared_error(self):
        
        mse = sum([(self.ys[i] -self.f(self.xs[i], i))**2 for i in range(len(self.ys))]) 
        
        return mse

pr = PolynomialRegression([[1, 2], [3, 7]], [5, 12])
print(pr.mean_squared_error())
