import numpy as np

class LinearRegression(object):

    def __init__(self, xs: list, ys: list, learning_rate: float=0.01, epochs: int=1, normalize: bool=False):
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = np.random.random()
        self.b = np.random.random()
        self.normalize = normalize

    def fit(self):
        
        history = {"coef": self.w, "intercept": self.b, "errors": []}

        for epoch in range(self.epochs):
        
            error = self.mean_squared_error()
            
            history["errors"].append((epoch, int(error)))
            dw = self.derivative_w()
            db = self.derivative_b()
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
        history["coef"] = self.w
        history["intercept"] = self.b

        return history

    def predict(self, value):

        return self.w * value + self.b

    def mean_squared_error(self):
        
        mse = sum([(y - (x*self.w + self.b))**2 for x, y in zip(self.xs, self.ys)])
        
        return mse / len(self.xs)

    def derivative_b(self):

        return (-2 * sum([(y - (x*self.w + self.b)) for x, y in zip(self.xs, self.ys)])) / len(self.xs)

    def derivative_w(self):
        
        return (-2 * sum([(y - (x*self.w + self.b)) * x for x, y, in zip(self.xs, self.ys)])) / len(self.xs)


lr = LinearRegression([30, 40, 50], [40, 50, 60], learning_rate=0.0001, epochs=10)
result = lr.fit()
print(result)
