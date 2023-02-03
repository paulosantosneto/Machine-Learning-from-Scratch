import numpy as np
from typing import Dict

class LogisticRegression():

    def __init__(self, xs: list, ys: list, learning_rate: float=1., epochs: float=100):
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.ws = np.random.random(size=(1, len(self.xs[0]))) 
        self.epochs = epochs
        self.b = np.random.random()


    def fit(self) -> Dict:
        print(self.xs)
        history = {"loss": []}

        for epoch in range(self.epochs):
            loss = self.binary_cross_entropy()
            history["loss"].append(loss)

            new_weights = [] 

            for weight in range(len(self.ws)):
                new_weights.append(self.ws[weight] - self.learning_rate * self.derivative_w(weight))
            
            self.b = self.b - self.learning_rate * self.derivative_b()
            self.ws = new_weights
        self.b = -self.b
       
        return history

    def predict(self, x):
        
        return 1 / (1 + np.exp(-np.dot([x], self.ws[0])+self.b))
    
    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy(self):
        
        loss = []
        for i in range(len(self.ys)):
            y_hat = self.sigmoid(np.dot(self.xs[i], self.ws[0])+self.b)
            y = self.ys[i]

            partial_loss = -(1 - y) * np.log(1 - y_hat) - y * np.log(y_hat)

            loss.append(partial_loss)

        return np.mean(loss)

    def derivative_w(self, w):
        
        dw = []
        for i in range(len(self.xs)):

            z = self.sigmoid(np.dot(self.xs[i], self.ws[0])+self.b)
            y = self.ys[i]
            x = self.xs[i][w]
            
            dw.append((z - y) * x)

        return np.mean(dw)
    
    def derivative_b(self):

        db = []

        for i in range(len(self.xs)):
            z = self.sigmoid(np.dot(self.xs[i], self.ws[0])+self.b)
            y = self.ys[i]

            db.append(z - y)
        
        return np.mean(db) 

#xs = [[0], [1], [3], [5]]
#ys = [0, 0, 1, 1]
#lr = LogisticRegression(xs, ys)
#history = lr.fit()
#print(history)
#print(lr.predict(0))
#print(lr.predict(6))
#print(lr.predict(1))
