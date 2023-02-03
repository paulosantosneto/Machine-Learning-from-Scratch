import numpy as np
from collections import Counter

class KNearestNeighbor(object):

    def __init__(self, n_neighbors: int=3):
        self.xs = None
        self.ys = None
        self.n_neighbors = n_neighbors

    def fit(self, xs: list, ys: list):

        self.xs = np.array(xs)
        self.ys = np.array(ys)
    
    def predict(self, value: list):

        value = np.array(value)
        distances = [(self.euclidian_distance(sample, value), self.ys[i]) for i, sample in enumerate(self.xs)]
        distances = sorted(distances, key=lambda x: x[0])[:self.n_neighbors]
        _, labels = zip(*distances)

        return Counter(labels).most_common(1)[0][0]
        
    def euclidian_distance(self, sample: list, value: list):
        print(sample) 
        print(value)
        return sum([(sample[i] - value[i]) ** 2 for i in range(len(sample))]) ** (1/2) 

#xs = [[1, 2], [2, 2], [3, 3], [3, 4], [5, 6], [6, 6]]
#ys = [0, 0, 1, 1, 2, 2]
#knn = KNearestNeighbor(n_neighbors=3)
#knn.fit(xs, ys)
#print(knn.predict([6, 6]))
