import numpy as np
import copy

class Node():

    def __init__(self, feature: any, x: list, y: list):
        self.feature = feature
        self.left = None
        self.right = None
        self.x = None
        self.y = None

class DecisionTree():

    def __init__(self, X: list, y: list, labels: list):
        self.X = np.array(X)
        self.y = np.array(y)
        self.entropy = {}
        self.root = None
        self.num_features = len(self.X.T)
        self.labels = labels
        self.labels_right = copy.deepcopy(labels)

    def fit(self, X: list, y: list, root: any=None, threshold: float=0.5, depth: int=4):
        ''' 
        for feature in range(self.num_features):
            entropy = self.calc_entropy(X, y)

            choose = max(entropy, key=lambda x: x[1])
            print(choose)

            right_x = np.delete(X, choose[2], 0)
            left_x = np.delete(X, choose[3], 0)

            right_x = np.delete(right_x, choose[0], 1) 
            left_x = np.delete(left_x, choose[0], 1)

            print(right_x)
            print(left_x)
        '''
        entropy = self.calc_entropy(X, y)
        choose = max(entropy, key=lambda x: x[1])
        label = self.labels[choose[0]]
        print(label)
        if len(X.T) == 1:
            label = 'end'
        else:
            del self.labels[choose[0]] 

        right_x = np.delete(X, choose[2], 0)
        left_x = np.delete(X, choose[3], 0)

        right_x = np.delete(right_x, choose[0], 1)
        left_x = np.delete(left_x, choose[0], 1)

        if self.root is None:
            self.root = Node(label, X, y)
            new_y = np.delete(y, choose[3], 0)
            print('root left')
            self.fit(root=self.root, X=left_x, y=new_y) 
            new_y = np.delete(y, choose[2], 0)
            print('root right')
            self.labels = self.labels_right
            del self.labels[self.labels.index(self.root.feature)]
            print(self.labels)
            self.root.right = None
            self.fit(root=self.root, X=right_x, y=new_y)
        else:
            if root.left == None:
                print('left')
                new_y = np.delete(y, choose[3], 0)
                root.left = Node(label, X, y)
                if len(new_y) >= 3:

                    self.fit(root=root.left, X=left_x, y=new_y)
            
            if root.right == None:
                print('right')
                new_y = np.delete(y, choose[2], 0)
                if len(new_y) < 3:
                    label = 'end'
                root.right = Node(label, X, y)
                print(X)
                print(y)
                print(left_x)
                print(right_x)
                print(new_y)
                if len(new_y) >= 3:
                    
                    self.fit(root=root.right, X=right_x, y=new_y)
                

    def calc_entropy(self, x: list, y: list):
        transpose = x.T
        entropy = []
        print("-"*50)
        for idx, feature in enumerate(transpose):
            # sets
            len_samples = len(feature)
            
            left_idxs = [idx for idx in range(len_samples) if feature[idx] == 1]
            right_idxs = [idx for idx in range(len_samples) if feature[idx] == 0]
            
            d_left = sum(feature)
            d_right = len_samples - d_left
            p1_left = (feature @ y) / d_left
            p1_right = (abs(feature - 1) @ y) / d_right
            p1_root = sum(y) / len_samples
            # calculate entropy
            h_left = self.H(p1_left)
            h_right = self.H(p1_right)
            h_root = self.H(p1_root)
            # choosing a split
            gain = h_root - ((d_left / len_samples) * h_left + (d_right / len_samples) * h_right)
            print(f"Feature {idx} | gain: {gain:.2f}")
            entropy.append((idx, gain, left_idxs, right_idxs))
        
        print("-"*50)
         
        return entropy
       
    def search(self, root):
        if root == self.root:
            print(root.feature)
            
        if root.left is not None:
            print('left')
            print(root.left.feature)
            self.search(root.left)

        if root.right is not None:
            print('right')
            print(root.right.feature)
            self.search(root.right)

    def H(self, p: float):
        
        return max(-p*np.log2(p + 1e-3) - (1 - p)*np.log2(1 - p + 1e-3), 0)




X = [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0],
        [0, 1, 0], [0, 1, 0]]
y = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0]
labels = ['ear', 'face', 'whiskers']
dt = DecisionTree(X, y, labels)
dt.fit(X=dt.X, y=dt.y)
#print(dt.root)
#print('search')
#dt.search(dt.root)
