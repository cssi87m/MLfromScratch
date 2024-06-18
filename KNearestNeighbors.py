import numpy as np 
import random
from collections import Counter
class Distance: 
    def __init__(self): 
        pass

    def Euclidean(self, X1: np.ndarray, X2: np.ndarray): 
        if X1.shape != X2.shape: 
            return False
        else: 
            diff_square = np.multiply(X2 - X1, X2-X1) 
            return (np.sum(diff_square))**1/2
     
    def Minkowski(self, X1:np.ndarray, X2: np.ndarray, p=2, w = None): 
        if X1.shape != X2.shape: 
            return False
        
        else: 
            diff_vect = np.power(x1 = X1 - X2, x2 = np.array([p for _ in range(len(X1))]))
            return np.sum(diff_vect)**1/p
        
class Algorithm: 
    '''class of algorithm use for compute nearest neighbors'''
    def __init__(self): 
        pass 

    def brute_force(self): 
        pass 

    def BallTree(self): 
        pass 

    def KDtree(self): 
        pass

class KNeighborsClassifier: 
    def __init__(self,n_neighbors=5, distance :str='euclidean', random_state = 42): 
        if distance == 'euclidean': 
            self.distance = Distance.Euclidean()
        else:
            self.distance = Distance.Minkowski()
        
        self.neighbors = n_neighbors
        random.seed(random_state)
    
    def predict(self, X, y, ins): 
        '''
        X: data
        y: label 
        ins: new instance, need to predict 
        '''
        dist = []
        try: 
            for i in range(len(X)): 
                dist.append((y[i], self.distance(X[i], ins)))
            dist.sort(key = lambda x: x[1]) 
            # choose n_neighbors points with smallest distances 
            nearest_points = dist[:self.neighbors]
            labels = [point[0] for point in nearest_points]

            label_counts = Counter(labels)
            # Determine the most common label 
            most_common_label = label_counts.most_common(1)[0][0]
            return most_common_label    
        except ValueError: 
            print("Dimension of new instance is not as same as data.")

class KNeighborsRegression: 
    def __init__(self, n_neighbors = 5, distance: str = 'euclidean'): 
        if distance == 'euclidean': 
            self.distance = Distance.Euclidean()
        else:
            self.distance = Distance.Minkowski()
        self.neighbors = n_neighbors
    
    def fit(self, X): 
        y = [random.random() for _ in range(len(X))]
        for i in range(len(X)):
            dist = []
            for j in range(len(X)): 
                if(i!=j): 
                    dist.append((X[j], y[j], self.distance(X[i], X[j])))
                dist.sort(key = lambda x: x[1])
                nearest_points = dist[:self.neighbors]
                labels = [point[0] for point in nearest_points]
                y[i] = np.mean(labels)
        
        return y 

    def predict(self, X, ins, y = None): 
        dist = []
        if y is None: 
            y = [random.random() for _ in range(len(X))]
        try: 
            for i in range(len(X)): 
                dist.append((y[i], self.distance(X[i], ins)))
            dist.sort(key = lambda x: x[1]) 
            # choose n_neighbors points with smallest distances 
            nearest_points = dist[:self.neighbors]
            labels = [point[0] for point in nearest_points]

            y_pred = np.mean(labels) 
            return y_pred
        except ValueError: 
            print("Dimension of new instance is not as same as data.")
