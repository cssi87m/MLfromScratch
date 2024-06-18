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

def cdist(XA, XB, metrics = Distance().Euclidean): 
    dist = np.array([[0 for _ in range(len(XB))] for _ in range(len(XA))])
    print(len(XA), len(XB))
    for i in range(len(XA)): 
        for j in range(len(XB)): 
            print(i, j)
            dist[i][j] = metrics(XA[i], XB[j])
    
    return dist
class KMeans: 
    def __init__(self, k, random_state = 42, distance: str = 'euclidean'): 
        random.seed(random_state)
        np.random.seed(random_state)
        self.k = k 
        if distance == 'euclidean':
            self.distance = Distance().Euclidean
        
        else: 
            self.distance = Distance().Minkowski
    
    def init_centers(self, X): 
        # randomly pick k rows of X as initial centers
        return X[np.random.choice(X.shape[0], self.k, replace=False)]

    def assign_labels(self, X, centers): 
        # calculate pairwise distances btw data and centers
        D = cdist(X, centers) 
        return np.argmin(D, axis = 1)
    
    def update_centers(self, X, labels): 
        centers = np.zeros((self.k, X.shape[1])) 
        for k in range(self.k): 
            # collect all points belonging to k-th cluster, than take average
            Xk = X[labels == k, :]
            centers[k, :] = np.mean(Xk, axis=0)
        
        return centers
    
    def has_converged(centers, new_centers):
        # return True if two sets of centers are the same
        return (set([tuple(a) for a in centers]) == 
            set([tuple(a) for a in new_centers]))
    
    def kmeans(self, X, max_iter = 100, return_iter = True): 
        centers = [self.init_centers(X, self.k)]
        labels = []
        it = 0 
        while True:
            labels.append(self.assign_labels(X, centers[-1]))
            new_centers = self.update_centers(X, labels[-1], self.k)
            if self.has_converged(centers[-1], new_centers) or it == max_iter:
                break
            centers.append(new_centers)
            it += 1
        
        if return_iter: 
            return (centers, labels, it)
        else: 
            return centers, labels