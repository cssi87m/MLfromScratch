import numpy as np
from DecisionTree import DecisionTree
def random_sampling(X, y, size): 
    s = X.shape[1]
    random_idx = np.random.choice(s, size)
    new_X = np.array(X[random_idx, :]) 
    new_y = np.array(y[random_idx]) 
    return new_X, new_y

class RandomForest: 
    def __init__(self, n_data = 50, num_trees = 5, n_features = 2, max_depth = 20, min_samples_split = 2, total_features = 50): 
        self.num_trees = num_trees 
        self.n_data = n_data 
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.total_features = total_features
        self.trees:list[DecisionTree] = []
    
    def fit(self, X, y):
        n_data = min(self.n_data, X.shape[0]) 
        total_features = min(self.total_features, X.shape[1])
        for _ in range(self.num_trees): 
            # random sampling n samples of data
            new_X, new_y = random_sampling(X, y, n_data)
            # random sampling k attributes
            att = np.random.choice(total_features, self.n_features)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=att)
            tree.fit(new_X, new_y)
            self.trees.append(tree)

    def predict(self, ins): 
        y = [tree.predict(ins) for tree in self.trees]
        return np.mean(y)