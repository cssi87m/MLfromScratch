import numpy as np 
'''This code is implemented based on https://www.youtube.com/watch?v=wF5t4Mmv5us'''
class DecisionStump: 
    def __init__(self): 
        self.polarity = 1 
        self.feature_idx = None 
        self.threshold = None 
        self.alpha = 0
    
    def predict(self, X): 
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples) 
        if self.polarity == 1: 
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1 
        
        return predictions
        
'''AdaBoost for binary classifier'''
class AdaBoost: 
    def __init__(self, n_clf = 5): 
        self.n_clf = 5

    def init_algo(self, weights, features): 
        pass 

    def fit(self, X, y): 
        '''Initialize some params'''
        N, features = X.shape
        w = np.full(N, 1/N)
        
        # list to store all classifier 
        self.clfs = []
        for _ in range(self.n_clf): 
            clf = DecisionStump()
            min_error = float('inf') 

            for feature in features: 
                X_column = X[:, feature] 
                thresholds = np.unique(X_column) 
                for threshold in thresholds: 
                    p = 1 
                    predictions = np.ones(N) 
                    predictions[X_column < threshold] = -1 

                    # calculate misclassified weights
                    misclassified = w[y != predictions]
                    error = sum(misclassified) 

                    # if error > 0.5, change error and sign of polarity: 
                    if error > 0.5: 
                        error = 1-error
                        p = -1 
                    
                    # if error < min_error, save classifier
                    if error < min_error: 
                        min_error = error 
                        clf.polarity = p 
                        clf.threshold = threshold
            # calculate performance
            EPS = 1e-10
            clf.alpha = 0.5*np.log((1-error-EPS)/(error + EPS))

            predict = clf.predict(X)
            w = w*np.exp(-clf.alpha*predict*y)/np.sum(w)

            # store the classifier 
            self.clfs.append(clf)

    def predict(self, X):
        clf_pred = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_pred)
        y_pred = np.sign(y_pred) 

        return y_pred