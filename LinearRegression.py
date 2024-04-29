import numpy as np 

class LinearRegression: 
    def __init__(self, lr: float = 0.00001, n_iters: int = 10000): 
        self.lr = lr 
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 
    
    def __repr__(self) -> str:
        return self.weights, self.bias

    def fit(self, X: np.ndarray, y: np.ndarray): 
        num_samples, num_features = X.shape # X.shape: m*n, y.shape: m*1
        # initialize self.weights and self.bias
        self.weights = np.random.randn(num_features) 
        self.bias = 0

        for _ in range(self.n_iters): 
            y_pred = np.dot(X, self.weights) + self.bias
            if y_pred.shape != y.shape: 
                y_pred = np.expand_dims(y_pred, axis = 1)
            dw = 1/num_samples*(np.dot(X.T, y_pred - y))
            db = 1/num_samples*(np.sum(y_pred - y)) 

            self.weights -= self.lr * dw 
            self.bias -= self.lr * db
    
    def predict(self, X: np.ndarray): 
        if self.weights is None: 
            num_samples, num_features = X.shape
             # initialize self.weights and self.bias
            self.weights = np.random.randn(num_features) 
            self.bias = 0
        return np.dot(X, self.weights) + self.bias
    
    def loss_func(self, y_true: np.ndarray, y_pred: np.ndarray): 
        # Using mean square error
        num_samples= len(y_true)
        return 1/num_samples*(np.dot((y_true - y_pred).T, (y_true - y_pred)))
    
    # R Square score
    def score(self, X: np.ndarray, y: np.ndarray):
        return 1 - self.loss_func(X, y)/ (y.std() * y.std())
    
# def test(): 
#     X = np.array([[1, 2, 3, 4], [4, 3, 2, 1]]) 
#     y = np.array([9, 4])
#     lin_regr = LinearRegression() 
#     lin_regr.fit(X, y) 
#     y_pred = lin_regr.predict(X) 
#     print(lin_regr.loss_func(y, y_pred))
# if __name__ == "__main__": 
#     test()