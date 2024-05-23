import numpy as np

class SVM: 
    def __init__(self, learning_rate = 1e-5, lambda_param = 0.01, n_iters = 1000): 
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iters = n_iters 
        self.w: np.ndarray = None 
        self.b = None
    
    def __repr__(self) -> str:
        return self.w, self.b
    
    def poly_kernel(self, d: int, X1: np.ndarray, X2: np.ndarray): 
        mat = X1.T @ X2 + 1
        # pow = np.ones(mat.shape) * d
        return np.power(mat, d)
    def sigmoid_kernel(self, gamma, r, X1: np.ndarray, X2: np.ndarray): 
        return np.tanh(gamma * X1.T @ X2 + r)
    def rbf_kernel(self, gamma, X1: np.ndarray, X2: np.ndarray): 
        return np.exp(-(np.linalg.norm(X1-X2))*gamma)
    

    def fit(self, X: np.ndarray, y: np.ndarray, kernel = None): 
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) 
        self.b = 0 

        y_ = np.where(y <= 0, -1, 1) 
        for _ in range(self.iters): 
            for idx, x_i in enumerate(X): 
                condition = y_[idx]*(np.dot(x_i, self.w)-self.b) >= 1 
                if condition: 
                    self.w -= self.learning_rate*(2*self.lambda_param*self.w) 
                
                else: 
                    self.w -= self.learning_rate*(2*self.lambda_param*self.w - np.dot(x_i, self.w))
                    self.b -= self.learning_rate* y_[idx]
        
    def predict(self, X: np.ndarray): 
        return np.sign(np.dot(self.w @ X) + self.b)