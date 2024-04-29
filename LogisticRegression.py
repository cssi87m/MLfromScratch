import numpy as np 
from sklearn.model_selection import train_test_split

class LogisticRegression: 
    def __init__(self, lr = 0.0000001, n_iters = 100000, threshold = 0.5): 
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = 0 
        self.threshold = threshold
    
    def __repr__(self) -> str:
        return self.weights, self.bias

    def sigmoid(self, X: np.ndarray): 
        return 1/(1+ np.exp(-X))
    
    def loss_fnc(self, y_true: np.ndarray, y_pred: np.ndarray): 
        # binary cross entropy loss
        num_samples= len(y_true)
        return -1/num_samples*(np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)))
    def predict(self, X: np.ndarray): 
        if self.weights is None: 
            num_samples, num_features = X.shape
            self.weights = np.random.randn(num_features)
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    def fit(self, X: np.ndarray, y: np.ndarray): 
        num_samples, num_features = X.shape
        if self.weights is None: 
            self.weights = np.random.randn(num_features)
            self.bias = 0

        for _ in range(self.n_iters):
            # calc gradient of binary cross entropy func 
            y_pred = self.predict(X) 
            if y_pred.shape != y.shape: 
                y_pred = np.expand_dims(y_pred, axis = 1)
            # print((y_pred-y).shape)
            # print(X.T.shape)
            dw = 1/num_samples*(np.dot(X.T, y_pred - y)) 
            db = 1/num_samples*(np.sum(y_pred-y)) 
            dw = dw.reshape(self.weights.shape)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
        
    def accuracy(self, y_pred: np.ndarray, y:np.ndarray): 
        num_samples = len(y) 
        num_acc = sum((y_pred[i] >= self.threshold) == y[i] for i in range(num_samples))
        return num_acc/num_samples

def test(): 
    import pandas as pd 
    df = pd.read_csv("ML-algo/diabetes.csv")
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values

    X_train, X_test, Y_train, Y_test = train_test_split( 
    X, Y, random_state = 42) 

    model = LogisticRegression()
    model.fit(X_train, Y_train) 

    y_pred = model.predict(X_test) 
    print(model.accuracy(y_pred, Y_test))

if __name__ == "__main__": 
    test()