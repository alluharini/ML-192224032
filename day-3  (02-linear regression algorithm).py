import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            # calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (linear_model - y))
            db = (1 / num_samples) * np.sum(linear_model - y)
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return linear_model
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)  # Mean Squared Error
        return mse

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[2, 3], [1, 2], [3, 4], [5, 6], [4, 5]])
    y_train = np.array([4, 2, 3, 8, 7])
    
    X_test = np.array([[6, 7], [1, 3]])
    y_test = np.array([10, 4])
    
    # Training the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Evaluating the model
    train_mse = lr_model.evaluate(X_train, y_train)
    test_mse = lr_model.evaluate(X_test, y_test)
    
    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)
