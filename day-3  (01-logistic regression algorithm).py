import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)
            
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[2, 3], [1, 2], [3, 4], [5, 6], [4, 5]])
    y_train = np.array([0, 0, 0, 1, 1])
    
    X_test = np.array([[6, 7], [1, 3]])
    y_test = np.array([1, 0])
    
    # Training the model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    # Evaluating the model
    train_accuracy = lr_model.evaluate(X_train, y_train)
    test_accuracy = lr_model.evaluate(X_test, y_test)
    
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
