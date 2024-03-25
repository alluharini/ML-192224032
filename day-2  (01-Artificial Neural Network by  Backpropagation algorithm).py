import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.randn(1, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output
    
    def backward(self, X, y, output, learning_rate):
        # Backpropagation
        error = y - output
        d_output = error
        d_hidden_output = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden_input = d_hidden_output * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias_hidden_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(X.T, d_hidden_input)
        self.bias_input_hidden += learning_rate * np.sum(d_hidden_input, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Example usage:
# Let's use a simple dataset for demonstration
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs, learning_rate)

# Test the trained network
print("\nTesting the trained network:")
for i in range(len(X)):
    input_data = X[i]
    true_output = y[i]
    predicted_output = nn.forward(input_data)
    print(f"Input: {input_data}, True Output: {true_output}, Predicted Output: {predicted_output}")
