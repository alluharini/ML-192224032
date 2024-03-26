import numpy as np

class EMAlgorithm:
    def __init__(self, num_clusters, num_iterations=100, tol=1e-4):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None
    
    def _initialize_parameters(self, X):
        n_samples, _ = X.shape
        # Initialize means randomly
        random_indices = np.random.choice(n_samples, size=self.num_clusters, replace=False)
        self.means = X[random_indices]
        # Initialize covariances as identity matrices
        self.covariances = [np.eye(X.shape[1]) for _ in range(self.num_clusters)]
        # Initialize weights uniformly
        self.weights = np.ones(self.num_clusters) / self.num_clusters
    
    def _gaussian_pdf(self, X, mean, covariance):
        d = X.shape[1]
        det_cov = np.linalg.det(covariance)
        inv_cov = np.linalg.inv(covariance)
        coeff = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))
        exponent = np.exp(-0.5 * np.sum(np.dot(X - mean, inv_cov) * (X - mean), axis=1))
        return coeff * exponent
    
    def _compute_expectation(self, X):
        n_samples, _ = X.shape
        # Compute responsibilities (E-step)
        responsibilities = np.zeros((n_samples, self.num_clusters))
        for k in range(self.num_clusters):
            responsibilities[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
        # Normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities
    
    def _compute_maximization(self, X, responsibilities):
        n_samples, _ = X.shape
        # Update parameters (M-step)
        total_resp = np.sum(responsibilities, axis=0)
        self.weights = total_resp / n_samples
        for k in range(self.num_clusters):
            # Update mean
            self.means[k] = np.sum(X * responsibilities[:, k].reshape(-1, 1), axis=0) / total_resp[k]
            # Update covariance
            diff = X - self.means[k]
            self.covariances[k] = np.dot((diff * responsibilities[:, k].reshape(-1, 1)).T, diff) / total_resp[k]
    
    def fit(self, X):
        self._initialize_parameters(X)
        prev_likelihood = None
        for i in range(self.num_iterations):
            responsibilities = self._compute_expectation(X)
            self._compute_maximization(X, responsibilities)
            likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            if prev_likelihood is not None and np.abs(likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = likelihood
    
    def predict(self, X):
        responsibilities = self._compute_expectation(X)
        return np.argmax(responsibilities, axis=1)

# Example usage:
if __name__ == "__main__":
    # Generate some sample data from a Gaussian Mixture Model
    np.random.seed(0)
    num_samples = 1000
    cluster_means = np.array([[2, 2], [8, 8]])
    cluster_covariances = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]])
    cluster_weights = np.array([0.5, 0.5])
    labels = np.random.choice(2, size=num_samples, p=cluster_weights)
    data = np.vstack([np.random.multivariate_normal(cluster_means[label], cluster_covariances[label], size=1) for label in labels])
    
    # Fit Gaussian Mixture Model using EM Algorithm
    em_model = EMAlgorithm(num_clusters=2)
    em_model.fit(data)
    predicted_labels = em_model.predict(data)
    
    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
    plt.title('Gaussian Mixture Model (EM Algorithm)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar(label='Cluster')
    plt.show()
