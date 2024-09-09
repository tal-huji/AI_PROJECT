import numpy as np

class OnlineNormalization:
    def __init__(self, num_features):
        """
        Initialize with the number of features.
        num_features: the number of features in the state space.
        """
        self.n = 0  # Scalar to track the number of data points across all features
        self.mean = np.zeros(num_features)  # Running mean for each feature
        self.M2 = np.zeros(num_features)  # Running sum of squared differences from the mean

    def update(self, x):
        """
        Update the running mean and variance for each feature.
        x: a numpy array representing the state (features), expected shape (num_features,).
        """
        x = np.asarray(x).flatten()  # Ensure x is a 1D array
        self.n += 1  # Increment the count
        delta = x - self.mean
        self.mean += delta / self.n  # Update the mean for each feature
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self):
        """
        Calculate the current standard deviation for each feature.
        """
        variance = self.M2 / (self.n - 1 + 1e-8)  # Small epsilon to avoid division by zero
        return np.sqrt(variance)

    def normalize(self, x):
        """
        Apply Z-score normalization to the input state.
        x: a numpy array representing the state (features).
        """
        x = np.asarray(x).flatten()  # Ensure x is a 1D array
        return (x - self.mean) / (self.std() + 1e-8)  # Normalize and add epsilon to avoid division by zero
