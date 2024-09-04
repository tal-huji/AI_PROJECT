import numpy as np
from math import sqrt

class OnlineNormalization:
    def __init__(self, num_features):
        """
        Initialize with the number of features.
        num_features: the number of features in the state space.
        """
        self.n = np.zeros(num_features)  # To track the number of data points for each feature
        self.mean = np.zeros(num_features)  # Running mean for each feature
        self.M2 = np.zeros(num_features)  # Running sum of squared differences from the mean

    def update(self, x):
        """
        Update the running mean and variance for each feature.
        x: a numpy array representing the state (features).
        """
        self.n += 1  # Increment the count for each feature
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self):
        """
        Calculate the current standard deviation for each feature.
        """
        variance = self.M2 / (self.n - 1 + 1e-8)  # Use small epsilon to avoid division by zero
        return np.sqrt(variance)

    def normalize(self, x):
        """
        Apply Z-score normalization to the input state.
        x: a numpy array representing the state (features).
        """
        return (x - self.mean) / (self.std() + 1e-8)  # Add epsilon to avoid division by zero
