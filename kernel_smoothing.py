"""
implement and evaluate various kernel smoothing techniques
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1001)
X = np.random.uniform(size=100)
X.sort()
def f(x):
    return np.sin(5*x)
# Generate noisy observations
eps = np.random.normal(scale=0.25, size=100)

Y_observed = f(X) + eps
unit_interval = np.linspace(0,1,1001)
Y_true = f(unit_interval)

plt.plot(unit_interval, Y_true, c='red', label="True values")
plt.scatter(X, Y_observed, marker='.', c='green', label="Observations")
ax = plt.gca()

# Generate a matrix of distances?
distances = abs(unit_interval[:,np.newaxis] - X)
def knn(linspace, X, observations, k=10):
    distance_matrix = abs(linspace[:,np.newaxis] - X)
    nearest_neighbors = np.argpartition(distance_matrix, k)[:,:k]
    knn_vals = observations[nearest_neighbors]
    knn_means = np.mean(knn_vals, axis=1)
    return knn_means

knn_regr = knn(unit_interval, X, Y_observed, k=10)
plt.plot(unit_interval, knn_regr, c='blue', label="K=10 nearest neighbors")

ax.legend()
plt.show()
