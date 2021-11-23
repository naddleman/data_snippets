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


# Generate a matrix of distances?
def dmatrix(linspace, obs_points):
    return abs(linspace[:,np.newaxis] - obs_points)

#distances = abs(unit_interval[:,np.newaxis] - X)
def knn(linspace, X, observations, k=10):
    #distance_matrix = abs(linspace[:,np.newaxis] - X)
    distance_matrix = dmatrix(linspace, X)
    nearest_neighbors = np.argpartition(distance_matrix, k)[:,:k]
    knn_vals = observations[nearest_neighbors]
    knn_means = np.mean(knn_vals, axis=1)
    return knn_means

knn_regr = knn(unit_interval, X, Y_observed, k=10)
knn_regr30 = knn(unit_interval, X, Y_observed, k=30)

def epanechnikov(linspace, X, observations, lam=0.2):
    """From Elements of Statistical Learning, pg 193"""
    def D(mat):
        out_matrix = np.empty(mat.shape)
        mask = mat > 1
        out_matrix = np.where(mask, 0, 3/4 * (1-mat**2))
        return out_matrix
    distances = dmatrix(linspace, X)
    K = D(distances / lam)
    val = (K * Y_observed).sum(axis=1)
    normalizer = K.sum(axis=1)
    return val / normalizer

epan_lam_2 = epanechnikov(unit_interval, X, Y_observed)
epan_lam_1 = epanechnikov(unit_interval, X, Y_observed, lam=0.1)

fig, axs = plt.subplots(2)
for ax in axs:
    ax.plot(unit_interval, Y_true, c='red', label="True values")
    ax.scatter(X, Y_observed, marker='.', c='green', label="Observations")

axs[0].plot(unit_interval, knn_regr, c='blue', label="K=10 nearest neighbors")
axs[0].plot(unit_interval, knn_regr30, c='orange', label="K=30 nearest neighbors")

axs[1].plot(unit_interval, epan_lam_2, c='blue', label="Epanechnikov kernel λ=.2")
axs[1].plot(unit_interval, epan_lam_1, c='orange', label="Epanechnikov kernel λ=.1")

for ax in axs:
    ax.legend()

plt.show()
