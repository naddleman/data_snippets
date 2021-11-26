"""
Show kernel density estimates and see how they perform in fat-tailed
distributions, maybe ones without a finite EV
"""

import numpy as np
import matplotlib.pyplot as plt

def dmatrix(linspace, obs_points):
    return abs(linspace[:,np.newaxis] - obs_points)

def gauss_kernel(points, observations, sigma=1):
    ds = dmatrix(points, observations)
    gauss = np.exp(-0.5 * np.square(ds) / np.square(sigma))
    return gauss / np.sum(gauss)


unit_interval = np.linspace(0,1,1001)
obs1 = np.random.normal(loc=0.2, scale=0.1, size=100)
obs2 = np.random.normal(loc=0.6, scale=0.2, size=50)
obs = np.concatenate((obs1, obs2))
Y1 = np.sum(gauss_kernel(unit_interval, obs, sigma=0.2), axis=1)
Y2 = np.sum(gauss_kernel(unit_interval, obs, sigma=0.1), axis=1)
Y3 = np.sum(gauss_kernel(unit_interval, obs, sigma=0.01), axis=1)

plt.plot(unit_interval, Y1, label='Parzen estimate, sigma=0.2')
plt.plot(unit_interval, Y2, label='Parzen estimate, sigma=0.1')
plt.plot(unit_interval, Y3, label='Parzen estimate, sigma=0.01')
plt.legend()
plt.show()
