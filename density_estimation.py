"""
Show kernel density estimates and see how they perform in fat-tailed
distributions, maybe ones without a finite EV
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1001)

def dmatrix(linspace, obs_points):
    return abs(linspace[:,np.newaxis] - obs_points)

def gauss_kernel(points, observations, sigma=1):
    ds = dmatrix(points, observations)
    gauss = np.exp(-0.5 * np.square(ds) / np.square(sigma))
    return gauss / np.sum(gauss)

nobs = 150
unit_interval = np.linspace(0,1,1001)

def gen_observations(nobs):
    obs1 = np.random.normal(loc=0.2, scale=0.2, size=2*nobs//3)
    obs2 = np.random.normal(loc=0.6, scale=0.1, size=nobs//3)
    obs = np.concatenate((obs1, obs2))
    return obs[np.where((obs<1) & (obs>=0))]

obs = gen_observations(150)
obs2 = gen_observations(1500)

pdf = 2/(0.2 * np.sqrt(np.pi * 2)) * \
    np.exp(-0.5 * ((unit_interval - 0.2)/0.2)**2)
pdf += 1/(0.1 * np.sqrt(np.pi * 2)) * \
        np.exp(-0.5 * ((unit_interval - 0.6)/0.1)**2)
pdf /= np.sum(pdf)

sigmas = [0.1, 0.05, 0.01]
Y1s = [np.sum(gauss_kernel(unit_interval, obs, sigma=s), axis=1) 
        for s in sigmas]

Y2s = [np.sum(gauss_kernel(unit_interval, obs2, sigma=s), axis=1) 
        for s in sigmas]

fig,axs = plt.subplots(2)
fig.suptitle('Kernel density estimation')
for Y,s in zip(Y1s,sigmas):
    axs[0].plot(unit_interval, Y, label=f'σ={s}')
axs[0].set_title("Parzen kernel density estimates, n=150")
for Y,s in zip(Y2s,sigmas):
    axs[1].plot(unit_interval, Y, label=f'σ={s}')
axs[1].set_title("Parzen kernel density estimates, n=1500")
for ax in axs:
    ax.plot(unit_interval, pdf, 'r:', label='theoretical pdf')
    ax.legend(loc='lower left')
    ax.set_yticks([]) # the y-axis is all relative
axs[0].plot(obs, np.zeros(obs.shape), 'b|', ms=20)
axs[1].plot(obs2, np.zeros(obs2.shape), 'b|', ms=20)


plt.show()
