import numpy as np
import random

def add_noise(I_t):
    mu = np.mean(I_t)
    sigma = np.std(I_t)
    dB = 3
    I_noise = 10**(-dB/20)*np.reshape([random.gauss(mu, sigma) for i in range(np.size(I_t))], np.shape(I_t))
    I = I_t + I_noise
    max_I  = np.max(I)
    min_I = np.min(I)
    I = np.round((I - min_I)*255/(max_I - min_I))
    return I