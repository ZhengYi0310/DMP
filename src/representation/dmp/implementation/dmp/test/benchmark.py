import numpy as np
import dmp


n_features = 9
T = np.linspace(0, 1, 101)
Y = np.hstack((T[:, np.newaxis], np.cos(2 * np.pi * T)[:, np.newaxis]))
alpha = 25.0


def imitate(T, Y, n_features, alpha):
    widths = np.empty(n_features)
    centers = np.empty(n_features)
    weights = np.empty(n_features * 2)
    dmp.initializeRBF(widths, centers, 1.0, 0.8, 25.0 / 3.0)
    dmp.LearnFromDemo(T, Y.ravel(), weights, widths, centers, 1e-10, alpha, alpha / 4.0, alpha / 3.0, False)
    return widths, centers, weights