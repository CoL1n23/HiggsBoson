import numpy as np
import numpy.linalg as la

"""
Colin Wu, wu1418@purdue.edu
CS 373, HW5
Mar 26, 2021

Input: number of features F
       numpy matrix X, with n rows (samples), d columns (features)
Output: numpy vector mu, with d rows, 1 column
        numpy matrix Z, with d rows, F columns
"""


def run(F, X):
    X = np.copy(X)
    n, d = X.shape

    mu = np.zeros((d, 1))
    for i in range(d):
        summation = 0
        for j in range(n):
            summation = summation + X[j][i]
        mu[i] = summation / n

    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]

    U, s, Vt = la.svd(X, False)

    g = np.zeros(F)
    for i in range(F):
        if s[i] > 0:
            g[i] = 1 / s[i]

    W = Vt[:F]
    Z = W.T.dot(np.diag(g))

    return mu, Z
