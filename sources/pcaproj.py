import numpy as np

"""
Colin Wu, wu1418@purdue.edu
CS 373, HW5
Mar 26, 2021

Input: numpy matrix X, with n rows (samples), d columns (features)
       numpy vector mu, with d rows, 1 column
       numpy matrix Z, with d rows, F columns
Output: numpy matrix P, with n rows, F columns
"""


def run(X, mu, Z):
    X = np.copy(X)
    n, d = X.shape

    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]

    P = X.dot(Z)

    return P
