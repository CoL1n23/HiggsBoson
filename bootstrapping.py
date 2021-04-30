import numpy as np
from sklearn.neighbors import KNeighborsClassifier

"""
Colin Wu, wu1418@purdue.edu
CS 373, Project
Apr 29, 2021

Input: number of bootstraps B
       numpy matrix X, with n rows (samples), d columns (features)
       numpy matrix y, with n rows (samples), 1 column (feature)
       int k, the k-th nearest neighbor
Output: float err, the classification error of bootstrapping
"""


def run(B, X_subset, y_subset, k):
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))
        alg = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
        alg.fit(X_subset[train_samples], np.ravel(y_subset[train_samples]))
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)
    return err
