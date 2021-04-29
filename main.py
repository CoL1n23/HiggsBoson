import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier
import bootstrapping
import pcalearn
import pcaproj


def run():
    """
    LOADING DATA
    """
    # Import data from source file
    raw_data = pd.read_csv('training.csv').values

    # Select 1,000 samples from dataset
    data = raw_data[:2000, :]
    n, d = data.shape

    """
    CLEANING DATA
    """
    # Extract label and convert the last column (label) to -1 and 1
    y = data[:, d - 1]
    y = np.array([y]).T

    samples_with_s = list(np.where(y == 's')[0])
    y[samples_with_s] = 1
    samples_with_b = list(np.where(y == 'b')[0])
    y[samples_with_b] = -1
    y = y.astype('int')

    X = data[:, 1:(d - 1)]
    X = X.astype('float')

    X_test = X[1000:, :]
    X = X[:1000]
    y_test = y[1000:, :]
    y = y[:1000]

    positive_samples_test = list(np.where(y_test == 1)[0])
    negative_samples_test = list(np.where(y_test == -1)[0])

    """
    HYPERPARAMETER TUNING
    """
    # Prepare folds
    positive_samples = list(np.where(y == 1)[0])
    negative_samples = list(np.where(y == -1)[0])

    # Separate positive and negative samples to create training and validation sets
    train_samples = positive_samples[:len(positive_samples)/2] + negative_samples[:len(negative_samples)/2]
    validation_samples = positive_samples[len(positive_samples)/2:] + negative_samples[len(negative_samples)/2:]

    B = 10
    best_F_1 = 1
    best_F_2 = 1
    best_k_1 = 0
    best_k_2 = 0
    best_err_1 = 1.5
    best_err_2 = 1.5

    X_fold1 = X[train_samples]
    X_fold2 = X[validation_samples]

    # Hyperparameter tuning for F and k
    all_combinations = np.zeros((10, 10))
    for F in range(10, 20):
        for k in range(20, 30):
            print "Current F and k: " + str(F) + " " + str(k)

            # Do PCA on both folds
            mu_fold1, Z_fold1 = pcalearn.run(F, X_fold1)
            X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
            mu_fold2, Z_fold2 = pcalearn.run(F, X_fold2)
            X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

            # Evaluate errors by running bootstrapping on KNeighborsClassifier
            err1 = bootstrapping.run(B, X_fold1_small, y[train_samples], k)
            if err1 < best_err_1:
                best_err_1 = err1
                best_k_1 = k
                best_F_1 = F

            err2 = bootstrapping.run(B, X_fold2_small, y[validation_samples], k)
            if err2 < best_err_2:
                best_err_2 = err2
                best_k_2 = k
                best_F_2 = F

            all_combinations[F - 10][k - 20] = (err1 + err2) / 2
    print "Final result is : " + str(best_k_1) + " " + str(best_k_2) + " " + str(best_err_1) + " " + str(
        best_err_2) + " " + str(best_F_1) + " " + str(best_F_2)

    # Retrieve the best F and k as the tuning result
    best_err = np.amin(all_combinations)
    best_F = np.where(all_combinations == best_err)[0][0] + 10
    best_k = np.where(all_combinations == best_err)[1][0] + 20

    # Compute classification error for the best F and k
    y_pred = np.zeros(len(X), int)
    mu_fold1, Z_fold1 = pcalearn.run(best_F, X_fold1)
    X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
    X_fold2_small = pcaproj.run(X_fold2, mu_fold1, Z_fold1)

    alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
    alg.fit(X_fold1_small, y[train_samples])
    y_pred[validation_samples] = alg.predict(X_fold2_small)

    mu_fold2, Z_fold2 = pcalearn.run(best_F, X_fold2)
    X_fold1_small = pcaproj.run(X_fold1, mu_fold2, Z_fold2)
    X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

    alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
    alg.fit(X_fold2_small, y[validation_samples])
    y_pred[train_samples] = alg.predict(X_fold1_small)

    err = np.mean(y != y_pred)

    print "Classification Error:" + str(err)
    print len(positive_samples)
    print len(negative_samples)


if __name__ == '__main__':
    run()
