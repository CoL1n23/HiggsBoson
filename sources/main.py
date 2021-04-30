import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier
import bootstrapping
import pcalearn
import pcaproj


def run():
    """
    -----------------------------------------------------------------------------------------------
    LOADING DATA
    -----------------------------------------------------------------------------------------------
    """
    # Import data from source file from target directory
    raw_data = pd.read_csv('../dataset/training.csv').values

    # Select 1,000 samples from dataset
    data = raw_data[:1500, :]
    n, d = data.shape

    """
    -----------------------------------------------------------------------------------------------    
    CLEANING DATA
    -----------------------------------------------------------------------------------------------    
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

    """
    -----------------------------------------------------------------------------------------------
    HYPERPARAMETER TUNING WITH TWO-FOLD CROSS VALIDATION
    -----------------------------------------------------------------------------------------------
    """
    # Prepare folds
    positive_samples = list(np.where(y == 1)[0])
    negative_samples = list(np.where(y == -1)[0])

    # Separate positive and negative samples to create training and validation sets
    samples_in_fold1 = positive_samples[:len(positive_samples)/2] + negative_samples[:len(negative_samples)/2]
    samples_in_fold2 = positive_samples[len(positive_samples)/2:] + negative_samples[len(negative_samples)/2:]

    B = 30
    best_err_1 = 1.5
    best_err_2 = 1.5

    X_fold1 = X[samples_in_fold1]
    X_fold2 = X[samples_in_fold2]

    # Hyperparameter tuning for F and k
    all_combinations = np.zeros((10, 10))
    for k in range(20, 30):
        for F in range(5, 15):
            print "Current k and F: " + str(k) + " " + str(F)

            # Do PCA on both folds
            mu_fold1, Z_fold1 = pcalearn.run(F, X_fold1)
            X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
            mu_fold2, Z_fold2 = pcalearn.run(F, X_fold2)
            X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

            # Evaluate errors by running bootstrapping on KNeighborsClassifier
            err1 = bootstrapping.run(B, X_fold1_small, y[samples_in_fold1], k)
            if err1 < best_err_1:
                best_err_1 = err1

            err2 = bootstrapping.run(B, X_fold2_small, y[samples_in_fold2], k)
            if err2 < best_err_2:
                best_err_2 = err2

            all_combinations[F - 5][k - 20] = (err1 + err2) / 2

    # Retrieve the best F and k as the tuning result
    best_err = np.amin(all_combinations)
    best_F = np.where(all_combinations == best_err)[0][0] + 5
    best_k = np.where(all_combinations == best_err)[1][0] + 20
    print "Obtained result! best_k=" + str(best_k) + ", best_F=" + str(best_F)

    # Compute classification error for the best F and k
    y_pred = np.zeros(len(X), int)
    mu_fold1, Z_fold1 = pcalearn.run(best_F, X_fold1)
    X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
    X_fold2_small = pcaproj.run(X_fold2, mu_fold1, Z_fold1)

    alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
    alg.fit(X_fold1_small, np.ravel(y[samples_in_fold1]))
    y_pred[samples_in_fold2] = alg.predict(X_fold2_small)

    mu_fold2, Z_fold2 = pcalearn.run(best_F, X_fold2)
    X_fold1_small = pcaproj.run(X_fold1, mu_fold2, Z_fold2)
    X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)

    alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
    alg.fit(X_fold2_small, np.ravel(y[samples_in_fold2]))
    y_pred[samples_in_fold1] = alg.predict(X_fold1_small)

    err = np.mean(y != np.array([y_pred]).T)

    print "Classification Error:" + str(err)
    desired1 = float(len(positive_samples)) / len(X)
    desired2 = float(len(negative_samples)) / len(X)
    print "Threshold of classification error=" + str(min(desired1, desired2))
    if err < min(desired1, desired2):
        print "Congrats! The training is successful!"
    else:
        print "Sorry, the model needs to be improved..."

    """
    -----------------------------------------------------------------------------------------------
    GRAPH DRAWING
    -----------------------------------------------------------------------------------------------
    """
    # Figure 1
    pp.figure(1)
    horizontal = range(5, 15)
    vertical = range(20, 30)
    values = all_combinations
    cp = pp.contourf(horizontal, vertical, values, cmap='YlOrRd')
    pp.colorbar(cp)
    pp.title("Training Error vs. Hyperparameter")
    pp.xlabel("F: number of features after applying PCA")
    pp.ylabel("k: the k-th nearest neighbor")

    # ------------------------------------------
    # |                                        |
    # | Comment starts from here               |
    # |                                        |
    # ------------------------------------------

    # Comment out the code below will expedite the execution
    # Figure 2
    alg = KNeighborsClassifier(n_neighbors=best_k, algorithm='brute')
    alg.fit(X, np.ravel(y))
    sensitivity = np.zeros(1000)
    specificity = np.zeros(1000)

    # Compute 1,000 data points
    # Based on our result, the size of data points should be larger
    # However, considering the computation power, we decided to set 1,000 as the size
    for k in range(1, 1000):
        alg = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
        alg.fit(X, np.ravel(y))
        y_pred = alg.predict(X_test)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for j in range(0, len(y_test)):
            if y_test[j] == 1 and y_pred[j] == 1:
                TP += 1
            if y_test[j] == -1 and y_pred[j] == 1:
                FP += 1
            if y_test[j] == 1 and y_pred[j] == -1:
                FN += 1
            if y_test[j] == -1 and y_pred[j] == -1:
                TN += 1
        sensitivity[k] = float(TP) / (TP + FN)
        specificity[k] = float(TN) / (TN + FP)

    # Manually insert one point
    # Since the number of data points is not large enough, add this point to simulate the plot
    sensitivity[0] = 1
    specificity[0] = 0

    # Sort x and y for better presentation
    new_spe, new_sen = zip(*sorted(zip(specificity, sensitivity)))

    pp.figure(2)
    pp.plot(new_spe, new_sen, color='magenta', label='K-Nearest Neighbors')
    pp.plot([0, 1], [1, 0], color='blue', linestyle='--')
    pp.xlabel('Specificity')
    pp.ylabel('Sensitivity')
    pp.title('Receiver Operating Characteristic (ROC) Curve')
    pp.legend()

    # ------------------------------------------
    # |                                        |
    # | Comment ends here                      |
    # |                                        |
    # ------------------------------------------

    pp.show()


if __name__ == '__main__':
    run()
