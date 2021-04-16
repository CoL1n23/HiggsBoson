import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier


def run():
    """
    LOADING DATA
    """
    # Import data from source file
    raw_data = pd.read_csv('training.csv').values

    # Select 1,000 samples from dataset
    data = raw_data[:1000, :]
    n, d = data.shape

    """
    CLEANING DATA
    """
    # Extract label and convert the last column (label) to -1 and 1
    label = data[:, d - 1]
    label = np.array([label]).T

    samples_with_s = list(np.where(label == 's')[0])
    label[samples_with_s] = 1.0
    samples_with_b = list(np.where(label == 'b')[0])
    label[samples_with_b] = -1.0

    """
    HYPERPARAMETER TUNING
    """
    # Prepare folds
    positive_samples = list(np.where(y == 1)[0])
    negative_samples = list(np.where(y == -1)[0])

    # Form different datasets
    X = data[:500, 1:31]
    y = label[:500]
    X_test = data[500:, 1:31]
    y_test = label[500:]


if __name__ == '__main__':
    run()
