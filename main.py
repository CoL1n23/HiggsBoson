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
    y = data[:, d - 1]
    y = np.array([y]).T

    samples_with_s = list(np.where(y == 's')[0])
    y[samples_with_s] = 1.0
    samples_with_b = list(np.where(y == 'b')[0])
    y[samples_with_b] = -1.0

    X = data[:, 1:(d - 1)]

    """
    HYPERPARAMETER TUNING
    """
    # Prepare folds
    positive_samples = list(np.where(y == 1)[0])
    negative_samples = list(np.where(y == -1)[0])

    # Split data into three sets
    train_samples = positive_samples[0:400] + negative_samples[0:400]
    validation_samples = positive_samples[400:700] + negative_samples[400:700]
    test_samples = positive_samples[700:] + negative_samples[700:]


if __name__ == '__main__':
    run()
