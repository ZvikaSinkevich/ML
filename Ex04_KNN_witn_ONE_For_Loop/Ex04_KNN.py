from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
from scipy.stats import mode


def calc_distance(sample, train_ds):
    return np.sqrt(np.square(train_ds - sample).sum(axis=1))


def calc_nearest_neighbours(train_ds, test_ds, k):
    nearest_neighbours_indexes = np.empty((test_ds.shape[0], k))

    # initialize with NaN to recognize bugs/problems
    nearest_neighbours_indexes[:] = np.nan

    # calc distances between instances
    for index, sample in enumerate(test_ds):
        # measure distances relative ONLY to the train dataset
        distances = calc_distance(train_ds, sample)

        # Return the k nearest indexes
        sorted_distances = np.argsort(distances)
        nearest_neighbours_indexes[index] = sorted_distances[0: k]

    return nearest_neighbours_indexes.astype(int)


def calc_predict(nearest_neighbours_indexes, neighbours_labels):
    p_mode, _ = mode(neighbours_labels[nearest_neighbours_indexes], axis=1)
    return np.squeeze(p_mode)


def calc_accuracy(true_labels, predict):
    predict = np.squeeze(predict)
    return np.mean(true_labels == predict)


if __name__ == '__main__':

    # Hyper parameter
    n_neighbors = 7
    random_state = 22
    test_size = 0.3

    # Preprocessing
    features, labels = load_iris(return_X_y=True)
    features = normalize(features, axis=0)
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    # Main Algorithm
    k_nearest_indexes = calc_nearest_neighbours(features_train, features_test, n_neighbors)
    predictions = calc_predict(k_nearest_indexes, labels_train)
    accuracy = calc_accuracy(labels_test, predictions)
