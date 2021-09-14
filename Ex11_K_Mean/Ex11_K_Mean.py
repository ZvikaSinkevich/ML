import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


def calc_closest_centroid(ds, centers):

    # initialize distances matrix
    distances = np.zeros((ds.shape[0], centers.shape[0]))

    # calc distances between instances
    for index, sample in enumerate(centers):

        # measure distances
        distances[:, index] = np.sqrt(np.square(ds - sample).sum(axis=1))

    # The closest centroid of each sample
    return distances.argmin(axis=1)


def calc_new_centers(ds, clusters_indexes, centers):

    new_centers = np.zeros_like(centers)
    for index, _ in enumerate(new_centers):

        # create cluster with its samples
        cluster = ds[clusters_indexes == index, :]

        # Run average to find the new center of each cluster
        new_centers[index] = np.mean(cluster, axis=0)

    return new_centers


def plot_clusters(ds, labels, clusters, centers):
    plt.figure(1, figsize=(40, 10))
    plt.subplot(121)
    plt.scatter(ds[:, 0], ds[:, 1], c=labels)
    plt.title('True Classes')
    plt.subplot(122)
    plt.scatter(ds[:, 0], ds[:, 1], c=clusters)
    plt.title('Estimated Clusters + Centroids')
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.show()


if __name__ == '__main__':

    # Hyper Parameters
    k = 3
    np.random.seed = 79

    # Load dataset and initialize variables
    dataset, true_labels = load_iris(return_X_y=True)
    counter = 0
    stop = 0

    # Randomly select 3 centers from the dataset
    means = dataset[np.random.randint(len(dataset), size=k), :]
    update_means = np.zeros_like(means)

    # Run the Algorithm
    while stop == 0:

        # Step 1 - Assign each data instance to the closest centroid.
        closest_centroid_indexes = calc_closest_centroid(dataset, means)

        # Step 2 - Set position of each centroid to be the mean of all its attached data.
        update_means = calc_new_centers(dataset, closest_centroid_indexes, means)

        # Step 3 - Stopping Criteria
        stop = np.all(means == update_means)

        # Update matrix for the next iteration
        means = update_means
        counter += 1

    plot_clusters(dataset, true_labels, closest_centroid_indexes, means)
    print(f'Number of Epochs: {counter}')
