#####################################################################################
# Logistic Regression
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy import log

np.random.seed(8)
plt.clf()

lr = 1e-4
n_samples = 100
n_epoch = 1000
sigma0 = 0.5
sigma1 = 0.3
mu0 = [3, 2]
mu1 = [2, 2]
r0 = 2  # For Q2
r1 = 3  # For Q2


def generate_2d_normal_dist(sigma, mu, n_samples):
    return np.random.normal(mu, sigma, size=(n_samples, 2))


def print_distribution(dist_0, dist_1):
    plt.figure()
    plt.scatter(dist_0[:, 0], dist_0[:, 1], c='r')
    plt.scatter(dist_1[:, 0], dist_1[:, 1], c='b')
    plt.title('2 Groups - Normal Distribution ')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def calc_logic_prediction(x, theta):
    return sigmoid((x @ np.reshape(theta, -1)).reshape(-1, 1))


def calc_negative_log_likelihood(h, y):
    return (-y.T @ log(h) - np.transpose((1 - y)) @ log(1 - h)) / n_samples


def calc_gradient(y, x, h):
    return ((y - h).T @ x).T


def train(x, y, weights, n_epoch):
    for index in range(n_epoch):
        h = calc_logic_prediction(x, weights)
        likelihood = calc_negative_log_likelihood(h, y)
        weights += lr * calc_gradient(y, x, h)
        # print('Loss index ' + str(index) + ': ' + str(likelihood))
    return weights, likelihood


#####################################################################################
# Main Q1
#####################################################################################


# Create 2 classes - normal distribution
class_0 = generate_2d_normal_dist(sigma0, mu0, n_samples)
class_1 = generate_2d_normal_dist(sigma1, mu1, n_samples)

# Plot distribution
print_distribution(class_0, class_1)

# Dataset
x = np.vstack((class_0, class_1))
y = np.vstack((np.zeros([n_samples, 1]), np.ones([n_samples, 1])))

# Dataset
random_weights = np.random.normal(0, 1, 2).reshape(-1, 1)

# Training
weights, _ = train(x, y, random_weights, n_epoch)

# Predict
predict = np.sum(np.round(calc_logic_prediction(x, weights)) == y) / len(y)
print('\nPredict: ' + str(predict))


#####################################################################################
# Main - Q2
#####################################################################################


# Create 2 uniform circles + normal noise
t = np.random.uniform(0.0, 2.0 * np.pi, n_samples).reshape(-1, 1)
x0 = np.hstack((r0 * np.cos(t), r0 * np.sin(t))) + generate_2d_normal_dist(0.1, [0, 0], n_samples)
x1 = np.hstack((r1 * np.cos(t), r1 * np.sin(t))) + generate_2d_normal_dist(0.1, [0, 0], n_samples)

# Plot distribution
print_distribution(x0, x1)

# Dataset
circle_matrix = np.hstack((np.vstack((x0, x0**2)), np.vstack((x1, x1**2))))
circle_labels = np.vstack((np.zeros([n_samples, 1]), np.ones([n_samples, 1])))
circle_random_weights = np.random.normal(0, 1, 4).reshape(-1, 1)

# Training
circle_weights, _ = train(circle_matrix, circle_labels, circle_random_weights, n_epoch)

# Predict
predict = np.sum(np.round(calc_logic_prediction(circle_matrix, circle_weights)) == circle_labels) / len(circle_labels)
print('\nCircle Predict: ' + str(predict))
