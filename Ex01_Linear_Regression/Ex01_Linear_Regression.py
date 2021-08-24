#####################################################################################
# Q1
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


def calc_linear_regression(x_matrix, y_label):
    return inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y_label


def print_result_linear_regression():

    # this func will plot the results for the entire exercise
    plt.clf()
    plt.figure(1)

    # print line1
    plt.subplot(221)
    plt.scatter(x, y_line_1, label='y_line_1')
    plt.plot(x_plot, y_1, label='y_1')
    plt.title('Q1 - line 1')

    # print line2
    plt.subplot(222)
    plt.scatter(x, y_line_2, label='y_line_2')
    plt.plot(x_plot, y_2)
    plt.title('Q1 - line 2')

    # print line3
    plt.subplot(223)
    plt.scatter(x, y_line_3, label='y_line_3')
    plt.plot(x_plot, y_3)
    plt.title('Q1 - line 3')

    # print Question 2
    plt.subplot(224)
    plt.title('Q2')
    plt.scatter(x_q2, y, label='Q2 - scatter')
    plt.plot(x_plot_q2, prediction, label='Q2 - Prediction')

    plt.legend()
    plt.tight_layout(pad=2)
    plt.show()


np.random.seed(1)
size = 10

x_int = np.random.randint(7, size=10)
x_int_3 = 3 * np.random.randint(9, size=5)

x = np.random.random(size=size)

# add noise to samples
y_line_1 = 2 * x
y_line_1 += np.random.normal(0, 1, size)

y_line_2 = 5 * x + 8
y_line_2 += np.random.normal(0, 1, size)

y_line_3 = 3 * (x ** 2) + 8 * x + 5
y_line_3 += np.random.normal(0, 1, size)

# calc X matrix
x_1 = x.reshape(-1, 1)
x_2 = np.column_stack((x, np.ones_like(x_1)))
x_3 = np.column_stack((x ** 2, x, np.ones_like(x_1)))

# calc weights
h_1 = calc_linear_regression(x_1, y_line_1)
h_2 = calc_linear_regression(x_2, y_line_2)
h_3 = calc_linear_regression(x_3, y_line_3)

# calc predictions line
x_plot = np.linspace(0, 1.2, size)
x_plot = x_plot.reshape(-1, 1)
y_1 = h_1[0] * x_plot
y_2 = h_2[0] * x_plot + h_2[1]
y_3 = h_3[0] * x_plot ** 2 + h_3[1] * x_plot + h_3[2]

#####################################################################################
# Q2
#####################################################################################

x_q2 = np.array(
    [0.08750722, 0.01433097, 0.30701415, 0.35099786, 0.80772547, 0.16525226, 0.46913072, 0.69021229, 0.84444625,
     0.2393042, 0.37570761, 0.28601187, 0.26468939, 0.54419358, 0.89099501, 0.9591165, 0.9496439, 0.82249202,
     0.99367066, 0.50628823])

y = np.array(
    [4.43317755, 4.05940367, 6.56546859, 7.26952699, 33.07774456, 4.98365345, 9.93031648, 20.68259753, 38.74181668,
     5.69809299, 7.72386118, 6.27084933, 5.99607266, 12.46321171, 47.70487443, 65.70793999, 62.7767844, 35.22558438,
     77.84563303, 11.08106882])

y_log = np.log(
    [4.43317755, 4.05940367, 6.56546859, 7.26952699, 33.07774456, 4.98365345, 9.93031648, 20.68259753, 38.74181668,
     5.69809299, 7.72386118, 6.27084933, 5.99607266, 12.46321171, 47.70487443, 65.70793999, 62.7767844, 35.22558438,
     77.84563303, 11.08106882])

x_q2 = x_q2.reshape(-1, 1)

# calc X matrix
matrix_x_q2 = np.column_stack((x_q2 ** 2, x_q2, np.ones_like(x_q2)))

# calc weights
h_log = calc_linear_regression(matrix_x_q2, y_log)
x_plot_q2 = np.linspace(0, 1.2, 30)

# calc predictions line
prediction = np.exp(h_log[0] * x_plot_q2 ** 2 + h_log[1] * x_plot_q2 + h_log[2])

# print all exercise results
print_result_linear_regression()
