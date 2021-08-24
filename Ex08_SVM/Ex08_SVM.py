import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits


########################################################################
# Hyper Parameters - For Iris & Digits Datasets
########################################################################

random_state = 30
test_size = 0.2
C = 100
gamma = 'scale'  # Options: 'scale' or 'auto'

########################################################################
# Iris Dataset
########################################################################

# Load datasets
x_iris, y_iris = load_iris(return_X_y=True)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris,
                                                    random_state=random_state,
                                                    test_size=test_size,
                                                    stratify=y_iris)

# RBF Kernel
model_rbf = SVC(kernel='rbf',
                random_state=random_state,
                C=C,
                gamma=gamma)
model_rbf.fit(x_train[:, [0, 2]], y_train)
scores = [model_rbf.score(x_test[:, [0, 2]], y_test)]

# Linear Kernel
model_linear = SVC(kernel='linear',
                   random_state=random_state,
                   C=C)
model_linear.fit(x_train[:, [0, 2]], y_train)
scores.append(model_linear.score(x_test[:, [0, 2]], y_test))


# Poly Kernel
model_poly = SVC(kernel='poly',
                 degree=3,
                 random_state=random_state,
                 C=C,
                 gamma=gamma)
model_poly.fit(x_train[:, [0, 2]], y_train)
scores.append(model_poly.score(x_test[:, [0, 2]], y_test))


# Print the number of support vectors for each kernel
print(f'\nIris Dataset:\nNNumber of support vectors - Poly: {model_poly.n_support_}, Score: {scores[0]}')
print(f'Number of support vectors - Linear: {model_linear.n_support_}, Score: {scores[1]}')
print(f'Number of support vectors - RBF: {model_rbf.n_support_}, Score: {scores[2]}')


# Print features 0 & 2 and Support Vectors
support_vectors = [model_poly.support_vectors_,
                   model_linear.support_vectors_,
                   model_rbf.support_vectors_]
models_names = {0: 'Poly', 1: 'Linear', 2: 'RBF'}

plt.subplots(2, 2, figsize=(24, 8))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(f'Iris - {models_names[i]} Model, C={C}')
    plt.scatter(x_train[:, 0],
                x_train[:, 2],
                c=y_train)
    plt.scatter(support_vectors[i][:, 0],
                support_vectors[i][:, 1],
                c='r',
                marker='^')
    plt.xlabel('Features: Column 0')
    plt.ylabel('Features: Column 2')
    plt.show()


########################################################################
# Digits Dataset
########################################################################

# Load datasets
x_digits, y_digits = load_digits(return_X_y=True)

# Split the dataset
x_train_digits, x_test_digits, y_train_digits, y_test_digits = train_test_split(x_digits, y_digits,
                                                                                random_state=random_state,
                                                                                test_size=test_size,
                                                                                stratify=y_digits)
# RBF Kernel
digits_model_rbf = SVC(kernel='rbf',
                       random_state=random_state,
                       C=C,
                       gamma=gamma)
digits_model_rbf.fit(x_train_digits, y_train_digits)
scores = [digits_model_rbf.score(x_test_digits, y_test_digits)]


# Linear Kernel
digits_model_linear = SVC(kernel='linear',
                          random_state=random_state,
                          C=C)
digits_model_linear.fit(x_train_digits, y_train_digits)
scores.append(digits_model_linear.score(x_test_digits, y_test_digits))


# Poly Kernel
digits_model_poly = SVC(kernel='poly',
                        degree=3,
                        random_state=random_state,
                        C=C,
                        gamma=gamma)
digits_model_poly.fit(x_train_digits, y_train_digits)
scores.append(digits_model_poly.score(x_test_digits, y_test_digits))


# Print the number of support vectors for each kernel
print(f'\nDigits Dataset:\nNumber of support vectors - Poly: {digits_model_poly.n_support_}, Score: {scores[0]}')
print(f'Number of support vectors - Linear: {digits_model_linear.n_support_}, Score: {scores[1]}')
print(f'Number of support vectors - RBF: {digits_model_rbf.n_support_}, Score: {scores[2]}')
