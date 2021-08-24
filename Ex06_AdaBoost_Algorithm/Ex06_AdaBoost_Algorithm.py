import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # TODO cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:

    def __init__(self, n_estimators, max_tree_depth):
        self.n_estimators = n_estimators
        self.max_tree_depth = max_tree_depth
        self.length = None
        self.models = []
        self.weights = []
        self.t_error = []
        self.error = []
        self.significance = []
        self.predictions = []
        self.epsilon = 1e-10
        self.features = None
        self.labels = None
        self.step = None  # For Debug

    def initialize_weights(self):
        # First weights - The sum is normalized to 1.
        weights = np.ones(self.length)
        self.weights.append(weights/self.length)

    def calc_t_error(self):
        # should be a vector
        self.predictions.append(self.models[-1].predict(self.features))
        self.t_error.append(self.predictions[-1] != self.labels)

    def calc_error(self):
        # Always should be between zero to one
        # self.error.append(self.weights[-1] @ self.t_error[-1])
        # self.error[-1] /= self.weights[-1].sum()
        self.error.append(np.average(self.t_error[-1], axis=0, weights=self.weights[-1]))

    def calc_significance(self):
        # Should be a scalar
        # If t_error is big, significance will be a big negative number
        # If t_error is small, significance will be a big positive number
        # If t_error is 1/2, significance will be 0, like a coin possibility
        self.significance.append(np.log((1-self.error[-1]+self.epsilon)/(self.error[-1]+self.epsilon)))
        print(f'accuracy {self.step}: {np.mean(self.predictions[-1] == self.labels)}, '
              f'Significance is: {self.significance[-1]}')

    def update_weights(self):
        self.weights.append(self.weights[-1]*np.exp(self.significance[-1]*self.t_error[-1]))
        self.weights[-1] /= self.weights[-1].sum()

    def fit(self, dataset_features, dataset_labels):

        # Initialize Dataset
        self.features = dataset_features  # For Debug
        self.labels = dataset_labels  # For Debug

        # Initialize weights
        self.length = dataset_features.shape[0]
        self.initialize_weights()

        # The fitting loop
        for step in range(self.n_estimators):
            self.step = step  # For Debug
            self.models.append(DecisionTreeClassifier(max_depth=self.max_tree_depth))
            self.models[step].fit(self.features, self.labels, sample_weight=self.weights[-1])
            self.calc_t_error()
            self.calc_error()
            self.calc_significance()
            self.update_weights()

            # For Debug
            if step % 20 == 0 and is_print:
                plt.figure(step)
                plot_tree(self.models[step])

    def predict(self, test_features, test_labels):

        # Predictions matrix - every LINE will be the predictions of a ONE weak-learner tree
        predicts = np.zeros((self.n_estimators, test_features.shape[0]))

        # Calculate predictions
        for step in range(self.n_estimators):
            predicts[step, :] = self.models[step].predict(test_features)

        predicts = np.sign(predicts.T @ np.array(self.significance))
        accuracy = np.mean(predicts == test_labels)

        print(f'\nMy Adaboost Model Accuracy is: {accuracy}')

        return predicts, accuracy


if __name__ == '__main__':

    # Hyper Parameters
    depth = 1
    n_epochs = 50
    test_size = 0.2
    random_state = 221
    np.random.seed(20)
    is_print = False

    # Pre processing dataframe
    df = pd.read_csv('Ex06_heart.csv')
    df.target[df.target == 0] = -1
    # df_get_dummies = pd.get_dummies(df, columns=['cp', 'fbs', 'slope', 'thal', 'restecg', 'ca'], drop_first=True)
    # reorder = df_get_dummies.columns.to_list()
    # reorder.pop(reorder.index('target'))
    # reorder.append('target')
    # df = df_get_dummies[reorder]
    # df.info()

    # change labels to be between 1 and -1, and split to train/test
    ds = df.to_numpy()
    ds_features = ds[:, :-1]
    ds_labels = ds[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(ds_features,
                                                        ds_labels,
                                                        test_size=test_size,
                                                        random_state=random_state)

    # Run Ada-boosting Algorithm
    my_adaboost = AdaBoost(n_epochs, depth)
    my_adaboost.fit(x_train, y_train)
    predictions, accuracy = my_adaboost.predict(x_test, y_test)

    # Run SKL Ada-boosting Algorithm
    skl_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth),
                                      n_estimators=n_epochs,
                                      learning_rate=0.01)
    skl_adaboost.fit(x_train, y_train)
    predictions_skl = skl_adaboost.predict(x_test)

    print(f'SKL Adaboost Class Accuracy: {np.mean(predictions_skl == y_test)}')
