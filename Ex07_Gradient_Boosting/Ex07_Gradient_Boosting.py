import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MyGradientBoosting:

    def __init__(self, depth, n_models, max_leaf_nodes, learning_rate):
        self.max_depth = depth
        self.models = []
        self.n_models = n_models
        self.train_predictions = None
        self.test_predictions = None
        self.residuals = []
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate

    def fit(self, features, labels):
        
        # Initialize train_predictions matrix
        self.train_predictions = np.zeros((self.n_models, features.shape[0]))

        # calc first prediction
        self.train_predictions[0, :] = labels.mean()
        self.models.append(self.train_predictions[0])

        # initialize first Pseudo Residuals
        self.residuals.append(self.train_predictions[0])

        # Create & Train ALL regression tree models
        for index in range(1, self.n_models):

            self.residuals.append(labels - self.train_predictions[index-1])

            # Add new regression tree
            self.models.append(DecisionTreeRegressor(max_depth=self.max_depth,
                                                     max_leaf_nodes=self.max_leaf_nodes,
                                                     criterion='mse'))

            # Train a model on the labels differences
            self.models[-1].fit(features, self.residuals[-1])

            # Calc Pseudo Residuals for the next iteration
            self.train_predictions[index] = \
                self.train_predictions[index-1] + self.learning_rate*self.models[-1].predict(features)

    def predict(self, features):
        
        # Initialize test_predictions matrix
        self.test_predictions = np.zeros((self.n_models, features.shape[0]))
        self.test_predictions[0, :] = self.models[0][0]
        
        # Calc test predictions (of n_models)
        for index in range(1, self.n_models):
            self.test_predictions[index] = self.test_predictions[index - 1] + \
                                           self.learning_rate * self.models[index].predict(features)

        return self.test_predictions[-1]


if __name__ == '__main__':

    # Hyper Parameters
    lr = 0.1
    n_estimators = 100
    tree_max_depth = 4
    max_leaves = 32
    np.random.seed(10)
    random_state = 20
    test_size = 0.2

    # Load dataset
    df = pd.read_csv('Ex07_Fish.csv')
    df = pd.get_dummies(df,
                        columns=['Species'],
                        drop_first=True)

    # Labels are the FIRST Column
    dataset = df.to_numpy()

    # Split data to train/test datasets
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:],
                                                        dataset[:, 0],
                                                        random_state=random_state,
                                                        test_size=test_size)
    # Run Gradient Boosting model
    model = MyGradientBoosting(tree_max_depth, n_estimators, max_leaves, lr)
    model.fit(x_train, y_train)

    # calc train error
    predictions_train = model.predict(x_train)
    mse_train = mean_absolute_error(y_train, predictions_train)

    # calc test error
    predictions_test = model.predict(x_test)
    mse_test = mean_absolute_error(y_test, predictions_test)

    # Compare to skl class
    model_skl = GradientBoostingRegressor(max_depth=tree_max_depth,
                                          n_estimators=n_estimators,
                                          max_leaf_nodes=max_leaves,
                                          learning_rate=lr)
    model_skl.fit(x_train, y_train)
    predictions_skl = model_skl.predict(x_test)
    mse_skl = mean_absolute_error(y_test, predictions_skl)

    print(f'\nHyper Parameters:'
          f'\n  lr: {lr} '
          f'\n  n_estimators: {n_estimators} '
          f'\n  tree_max_depth: {tree_max_depth}:'
          f'\n\nResults:'
          f'\n  Train MAE: {mse_train} '
          f'\n  My Model MAE: {mse_test} '
          f'\n  SKL Model MAE: {mse_skl}'
          f'\n  y_test STD: {np.std(y_test)}'
          f'\n  y_train STD: {np.std(y_train)}')
