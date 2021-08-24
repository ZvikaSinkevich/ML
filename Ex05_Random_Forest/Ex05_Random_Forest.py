#####################################################################################
# Random Forest - Zvika Sinkevich
#####################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode


class Node:

    def __init__(self, dataset, depth, is_random_forest, subset_percentage):
        self.dataset = dataset
        self.left_node = None
        self.right_node = None
        self.threshold = None
        self.feature_index = None
        self.leaf_label = None
        self.depth_in_tree = depth
        self.is_trained = False  # TODO: Implement as an evaluation condition
        self.best_jini = None  # A STOP recursion criteria
        self.is_random_forest = is_random_forest
        self.subset_percentage = subset_percentage

    @staticmethod
    def split(dataset, split_index, split_threshold):
        mask = dataset[:, split_index] <= split_threshold
        left_split = dataset[mask]
        right_split = dataset[~mask]
        return left_split, right_split

    @staticmethod
    # Calc the Jini score for a given dataset
    def calc_gini(dataset):
        n_all = dataset.shape[0]
        values, freq = np.unique(dataset[:, -1], return_counts=True)
        return 1 - ((freq / n_all) ** 2).sum()

    def get_features_indexes(self):
        n_features = self.dataset.shape[1]
        if self.is_random_forest:
            selected_features = int(np.sqrt(n_features))

            # minus 1 to ignore the label column
            return np.random.randint(0, n_features-1, selected_features)
        else:
            # minus 1 to ignore the label column
            return range(0, n_features-1)

    # Calc the Node best feature index & threshold using Jini criteria
    def get_split(self):
        best_jini = np.inf
        best_feature_index = np.inf
        best_threshold = np.inf
        dataset_left = None
        dataset_right = None
        features = self.get_features_indexes()

        # Create random data subset in Random Forest state
        if self.is_random_forest:
            self.dataset = self.create_subsample()

        # Loop on the features
        for feature_index in features:

            # Calc the list of thresholds to check for a given feature_index
            thresholds = np.sort(np.unique(self.dataset[:, feature_index]))
            if thresholds.shape[0] > 1:
                thresholds = (thresholds[1:len(thresholds)] + thresholds[0:-1]) / 2

            # find the best threshold on a given feature_index
            for value in thresholds:

                # Split data for the Jini calculation
                left, right = self.split(self.dataset, feature_index, value)

                # calculate Jini score
                l_gini = self.calc_gini(left)
                r_gini = self.calc_gini(right)
                total_jini = (l_gini * left.shape[0] + r_gini * right.shape[0]) / self.dataset.shape[0]

                # update properties for best split
                if total_jini < best_jini:
                    best_jini = total_jini
                    best_feature_index = feature_index
                    best_threshold = value
                    dataset_left = left
                    dataset_right = right
        return best_feature_index, best_threshold, dataset_left, dataset_right, best_jini

    def split_node(self, max_depth):

        # find the decision parameter, and save sub-datasets for the next depth
        self.feature_index, self.threshold, left_dataset, right_dataset, self.best_jini = self.get_split()

        # Stop Criteria - You are in a leave
        if max_depth <= self.depth_in_tree or self.best_jini < min_jini or self.dataset.shape[0] <= 1:

            # Calc leave label
            val, freq = np.unique(self.dataset[:, -1], return_counts=True)
            self.leaf_label = val[np.argmax(freq)]
            self.is_trained = True

            # Free memory
            del self.dataset

        # Create the branch below
        else:
            # Free memory
            del self.dataset

            # Activate recursion to create the next 2 nodes below
            self.left_node = Node(left_dataset, self.depth_in_tree+1, self.is_random_forest, self.subset_percentage)
            self.left_node.split_node(max_depth)  # recursion
            self.right_node = Node(right_dataset, self.depth_in_tree+1, self.is_random_forest, self.subset_percentage)
            self.right_node.split_node(max_depth)  # recursion
            self.is_trained = True

    def print_tree(self):

        # Print the leaves (stop recursion criteria)
        if self.leaf_label is not None:
            print(self.depth_in_tree * '\t',
                  f'Leaf Depth: {self.depth_in_tree},'
                  f'Label: {self.leaf_label},',
                  f'Jini: {self.best_jini}')

        # Print the nodes (using recursion)
        else:
            print(self.depth_in_tree * '\t',
                  f'Node Depth: {self.depth_in_tree},'
                  f' Feature:{self.feature_index} ,'
                  f'Threshold:{self.threshold} ,'
                  f'Jini: {self.best_jini}')
            self.left_node.print_tree()
            self.right_node.print_tree()

    # Calc predict for only ONE instance
    def calc_predicts(self, test_sample):
        if self.leaf_label is not None:
            return self.leaf_label
        else:
            if test_sample[self.feature_index] < self.threshold:
                return self.left_node.calc_predicts(test_sample)
            else:
                return self.right_node.calc_predicts(test_sample)

    # Calc predictions & accuracy
    def calc_evaluation(self, test_ds, true_labels):
        predict = []
        for sample in test_ds:
            predict.append(self.calc_predicts(sample))
        evaluate = np.mean(true_labels == predict)
        return predict, evaluate

    # Create a sub-dataset using Bagging
    def create_subsample(self):
        n_subsample = int(np.ceil(self.subset_percentage*self.dataset.shape[0]))
        return self.dataset[np.random.randint(0, self.dataset.shape[0], n_subsample)]


class Forest:
    def __init__(self, train_ds, n_trees, subset_size):

        self.train_ds = train_ds
        self.n_trees = n_trees
        self.subset_size = subset_size
        self.trees = []
        self.trees_accuracies = []
        self.trees_predicts = np.inf
        self.forest_accuracy = np.inf
        if not random_forest_state:
            self.n_trees = 1
        self.create_tree()  # Last so it will have all the attributes

    def create_tree(self):

        # Create the trees in a self list
        for index in range(self.n_trees):

            # Create the root
            self.trees.append(Node(self.train_ds, 0, random_forest_state, self.subset_size))

            # Create branches
            self.trees[index].split_node(n_depth)

            # Print the Trees
            if is_print_tree:
                print('\n\nTree #{index}')
                self.trees[index].print_tree()

    def bagging_predict(self, features_test, labels_test):

        # create a matrix with all trees predictions & accuracies
        self.trees_predicts = np.chararray((self.n_trees, len(labels_test)), unicode=True)
        for index in range(self.n_trees):
            predictions, accuracy = self.trees[index].calc_evaluation(features_test, labels_test)
            self.trees_predicts[index] = np.array(predictions)
            self.trees_accuracies.append(accuracy)

        # Select the most common prediction
        value, count = mode(self.trees_predicts, axis=0)

        # Calc the entire FOREST Accuracy
        self.forest_accuracy = np.mean(labels_test == value)
        print(self.forest_accuracy)


def preprocessing(random_state, test_size):

    # read the file
    df = pd.read_csv('Ex03_wdbc.data', header=None)

    # Save labels
    labels = np.array(df[1])

    # delete the id and labels columns
    df.drop([0, 1], axis=1, inplace=True)

    # Split to train/test
    data = np.array(df)
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)


if __name__ == '__main__':

    # Hyper parameters - Random Forest
    random_forest_state = True
    n_trees = 50
    subset_size = 0.9
    is_print_tree = True

    # Hyper parameters - Tree
    n_depth = 6  # split_node recursion stopping criteria
    min_jini = 0.05  # split_node recursion stopping criteria

    # Hyper parameters - General
    np.random.seed(18)
    train_test_split_random_seed = 21
    test_percentage = 0.2

    # Pre Processing
    x_train, x_test, y_train, y_test = preprocessing(train_test_split_random_seed, test_percentage)
    train_data = np.column_stack((x_train, y_train))
    
    # Create & Print the Forest
    forest = Forest(train_data, n_trees, subset_size)

    # Calc predictions & accuracy
    forest.bagging_predict(x_test, y_test)
