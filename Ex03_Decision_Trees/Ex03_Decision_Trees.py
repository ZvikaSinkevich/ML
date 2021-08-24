#####################################################################################
# Decision Trees - Zvika Sinkevich
#####################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:

    def __init__(self, dataset, depth):
        self.dataset = dataset
        self.left_node = None
        self.right_node = None
        self.threshold = np.inf
        self.feature_index = np.inf
        self.leaf_label = None
        self.depth_in_tree = depth
        self.is_trained = False  # TODO: Implement as an evaluation condition
        self.best_jini = None  # A STOP recursion criteria

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

    # Calc the Node best feature index & threshold using Jini criteria
    def get_split(self):
        best_jini = np.inf
        best_feature_index = None
        best_threshold = np.inf
        dataset_left = None
        dataset_right = None
        
        # Loop on the features
        for feature_index in range(0, self.dataset.shape[1] - 1):

            # Calc the list of thresholds to check for a given feature_index
            thresholds = np.sort(np.unique(self.dataset[:, feature_index]))
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
        if max_depth <= self.depth_in_tree or self.best_jini < min_jini or self.dataset.shape[0] == 1:

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
            self.left_node = Node(left_dataset, self.depth_in_tree+1)
            self.left_node.split_node(max_depth)  # recursion
            self.right_node = Node(right_dataset, self.depth_in_tree+1)
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

    @staticmethod
    # Calc predictions & accuracy
    def calc_evaluation(test_ds, true_labels):
        predict = []
        for sample in test_ds:
            predict.append(tree.calc_predicts(sample))
        evaluate = np.mean(true_labels == predict)
        return predict, evaluate


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

    # Hyper parameter
    n_depth = 6  # split_node recursion stopping criteria
    min_jini = 0.005  # split_node recursion stopping criteria
    np.random.seed(18)
    random_seed = 21
    test_percentage = 0.2

    # Pre Processing
    x_train, x_test, y_train, y_test = preprocessing(random_seed, test_percentage)
    train_data = np.column_stack((x_train, y_train))
    
    # Create the Tree
    tree = Node(train_data, 0)
    tree.split_node(n_depth)

    # Print the tree
    tree.print_tree()

    # Calc predictions & accuracy
    predictions, accuracy = tree.calc_evaluation(x_test, y_test)
