import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm


class MyNaiveBayes:

    def __init__(self):

        # Probability of labels ('Outcome')
        self.initial_guess = None
        self.f_mean_0 = None
        self.f_std_0 = None
        self.f_mean_1 = None
        self.f_std_1 = None

    def fit(self, x_train_ds, y_train_ds):

        # Statistics calculations
        self.initial_guess = np.unique(y_train_ds, return_counts=True)[1]/len(y_train_ds)
        self.f_mean_0 = x_train_ds[y_train_ds == 0].mean(axis=0)
        self.f_std_0 = x_train_ds[y_train_ds == 0].std(axis=0)
        self.f_mean_1 = x_train_ds[y_train_ds == 1].mean(axis=0)
        self.f_std_1 = x_train_ds[y_train_ds == 1].std(axis=0)

    def predict(self, x_ds, y_ds):

        # Calculate the Gaussian likelihood
        pdf_0 = norm(self.f_mean_0, self.f_std_0).pdf(x_ds)
        pdf_1 = norm(self.f_mean_1, self.f_std_1).pdf(x_ds)

        # Calculate the full Naive Bayes
        score_0 = np.log(pdf_0).sum(axis=1) + np.log(self.initial_guess[0])
        score_1 = np.log(pdf_1).sum(axis=1) + np.log(self.initial_guess[1])

        # Predict
        y_predict = np.zeros_like(y_ds)
        y_predict[score_1 > score_0] = 1

        # Accuracy
        accuracy = np.mean(y_predict == y_ds)

        return y_predict, accuracy


if __name__ == '__main__':

    # Hyper Parameters
    RANDOM_STATE = 1
    TEST_SIZE = 0.2

    # Load & Split the dataset
    df = pd.read_csv('diabetes.csv')
    dataset = df.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                                        dataset[:, -1],
                                                        random_state=RANDOM_STATE,
                                                        test_size=TEST_SIZE,
                                                        stratify=dataset[:, -1])
    # Create & Fit My Bayes Model
    model = MyNaiveBayes()
    model.fit(x_train, y_train)

    # Train Predictions
    _, score = model.predict(x_train, y_train)
    print(f'My Model - Train Accuracy: {score}')

    # Test Predictions
    _, score = model.predict(x_test, y_test)
    print(f'My Model - Test Accuracy: {score}')

    # SKL Model
    model_skl = GaussianNB()
    model_skl.fit(x_train, y_train)
    score_skl = model_skl.score(x_train, y_train)
    print(f'SKL Model - Train Accuracy: {score_skl}')

    score_skl = model_skl.score(x_test, y_test)
    print(f'SKL Model - Test Accuracy: {score_skl}')
