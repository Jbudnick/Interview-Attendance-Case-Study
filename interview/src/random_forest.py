import numpy as np
import pandas as pd

import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error

from scipy.interpolate import make_interp_spline
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.dates import (DAILY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)


class reg_model(object):
    def __init__(self, X, y, train_test_split_size=0.75):
        '''
            Parameters:
                X (Pandas DataFrame): Data to be used for regression model - before train/test split
                y (Series): Target values for X
                train_test_split (int/float): Days elapsed value to separate dataset into train/testing set. If int, will use days_elapsed number. If float, will separate based on percentage (0.80 = 80% of data will go into training, 20% will be testing)
        '''
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_test_split_size, random_state=42)

    def rand_forest(self, n_trees=100):
        '''
        Applies random forest to reg_model object.
        '''
        if n_trees == 'optimize':
            '''
            If set to optimize, will take a selection of 1 to max_trees and uses number that minimizes error in training set.
            This can be plotted by uncommenting out the plt.plot(n, error) line.
            '''
            max_trees = 100
            n = np.arange(1, max_trees + 1, 1)
            error = []
            for each in n:
                self.model = RandomForestClassifier(
                    n_estimators=each, n_jobs=-1, random_state=1)
                self.model.fit(self.X_train, self.y_train)
                self.error_metric = 'rmse'
                error.append(self.evaluate_model())
            plt.plot(n, error)
            n_trees = n[error.index(min(error))]
        self.model = RandomForestClassifier(
            n_estimators=n_trees, random_state=None)
        self.model.fit(self.X_train.values.astype(
            int), self.y_train.values.astype(int).flatten())
        self.error_metric = 'rmse'

    def evaluate_model(self, print_err_metric=True):
        '''
        Determine validity of model on test set.
        '''
        self.y_hat = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test.values.astype(int), self.y_hat)
        prec = precision_score(self.y_test.values.astype(int), self.y_hat)
        recall = recall_score(self.y_test.values.astype(int), self.y_hat)
        print("Random Forest Accuracy: {}".format(acc))
        print("Random Forest Precision: {}".format(prec))
        print("Random Forest Recall: {}".format(recall))

    def get_feature_importances(self):
        features = self.X_train.columns
        imps = self.model.feature_importances_
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        ax.bar(features, imps)
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel('Feature Importance')
        ax.set_title('Random Forest Feature Importances')
        fig.savefig('../images/rf_feature_importances.png', dpi=500)
