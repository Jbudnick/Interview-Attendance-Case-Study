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
plt.rcParams.update({'font.size': 16})
plt.style.use('fivethirtyeight')
plt.tight_layout()


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
        self.model_type = 'Random Forest'
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

    def log_reg(self):
        self.model_type = 'Logistic Regression'
        self.model = LogisticRegression()
        self.model.fit(self.X_train.values.astype(
            int), self.y_train.values.astype(int).flatten())

    def evaluate_model(self, print_err_metric=True):
        '''
        Determine validity of model on test set.
        '''
        self.y_hat = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test.values.astype(int), self.y_hat)
        prec = precision_score(self.y_test.values.astype(int), self.y_hat)
        recall = recall_score(self.y_test.values.astype(int), self.y_hat)
        print("{} Accuracy: {}".format(self.model_type, acc))
        print("{} Precision: {}".format(self.model_type, prec))
        print("{} Recall: {}".format(self.model_type, recall))

    def get_feature_importances(self):
        features = self.X_train.columns
        if self.model_type == 'Random Forest':
            imps = self.model.feature_importances_
        elif self.model_type == 'Logistic Regression':
            imps = self.model.coef_.flatten()          
        fig, ax = plt.subplots(1, 1, figsize=(16, 7))
        fig.subplots_adjust(bottom=0.2)
        ax.bar(features, imps)
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel('Feature Importance')
        ax.set_title('{} Feature Importances'.format(self.model_type))
        fig.tight_layout()
        fig.savefig('../images/{}_feature_importances.png'.format(self.model_type, dpi = 500))
