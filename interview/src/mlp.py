import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import (DAILY, DateFormatter, RRuleLocator, drange,
                              rrulewrapper)
from pandas.plotting import register_matplotlib_converters
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


class Mlp_Model(object):
    def __init__(self):
        '''
            Parameters:
                X (Pandas DataFrame): Data to be used for regression model - before train/test split
                y (Series): Target values for X
                train_test_split (int/float): Days elapsed value to separate dataset into train/testing set. If int, will use days_elapsed number. If float, will separate based on percentage (0.80 = 80% of data will go into training, 20% will be testing)
        '''
        self.num_epochs = 500 # number of times to train on the entire training set
        self.batch_size = 128 # using batch gradient descent
        self.y = None
        self.X = None

    def model_build(self, n_trees=100):
        '''
        Builds, compiles, and trains MLP object.
        '''
        nn_hl = 10 # number of neurons in the hidden layer
        activ='tanh'

        num_coef = self.X.shape[1]
        model = Sequential() # sequential model is a linear stack of layers
        model.add(Dense(units=nn_hl,
                        input_shape=(num_coef,),
                        activation=activ, 
                        use_bias=True, 
                        kernel_initializer='ones', 
                        bias_initializer='ones', 
                        kernel_regularizer=None, 
                        bias_regularizer=None, 
                        activity_regularizer=None, 
                        kernel_constraint=None, 
                        bias_constraint=None))
        model.add(Dense(units=nn_hl,
                        input_shape=(num_coef,),
                        activation=activ, 
                        use_bias=True, 
                        kernel_initializer='ones', 
                        bias_initializer='ones', 
                        kernel_regularizer=None, 
                        bias_regularizer=None, 
                        activity_regularizer=None, 
                        kernel_constraint=None, 
                        bias_constraint=None))
        model.add(Dense(units=1,
                        activation=activ,
                        use_bias=True, 
                        kernel_initializer='glorot_uniform')) 
        sgd = SGD(lr=1e2, decay=1e-7, momentum=0.9) # using stochastic gradient descent
        model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["mse"] )

    def mlp_fit(self):
        self.model.fit(self.X, self.y, batch_size=self.batch_size, epochs=self.num_epochs, verbose=1, shuffle=True)


    def evaluate_model(self):
        '''
        Determine validity of model on test set.
        '''
        y_pred = self.model.predict(self.X)

        for yp in y_pred:
            yp[0] = int(round(yp[0]))

        print("accuracy is: ", accuracy_score(y, y_pred))
        print("Precision is: ", precision_score(y, y_pred))
        print("Recall is: ", recall_score(y, y_pred))
