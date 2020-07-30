'''
Married : 1  = yes, 0 = no
Gender: 1 = Male, 0 = Female

'''

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateparser import parse
from pandas.plotting import scatter_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from statsmodels.tools.tools import add_constant

# comment out the tensorflow imports if you want to run without modeling
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from plot_creator import get_stacked_bars
from random_forest import reg_model
from mlp import Mlp_Model

plt.rcParams.update({'font.size': 16})
plt.style.use('fivethirtyeight')
plt.close('all')

def load_df():
    cols = np.arange(0,23)
    df = pd.read_csv("../data/Interview.csv", usecols=cols)
    df.drop(1233, inplace=True)
    df.drop(['Candidate Current Location', 'Name(Cand ID)'], axis=1, inplace=True)
        
    return df

def fix_date(df):
    '''
    Replaces Na dates
    '''
    default_date = datetime.date(2000, 1, 1)
    Na_indices = df[df['Date'] == 'Na'].index
    df.loc[Na_indices, 'Date'] = default_date
    df['Date'] = df['Date'].apply(parse)
    df.loc[505:524, 'Date'] = parse('28.08.2016')
    break_down_date(df, default_date)
    return df

def break_down_date(df, default_date, min_year = 2015, max_year = 2016):
    df['Year'] = df.loc[df['Date'] > default_date, 'Date'].apply(lambda x: x.year)
    df.loc[(df['Year'] < min_year) | (df['Year'] > max_year), 'Year'] = 'Unknown'
    df['Month'] = df.loc[df['Date'] > default_date,
                        'Date'].apply(lambda x: x.month)
    df['Day'] = df.loc[df['Date'] > default_date,
                         'Date'].apply(lambda x: x.day)
    #0 is Monday
    df['Day of Week'] = df['Date'].apply(lambda x: x.dayofweek)
    df.loc[:, ['Year', 'Month', 'Day']].fillna('Unknown', inplace = True)
    return df

def combine_types(df, col, other_entries, combine_to):
    for entry in other_entries:
        df[col].replace(entry, combine_to, inplace = True)
    return df


def df_col_setup(df):
    col_rename_dict = {
        'Date of Interview': 'Date',
        'Client name': 'Company',
        'Position to be closed': 'Position',
        'Nature of Skillset': 'Skillset',
        "Have you obtained the necessary permission to start at the required time": "Permissions",
        'Hope there will be no unscheduled meetings': 'No Unscheduled Meetings',
        "Can I Call you three hours before the interview and follow up on your attendance for the interview": '3 Hour Confirmation Call',
        "Can I have an alternative number/ desk number. I assure you that I will not trouble you too much": "Alternate Phone Number",
        'Have you taken a printout of your updated resume. Have you read the JD and understood the same': "Took Resume and Read JD",
        'Are you clear with the venue details and the landmark.' : 'Confirmed Location',
        'Has the call letter been shared' : 'Call Letter Shared',
        'Marital Status' : 'Married'
    }

    df.rename(columns = col_rename_dict, inplace = True)
    cols_with_NaN = ['No Unscheduled Meetings', 'Permissions',
                     'Alternate Phone Number', 'Took Resume and Read JD', 'Confirmed Location', 'Call Letter Shared', '3 Hour Confirmation Call', 'Expected Attendance']
    for col in cols_with_NaN:
        df[col].fillna('Na', inplace = True)

    clean_col(df, ["Observed Attendance", 'No Unscheduled Meetings', "Candidate Job Location", 
                    "Location", 'Permissions', 'Alternate Phone Number', 'Call Letter Shared', '3 Hour Confirmation Call', 'Took Resume and Read JD',
                    'Confirmed Location'])
    
    # If Not sure or NA is provided, grouped into No for boolean values - may reconsider adding third "Unknown" entry later
    cols_to_bool = [('Observed Attendance', 'yes'),
                    ('Married', 'Married'), 
                    ('Gender', 'Male'),
                    ('No Unscheduled Meetings', 'yes'),
                    ('Permissions', 'yes'),
                    ('Alternate Phone Number', 'yes'),
                    ('3 Hour Confirmation Call', 'yes'),
                    ('Call Letter Shared', 'yes'),
                    ('Took Resume and Read JD', 'yes'),
                    ('Confirmed Location', 'yes')
                    ]
    for col, def_1 in cols_to_bool:
        convert_to_boolean(df, col, def_1)
    

    #Combine redundant IT variations for Company into IT
    combine_types(df, col='Industry', other_entries=('IT Services', 'IT Products and Services'), combine_to = 'IT')
    combine_types(df, col='Location', other_entries=(
        'gurgaonr'), combine_to='gurgaon')
    combine_types(df, 'Expected Attendance', ['NO', 'Uncertain', 'Na'], 'No')
    combine_types(df, 'Expected Attendance', ['yes', '10.30 Am', '11:00 AM'], 'Yes')
    combine_types(df, 'Interview Type', ['Walkin', 'Walkin '], 'Walk In')
    combine_types(df, 'Interview Type', ['Scheduled Walkin', 'Sceduled walkin'], 'Scheduled Walk In')
    for col in ['Location', 'Candidate Job Location', 'Interview Venue']:
        df[col] = df[col].apply(lambda x: x.title())
    create_local_col(df)
    fix_date(df)

def convert_to_boolean(df, col, def_1):
    df[col][df[col] == def_1] = 1
    df[col][df[col] != 1] = 0
    return df

def clean_col(df, cols):
    '''
    Lowercase all possible letters
    Remove spaces
    '''
    for col in cols:
        df[col] = df[col].apply(lambda x: x.lower().strip())
    return df

def create_local_col(df):
    df["Local Candidate"] = -1
    for i in range(df.shape[0]):
        if df["Location"][i] == df["Candidate Job Location"][i]:
            df.loc[i, "Local Candidate"] = 1
        else:
            df.loc[i, "Local Candidate"] = 0

def compare_bools(df, col1, col2):
    col1y_col2y = df[(df[col1] == 1) & (df[col2] == 1)].shape[0]
    col1y_col2n = df[(df[col1] == 1) & (df[col2] == 0)].shape[0]
    col1n_col2y = df[(df[col1] == 0) & (df[col2] == 1)].shape[0]
    col1n_col2n = df[(df[col1] == 0) & (df[col2] == 0)].shape[0]
    return col1y_col2y, col1y_col2n, col1n_col2y, col1n_col2n

def hot_encode():
    # Industry column has ~20 for one hot encoding
    X = df.select_dtypes(exclude=['number']).apply(LabelEncoder().fit_transform).join(df.select_dtypes(include=['number']))


def mlp_model(num_neur_hid = 10, num_epochs = 500):
    df_X = pd.read_csv("../data/onehot_dataframe_notime.csv",index_col="Unnamed: 0")
    df_y = pd.read_csv("../data/onehot_y.csv",index_col="Unnamed: 0")
    # df_X.drop(['Gender', 'Permissions', 'No_Unscheduled_Meetings', 
            # 'alt_phone', 'Take_Resume', 'Confirmed_Location', 
            # 'Call_Letter', 'Married', 'is_local'],
            # axis = 1, inplace=True)
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    nn_hl = 4 # number of neurons in the hidden layer
    num_epochs = 500 # number of times to train on the entire training set
    batch_size = X.shape[0] # using batch gradient descent
    mlp = define_hl_mlp_model(X, nn_hl)
    activ='sigmoid'

    num_coef = X.shape[1]
    model = Sequential() # sequential model is a linear stack of layers
    model.add(Dense(units=nn_hl,
                    input_shape=(num_coef,),
                    activation=activ, 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', 
                    kernel_regularizer=None, 
                    bias_regularizer=None, 
                    activity_regularizer=None, 
                    kernel_constraint=None, 
                    bias_constraint=None))
    model.add(Dense(units=1,
                    activation=activ, 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform')) 
    sgd = SGD(lr=1e-3, decay=1e-7, momentum=0.9) # using stochastic gradient descent
    model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["mse"] )

    mlp.fit(X, y, batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True)
    y_pred = mlp.predict(X)

    for yp in y_pred:
        yp[0] = int(round(yp[0]))

    print("accuracy is: ", accuracy_score(y, y_pred))


if __name__ == "__main__":
    df = load_df()
    df_col_setup(df)
    create_plots = True
    get_random_forest = True
    logistic_regression_analysis = True

    # if create_plots == True:
    #     for col in df.columns:
    #         try:
    #             get_stacked_bars(df, x=col, y='Observed Attendance')
    #         except:
    #             continue
    
    if get_random_forest == True:
        rf_df = df.copy()
        rf_df = rf_df.drop(['Date', 'Company', 'Industry', 'Location', 'Position', 'Skillset', 'Interview Type', 'Candidate Job Location', 'Interview Venue', 'Candidate Native location', 'Year', 'Month', 'Day', 'Day of Week', 'Expected Attendance'], axis = 1)
        y = rf_df.pop('Observed Attendance')
        X = rf_df
        rand_forest_model = reg_model(X, y)
        rand_forest_model.rand_forest()
        rand_forest_model.evaluate_model()
        rand_forest_model.get_feature_importances()
    
    if logistic_regression_analysis == True:
        logreg_df = df.copy()
        logreg_df = logreg_df.drop(['Date', 'Company', 'Industry', 'Location', 'Position', 'Skillset', 'Interview Type',
                                    'Candidate Job Location', 'Interview Venue', 'Candidate Native location', 'Year', 'Month', 'Day', 'Day of Week'], axis=1)
        y = logreg_df.pop('Observed Attendance')
        X = logreg_df
        logreg_model = reg_model(X, y)
        logreg_model.log_reg()
        logreg_model.evaluate_model()

    original_exp_attend = df.pop('Expected Attendance')
