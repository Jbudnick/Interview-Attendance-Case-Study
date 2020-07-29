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
    default_date = '2000-1-1'
    Na_indices = df[df['Date'] == 'Na'].index
    df.loc[Na_indices, 'Date'] = default_date
    df['Date'] = df['Date'].apply(parse)
    df.loc[505:524, 'Date'] = parse('28.08.2016')
    break_down_date(df)
    return df

def get_stacked_bars(df, x='dayofweek', y='Observed Attendance', order=None):
    specific_df = df.loc[:, [x, y, 'Date']]
    specific_df = specific_df.groupby([x, y]).count().reset_index()
    specific_df.rename(columns={'Date': 'Count'}, inplace=True)
    showDict = {0: 'No Show', 1: 'Show'}
    specific_df[y] = specific_df[y].apply(lambda z: showDict[z])

    pivot_df = specific_df.pivot(index=x, columns=y, values='Count')
    pivot_df.fillna(0, inplace = True)
    if order == None:
        pivot_df = pivot_df.loc[:, ['Show', 'No Show']]
    else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pivot_df.plot.bar(stacked=True, color=[
                      'blue', 'red'], figsize=(16, 7), ax=axes[0])
    pivot_df.plot.bar(stacked = True, color = ['blue', 'red'], figsize=(16,7), ax = axes[0])

    pivot_df['percent show'] = round(
        100 * pivot_df['Show'] / (pivot_df['Show'] + pivot_df['No Show']), 2)
    pivot_df['percent no show'] = round(
        100 * pivot_df['No Show'] / (pivot_df['Show'] + pivot_df['No Show']), 2)
    pivot_df[['percent show', 'percent no show']].plot.bar(
        stacked=True, color=['blue', 'red'], figsize=(16, 7), ax=axes[1])

    
    if override_x != None:
        x = override_x
    axes[0].set_xlabel(x)
    axes[0].set_ylabel('Count')
    axes[0].set_title('{} Based on {}'.format(y, x))
    
    axes[1].set_xlabel(x)
    axes[1].set_ylabel('%')

    axes[1].set_title('Percent {} Based on {}'.format(y, x))
    
        df[col].replace(entry, combine_to, inplace=True)
        fig.savefig('../images/{}.png'.format(x), dpi = 300)

def break_down_date(df):
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['day'] = df['Date'].apply(lambda x: x.day)
    #0 is Monday
    df['dayofweek'] = df['Date'].apply(lambda x: x.dayofweek)
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
        'Hope there will be no unscheduled meetings': 'No_Unscheduled_Meetings',
        "Can I Call you three hours before the interview and follow up on your attendance for the interview": '3hr_call',
        "Can I have an alternative number/ desk number. I assure you that I will not trouble you too much": "alt_phone",
        'Have you taken a printout of your updated resume. Have you read the JD and understood the same': "Take_Resume",
        'Are you clear with the venue details and the landmark.' : 'Confirmed_Location',
        'Has the call letter been shared' : 'Call_Letter',
        'Marital Status' : 'Married'
    }

    df.rename(columns = col_rename_dict, inplace = True)
    cols_with_NaN = ['No_Unscheduled_Meetings', 'Permissions',
                     'alt_phone', 'Take_Resume', 'Confirmed_Location', 'Call_Letter', '3hr_call']
    for col in cols_with_NaN:
        df[col].fillna('Na', inplace = True)

    clean_col(df, ["Observed Attendance", 'No_Unscheduled_Meetings', "Candidate Job Location", 
                    "Location", 'Permissions', 'alt_phone', 'Call_Letter', '3hr_call', 'Take_Resume',
                    'Confirmed_Location'])
    
    # If Not sure or NA is provided, grouped into No for boolean values - may reconsider adding third "Unknown" entry later
    cols_to_bool = [('Observed Attendance', 'yes'),
                    ('Married', 'Married'), 
                    ('Gender', 'Male'),
                    ('No_Unscheduled_Meetings', 'yes'),
                    ('Permissions', 'yes'),
                    ('alt_phone', 'yes'),
                    ('3hr_call', 'yes'),
                    ('Call_Letter', 'yes'),
                    ('Take_Resume', 'yes'),
                    ('Confirmed_Location', 'yes')
                    ]
    for col, def_1 in cols_to_bool:
        convert_to_boolean(df, col, def_1)
    

    #Combine redundant IT variations for Company into IT
    combine_types(df, col='Industry', other_entries=('IT Services', 'IT Products and Services'), combine_to = 'IT')
    combine_types(df, col='Location', other_entries=(
        'gurgaonr'), combine_to='gurgaon')
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
    df["is_local"] = -1
    for i in range(df.shape[0]):
        if df["Location"][i] == df["Candidate Job Location"][i]:
            df.loc[i, "is_local"] = 1
        else:
            df.loc[i, "is_local"] = 0

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
    """ 
    taken from Perceptron Lecture - 3_layer_keras.py
    https://github.com/GalvanizeDataScience/lectures/blob/Denver/perceptrons/frank-burkholder/3_layer_keras.py
    """

    df_X = pd.read_csv("../data/onehot_dataframe_notime.csv",index_col="Unnamed: 0")
    df_y = pd.read_csv("../data/onehot_y.csv",index_col="Unnamed: 0")
    # Code to drop all columns:
    # df_X.drop(['Gender', 'Permissions', 'No_Unscheduled_Meetings', 
    #       'alt_phone', 'Take_Resume', 'Confirmed_Location', 
    #       'Call_Letter', 'Married', 'is_local', '3hr_call']

    X = df_X.to_numpy()
    y = df_y.to_numpy()
    
    batch_size = X.shape[0] # using batch gradient descent
    mlp = define_hl_mlp_model(X, num_neur_hid)
    activ='sigmoid'

    num_coef = X.shape[1]
    model = Sequential() # sequential model is a linear stack of layers
    model.add(Dense(units=num_neur_hid,
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


    print("Accuracy:", np.average(accuracies))
    print("Precision:", np.average(precisions))
    print("Recall:", np.average(recalls))


if __name__ == "__main__":
    df = load_df()
    df_col_setup(df)

    # log_reg(df)

    # for bool_col in ['is_local', '3hr_call', 'alt_phone', 'Permissions', 'Married']:
    #     col1y_col2y, col1y_col2n, col1n_col2y, col1n_col2n = compare_bools(df, bool_col, 'Observed Attendance')
    #     if (col1y_col2y + col1y_col2n == 0):
    #         attend_ratio_col1 = col1y_col2y / 0.0001
    #     else: 
    #         attend_ratio_col1 = col1y_col2y / (col1y_col2y + col1y_col2n)
    #     if (col1y_col2y + col1y_col2n) == 0:
    #         attend_ratio_col2 = col1n_col2y / 0.0001
    #     else:
    #         attend_ratio_col2 = col1n_col2y / (col1n_col2y + col1n_col2n)
            
    #     fig, ax = plt.subplots(figsize = (12,8))

    #     not_label = 'not ' + str(bool_col)
    #     ax.bar(x = [bool_col, not_label], height = [attend_ratio_col1, attend_ratio_col2])
        
    #     file_name = "../images/" + str(bool_col) + "_ratio.png"
    #     ax.set_ylabel("Attendance Ratio (Attend vs Not Attend)")
    #     plt.savefig(file_name)
    

    original_exp_attend = df.pop('Expected Attendance')
