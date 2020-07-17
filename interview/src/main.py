'''
Married : 1  = yes, 0 = no
Gender: 1 = Male, 0 = Female

'''

from dateparser import parse
# Marc please run conda install dateparser in your terminal if this doesn't work

import numpy as np
import pandas as pd
import datetime
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statsmodels.tools.tools import add_constant

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
    others = ('IT Services', 'IT Products and Services')
    for each in others:
        df['Industry'][df['Industry'] == each] = 'IT'

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


def log_reg(df):
    # taken from Logistic Regression Solutions Notebook
    # https://github.com/GalvanizeDataScience/solutions-g114/blob/master/logistic-regression/logistic_regression_solutions.ipynb
    X = df[['3hr_call', 'alt_phone', 'Married','is_local']].values
    X_const = add_constant(X, prepend=True)
    y = df['Observed Attendance'].values

    logit_model = Logit(y, X_const).fit()

    logit_model.summary()

    kfold = KFold(n_splits=5)

    accuracies = []
    precisions = []
    recalls = []

    # why splitting after fitting - leakage?
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    for train_index, test_index in kfold.split(X_train):
        model = LogisticRegression(solver="lbfgs")
        model.fit(X[train_index], y[train_index])
        y_predict = model.predict(X[test_index])
        y_true = y[test_index]
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

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
    
    


















    
    
