import matplotlib.pyplot as plt
import numpy as np


def x_val_analyzer(df, x):
    if x == 'Day of Week':
        answer_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                       3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    elif x == 'Month':
        answer_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                       7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    elif x == 'Gender':
        answer_dict = {1: 'Male', 0: 'Female'}
    elif len(df[x].unique()) == 2:
        answer_dict = {1: 'Yes', 0: 'No'}
    else:
        order = None
        return df, order
    df[x] = df[x].apply(lambda x: answer_dict[x])
    order = list(answer_dict.values())
    return df, order

def get_stacked_bars(df, x='Day of Week', y='Observed Attendance', override_x = None):
    specific_df = df.loc[:, [x, y, 'Date']]
    specific_df = specific_df.groupby([x, y]).count().reset_index()
    specific_df.rename(columns={'Date': 'Count'}, inplace=True)
    showDict = {0: 'No Show', 1: 'Show'}
    specific_df[y] = specific_df[y].apply(lambda z: showDict[z])
    specific_df, order = x_val_analyzer(specific_df, x)

    pivot_df = specific_df.pivot(index=x, columns=y, values='Count')
    pivot_df.fillna(0, inplace=True)

    if order == None:
        pivot_df = pivot_df.loc[:, ['Show', 'No Show']]
    else:
        pivot_df = pivot_df.loc[order, ['Show', 'No Show']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pivot_df.plot.bar(stacked=True, color=[
        'blue', 'red'], figsize=(16, 7), ax=axes[0])

    pivot_df['Percent Show'] = round(
        100 * pivot_df['Show'] / (pivot_df['Show'] + pivot_df['No Show']), 2)
    pivot_df['Percent No Show'] = round(
        100 * pivot_df['No Show'] / (pivot_df['Show'] + pivot_df['No Show']), 2)
    pivot_df[['Percent Show', 'Percent No Show']].plot.bar(
        stacked=True, color=['blue', 'red'], figsize=(16, 7), ax=axes[1])

    if override_x != None:
        x = override_x
    axes[0].set_xlabel(x)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('Count')
    axes[0].set_title('{} Based on\n {}'.format(y, x))
    axes[1].set_xlabel(x)
    axes[1].set_ylabel('%') 
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('Percent {} Based on\n {}'.format(y, x))

    x = x.replace(' ', '_')
    fig.tight_layout()
    fig.savefig('../images/{}.png'.format(x), dpi=500)
