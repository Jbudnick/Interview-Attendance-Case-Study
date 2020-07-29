import matplotlib.pyplot as plt

def get_stacked_bars(df, x='dayofweek', y='Observed Attendance', order=None, override_x = None):
    specific_df = df.loc[:, [x, y, 'Date']]
    specific_df = specific_df.groupby([x, y]).count().reset_index()
    specific_df.rename(columns={'Date': 'Count'}, inplace=True)
    showDict = {0: 'No Show', 1: 'Show'}
    specific_df[y] = specific_df[y].apply(lambda z: showDict[z])

    pivot_df = specific_df.pivot(index=x, columns=y, values='Count')
    pivot_df.fillna(0, inplace=True)
    if order == None:
        pivot_df = pivot_df.loc[:, ['Show', 'No Show']]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pivot_df.plot.bar(stacked=True, color=[
        'blue', 'red'], figsize=(16, 7), ax=axes[0])
    pivot_df.plot.bar(stacked=True, color=[
                        'blue', 'red'], figsize=(16, 7), ax=axes[0])

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

    x = x.replace(' ', '_')
    fig.savefig('../images/{}.png'.format(x), dpi=300)