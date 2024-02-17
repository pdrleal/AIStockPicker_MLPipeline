# Function to create lag features
import pandas as pd


def series_to_supervised(data, n_in_start=1, n_in_end=8, n_out=1, dropnan=True, varNames=None):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in_start: Minimum number of lag observations as input (X).
        n_in_end: Maximum umber of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether to drop rows with NaN values.
        varNames: List of column names (same size as the number of variables).
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in_end, n_in_start - 1, -1):
        cols.append(df.shift(i))
        names += [(varNames[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(varNames[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(varNames[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
