# Function to create lag features
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, root_mean_squared_error, \
    mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


# python colors: https://mail.python.org/pipermail/python-list/2009-August/546528.html

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


def minimize_evaluation_metrics() -> list:
    """
    Get a list of evaluation metrics to minimize.

    Returns:
        metrics: List of evaluation metrics to minimize
    """
    return [metric for metric, description in available_evaluation_metrics().items() if
            'error' in description.lower() or 'loss' in description.lower()]


def maximize_evaluation_metrics() -> list:
    """
    Get a list of evaluation metrics to maximize.

    Returns:
        metrics: List of evaluation metrics to maximize
    """
    return [metric for metric in available_evaluation_metrics() if metric not in minimize_evaluation_metrics()]


def available_evaluation_metrics() -> dict:
    """
    Get a dictionary of available evaluation metrics with their descriptions.

    Returns:
        metrics: Dictionary with evaluation metrics as keys and their descriptions as values
    """
    metrics = {
        'rmse': 'Root Mean Squared Error (RMSE)',
        'mape': 'Mean Absolute Percentage Error (MAPE)',
        'r2_score': 'R-squared Score (R2)',
        'information_ratio': 'Information Ratio',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    return metrics


def inverse_transform_returns(true_returns, predicted_returns, scaler_object):
    """
    Inverse transform the true and predicted returns.

    Arguments:
        scaler_object: Scaler object
        true_returns: True returns
        predicted_returns: Predicted returns
    """
    # scale back the returns
    inverse_scaler_object = MinMaxScaler()
    inverse_scaler_object.min_, inverse_scaler_object.scale_ = scaler_object.min_[0], scaler_object.scale_[0]

    if true_returns is None:
        return inverse_scaler_object.inverse_transform(np.array(predicted_returns).reshape(-1, 1)).flatten()

    true_returns = inverse_scaler_object.inverse_transform(np.array(true_returns).reshape(-1, 1)).flatten()
    predicted_returns = inverse_scaler_object.inverse_transform(np.array(predicted_returns).reshape(-1, 1)).flatten()
    return true_returns, predicted_returns


def generate_scores_from_returns(true_returns, predicted_returns, scaler_object) -> dict:
    """
    Generate scores from true and predicted returns.

    Arguments:
        scaler_object: Scaler object
        true_returns: True returns
        predicted_returns: Predicted returns

    Returns:
        scores: Dictionary with scores for the specified metrics
    """
    # scale back the returns
    true_returns, predicted_returns = inverse_transform_returns(true_returns=true_returns,
                                                                predicted_returns=predicted_returns,
                                                                scaler_object=scaler_object)

    true_labels = [1 if val > 0 else 0 for val in true_returns]
    predicted_labels = [1 if val > 0 else 0 for val in predicted_returns]

    scores = {}
    available_metrics = available_evaluation_metrics()
    for metric in available_metrics:
        if metric == 'rmse':
            scores[metric] = root_mean_squared_error(true_returns, predicted_returns)
        elif metric == 'mape':
            scores[metric] = mean_absolute_percentage_error(true_returns, predicted_returns)
        elif metric == 'r2_score':
            scores[metric] = r2_score(true_returns, predicted_returns)
        elif metric == 'information_ratio':
            scores[metric] = (np.mean(predicted_returns) / np.std(predicted_returns)) if np.std(
                predicted_returns) != 0 else 0
        elif metric == 'accuracy':
            scores[metric] = accuracy_score(true_labels, predicted_labels)
        elif metric == 'precision':
            scores[metric] = precision_score(true_labels, predicted_labels, zero_division=0)
        elif metric == 'recall':
            scores[metric] = recall_score(true_labels, predicted_labels, zero_division=0)
        elif metric == 'f1_score':
            scores[metric] = f1_score(true_labels, predicted_labels, zero_division=0)

    return scores
