import itertools
import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from statsmodels.tsa import ar_model


def generate_optimized_lagged_features(stock_df: pd.DataFrame, sql_variables_table: pd.DataFrame,
                                       parameters: Dict) -> pd.DataFrame:
    """Creates optimized lags for the features.

    Args:
        stock_df: Data containing features and target.
        sql_variables_table: Data containing variables from SQL.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Data with optimized lags.
    """

    def _get_best_lags(stock_df: pd.DataFrame, parameters: Dict) -> dict:
        """
        Get the best number of lags for each feature.
        Args:
            stock_df: Stock data.
            parameters: Parameters defined in parameters/data_science.yml.

        Returns:
            best_feature_lags dictionary: feature1: [lag1, lag2, ...], feature2: [lag1, lag2, ...], ...
        """
        best_lags = {}
        for feature in stock_df.columns:
            best_nr_lags = []
            for i in range(0, parameters["validation_size"]):
                val_starting_index = len(stock_df) - parameters["validation_size"] + i
                train_indexes = range(val_starting_index - parameters['training_size'], val_starting_index)

                y_train = stock_df[feature].iloc[train_indexes]

                search = ar_model.ar_select_order(endog=y_train, maxlag=parameters["max_number_lags"], ic="aic")
                if search is not None and search.ar_lags is not None:
                    best_nr_lags.extend(search.ar_lags)

            best_lags[feature] = list(set(best_nr_lags))

        return best_lags

    holidays = sql_variables_table.loc[sql_variables_table["key"] == "Market Holidays", 'value'].values[0].split(",")
    custom_business_day = pd.tseries.offsets.CustomBusinessDay(holidays=holidays)
    stock_df.index.freq = custom_business_day

    best_lags = _get_best_lags(stock_df, parameters)
    for feature, lags in best_lags.items():
        for lag in lags:
            stock_df[f"{feature}_lag_{lag}"] = stock_df[feature].shift(lag)
        if feature != parameters["target"]:
            stock_df.drop(feature, axis=1, inplace=True)

    return stock_df.dropna(), best_lags


def split_data(stock_df: pd.DataFrame, parameters: dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        stock_df: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = stock_df.drop([parameters["target"]], axis=1)
    y = stock_df[parameters["target"]].values

    return X, y


def perform_grid_search(X: pd.DataFrame, y, parameters: Dict) -> Tuple:
    """Performs grid search to find the best model with the best parameters.

    Args:
        X: Data containing features.
        y: Target array.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Best model
    """

    def _get_best_svm_regressor(X: pd.DataFrame, y, parameters_model: Dict, parameters: Dict) -> Tuple:
        param_combinations = itertools.product(parameters_model["kernel"],
                                               parameters_model["C"],
                                               parameters_model["degree"],
                                               parameters_model["gamma"])

        best_regressor: SVR
        best_rmse = math.inf
        best_regressor_returns = []
        for kernel, C, degree, gamma in param_combinations:
            returns = []
            rmse = []
            for i in range(0, parameters["validation_size"]):
                val_starting_index = len(X) - parameters["validation_size"] + i
                train_indexes = range(val_starting_index - parameters['training_size'], val_starting_index)

                X_train = X.iloc[train_indexes]
                y_train = y[train_indexes]
                X_val = X.iloc[[val_starting_index]]
                y_val = y[val_starting_index]

                # Train the model
                regressor = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)
                regressor.fit(X_train, y_train)

                # Evaluate the model
                y_pred = regressor.predict(X_val)
                rmse_score = np.sqrt(mean_squared_error([y_val], [y_pred]))
                rmse.append(rmse_score)

                returns.append(y_pred[0])

            if np.mean(rmse) < best_rmse:
                best_rmse = np.mean(rmse)
                best_regressor = regressor
                best_regressor_returns = returns

        return best_regressor, best_regressor_returns, best_rmse

    def _get_best_linear_regressor(X: pd.DataFrame, y, parameters_model: Dict, parameters: Dict) -> Tuple:

        param_combinations = itertools.product(parameters_model["fit_intercept"], parameters_model["positive"])
        best_regressor: LinearRegression
        best_rmse = math.inf
        best_regressor_returns = []
        for fit_intercept, positive in param_combinations:
            returns = []
            rmse = []
            for i in range(0, parameters["validation_size"]):
                val_starting_index = len(X) - parameters["validation_size"] + i
                train_indexes = range(val_starting_index - parameters['training_size'], val_starting_index)

                X_train = X.iloc[train_indexes]
                y_train = y[train_indexes]
                X_val = X.iloc[[val_starting_index]]
                y_val = y[val_starting_index]

                # Train the model
                regressor = LinearRegression(fit_intercept=fit_intercept,
                                             positive=positive)
                regressor.fit(X_train, y_train)

                # Evaluate the model
                y_pred = regressor.predict(X_val)
                rmse_score = np.sqrt(mean_squared_error([y_val], [y_pred]))
                rmse.append(rmse_score)

                returns.append(y_pred[0])

            if np.mean(rmse) < best_rmse:
                best_regressor = regressor
                best_rmse = np.mean(rmse)
                best_regressor_returns = returns

        return best_regressor, best_regressor_returns, best_rmse

    def _get_best_lstm_regressor(X: pd.DataFrame, y, parameters_model: Dict, parameters: Dict) -> Tuple:

        num_rows_validation = int(len(X) * parameters["validation_size"])
        # do walk forward validation
        train_size = len(X) - num_rows_validation

        best_regressor: SVR
        best_rmse = math.inf

    logger = logging.getLogger(__name__)
    models = ["linear_regression"]

    best_score = math.inf
    best_model = None
    best_model_returns = []
    best_model_name = "None"
    for model in models:
        if model == "linear_regression":
            parameters_linear_regression = {
                "fit_intercept": [True, False],
                "positive": [False, True]
            }
            model, returns, score = _get_best_linear_regressor(X, y, parameters_linear_regression, parameters)
            if score < best_score:
                best_model = model
                best_model_returns = returns
                best_score = score
                best_model_name = model.__class__.__name__
        elif model == "svm":
            parameters_svm = {
                "kernel": ["linear", "poly", "rbf"],
                "C": [1, 10, 100],
                "degree": [3, 4, 5],
                "gamma": ["scale", "auto"]
            }
            model, returns, score = _get_best_svm_regressor(X, y, parameters_svm, parameters)
            if score < best_score:
                best_model = model
                best_model_returns = returns
                best_score = score
                best_model_name = model.__class__.__name__
        elif model == "lstm":
            parameters_lstm = {
                "n_hidden": [1, 2, 3],
                "n_neurons": [10, 50, 100],
                "n_input": [1, 2, 3],
                "n_output": [1, 2, 3]
            }
            """ 
            model, returns, score = __get_best_lstm_regressor(X,y, parameters_lstm, parameters)
            if score < best_score:
                best_model = model
                best_model_returns = returns
                best_score = score
                best_model_name = "Long Short-Term Memory Regressor"
            """

    true_signals = [1 if return_val > 0 else 0 for return_val in y[-parameters["validation_size"]:]]
    predicted_signals = [1 if return_val > 0 else 0 for return_val in best_model_returns]
    accuracy = accuracy_score(true_signals, predicted_signals)
    information_ratio = np.mean(best_model_returns) / np.std(best_model_returns) if np.std(best_model_returns) != 0 else 0

    logger.info("\x1b[34mBest model is %s with Walk-Forward Validation RMSE of %.3f\x1b[39m", best_model_name,
                best_score)
    return best_model, {
        "RMSE": best_score.round(5),
        "Accuracy": accuracy.round(5),
        "Information Ratio": information_ratio.round(5)
    }


def predict_return(original_df: pd.DataFrame, processed_df: pd.DataFrame, best_lags: dict, regressor,
                   regressor_validation_scores: dict,
                   sql_variables_table: pd.DataFrame, parameters: Dict) -> dict:
    """Predicts the return of a stock using the model.

    Args:
        original_df: Original data containing features and target.
        processed_df: Processed data containing features and target.
        best_lags: Best lags for each feature.
        regressor: Model to use for prediction.
        regressor_validation_score: Validation score of the model.
        sql_variables_table: Data containing variables from SQL.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Predicted returns.
    """
    holidays = sql_variables_table.loc[sql_variables_table["key"] == "Market Holidays", 'value'].values[0].split(",")
    custom_business_day = pd.tseries.offsets.CustomBusinessDay(holidays=holidays)
    processed_df.index.freq = custom_business_day

    # Concatenate the new row to the stock_df
    processed_df.loc[processed_df.index[-1] + custom_business_day] = pd.Series(index=processed_df.columns,
                                                                               dtype=processed_df.dtypes.values)

    for feature, lags in best_lags.items():
        for lag in lags:
            processed_df[f"{feature}_lag_{lag}"] = processed_df[feature].shift(lag)
        if feature != parameters["target"]:
            processed_df.drop(feature, axis=1, inplace=True)
    processed_df.dropna(subset=processed_df.columns.drop(parameters["target"]), inplace=True)

    training_starting_index = len(processed_df) - parameters["training_size"]
    X = processed_df.drop(parameters["target"], axis=1)
    y = processed_df[parameters["target"]].values
    X_train = X.iloc[training_starting_index: - 1]
    y_train = y[training_starting_index: - 1]
    X_test = X.iloc[[-1]]

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)[0]

    predicted_close_price = original_df['close'].iloc[-1] * (1 + y_pred)
    return {
        "Date": pd.to_datetime(X_test.index[-1]).strftime("%Y-%m-%d"),
        "Predicted Return": y_pred,
        "Predicted Close Price": predicted_close_price.round(2),
        "Regressor": regressor.__class__.__name__,
        "Validation Scores": regressor_validation_scores,
        "Best Lags": best_lags
    }
