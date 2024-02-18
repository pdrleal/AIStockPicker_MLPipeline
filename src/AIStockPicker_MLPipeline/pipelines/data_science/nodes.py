import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def __get_best_svm_regressor(stock_df: pd.DataFrame, parameters: Dict, parameters_model: Dict) -> Tuple:
    # Perform grid search to find the best parameters
    X = stock_df.drop(["return"], axis=1)
    y = stock_df[["return"]]

    num_rows_validation = int(len(X) * parameters["validation_size"])
    # do walk forward validation
    train_size = len(X) - num_rows_validation

    best_regressor: SVR
    best_rmse = math.inf
    for kernel in parameters_model["kernel"]:
        for C in parameters_model["C"]:
            for degree in parameters_model["degree"]:
                for gamma in parameters_model["gamma"]:
                    rmse = []
                    for i in range(0, num_rows_validation):
                        X_train = X.iloc[i:train_size + i]
                        y_train = y.iloc[i:train_size + i]
                        X_val = X.iloc[train_size + i:train_size + i + 1]
                        y_val = y.iloc[train_size + i:train_size + i + 1]

                        # Train the model
                        regressor = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)
                        regressor.fit(X_train, y_train)

                        # Evaluate the model
                        y_pred = regressor.predict(X_val)
                        rmse_score = np.sqrt(mean_squared_error(y_val, y_pred))
                        rmse.append(rmse_score)

                    if np.mean(rmse) < best_rmse:
                        best_rmse = np.mean(rmse)
                        best_regressor = regressor

    return best_rmse, best_regressor.get_params()


def __get_best_linear_regressor(stock_df: pd.DataFrame, parameters: Dict, parameters_model: Dict) -> Tuple:
    # Perform grid search to find the best parameters
    X = stock_df.drop(["return"], axis=1)
    y = stock_df[["return"]]

    num_rows_validation = int(len(X) * parameters["validation_size"])
    # do walk forward validation
    train_size = len(X) - num_rows_validation

    best_regressor: LinearRegression
    best_rmse = math.inf
    for fit_intercept in parameters_model["fit_intercept"]:
        for positive in parameters_model["positive"]:
            rmse = []
            for i in range(0, num_rows_validation):
                X_train = X.iloc[i:train_size + i]
                y_train = y.iloc[i:train_size + i]
                X_val = X.iloc[train_size + i:train_size + i + 1]
                y_val = y.iloc[train_size + i:train_size + i + 1]

                # Train the model
                regressor = LinearRegression(fit_intercept=fit_intercept,
                                             positive=positive)
                regressor.fit(X_train, y_train)

                # Evaluate the model
                y_pred = regressor.predict(X_val)
                rmse_score = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse.append(rmse_score)

            if np.mean(rmse) < best_rmse:
                best_rmse = np.mean(rmse)
                best_regressor = regressor
    return best_rmse, best_regressor


def __get_best_lstm_regressor(stock_df: pd.DataFrame, parameters: Dict, parameters_model: Dict) -> Tuple:
    # Perform grid search to find the best parameters
    X = stock_df.drop(["return"], axis=1)
    y = stock_df[["return"]]

    num_rows_validation = int(len(X) * parameters["validation_size"])
    # do walk forward validation
    train_size = len(X) - num_rows_validation

    best_regressor: SVR
    best_rmse = math.inf


def perform_grid_search(stock_df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Performs grid search to find the best parameters for the model.

    Args:
        stock_df: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Best parameters.
    """
    logger = logging.getLogger(__name__)
    models = ["linear_regression", "svm"]

    best_score = math.inf
    best_model = None
    best_model_name = "None"
    for model in models:
        if model == "linear_regression":
            parameters_linear_regression = {
                "fit_intercept": [True, False],
                "positive": [True, False]
            }
            score, model = __get_best_linear_regressor(stock_df, parameters, parameters_linear_regression)
            if score < best_score:
                best_score = score
                best_model = model
                best_model_name = "Linear Regression"
        elif model == "svm":
            parameters_svm = {
                "kernel": ["linear","poly", "rbf"],
                "C": [1, 10, 100],
                "degree": [3, 4, 5],
                "gamma": ["scale", "auto"]
            }
            score, model = __get_best_svm_regressor(stock_df, parameters, parameters_svm)
            if score < best_score:
                best_score = score
                best_model = model
                best_model_name = "Support Vector Machine Regressor"
        elif model == "lstm":
            parameters_lstm = {
                "n_hidden": [1, 2, 3],
                "n_neurons": [10, 50, 100],
                "n_input": [1, 2, 3],
                "n_output": [1, 2, 3]
            }
            """ 
            score, model = __get_best_lstm_regressor(stock_df, parameters, parameters_lstm)
            if score < best_score:
                best_score = score
                best_model = model
                best_model_name = "Long Short-Term Memory Regressor"
            """

    logger.info("\x1b[34mBest model is %s with Walk-Forward Validation RMSE of %.3f\x1b[39m", best_model_name,
                best_score)
    return best_score, best_model


def split_data(stock_df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        stock_df: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = stock_df.drop(["return"], axis=1)
    y = stock_df["return"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=parameters["validation_size"], shuffle=False
    )
    return X_train, X_val, y_train, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
        regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return score
