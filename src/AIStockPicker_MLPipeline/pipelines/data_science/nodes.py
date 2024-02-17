import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from AIStockPicker_MLPipeline.utils import series_to_supervised


def perform_grid_search(stock_df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Performs grid search to find the best parameters for the model.

    Args:
        stock_df: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Best parameters.
    """
    # Perform grid search to find the best parameters
    X = stock_df.drop(["return"], axis=1)
    y = stock_df[["return"]]

    num_rows_validation = int(len(X) * parameters["validation_size"])
    # do walk forward validation
    train_size = len(X) - num_rows_validation

    average_rmse = 0
    for i in range(0, num_rows_validation):
        X_train = X.iloc[i:train_size + i]
        y_train = y.iloc[i:train_size + i]
        X_val = X.iloc[train_size + i:train_size + i + 1]
        y_val = y.iloc[train_size + i:train_size + i + 1]

        # Train the model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Evaluate the model
        y_pred = regressor.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        average_rmse += score
        logger = logging.getLogger(__name__)

    average_rmse = average_rmse / num_rows_validation
    logger.info("Model has a RMSE of %.3f on validation data.", average_rmse)
    return X_train, X_val, y_train, y_val


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
