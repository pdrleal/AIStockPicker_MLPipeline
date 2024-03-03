import itertools
import logging
import math

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.ar_model import AutoReg

from AIStockPicker_MLPipeline.utils import generate_scores_from_returns, available_evaluation_metrics, \
    minimize_evaluation_metrics, inverse_transform_returns


def generate_optimized_lagged_features(stock_df: pd.DataFrame, sql_variables_table: pd.DataFrame,
                                       parameters: dict) -> tuple:
    """Creates optimized lags for the features.

    Args:
        stock_df: Data containing features and target.
        sql_variables_table: Data containing variables from SQL.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Data with optimized lags.
    """

    def _get_best_lags(stock_df: pd.DataFrame, parameters: dict) -> dict:
        """
        Using AutoReg to find the best lags for each feature by minimizing the AIC.
        Args:
            stock_df: Stock data.
            parameters: Parameters defined in parameters/data_science.yml.

        Returns:
            best_feature_lags dictionary: feature1: [lag1, lag2, ...], feature2: [lag1, lag2, ...], ...
        """
        best_lags = {}
        for feature in stock_df.columns:
            val_starting_index = len(stock_df) - parameters["validation_size"]
            train_indexes = range(val_starting_index - parameters['training_size'], val_starting_index)

            y_train = stock_df[feature].iloc[train_indexes]

            min_aic = float('inf')
            best_lag = 1
            for lag in range(1, parameters["max_number_lags"] + 1):
                aic = AutoReg(y_train, lags=lag).fit().aic
                # TODO VALIDAR none aic
                if aic is not None and aic < min_aic:
                    min_aic = aic
                    best_lag = lag

            best_lags[feature] = list(range(1, best_lag + 1))

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


def split_data(stock_df: pd.DataFrame, parameters: dict) -> tuple:
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


def perform_grid_search(X: pd.DataFrame, y, parameters: dict, scaler_object) -> tuple:
    """Performs grid search to find the best model with the best parameters.

    Args:
        X: Data containing features.
        y: Target array.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Best model
    """

    def _get_best_linear_regressor(X: pd.DataFrame, y, parameters_model: dict, parameters: dict) -> tuple:

        param_combinations = itertools.product(parameters_model["fit_intercept"], parameters_model["positive"])
        best_regressor: LinearRegression
        best_optimize_score = math.inf if parameters[
                                              "evaluation_metric"] in minimize_evaluation_metrics() else -math.inf
        best_all_scores = {}
        best_regressor_returns = []
        for fit_intercept, positive in param_combinations:
            true_returns = []
            predicted_returns = []
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

                true_returns.append(y_val)
                predicted_returns.append(y_pred[0])

            all_scores = generate_scores_from_returns(true_returns, predicted_returns, scaler_object)

            if parameters["evaluation_metric"] in minimize_evaluation_metrics():
                if all_scores[parameters["evaluation_metric"]] < best_optimize_score:
                    best_regressor = regressor
                    best_optimize_score = all_scores[parameters["evaluation_metric"]]
                    best_all_scores = all_scores
                    best_regressor_returns = predicted_returns
            else:
                if all_scores[parameters["evaluation_metric"]] > best_optimize_score:
                    best_regressor = regressor
                    best_optimize_score = all_scores[parameters["evaluation_metric"]]
                    best_all_scores = all_scores
                    best_regressor_returns = predicted_returns

        logger.info("Best Linear Regressor with Walk-Forward Validation %s of %.3f",
                    available_evaluation_metrics()[parameters["evaluation_metric"]], best_optimize_score)
        return best_regressor, best_regressor_returns, best_optimize_score, best_all_scores

    def _get_best_svm_regressor(X: pd.DataFrame, y, parameters_model: dict, parameters: dict) -> tuple:
        param_combinations = itertools.product(parameters_model["kernel"],
                                               parameters_model["C"],
                                               parameters_model["degree"],
                                               parameters_model["gamma"])

        best_regressor: SVR
        best_optimize_score = math.inf if parameters[
                                              "evaluation_metric"] in minimize_evaluation_metrics() else -math.inf
        best_all_scores = {}
        best_regressor_returns = []
        for kernel, C, degree, gamma in param_combinations:
            true_returns = []
            predicted_returns = []
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
                y_pred = regressor.predict(X_val)

                true_returns.append(y_val)
                predicted_returns.append(y_pred[0])

            all_scores = generate_scores_from_returns(true_returns, predicted_returns, scaler_object)

            if parameters["evaluation_metric"] in minimize_evaluation_metrics():
                if all_scores[parameters["evaluation_metric"]] < best_optimize_score:
                    best_regressor = regressor
                    best_optimize_score = all_scores[parameters["evaluation_metric"]]
                    best_all_scores = all_scores
                    best_regressor_returns = predicted_returns
            else:
                if all_scores[parameters["evaluation_metric"]] > best_optimize_score:
                    best_regressor = regressor
                    best_optimize_score = all_scores[parameters["evaluation_metric"]]
                    best_all_scores = all_scores
                    best_regressor_returns = predicted_returns

        logger.info("Best SVM regressor with Walk-Forward Validation %s of %.3f",
                    available_evaluation_metrics()[parameters["evaluation_metric"]], best_optimize_score)
        return best_regressor, best_regressor_returns, best_optimize_score, best_all_scores

    logger = logging.getLogger(__name__)
    models = ["linear_regression", "svm"]

    best_model = None
    best_model_name = "None"
    best_optimize_score = math.inf if parameters["evaluation_metric"] in minimize_evaluation_metrics() else -math.inf
    best_all_scores = {}
    for model in models:
        if model == "linear_regression":
            parameters_linear_regression = {
                "fit_intercept": [True, False],
                "positive": [False, True]
            }
            model, returns, optimize_score, all_scores = _get_best_linear_regressor(X, y, parameters_linear_regression,
                                                                                    parameters)
            if parameters["evaluation_metric"] in minimize_evaluation_metrics():
                if optimize_score < best_optimize_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_optimize_score = optimize_score
                    best_all_scores = all_scores
            else:
                if optimize_score > best_optimize_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_optimize_score = optimize_score
                    best_all_scores = all_scores
        elif model == "svm":
            parameters_svm = {
                "kernel": ["linear", "poly", "rbf"],
                "C": [0.1, 1, 10],
                "degree": [3, 4, 5],
                "gamma": ["scale", "auto"]
            }
            model, returns, optimize_score, all_scores = _get_best_svm_regressor(X, y, parameters_svm, parameters)
            if parameters["evaluation_metric"] in minimize_evaluation_metrics():
                if optimize_score < best_optimize_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_optimize_score = optimize_score
                    best_all_scores = all_scores
            else:
                if optimize_score > best_optimize_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_optimize_score = optimize_score
                    best_all_scores = all_scores
        elif model == "lstm":
            parameters_lstm = {
                "n_hidden": [1, 2, 3],
                "n_neurons": [10, 50, 100],
                "n_input": [1, 2, 3],
                "n_output": [1, 2, 3]
            }
            """ 
            model, returns, score = __get_best_lstm_regressor(X,y, parameters_lstm, parameters)
            if score < best_optimize_score:
                best_model = model
                best_model_returns = returns
                best_optimize_score = score
                best_model_name = "Long Short-Term Memory Regressor"
            """

    logger.info("\x1b[34mBest model is %s with Walk-Forward Validation %s of %.3f\x1b[39m", best_model_name,
                available_evaluation_metrics()[parameters["evaluation_metric"]], best_optimize_score)

    best_all_scores = {available_evaluation_metrics()[key]: value for key, value in best_all_scores.items()}
    return best_model, best_all_scores


def predict_return(original_df: pd.DataFrame, processed_df: pd.DataFrame, best_lags: dict, regressor,
                   regressor_validation_scores: dict, sql_variables_table: pd.DataFrame, parameters: dict,
                   scaler_object) -> dict:
    """Predicts the return of a stock using the model.

    Args:
        original_df: Original data containing features and target.
        processed_df: Processed data containing features and target.
        best_lags: Best lags for each feature.
        regressor: Model to use for prediction.
        regressor_validation_score: Validation score of the model.
        sql_variables_table: Data containing variables from SQL.
        parameters: Parameters defined in parameters/data_science.yml.
        scaler_object: Scaler object to inverse transform the data.
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

    predicted_return = inverse_transform_returns(true_returns=None, predicted_returns=y_pred,
                                                 scaler_object=scaler_object)[0]

    predicted_close_price = original_df['close'].iloc[-1] * (1 + predicted_return)
    return {
        "Date": pd.to_datetime(X_test.index[-1]).strftime("%Y-%m-%d"),
        "Metric Optimized": available_evaluation_metrics()[parameters["evaluation_metric"]],
        "Predicted Return": predicted_return,
        "Predicted Close Price": predicted_close_price.round(2),
        "Regressor": regressor.__class__.__name__,
        "Validation Scores": regressor_validation_scores,
        "Used Lags": best_lags,
        "Used Features": list(set([column.split("_lag")[0] for column in X.columns]))
    }
