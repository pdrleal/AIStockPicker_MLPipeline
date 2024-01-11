import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def filter_stocks_table_by_index(stocks_df: pd.DataFrame, stock_index) -> pd.DataFrame:
    """
    Filter and extract data for a specific stock index from Stocks Table.

    Args:
        stocks_df: A DataFrame containing stock data, including a 'stock_index' column.
        stock_index: The stock index to filter and extract data for.

    Returns:
        pd.DataFrame: A DataFrame containing data for the specified stock index.

    Notes:
        If no data is found for the given 'stock_index', an empty DataFrame is returned.
    """

    logger.info("Fetching data for stock index: %s", stock_index)
    stocks_df = stocks_df.loc[stocks_df['stock_index'] == stock_index]

    n_rows = stocks_df.shape[0]
    if n_rows == 0:
        logger.info("No data found for stock index: %s", stock_index)
        return pd.DataFrame()

    min_date = stocks_df['datetime'].min()
    max_date = stocks_df['datetime'].max()
    logger.info("There are %s rows raging from %s to %s", str(n_rows), min_date, max_date)

    return stocks_df;


def fill_sentiment_missing_values(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the sentiment columns of filtered_stocks_table.

    Args:
        stock_df (pd.DataFrame): A DataFrame containing stock data.

    Returns:
        pd.DataFrame: A DataFrame containing data with no missing values.

    """
    logger.info("Filling missing values in sentiment columns")
    stock_df = stock_df.ffill().bfill()
    return stock_df


def compute_simple_moving_averages(stock_df: pd.DataFrame, moving_averages) -> pd.DataFrame:
    """
    Create new features from existing data.

    Args:
        stock_df (pd.DataFrame): A DataFrame containing stock data.
        moving_averages: A list of moving averages to calculate.

    Returns:
        pd.DataFrame: A DataFrame containing data with new features

    """
    logger.info("Engineering new features for stock data")
    for moving_averages in moving_averages:
        stock_df[f'SMA{moving_averages}'] = stock_df['close'].rolling(moving_averages).mean()

    return stock_df
