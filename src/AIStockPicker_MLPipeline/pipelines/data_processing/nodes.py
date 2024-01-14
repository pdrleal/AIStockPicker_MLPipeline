import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)
first_date_with_sentiment = None


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

    logger.info("Fetching data for %s...", stock_index)

    # save minimum date with news_sentiment different from np.nan
    global first_date_with_sentiment
    first_date_with_sentiment = stocks_df.loc[stocks_df['news_sentiment'].notnull(), 'datetime'].min()

    stocks_df = stocks_df.loc[stocks_df['stock_index'] == stock_index]
    if stocks_df.shape[0] == 0:
        logger.info("No data found for %s.", stock_index)
        return pd.DataFrame()

    return stocks_df;


def compute_percentage_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing percentage returns...")
    stock_df['return'] = stock_df['close'].pct_change()

    return stock_df


def compute_simple_moving_averages(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing simple moving averages...")
    # https://www.elearnmarkets.com/school/units/swing-trading/moving-average-strategy
    stock_df['sma21'] = ta.sma(stock_df['close'], 21)
    stock_df['sma50'] = ta.sma(stock_df['close'], 50)
    stock_df['sma100'] = ta.sma(stock_df['close'], 100)
    # Golden Cross-over can be achieved when SMA21 crosses SMA50 from below
    stock_df['bullish_sma'] = (stock_df['sma21'] > stock_df['sma50']).astype(int)
    # Death Cross-over can be achieved when SMA21 crosses SMA50 from above
    stock_df['bearish_sma'] = (stock_df['sma21'] < stock_df['sma50']).astype(int)

    return stock_df[['sma21', 'sma50', 'sma100', 'bullish_sma']]


def compute_relative_strength_indexes(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing relative strength index...")
    # https://admiralmarkets.com/education/articles/forex-indicators/relative-strength-index-how-to-trade-with-an-rsi-indicator
    stock_df['rsi'] = ta.rsi(stock_df['close'], 14)
    stock_df['rsi_overbought'] = (stock_df['rsi'] > 70).astype(int)
    stock_df['rsi_oversold'] = (stock_df['rsi'] < 30).astype(int)

    return stock_df[['rsi', 'rsi_overbought', 'rsi_oversold']]


def merge_dataframes(*df_list) -> pd.DataFrame:
    logger.info("Merging dataframes...")
    stock_df = pd.concat(df_list, axis=1)
    return stock_df


def treat_missing_values(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Limiting timeframe from the first news forward...")
    stock_df = stock_df.loc[stock_df['datetime'] >= first_date_with_sentiment]
    logger.info("Propagating sentiment values forward...")
    stock_df = stock_df.ffill().bfill()

    return stock_df


def perform_feature_selection(stock_df) -> pd.DataFrame:
    logger.info("Selecting automatically best features...")
    return stock_df[['return', 'news_sentiment', 'bullish_sma']]


def scale_data(stock_df) -> pd.DataFrame:
    logger.info("Scaling numerical features...")
    return stock_df
