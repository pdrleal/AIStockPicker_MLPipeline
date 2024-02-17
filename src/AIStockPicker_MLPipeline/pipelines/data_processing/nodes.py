import logging

import numpy as np
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

    stock_df = stocks_df.loc[stocks_df['stock_index'] == stock_index].copy()

    stock_df['datetime'] = pd.to_datetime(stock_df['datetime'])
    stock_df = stock_df.sort_values('datetime')
    stock_df = stock_df.set_index('datetime', drop=True)
    stock_df = stock_df.drop(columns=['stock_index'])
    return stock_df


def treat_missing_values(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Replacing sentiments missing values by 0..")
    stock_df.loc[:, 'news_sentiment'] = stock_df['news_sentiment'].fillna(0)

    return stock_df


def compute_percentage_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing percentage returns...")
    stock_df['return'] = stock_df['close'].pct_change()

    return stock_df


# https://www.elearnmarkets.com/school/units/swing-trading/moving-average-strategy
# https://www.investopedia.com/terms/g/goldencross.asp
def compute_simple_moving_averages(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating optimal moving averages...")

    # Testing with short moving averages of 1,2 and 3 months and long moving averages of 4,5 and 6 months
    short_sma_days = [22, 44, 66]
    long_sma_days = [88, 110, 132]
    bands = np.arange(0.001, 0.01, 0.001)

    best_sma_returns_information_ratio = -1000
    best_sma = (0, 0, 0)
    best_sma_signals = []
    for short_sma in short_sma_days:
        for long_sma in long_sma_days:
            for band in bands:

                sma_signals = (ta.sma(stock_df['close'], short_sma) > (1 + band) * ta.sma(stock_df['close'],
                                                                                          long_sma)).astype(int)
                sma_returns = stock_df['return'] * sma_signals
                sma_returns_information_ratio = sma_returns.mean() / sma_returns.std()
                if sma_returns_information_ratio > best_sma_returns_information_ratio:  # maximization
                    best_sma_returns_information_ratio = sma_returns_information_ratio
                    best_sma = (short_sma, long_sma, band)
                    best_sma_signals = sma_signals
                logger.info(
                    "Testing SMA strategy with short_sma=%d, long_sma=%d and band=%.3f. Information Ratio: %.2f",
                    short_sma, long_sma,
                    band, sma_returns_information_ratio)

    logger.info("Best SMA strategy found: short_sma=%d, long_sma=%d and band=%.3f. Information Ratio: %.2f",
                best_sma[0], best_sma[1],
                best_sma[2], best_sma_returns_information_ratio)
    stock_df['sma_signals'] = best_sma_signals
    return stock_df[['sma_signals']]


def compute_relative_strength_indexes(stock_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating optimal relative strength indexes...")
    # https://admiralmarkets.com/education/articles/forex-indicators/relative-strength-index-how-to-trade-with-an-rsi-indicator
    # https://medium.com/mudrex/rsi-masterclass-part-3-9f9a5dfe3fca

    # Testing with RSIs of 14, 28 and 42 days for lower levels of 20,25,30,35,40 and upper leves of 80,75,70,65,60
    rsi_short_lengths = [5, 8, 11, 14, 17]
    rsi_long_lengths = [20, 25, 30, 35, 40, 45]
    lower_levels = [20, 25, 30, 35, 40, 45]
    upper_levels = [80, 75, 70, 65, 60, 55]

    best_rsi_returns_information_ratio = -1000
    best_rsi = (0, 0, 0, 0)
    best_rsi_signals = []

    for rsi_short_length in rsi_short_lengths:
        for rsi_long_length in rsi_long_lengths:
            for lower_level, upper_level in zip(lower_levels, upper_levels):

                rsi_oversold = (ta.rsi(stock_df['close'], length=rsi_short_length) < lower_level)
                rsi_overbought = (ta.rsi(stock_df['close'], length=rsi_short_length) > upper_level)

                rsi_positive_trend = (ta.rsi(stock_df['close'], length=rsi_long_length) > 50)
                rsi_negative_trend = (ta.rsi(stock_df['close'], length=rsi_long_length) < 50)

                # rsi_signals_buy = (rsi_oversold & rsi_positive_trend)
                rsi_signals_buy = rsi_oversold
                # rsi_signals_sell = (rsi_overbought & rsi_negative_trend)
                rsi_signals_sell = rsi_overbought

                rsi_signals = pd.Series(index=stock_df.index)
                rsi_signals.loc[rsi_signals_buy] = 1
                rsi_signals.loc[rsi_signals_sell] = 0

                # Forward fill the signal column
                rsi_signals = rsi_signals.ffill().bfill().fillna(0)

                rsi_returns = stock_df['return'] * rsi_signals
                if rsi_returns.std() != 0:
                    rsi_returns_information_ratio = rsi_returns.mean() / rsi_returns.std()
                else:
                    rsi_returns_information_ratio = 0
                if rsi_returns_information_ratio > best_rsi_returns_information_ratio:
                    best_rsi_returns_information_ratio = rsi_returns_information_ratio
                    best_rsi = (rsi_short_length, rsi_long_length, lower_level, upper_level)
                    best_rsi_signals = rsi_signals
                logger.info(
                    "Testing RSI strategy with rsi_short_length=%d, rsi_long_length=%d, lower_level=%d and "
                    "upper_level=%d. Information ratio: %.2f",
                    rsi_short_length, rsi_long_length, lower_level, upper_level, rsi_returns_information_ratio)

    logger.info(
        "Best RSI strategy found: rsi_short_length=%d, rsi_long_length=%d, lower_level=%d and upper_level=%d. Information ratio: %.2f",
        best_rsi[0], best_rsi[1], best_rsi[2], best_rsi[3], best_rsi_returns_information_ratio)
    stock_df['rsi_signals'] = best_rsi_signals

    return stock_df[['rsi_signals']]


def merge_dataframes(*df_list) -> pd.DataFrame:
    logger.info("Merging dataframes...")
    stock_df = pd.concat(df_list, axis=1)
    logger.info("Limiting timeframe from the first news forward...")
    stock_df = stock_df.loc[stock_df.index >= first_date_with_sentiment]
    #stock_df.to_csv('notebooks/aapl_df.csv')
    return stock_df


def perform_feature_selection(stock_df) -> pd.DataFrame:
    logger.info("Selecting automatically best features...")
    return stock_df


def scale_data(stock_df) -> pd.DataFrame:
    logger.info("Scaling numerical features...")
    return stock_df
