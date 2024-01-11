from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_stocks_table_by_index,
                inputs=["raw_stocks_table", "params:stock_index"],
                outputs="stock_df",
                name="filter_stocks_table_by_index",
            ),
            node(
                func=fill_sentiment_missing_values,
                inputs=["stock_df"],
                outputs="preprocess_stock_df",
                name="fill_sentiment_missing_values",
            ),
            node(
                func=compute_simple_moving_averages,
                inputs=["preprocess_stock_df", "params:simple_moving_averages"],
                outputs="preprocess_stock_df_sma",
                name="compute_simple_moving_averages",
            ),
        ]
    )
