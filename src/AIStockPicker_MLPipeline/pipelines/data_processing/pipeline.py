from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_stocks_table_by_index,
                inputs=["raw_stocks_table", "params:stock_index"],
                outputs="stock_table",
                name="filter_stocks_table_by_index",
            ),
            node(
                func=treat_missing_values,
                inputs=["stock_table"],
                outputs="clean_stock_table",
                name="treat_sentiment_missing_values",
            ),
            node(
                func=compute_percentage_returns,
                inputs=["clean_stock_table"],
                outputs="percentage_returns",
                name="compute_percentage_return",
            ),
            node(
                func=compute_simple_moving_averages,
                inputs=["percentage_returns", "params:modelling"],
                outputs="simple_moving_averages",
                name="compute_simple_moving_average",
            ),
            node(
                func=compute_relative_strength_indexes,
                inputs=["percentage_returns", "params:modelling"],
                outputs="relative_strength_indexes",
                name="compute_relative_strength_index",
            ),
            node(
                func=merge_dataframes,
                inputs=["percentage_returns","simple_moving_averages", "relative_strength_indexes"],
                outputs="stock_table_feature_engineering",
                name="merge_features",
            ),

            node(
                func=perform_feature_selection,
                inputs=["stock_table_feature_engineering", "params:include_sentiments"],
                outputs="stock_table_feature_selection",
                name="perform_feature_selection"
            ),
            node(
                func=scale_data,
                inputs=["stock_table_feature_selection"],
                outputs=["stock_table_processed", "scaler_object"],
                name="scale_data"
            ),

        ]
    )
