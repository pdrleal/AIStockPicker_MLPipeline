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
                func=compute_percentage_returns,
                inputs=["stock_table"],
                outputs="percentage_returns",
                name="compute_percentage_return",
            ),
            node(
                func=compute_simple_moving_averages,
                inputs=["stock_table"],
                outputs="simple_moving_averages",
                name="compute_simple_moving_average",
            ),
            node(
                func=compute_relative_strength_indexes,
                inputs=["stock_table"],
                outputs="relative_strength_indexes",
                name="compute_relative_strength_index",
            ),
            node(
                func=merge_dataframes,
                inputs=["percentage_returns", "simple_moving_averages", "relative_strength_indexes"],
                outputs="stock_table_feature_engineering",
                name="merge_new_features",
            ),
            node(
                func=treat_missing_values,
                inputs=["stock_table_feature_engineering"],
                outputs="clean_stock_table",
                name="treat_missing_values",
            ),
            node(
                func=perform_feature_selection,
                inputs=["clean_stock_table"],
                outputs="stock_table_feature_selection",
                name="perform_feature_selection"
            ),
            node(
                func=scale_data,
                inputs=["stock_table_feature_selection"],
                outputs="model_input_table",
                name="scale_data"
            ),

        ]
    )
