from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_optimized_lagged_features, split_data, perform_grid_search, predict_return


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_optimized_lagged_features,
                inputs=["stock_table_processed", "sql_variables_table", "params:modelling"],
                outputs=["stock_table_with_lagged_features", "best_lags"],
                name="generate_optimized_lagged_features",
            ),
            node(
                func=split_data,
                inputs=["stock_table_with_lagged_features", "params:modelling"],
                outputs=["X", "y"],
                name="split_data",
            ),
            node(
                func=perform_grid_search,
                inputs=["X", "y", "params:modelling", "scaler_object"],
                outputs=["regressor", "regressor_validation_scores"],
                name="perform_grid_search",
            ),
            node(
                func=predict_return,
                inputs=["clean_stock_table", "stock_table_processed", "best_lags", "regressor",
                        "regressor_validation_scores", "sql_variables_table", "params:modelling", "scaler_object"],
                outputs='prediction_details',
                name="predict_return",
            )
        ]
    )
