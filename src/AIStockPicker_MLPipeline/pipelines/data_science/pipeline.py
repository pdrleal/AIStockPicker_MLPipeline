from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, perform_grid_search


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=perform_grid_search,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_val", "y_val"],
                outputs='score',
                name="evaluate_model_node",
            ),
        ]
    )
