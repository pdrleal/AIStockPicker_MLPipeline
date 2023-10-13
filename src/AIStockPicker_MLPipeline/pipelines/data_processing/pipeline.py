from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=do_nothing,
                inputs=["stocks_table", "params:stock_index"],
                outputs="model_input_table",
                name="do_nothing_node",
            ),

            # Disabled Nodes
            # node(
            #     func=preprocess_companies,
            #     inputs="companies",
            #     outputs="preprocessed_companies",
            #     name="preprocess_companies_node",
            # ),
            # node(
            #     func=preprocess_shuttles,
            #     inputs="shuttles",
            #     outputs="preprocessed_shuttles",
            #     name="preprocess_shuttles_node",
            # ),
            # node(
            #     func=create_model_input_table,
            #     inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
            #     outputs="model_input_table",
            #     name="create_model_input_table_node",
            # ),
        ]
    )
