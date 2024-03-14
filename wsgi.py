from pathlib import Path
from typing import Dict, Any

from flask import Flask, request
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from AIStockPicker_MLPipeline.utils import available_evaluation_metrics


def create_new_kedro_session(
        extra_params: Dict[str, Any] = None, env: str = None
) -> KedroSession:
    """Helper function to initialize a new `KedroSession` inside another python file.
    """
    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    new_session = KedroSession.create(
        metadata.package_name, project_path, extra_params=extra_params, env=env
    )
    return new_session


app = Flask(__name__)


@app.route('/forecast')
def forecast():
    stock_index = request.args.get('stock_index')
    evaluation_metric = request.args.get('evaluation_metric')
    include_sentiments = request.args.get('include_sentiments')
    if stock_index is None:
        return "Missing 'stock_index' parameter", 400

    if include_sentiments is None:
        include_sentiments = "yes"
    elif include_sentiments not in ["yes", "no", "news", "social"]:
        return "Invalid 'include_sentiments' parameter. Available options are: 'yes', 'no', 'news', 'social'.", 400

    if evaluation_metric is None:
        evaluation_metric = "information_ratio"
    elif evaluation_metric not in available_evaluation_metrics().keys():
        return f"Invalid 'evaluation_metric' parameter. Available options are: {', '.join(available_evaluation_metrics().keys())}.", 400

    new_session = create_new_kedro_session(
        extra_params={"stock_index": stock_index,
                      "include_sentiments": include_sentiments,
                      "modelling": {"evaluation_metric": evaluation_metric}})
    return new_session.run()


if __name__ == '__main__':
    app.run(debug=False, port=5001)
