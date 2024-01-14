from pathlib import Path
from typing import Dict, Any

from flask import Flask, request
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


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
    if stock_index is None:
        return "Missing 'stock_index' parameter", 400
    new_session = create_new_kedro_session(extra_params={"stock_index": stock_index})
    return new_session.run()


if __name__ == '__main__':
    app.run(debug=False, port=5001)
