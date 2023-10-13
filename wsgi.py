from typing import Dict, Any

from flask import Flask
from pathlib import Path
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


@app.route('/forecast/<string:stock_index>')
def forecast(stock_index):
    new_session = create_new_kedro_session(extra_params={"stock_index": stock_index})
    return new_session.run()

