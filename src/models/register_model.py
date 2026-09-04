import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging
import os

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers when the module is imported multiple times
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
repo_owner = os.getenv("MLFLOW_TRACKING_USERNAME", "bhargavivyshnavi04")
repo_name = os.getenv("REPO_NAME", "Swiggy-Delivery-Time-Prediction")

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/bhargavivyshnavi04/"
    "Swiggy-Delivery-Time-Prediction.mlflow"
)


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    # ========================================================
    # FIND LOGGED MODEL
    # ========================================================

    client = MlflowClient()

    run = client.get_run(run_id)
    experiment_id = run.info.experiment_id

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list"
    )

    logged_model = None

    for model in logged_models:
        if model.name == model_name:
            logged_model = model
            break

    if logged_model is None:
        raise RuntimeError(
            f"Could not find LoggedModel '{model_name}' "
            f"for run '{run_id}'"
        )

    logger.info(
        f"Found LoggedModel: {logged_model.name}"
    )

    logger.info(
        f"LoggedModel ID: {logged_model.model_id}"
    )

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    model_registry_path = f"models:/{logged_model.model_id}"

    model_version = mlflow.register_model(
        model_uri=model_registry_path,
        name=model_name
    )

    registered_model_version = model_version.version
    registered_model_name = model_version.name

    logger.info(
        f"The latest model version in model registry is "
        f"{registered_model_version}"
    )
