import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import dagshub
import mlflow
from mlflow import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize DagsHub & MLflow
repo_owner = os.getenv("MLFLOW_TRACKING_USERNAME", "bhargavivyshnavi04")
repo_name = os.getenv("REPO_NAME", "Swiggy-Delivery-Time-Prediction")

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)


def load_model_information(file_path):
    with open(file_path, "r") as f:
        run_info = json.load(f)
    return run_info


def promote_model():
    # Resolve root path and run_information.json path
    root_path = Path(__file__).resolve().parent.parent
    run_info_path = root_path / "run_information.json"

    # Get model information
    run_info = load_model_information(run_info_path)
    model_name = run_info.get("model_name", "delivery_time_pred_model")

    client = MlflowClient()

    # Retrieve staging version via alias or stage
    staging_version = None

    try:
        model_version_info = client.get_model_version_by_alias(name=model_name, alias="staging")
        staging_version = model_version_info.version
    except Exception as e:
        logger.warning(f"Could not get model version by alias 'staging': {e}")

    if not staging_version:
        try:
            latest_versions = client.get_latest_versions(name=model_name, stages=["Staging"])
            if latest_versions:
                staging_version = latest_versions[0].version
        except Exception as e:
            logger.warning(f"Could not get model version by stage 'Staging': {e}")

    if not staging_version and "model_version" in run_info:
        staging_version = str(run_info["model_version"])

    if not staging_version:
        raise RuntimeError(f"No model found in staging for model '{model_name}'")

    logger.info(f"Target model '{model_name}' staging version: {staging_version}")

    # Set prod and production aliases
    for alias in ["prod", "production"]:
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=staging_version
            )
            logger.info(f"Assigned alias '{alias}' to model '{model_name}' version {staging_version}")
        except Exception as e:
            logger.warning(f"Failed to set alias '{alias}': {e}")

    # Transition stage to Production
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Transitioned model '{model_name}' version {staging_version} to 'Production' stage")
    except Exception as e:
        logger.warning(f"Failed to transition stage to 'Production': {e}")

    logger.info(f"Model '{model_name}' version {staging_version} successfully promoted to production.")


if __name__ == "__main__":
    promote_model()


