import os
from dotenv import load_dotenv

load_dotenv()
if "DAGSHUB_USER_TOKEN" not in os.environ and os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

import pandas as pd 
import joblib
import logging
import mlflow
import dagshub
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json

# Suppress MLflow schema hint warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

repo_owner = os.getenv("MLFLOW_TRACKING_USERNAME", "bhargavivyshnavi04")
repo_name = os.getenv("REPO_NAME", "Swiggy-Delivery-Time-Prediction")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


#set mlflow experiment name
mlflow.set_experiment("DVC-Pipeline")

TARGET = "time_taken"

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("Evaluation")
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
    

def load_data(data_path:Path)-> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise

def make_X_and_y(data:pd.DataFrame, target_column: str):
    """
    Split the dataset into features (X) and target (y).
    """
    try:
        logger.info(f"Splitting data into features (X) and target (y)")
        X = data.drop(columns=[target_column])
        y = data[target_column]
        logger.info(f"Data split successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error while splitting data: {e}")
        raise

def load_model(model_path:Path) -> any:
    """
    Load the trained model from a joblib file.
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error while loading model: {e}")
        raise

def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id" : run_id,
        "artifact_path" : artifact_path,
        "model_name" : model_name,
    }

    with open(save_json_path, 'w') as f:
        json.dump(info_dict, f, indent=4)
    logger.info(f"Model info saved successfully to {save_json_path}")

if __name__ == "__main__":
    # get the paths 
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    
    # model path 
    model_path = root_path / "models" / "model.joblib"

    # load the training data
    train_data = load_data(train_data_path)

    # load the testing data
    test_data = load_data(test_data_path)

    # load the model
    model = load_model(model_path)
    logger.info("Model Loaded successfully")

    # make X and y
    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("predictions on data complete")

    # calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    logger.info("mae error calculated")

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info("r2 score calculated")

    # calculate cross val scores
    cv_scores = cross_val_score(
                            model,
                            X_train,
                            y_train,
                            scoring="neg_mean_absolute_error",
                            cv=5,
                            n_jobs=-1,
                            )
    logger.info("cross val scores calculated")

    # mean corss val score
    mean_cv_score = -(cv_scores.mean())

    # log with mlflow
    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag("model", "Food Delivery Time Regressor")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mean_cv_score", mean_cv_score)
        logger.info("Metrics logged successfully")
        
        # log individual cross validation scores
        for i, cv_score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_score_{i+1}", cv_score)
        logger.info("Individual cv scores logged successfully")

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)

        # log input
        mlflow.log_input(dataset=train_data_input, context="training")
        mlflow.log_input(dataset=test_data_input, context="validation")

        # model signature (convert integers to float to prevent schema warning)
        sample_input = X_train.sample(20, random_state=42).copy()

        integer_cols = sample_input.select_dtypes(include="integer").columns

        sample_input[integer_cols] = sample_input[integer_cols].astype("float64")

        sample_output = model.predict(sample_input)

        model_signature = mlflow.models.infer_signature(
            model_input=sample_input,
            model_output=sample_output
        )
        logger.info("Model signature inferred successfully")
        
        # log the final model
        logger.info("Uploading model to MLflow...")
        # trusted_types = [
        #     "collections.OrderedDict",
        #     "lightgbm.basic.Booster",
        #     "lightgbm.sklearn.LGBMRegressor",
        #     "sklearn.utils._bunch.Bunch",
        # ]
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     name="delivery_time_pred_model",
        #     signature=model_signature,
        #     serialization_format="skops",
        #     skops_trusted_types=trusted_types,

        # )
                # log the final model
        logger.info("Uploading model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="delivery_time_pred_model",
            signature=model_signature,
            serialization_format="cloudpickle",
        )
        logger.info("Model successfully logged to MLflow")


        # log model.joblib
        logger.info("Uploading model.joblib to MLflow artifacts...")
        mlflow.log_artifact(root_path / "models" / "model.joblib")
        logger.info("Model joblib artifact logged successfully")

        artifact_url = mlflow.get_artifact_uri()
        logger.info("Mlflow logging complete and model logged")

        # get the run id
        run_id = run.info.run_id
        model_name ="delivery_time_pred_model"

        # save the model run id
        save_json_path = root_path / "run_information.json"

        save_model_info(save_json_path=save_json_path,
                        run_id=run_id,
                        artifact_path=artifact_url,
                        model_name=model_name
                        )
        logger.info("Model Information saved")
        