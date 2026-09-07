# regression gate: loads the staging model + preprocessor, predicts on the 
# held-out test split , and fails if mean absolute error exceeds the threashold
# Run individually: pytest tests/test_model_perf.py -v -s
import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_owner='bhargavivyshnavi04', repo_name='Swiggy-Delivery-Time-Prediction', mlflow=True)

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    
    return run_info

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# set the root path
root_path = Path(__file__).parent.parent

# set model name
model_name = load_model_information(root_path / "run_information.json")["model_name"]
alias = "staging"

# load the latest model from model registry
try:
    model_path = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_path)
except Exception:
    model_path = f"models:/{model_name}/{alias.capitalize()}"
    model = mlflow.sklearn.load_model(model_path)

# load the processor
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ]
)

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                argvalues=[(model_pipe, test_data_path, 5)])

def test_model_performance(model_pipe, test_data_path, threshold_error):
    df = pd.read_csv(test_data_path)

    # drop the missing values
    df.dropna(inplace=True)

    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df["time_taken"]

    # get the predictions
    y_pred = model_pipe.predict(X)

    # compute the mean absolute error
    mean_error = mean_absolute_error(y, y_pred)
 
    # check the performance against the threshold
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes "
    print("The avg error is", mean_error)

    print(f"The {model_name} model has met the performance criteria")
