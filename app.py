from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
import os
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output to pandas
set_config(transform_output="pandas")

# initialize dagshub
import dagshub

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


from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, AliasChoices

class Data(BaseModel):
    ID: Optional[Any] = None
    Delivery_person_ID: Optional[str] = None
    Delivery_person_Age: Optional[Any] = None
    Delivery_person_Rating: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices(
            "Delivery_person_Rating",
            "Delivery_person_Ratings",
            "delivery_person_ratings",
            "delivery_person_rating",
            "ratings",
        ),
    )
    Restaurant_latitude: Optional[float] = None
    Restaurant_longitude: Optional[float] = None
    Delivery_location_latitude: Optional[float] = None
    Delivery_location_longitude: Optional[float] = None
    order_Date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("order_Date", "Order_Date", "order_date"),
    )
    Time_Ordered: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "Time_Ordered", "Time_Orderd", "time_ordered", "time_orderd"
        ),
    )
    Time_Order_picked: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Time_Order_picked", "time_order_picked"),
    )
    Weatherconditions: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Weatherconditions", "weatherconditions", "weather"),
    )
    Road_traffic_density: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "Road_traffic_density", "road_traffic_density", "traffic"
        ),
    )
    Vehicle_condition: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("Vehicle_condition", "vehicle_condition"),
    )
    Type_of_order: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Type_of_order", "type_of_order"),
    )
    Type_of_vehicle: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Type_of_vehicle", "type_of_vehicle"),
    )
    multiple_deliveries: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("multiple_deliveries"),
    )
    Festival: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("Festival", "festival"),
    )
    City: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("City", "city", "city_type"),
    )

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)

    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


# columns to preprocess in data
num_cols = [
    "age",
    "ratings",
    "pickup_time_minutes",
    "distance"
]

nominal_cat_cols = [
    "weather",
    "type_of_order",
    "type_of_vehicle",
    "festival",
    "is_weekend",
    "order_time_of_day"
]

ordinal_cat_cols = [
    "traffic",
    "distance_type"
]

# mlflow client
client = MlflowClient()

model_name = load_model_information(
    "run_information.json"
)["model_name"]

# load the candidate model
model_path = f"models:/{model_name}@candidate"

# load the candidate model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# create the app
app = FastAPI()


@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"


@app.post(path="/predict")
def predict(data: Union[Data, Dict[str, Any]]):
    if isinstance(data, BaseModel):
        input_data = data.model_dump(by_alias=True)
    else:
        input_data = data

    pred_data = pd.DataFrame([input_data])

    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)

    # get the predictions
    predictions = float(model_pipe.predict(cleaned_data)[0])

    return predictions


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )