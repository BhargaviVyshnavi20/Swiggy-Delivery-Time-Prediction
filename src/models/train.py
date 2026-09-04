import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from lightgbm import LGBMRegressor

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

TARGET = 'time_taken'


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("model_trainin")
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data(data_path: Path) -> pd.DataFrame:
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


# ============================================================
# READ PARAMETERS
# ============================================================

def read_params(file_path):
    with open(file_path,'r') as file:
        params_file = yaml.safe_load(file)
        return params_file


def save_model(model, save_dir:Path, model_name: str):
    # form the save location
    save_location = save_dir / model_name
    # save the model with compression
    joblib.dump(value=model, filename=save_location, compress=3)

def save_transformer(transformer, save_dir:Path, transformer_name: str):
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer with compression
    joblib.dump(value=transformer, filename=save_location, compress=3)

def train_model(model, X_train: pd.DataFrame, y_train):
    # fit the model
    model.fit(X_train, y_train)
    return model

def make_X_and_y(data:pd.DataFrame, target_column:str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    params_file_path = root_path/"params.yaml"

    training_data =  load_data(data_path)
    logger.info("Training Data read successfullly")

    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")

    model_params = read_params(params_file_path)['Train']

    rf_params = model_params['Random_Forest']
    logger.info("random forest parameter read")

    rf = RandomForestRegressor(**rf_params)
    logger.info("built random forest model")

    lgbm_params = model_params['LightGBM']
    logger.info("lightGBM parameter read")

    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("built lightGBM model")

    # meta modle
    lr = LinearRegression()
    logger.info("built meta model")
    
    power_transform = PowerTransformer()
    logger.info("Target Transformer built")

    # form the stacking regressor
    stacking_reg = StackingRegressor(estimators=[("rf_model", rf),
                                                ("lgbm_model", lgbm)],
                                    final_estimator=lr,
                                    cv=5, n_jobs=-1
                                    )
    logger.info("stacking regressor built")

    model = TransformedTargetRegressor(regressor=stacking_reg,transformer=power_transform)
    logger.info("Models wrapped inside wrapper")

    train_model(model, X_train, y_train)
    logger.info("Model trained")

    model_filename = 'model.joblib'

    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # extract the model from wrapper 
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save the model
    save_model(model=model,
                save_dir=model_save_dir,
                model_name=model_filename)

    save_model(model=stacking_model,
               save_dir=model_save_dir,
               model_name="stacking_regressor.joblib")

    save_transformer(transformer=transformer,
                     save_dir=model_save_dir,
                     transformer_name="power_transformer.joblib")