import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

TARGET = "time_taken"


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("data_preparation")
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



def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    """
    Split the dataset into training and testing sets.
    """
    try:
        logger.info(f"Splitting data into training and testing sets")
        train_data, test_data = train_test_split(data, 
            test_size=test_size,
            random_state=random_state,
            )  
        logger.info(f"Data split successfully. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error while splitting data: {e}")
        raise

def read_params(file_path):
    with open(file_path,'r') as file:
        params_file = yaml.safe_load(file)
        return params_file


def save_data(data: pd.DataFrame, save_path: Path):
    """
    Save the dataset to a CSV file.
    """
    try:
        logger.info(f"Saving data to {save_path}")
        data.to_csv(save_path, index=False)
        logger.info(f"Data saved successfully.")
    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise


if __name__ == "__main__":
    # set file paths
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data load path
    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv" 
    # save data directory
    save_data_dir = root_path / "data" / "interim"
    # create save data directory
    save_data_dir.mkdir(parents=True, exist_ok=True)
    # train and test data save paths
    # filenames
    train_filename = "train.csv"
    test_filename = "test.csv"

    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename

    # parameters file
    params_file_path = root_path  / "params.yaml"

    # load cleaned data
    data = load_data(data_path)
    logger.info("Data Loaded successfully")

    # read the parameters
    parameters = read_params(params_file_path)["Data_Preparation"]
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]
    logger.info("parameters read successfully")

    # split data into train and test sets
    train_data, test_data = split_data(
        data=data,
        test_size=test_size,
        random_state=random_state,
    )

    # save train and test datasets
    save_data(train_data, save_train_path)
    save_data(test_data, save_test_path)
    logger.info("Train and Test data prepared and saved successfully")