import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import joblib
from sklearn import set_config

# set the transformer outputs to pandas
set_config(transform_output="pandas")

# columd to preprocess in data

num_cols = ['age',
            'ratings',
            'pickup_time_minutes',
            'distance']

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    'festival',
                    'city_type',
                    'is_weekend',
                    'order_time_of_day']

ordinal_cat_cols = ['traffic', 'distance_type']

target_col = "time_taken"

# generate order for ordinal encoding

traffic_order = ["low", "medium", "high", "jam"]

distance_type_order = ["short", "medium", "long", "very_long"]


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("data_preprocessing")
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

def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info(f"The original dataset with missing values has {data.shape[0]} rows and {data.shape[1]} columns")
        df_dropped = data.dropna()
        logger.info(f"The dataset after removing missing values has {df_dropped.shape[0]} rows and {df_dropped.shape[1]} columns")
        missing_vals = df_dropped.isna().sum().sum()

        if missing_vals > 0:
            raise ValueError ("The dataframe has missing values")
        return df_dropped

    except Exception as e:
        logger.error(f"Error while checking for missing values: {e}")
        raise


def save_transfomer(transformer, save_dir: Path, transformer_name: str):
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    joblib.dump(value=transformer, filename=save_location)
    logger.info(f"Transformer {transformer_name} saved successfully")
 
def train_preprocessor(preprocessor, data: pd.DataFrame):
    # fit on the data
    preprocessor.fit(data)
    return preprocessor 

def perform_transformations(preprocessor, data: pd.DataFrame):
    # transform the data
    transformed_data = preprocessor.transform(data)
    return transformed_data

def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=target_column)
    y = data[target_column]
    return X, y  

def join_X_and_Y(X: pd.DataFrame, y: pd.Series):
    # join based on indexes
    joined_df = X.join(y, how='inner')
    return joined_df


if __name__ == "__main__":
    # set paths
    root_path = Path(__file__).parent.parent.parent

    # load data path
    train_data_path = root_path / "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"

    # define save data  location
    save_data_dir = root_path / "data" / "processed"
    save_data_dir.mkdir(parents=True, exist_ok=True)

    train_trans_filename = 'train_trans.csv'
    test_trans_filename = 'test_trans.csv'

    # define save files
    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename

    # preprocessor 
    preprocessor = ColumnTransformer(
    transformers=[
        ("scale", MinMaxScaler(), num_cols),
        (
            "nominal_encode",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ),
            nominal_cat_cols
        ),
        (
            "ordinal_encode",
            OrdinalEncoder(
                categories=[traffic_order, distance_type_order],
                encoded_missing_value=-999,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ordinal_cat_cols
        )
    ],
    remainder="passthrough",
    n_jobs=-1,
    verbose_feature_names_out=False,
    verbose=True
)

logger.info("Preprocessor created successfully")


# load the train and test data with missing values dropped
train_df = drop_missing_values(load_data(train_data_path))
logger.info("Train data loaded successfully")
test_df = drop_missing_values(load_data(test_data_path))
logger.info("Test data loaded successfully")

# split the train and test data
X_train, y_train = make_X_and_y(data=train_df, target_column=target_col)
X_test, y_test = make_X_and_y(data=test_df, target_column=target_col)
logger.info("Train and Test data split successfully")

# fit the preprocessor on X_train
train_preprocessor(preprocessor=preprocessor, data=X_train)
logger.info("Preprocessor is trained")

# transform the data
X_train_trans = perform_transformations(preprocessor=preprocessor, data=X_train)
logger.info("x_train is transformed")

X_test_trans = perform_transformations(preprocessor=preprocessor, data=X_test)
logger.info("x_test is transformed")

# join back X and y 
train_trans_df = join_X_and_Y(X_train_trans, y_train)
test_trans_df = join_X_and_Y(X_test_trans, y_test)
logger.info("Datasets joined")

# save the transfomed data
data_subsets = [train_trans_df, test_trans_df]
data_paths = [save_train_trans_path, save_test_trans_path]
filename_list = [train_trans_filename, test_trans_filename]
for data, path,filename in zip(data_subsets, data_paths, filename_list):
    save_data(data=data, save_path=path)
    logger.info(f"Dataset {filename} saved successfully")

# save the transformer
transformer_filename = "preprocessor.joblib"
# directory to save transformers
transformer_save_dir = root_path / "models"
# create save directory
transformer_save_dir.mkdir(parents=True, exist_ok=True)

# save the transformer
save_transfomer(transformer=preprocessor, save_dir=transformer_save_dir, transformer_name=transformer_filename)
logger.info("Preprocessor saved to location")
    
    
