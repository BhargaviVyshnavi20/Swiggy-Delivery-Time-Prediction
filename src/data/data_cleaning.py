import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger("data_cleaning")
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
# COLUMNS TO DROP
# ============================================================

columns_to_drop = [
    "rider_id",
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
    "order_date",
    "order_time_hour",
    "order_day",
    "city_name",
    "order_day_of_week",
    "order_month",
]


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
# CHANGE COLUMN NAMES
# ============================================================

def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase and rename inconsistent columns.
    """
    return (
        data.rename(str.lower, axis=1)
        .rename(
            columns={
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken",
            }
        )
    )


# ============================================================
# EXTRACT DATE FEATURES
# ============================================================

def extract_datetime_features(ser: pd.Series) -> pd.DataFrame:
    """
    Extract useful features from the order date.
    """
    date_col = pd.to_datetime(ser, dayfirst=True, errors="coerce")

    return pd.DataFrame(
        {
            "order_day": date_col.dt.day,
            "order_month": date_col.dt.month,
            "order_year": date_col.dt.year,
            "order_day_of_week": date_col.dt.day_name().str.lower(),
            "is_weekend": date_col.dt.dayofweek.isin([5, 6]).astype(int),
        },
        index=ser.index,
    )


# ============================================================
# TIME OF DAY FEATURE
# ============================================================

def time_of_day(ser: pd.Series) -> np.ndarray:
    """
    Categorize order time into periods of the day.
    """
    time_col = pd.to_datetime(
        ser,
        format="%H:%M:%S",
        errors="coerce",
    ).dt.hour

    return np.select(
        condlist=[
            time_col.between(6, 11),
            time_col.between(12, 16),
            time_col.between(17, 19),
            time_col.between(20, 23),
        ],
        choicelist=[
            "morning",
            "afternoon",
            "evening",
            "night",
        ],
        default="after_midnight",
    )


# ============================================================
# DATA CLEANING FUNCTION
# ============================================================

def data_cleaning(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Perform core data cleaning, type conversions, and feature extraction.
    """
    logger.info("Starting core data cleaning process")
    df = df_input.copy()

    # 1. Replace invalid string values with actual NaN
    df.replace(
        {
            "NaN ": np.nan,
            "conditions NaN": np.nan,
            "NaN": np.nan,
        },
        inplace=True,
    )

    # 2. Drop unnecessary ID column
    df.drop(
        columns=["id"],
        errors="ignore",
        inplace=True,
    )

    # 3. Extract city name from rider ID
    if "rider_id" in df.columns:
        df["city_name"] = (
            df["rider_id"]
            .astype("string")
            .str.split("RES")
            .str[0]
        )

    # 4. Convert numeric columns
    numeric_columns = [
        "age",
        "ratings",
        "multiple_deliveries",
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Clean time_taken column
    if "time_taken" in df.columns:
        df["time_taken"] = (
            df["time_taken"]
            .astype("string")
            .str.replace("(min)", "", regex=False)
            .str.strip()
        )
        df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")

    # 6. Clean weather column
    if "weather" in df.columns:
        df["weather"] = (
            df["weather"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("conditions ", "", regex=False)
        )

    # 7. Clean categorical columns
    categorical_columns = [
        "traffic",
        "type_of_order",
        "type_of_vehicle",
        "festival",
        "city_type",
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    # 8. Process order date
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(
            df["order_date"],
            dayfirst=True,
            errors="coerce",
        )
        date_features = extract_datetime_features(df["order_date"])
        df = pd.concat([df, date_features], axis=1)

    # 9. Process order and pickup times
    time_columns = ["order_time", "order_picked_time"]
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                format="%H:%M:%S",
                errors="coerce",
            )

    if "order_time" in df.columns and "order_picked_time" in df.columns:
        adjusted_pickup_time = df["order_picked_time"].where(
            df["order_picked_time"] >= df["order_time"],
            df["order_picked_time"] + pd.Timedelta(days=1),
        )

        df["pickup_time_minutes"] = (
            adjusted_pickup_time - df["order_time"]
        ).dt.total_seconds() / 60

        df["order_time_hour"] = df["order_time"].dt.hour
        df["order_time_of_day"] = time_of_day(
            df["order_time"].dt.strftime("%H:%M:%S")
        )
        df.drop(columns=["order_time", "order_picked_time"], errors="ignore", inplace=True)

    # 10. Filter invalid records
    initial_rows = len(df)
    if "age" in df.columns:
        df = df.loc[df["age"] >= 18]
    if "ratings" in df.columns:
        df = df.loc[df["ratings"] != 6.0]

    logger.info(
        f"Filtered records in data_cleaning: {initial_rows - len(df)} invalid rows removed"
    )

    return df


# ============================================================
# CLEAN LATITUDE AND LONGITUDE
# ============================================================

def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Clean latitude and longitude by replacing values below threshold with NaN,
    and filtering coordinates within valid geographic bounds of India.
    """
    logger.info("Cleaning latitude and longitude coordinates")

    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]

    loc_cols = [col for col in location_columns if col in data.columns]

    # Convert coordinates with absolute values < threshold to NaN
    data = data.assign(
        **{
            col: np.where(
                data[col].abs() < threshold,
                np.nan,
                data[col].abs(),
            )
            for col in loc_cols
        }
    )

    # India's approximate geographical boundaries
    min_lat = 8.0 + 4 / 60
    max_lat = 37.0 + 6 / 60
    min_long = 68.0 + 7 / 60
    max_long = 97.0 + 25 / 60

    valid_mask = (
        data["restaurant_latitude"].between(min_lat, max_lat)
        & data["restaurant_longitude"].between(min_long, max_long)
        & data["delivery_latitude"].between(min_lat, max_lat)
        & data["delivery_longitude"].between(min_long, max_long)
    )

    return data.loc[valid_mask]


# ============================================================
# CALCULATE HAVERSINE DISTANCE
# ============================================================

def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the great circle distance between restaurant and delivery location
    using the Haversine formula in kilometers.
    """
    logger.info("Calculating Haversine distance")

    lat1 = df["restaurant_latitude"]
    lon1 = df["restaurant_longitude"]
    lat2 = df["delivery_latitude"]
    lon2 = df["delivery_longitude"]

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's radius in kilometers
    distance = 6371 * c

    return df.assign(distance=distance)


# ============================================================
# CREATE DISTANCE TYPE
# ============================================================

def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize delivery distance into bins: short, medium, long, very_long.
    """
    logger.info("Creating distance categories")

    return data.assign(
        distance_type=pd.cut(
            data["distance"],
            bins=[0, 5, 10, 20, np.inf],
            labels=["short", "medium", "long", "very_long"],
            include_lowest=True,
        )
    )


# ============================================================
# DROP COLUMNS
# ============================================================

def drop_columns(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Drop specified columns from the dataset.
    """
    logger.info(f"Dropping columns: {columns}")
    if columns is None:
        return data
    return data.drop(columns=columns, errors="ignore")


# ============================================================
# PIPELINE AUTOMATION FUNCTION
# ============================================================

def perform_data_cleaning(data: pd.DataFrame, saved_data_path: Path) -> None:
    """
    Execute the automated data cleaning and feature engineering pipeline.
    """
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
        .pipe(drop_columns, columns=columns_to_drop)
    )

    # save the data
    cleaned_data.to_csv(saved_data_path, index=False)


# ============================================================
# MAIN SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data save directory
    cleaned_data_save_dir = root_path / "data" / "cleaned"
    # make directory if not exits
    cleaned_data_save_dir.mkdir(exist_ok=True, parents=True)
    # cleaned data file name
    cleaned_data_filename = "swiggy_cleaned.csv"
    # data save path
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
    # data load path
    data_load_path = root_path / "data" / "raw" / "swiggy.csv"

    # load the data
    df = load_data(data_load_path)
    logger.info("Data read successfully")

    # clean the data and save
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
    logger.info("Data cleaned and saved")