import numpy as np
import pandas as pd

columns_to_drop = [
    "id",
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
    "order_time",
    "order_picked_time",
]


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
                "delivery_person_rating": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_ordered": "order_time",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken",
            }
        )
    )


def time_of_day(ser: pd.Series) -> np.ndarray:
    """
    Categorize order hour/time into periods of the day.
    """
    if pd.api.types.is_numeric_dtype(ser):
        hour = ser
    else:
        hour = pd.to_datetime(ser, format="%H:%M:%S", errors="coerce").dt.hour

    return np.select(
        condlist=[
            hour.between(6, 11),
            hour.between(12, 16),
            hour.between(17, 19),
            hour.between(20, 23),
        ],
        choicelist=[
            "morning",
            "afternoon",
            "evening",
            "night",
        ],
        default="after_midnight",
    )


def data_cleaning(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Perform core data cleaning, type conversions, and feature extraction.
    """
    df = df_input.copy()

    # 1. Replace invalid string values with actual NaN
    df.replace(
        {
            "NaN ": np.nan,
            "conditions NaN": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
        },
        inplace=True,
    )

    # 2. Extract city name from rider ID
    if "rider_id" in df.columns:
        df["city_name"] = (
            df["rider_id"]
            .astype("string")
            .str.split("RES")
            .str[0]
        )

    # 3. Convert numeric columns
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

    # 4. Clean time_taken column if present
    if "time_taken" in df.columns:
        df["time_taken"] = (
            df["time_taken"]
            .astype("string")
            .str.replace("(min)", "", regex=False)
            .str.strip()
        )
        df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")

    # 5. Clean weather column
    if "weather" in df.columns:
        df["weather"] = (
            df["weather"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("conditions ", "", regex=False)
            .replace("nan", np.nan)
        )

    # 6. Clean categorical columns
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

    # 7. Process order date
    if "order_date" in df.columns:
        date_col = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
        df["order_day"] = date_col.dt.day
        df["order_month"] = date_col.dt.month
        df["order_year"] = date_col.dt.year
        df["order_day_of_week"] = date_col.dt.day_name().str.lower()
        df["is_weekend"] = date_col.dt.dayofweek.isin([5, 6]).astype(int)

    # 8. Process order and pickup times
    if "order_time" in df.columns and "order_picked_time" in df.columns:
        order_t = pd.to_datetime(df["order_time"], format="mixed", errors="coerce")
        picked_t = pd.to_datetime(df["order_picked_time"], format="mixed", errors="coerce")

        adjusted_picked_t = picked_t.where(
            picked_t >= order_t,
            picked_t + pd.Timedelta(days=1),
        )

        df["pickup_time_minutes"] = (
            adjusted_picked_t - order_t
        ).dt.total_seconds() / 60

        df["order_time_hour"] = order_t.dt.hour
        df["order_time_of_day"] = time_of_day(df["order_time_hour"])

    return df


def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Clean latitude and longitude by converting absolute values.
    """
    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]
    loc_cols = [col for col in location_columns if col in data.columns]

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
    return data


def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the great circle distance between restaurant and delivery location
    using the Haversine formula in kilometers.
    """
    lat1 = df["restaurant_latitude"]
    lon1 = df["restaurant_longitude"]
    lat2 = df["delivery_latitude"]
    lon2 = df["delivery_longitude"]

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)


def create_distance_type(data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize delivery distance into bins: short, medium, long, very_long.
    """
    return data.assign(
        distance_type=pd.cut(
            data["distance"],
            bins=[0, 5, 10, 20, np.inf],
            labels=["short", "medium", "long", "very_long"],
            include_lowest=True,
        )
    )


def drop_columns(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Drop specified columns from the dataset.
    """
    if columns is None:
        return data
    return data.drop(columns=columns, errors="ignore")


def perform_data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the data cleaning and feature engineering pipeline on input dataframe.
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
    return cleaned_data
if __name__ == "__main__":
    # data path for data
    DATA_PATH = "swiggy.csv"

    # read the data from path
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully")

    perform_data_cleaning(df)
    print("Data cleaned successfully")
    