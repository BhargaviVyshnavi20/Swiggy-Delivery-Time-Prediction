# Live check : sends a real row (minus the target) to the running FastAPI
# server and asserts it responds with 200. Requires `unicor app` to
# already be running on port 8000 in a separate terminal.
# Run individually: pytest tests/test_api_endpoint.py -v -s
import pandas as pd
import requests
from pathlib import Path
import pytest

# path for data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# sample row for testing
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ", ""))

# remove the target column 
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()

@pytest.mark.parametrize(argnames="url, data",
                            argvalues=[("http://localhost:8000/predict", data)])


def test_predict_endpoint(url,data):
    # get the response from API
    response = requests.post(url=url, json=data)
    # test for 200 code
    assert response.status_code == 200, "Prediction endpoint not giving response"
    