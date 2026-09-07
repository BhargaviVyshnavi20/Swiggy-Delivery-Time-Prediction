# set the base image
From python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set the working directory
WORKDIR /app

# copy the requirements file
COPY requirements.txt 

# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the app contents
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py
COPY ./run_information.json ./

# expose the port
EXPOSE 8000

# run the application
CMD ["python", "./app.py"]