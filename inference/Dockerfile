#Building on base image
FROM python:3.11.6

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY ../data /app/data

# Copy models folder to Docker
COPY ../models /app/models

# Copy the code itself
COPY inference /app/inference

# Copy usefull scripts and configs
COPY ../utils.py /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runs trained model on inference data
CMD ["python3","inference/run.py"]