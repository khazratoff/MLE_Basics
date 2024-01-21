#Building on base image
FROM python:3.11.6

#Adding MODEL_NAME as build argument
ARG MODEL_NAME="final_nn_model.h5"

# Setting the build arguments as environment variable to access through training.py script
ENV MODEL_NAME=${MODEL_NAME}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy the code itself
COPY training /app/training

#Copy utils.py
COPY ../utils.py /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 training/train.py
