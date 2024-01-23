# MLE Basics
Welcome to the `MLE_Basics` project. It's a homework of Basics of Machine Learning Engineering module at EPAM Data Science training.
## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_Basics
├── data                      # Data files used for training and inference (it can be generated with training/train.py script.
│   ├── train_data.csv
│   └── inference_data.csv       
├── inference                 # Scripts and Dockerfiles used for inference.
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where last trained model is stored.
│   └── model file (.h5)
├── training                  # Scripts and Dockerfiles used for training.
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── results                    # Folder where result of the model trained on inference data is stored.
│   └── result file (.csv)
├── utils.py                  # Utility functions and classes that are used in scripts.
└── README.md
```
## Training:
The training phase of this project can be done by running `training/train.py` script.
<br>To train the model using Docker following insturctions should be done: 
- Build the training Docker image using this command:
```bash
docker build -f ./training/Dockerfile  --build-arg MODEL_NAME=<your_model_name.h5> -t training_image .
```
Replace `model_name` whatever you want for the name of your model but make sure to give `.h5` extension (for keras model) to it.
- After successfull built you have to run the container that actually trains the model, and mounts `/models` and `/data` folders on container to your local machine by the following command:
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data training_image
```
If you get an error, probably it is because there are some whitespaces in your printed working directory(pwd).
To handle this instead of using `pwd` write your project path by hand like:`/Users/john/desktop/MLE_Basics`.
In the shell (terminal, powershell) you can see the training phase with `logging.info` created by the script along with model's `accuracy` score.
Make sure that `train_data.csv` and `inference_data.csv` created in `/data` folder along with your model in `/models` folder.
## Inference
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in inference/run.py.
- Build the inference Docker image:
```bash
  docker build -f ./inference/Dockerfile -t inference_image .
```
- Run the inference Docker container with following command:
``` bash
docker run -v $(pwd)/results:/app/results inference_image
```
Make sure you don't get any errors related to your pathname (pwd). After succesfull run, inference phase can be shown in shell, and what's more important is that `/results` folder in container will be mounted on your local `/results` folder which keeps result of the models prediction (csv file)  on inference data.
## What's more
- Some tests and exceptions are included in some problematic parts in the code.<br> *Looking forward to get your feedback, and if you don't get any desired results please let me know.*

