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
In the shell (terminal, powershell) you can the `logging.info` created by the script along with model's `accuracy` score.
Make sure that train and inference data created in `/data` folder along with model in `/models` folder.
## Inference

