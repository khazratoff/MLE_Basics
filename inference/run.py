"""
Script loads the latest trained model, data for inference, predicts results and stores them.
"""
import os 
import sys
import warnings
import logging 

import pandas as pd
import numpy as np

from keras.models import load_model

#Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting inference...\U0001f600")

#Configuring warnings
warnings.filterwarnings("ignore") 

#Defining ROOT directory, and appending it to the sys.path
# so that python know which file should be included within the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

#Defining data, models and results directories
from utils import get_project_dir
DATA_PATH = get_project_dir('data')
MODEL_PATH = get_project_dir('models')
RESULTS_PATH = get_project_dir('results')


def load_latest_model(model_path):
    '''Gets latest trained model'''
    logging.info("Loading latest model...")
    try:
        # Getting the list of models in the MODEL PATH
        models = [os.path.join(model_path, file) for file in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, file))]

        # Sorting models by their creation time
        sorted_models = sorted(models, key=os.path.getctime, reverse=True)

        #Checks for the avaliable models if there're no any raises Exception
        try:
            return sorted_models[0]
        except:
            logging.error(f"No pretrained model found... Make sure to train your model first!")
            sys.exit(1)

    except FileNotFoundError:
        logging.error(f"No pretrained model found... Make sure to train your model first!")
        sys.exit(1)

def load_inference_data(data_path: str):
    '''Loads inference data and returns it as DataFrame'''
    logging.info("Loading inference data...")
    try:
        data = pd.read_csv(os.path.join(data_path,"inference_data.csv"))
        return data

    except Exception as ex:
        logging.error(f"An error occurred while loading inference data: {ex}")
        sys.exit(1)

def get_results(model:object, data: pd.DataFrame):
    '''Makes model to predict on inference data and
       converts probabilites into actual classes (0.Setosa, 1.Versicolor, 2.Virginica)'''
    logging.info("Predicting results...")
    model = load_model(model)
    results = model.predict(data)
    classes = [np.argmax(result) for result in results]
    data['class'] = classes
    return data

def save_results(result: pd.DataFrame,latest_model_name:str):
    '''Store the prediction results in 'results' directory with the name of the model predicted those results'''
    logging.info("Saving results...")
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    path = os.path.join(RESULTS_PATH, os.path.basename(latest_model_name)+"_results.csv")
    pd.DataFrame(result).to_csv(path, index=False)
    logging.info("Results saved successfully \N{grinning face}")


def main():
    '''Main Function'''
    latest_model = load_latest_model(MODEL_PATH)
    data = load_inference_data(DATA_PATH)
    results = get_results(latest_model,data)
    save_results(result=results, latest_model_name = latest_model)
   

if __name__ == "__main__":
    main()