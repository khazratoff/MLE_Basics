"""
Script loads and prepares the data, runs the training, and saves the model.
"""

import os
import sys
import logging
import warnings

import pandas as pd
import numpy as np

from keras.models import Sequential 
from keras.layers import Dense, Dropout
from tensorflow.python.keras.utils import np_utils

import sklearn.preprocessing as pre
import sklearn.impute as imp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Configuring warnings
warnings.filterwarnings("ignore") 

#Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting...\U0001f600")

#Defining ROOT directory, and appending it to the sys.path
# so that python know which file should be included within the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

#Loading dataset
logging.info(f"Loading iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
X, X_infer, y, y_infer = train_test_split(X, y, test_size=0.2, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=101)

#Defining data and model directories
from utils import get_project_dir
DATA_PATH = get_project_dir('data')
MODEL_PATH = get_project_dir('models')

def define_train_data():
    '''Creates train data in data folder'''
    train_data = pd.DataFrame(data = X)
    train_data['class'] = y
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    train_data_path = os.path.join(DATA_PATH,"train_data.csv")
    train_data.to_csv(train_data_path,index=False)
    logging.info(f'Train data saved successfully!')


def define_inference_data() -> None:
    '''Creates inference data in data folder'''
    infer_data = pd.DataFrame(data = X_infer)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    infer_data_path = os.path.join(DATA_PATH,"inference_data.csv")
    infer_data.to_csv(infer_data_path,index=False)
    logging.info(f'Inference data saved successfully!')


def prepare_data(train_data,test_data,y_train,y_test) -> tuple:
    '''Imputes, scales features and one-hot encodes classes(labels)'''
    imputer = imp.SimpleImputer(strategy='median')
    scaler = pre.MinMaxScaler(feature_range=(0,1))
    train_data = imputer.fit_transform(train_data)
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    y_train=np_utils.to_categorical(y_train,num_classes=3)
    y_test=np_utils.to_categorical(y_test,num_classes=3)
    return train_data,test_data,y_train,y_test

def trainNN(X_train,X_test,y_train,y_test) -> object:
    '''Trains Deep Neural Network model on train data and evaluates it on the test data'''
    logging.info("Preparing data...")
    logging.info("Training the model...")

    X_train,X_test,y_train,y_test = prepare_data(X_train,X_test,y_train,y_test)

    # NN architecture
    model=Sequential()
    model.add(Dense(1000,input_dim=4,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(300,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #Fitting the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)

    #Evaluating on the test set
    prediction=model.predict(X_test)
    length=len(prediction)
    y_label=np.argmax(y_test,axis=1)
    predict_label=np.argmax(prediction,axis=1)
    accuracy=np.sum(y_label==predict_label)/length * 100 
    logging.info(f"Accuracy of the model: {accuracy:.1f}%" )
    return model

def save_trained_model():
    '''Saves the model in models directory'''
    #Training the model
    model = trainNN(X_train,X_test,y_train,y_test)

    #Saving the model to the models folder
    logging.info("Saving the model...")
    model_name = os.environ.get("MODEL_NAME", "final_nn_model.h5")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    latest_model_path = os.path.join(MODEL_PATH,model_name)
    model.save(latest_model_path)
    logging.info(f'"{model_name}" saved successfully! \N{grinning face}')



def main():
    '''Main Function'''
    define_train_data()
    define_inference_data()
    save_trained_model()



if __name__ == "__main__":
    main()