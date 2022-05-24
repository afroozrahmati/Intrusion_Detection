from typing import List
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import scatter_nd_non_aliasing_add_eager_fallback
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics.pairwise as smp
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
from fl_ss_data_processing import *
from csv import writer
import matplotlib.pyplot as plt
#from triangle_sector_similarity import Cosine_Similarity,Euclidean_Distance,TS_SS,Pairwise_TS_SS
import math
import torch
import csv
from itertools import zip_longest
import config
#from tensorflow.python.ops.numpy_ops import np_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_sim_model(timesteps,n_features):
    model = RandomForestClassifier(criterion='entropy')  

    return model

def model_sim_training(model,x_train,y_train):
    model.fit(x_train,y_train)
    return model

def model_sim_evaluate(path, attack, defense, log_name,model,x_train,y_train,x_test,y_test,epochs, num_sybils):
    y_predict = rf_clf.predict(x_test)
    tesetAcc = accuracy_score(y_test,y_predict)

    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write('\n############################################################################################\n')
            f.write('\n############################################################################################\n')
            f.write('\ncomm_round: {} || global_test_acc: {:.3%}'.format(epochs, testAcc))
    f.close()
    print('\ncomm_round: {} || global_test_acc: {:.3%}'.format(epochs, testAcc))

