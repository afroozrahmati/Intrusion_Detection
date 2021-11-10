import sys
import os
import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # plotting
from scipy import stats
import os
import pickle  # To load data int disk
from prettytable import PrettyTable  # To print in tabular format
from keras.utils import np_utils
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict



#file name as an input argument for each client
file_path_normal= './data/'+ sys.argv[0]
file_path_abnormal= './data/'+ sys.argv[1]
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#tensorflow version shoudl be 2.4.0

def normalize(img):
    '''
    Normalizes an array
    (subtract mean and divide by standard deviation)
    '''
    eps = 0.001
    #print(np.shape(img))
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img

def normalize_dataset(x):
    '''
    Normalizes list of arrays
    (subtract mean and divide by standard deviation)
    '''
    normalized_dataset = []
    for img in x:
        normalized = normalize(img)
        normalized_dataset.append(normalized)
    return normalized_dataset





if __name__ == "__main__":
    # Load dataset
    df_normal = pd.read_csv(file_path_normal)
    df_abnormal = pd.read_csv(file_path_abnormal)

    #data pre-processing step

    df = pd.concat([df_normal, df_abnormal]).reset_index(drop=True)
    df = df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco'])

    d = {'e': 1, 'e s': 2, 'e d': 3, 'e *': 4, 'e g': 5, 'eU': 6, 'e &': 7, 'e   t': 8, 'e    F': 9}
    df['flgs'] = df['flgs'].map(d)

    d = {'udp': 1, 'tcp': 2, 'arp': 3, 'ipv6-icmp': 4, 'icmp': 5, 'igmp': 6, 'rarp': 7}
    df['proto'] = df['proto'].map(d)

    df['bytes'] = df['bytes'].astype(int)
    df['pkts'] = df['pkts'].astype(int)
    df['spkts'] = df['spkts'].astype(int)
    df['dpkts'] = df['dpkts'].astype(int)
    df['sbytes'] = df['sbytes'].astype(int)
    df['dbytes'] = df['dbytes'].astype(int)
    df['sport'] = df.sport.fillna(value=0)
    df['dport'] = df.sport.fillna(value=0)

    df.loc[df.sport.astype(str).str.startswith('0x'), "sport"]
    df.loc[df.sport.astype(str).str.startswith('0x0303'), "sport"] = 771
    df.loc[df.dport.astype(str).str.startswith('0x'), "dport"]
    df.loc[df.dport.astype(str).str.startswith('0x0303'), "dport"] = 771
    df['sport'] = df['sport'].astype(float)
    df['dport'] = df['dport'].astype(float)
    d = {'CON': 1, 'INT': 2, 'FIN': 3, 'NRS': 4, 'RST': 5, 'URP': 6}
    df['state'] = df['state'].map(d)
    d = {'Normal': 0, 'UDP': 1, 'TCP': 2, 'Service_Scan': 3, 'OS_Fingerprint': 4, 'HTTP': 5}
    df['subcategory '] = df['subcategory '].map(d)
    df = df.drop(columns=['category'])

    train, test = train_test_split(df, test_size=0.3, random_state=16)
    #.........................................
    # creating x and y set from the dataset
    x_train, y_train = train.drop(columns=['attack']), train['attack']
    x_test, y_test = test.drop(columns=['attack']), test['attack']
    x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

    x_train = np.array(x_train)
    x_train = np.nan_to_num(x_train)
    x_train = normalize_dataset(x_train)
    x_test = np.array(x_test)
    x_test = np.nan_to_num(x_test)
    x_test = normalize_dataset(x_test)

    del df

    CLASSES = ['normal', 'abnormal']
    y_train = np.array(np_utils.to_categorical(y_train, len(CLASSES)))
    y_test = np.array(np_utils.to_categorical(y_test, len(CLASSES)))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    #........................
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("172.29.128.1:8080", client=CifarClient())