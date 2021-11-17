import sys
import os
import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
#from keras.utils import np_utils
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tqdm import tqdm_notebook
from datetime import datetime
import os, fnmatch
import tensorflow.keras
from ClusteringLayer import *

#path = os.getcwd()+'/Intrusion_Detection'
#os.chdir(path)
#file name as an input argument for each client
file_path_normal= sys.argv[1] # './data/normal.csv'   #+ sys.argv[0]
file_path_abnormal= sys.argv[2] #'./data/abnormal.csv'  #+ sys.argv[1]
# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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


def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


if __name__ == "__main__":
    # Load dataset
    print(file_path_normal)
    print(file_path_abnormal)
    df_normal = pd.read_csv(file_path_normal)
    df_abnormal = pd.read_csv(file_path_abnormal)
    print("df_normal shape: " ,df_normal.shape)
    print("df_abnormal shape: ", df_abnormal.shape)
    #data pre-processing step
    df = pd.concat([df_normal, df_abnormal], ignore_index=True)
    df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco','category'], axis=1 , inplace=True)

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
    y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
    y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    #........................
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Load and compile Keras model
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    tf.keras.backend.clear_session()
    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    n_classes = 2
    batch_size = 64
    epochs = 1000
    callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max', verbose=2, patience=800,
                              restore_best_weights=True)
    model_dir = './model/'
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]
    now = datetime.now()  # current date and time
    now = now.strftime("%m") + '_' + now.strftime("%d") + '_' + now.strftime("%Y") + '_' + now.strftime(
        "%H") + '_' + now.strftime("%M") + '_' + now.strftime("%S")

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)

    gamma = 1
    # tf.keras.backend.clear_session()
    print('Setting Up Model for training')
    print(gamma)
    model_name = now + '_' + 'Gamma(' + str(gamma) + ')-Optim(' + "Adam" + ')'
    print(model_name)

    model = 0

    inputs = encoder = decoder = hidden = clustering = output = 0
    inputs = keras.Input(shape=(timesteps, n_features))
    encoder = LSTM(32, activation='tanh')(inputs)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(100, activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder_out = Dense(100, activation=None, name='encoder_out')(encoder)
    clustering = ClusteringLayer(n_clusters=2, name='clustering', alpha=0.05)(encoder_out)
    hidden = RepeatVector(timesteps, name='Hidden')(encoder_out)
    decoder = Dense(100, activation='relu')(hidden)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = LSTM(32, activation='tanh', return_sequences=True)(decoder)
    output = TimeDistributed(Dense(n_features), name='decoder_out')(decoder)

    # kmeans = KMeans(n_clusters=2, n_init=100)

    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    # kmeans.fit(encoder_model.predict(x_train))

    model = Model(inputs=inputs, outputs=[clustering, output])

    clustering_model = Model(inputs=inputs, outputs=clustering)

    # plot_model(model, show_shapes=True)
    model.summary()
    q, _ = model.predict(x_train, verbose=2)
    q_t, _ = model.predict(x_test, verbose=2)
    p = target_distribution(q)

    y_pred = np.argmax(p, axis=1)
    y_arg = np.argmax(y_train, axis=1)
    acc = np.round(accuracy_score(y_arg, y_pred), 5)

    print('====================')
    print('====================')
    print('====================')
    print('====================')
    print('Pre Training Accuracy')
    print(acc)
    print('====================')
    print('====================')
    print('====================')
    print('====================')
    model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer='adam',
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            print('Model compiled.')
            print('Training Starting:')
            train_history = model.fit(x_train,
                                      y={'clustering': y_train, 'decoder_out': x_train},
                                      epochs=epochs,
                                      validation_split=0.2,
                                      # validation_data=(x_test, (y_test, x_test)),
                                      batch_size=batch_size,
                                      verbose=2,
                                      callbacks=callbacks)
            #model.fit(x_train, y_train, epochs=1, batch_size=32)
            results = {
                "clustering_loss": train_history.history["clustering_loss"][0],
                "decoder_out_loss": train_history.history["decoder_out_loss"][0],
                "clustering_accuracy": train_history.history["clustering_accuracy"][0],
                "decoder_out_mse": train_history.history["decoder_out_mse"][0],
                "val_clustering_loss": train_history.history["val_clustering_loss"][0],
                "val_decoder_out_loss": train_history.history["val_decoder_out_loss"][0],
                "val_clustering_accuracy": train_history.history["val_clustering_accuracy"][0],
            }

            return model.get_weights(), len(x_train), results

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            #loss, accuracy = model.evaluate(x_test, y_test)

            q, _ = model.predict(x_train, verbose=0)
            q_t, _ = model.predict(x_test, verbose=0)
            p = target_distribution(q)

            y_pred = np.argmax(q, axis=1)
            y_arg = np.argmax(y_train, axis=1)
            y_pred_test = np.argmax(q_t, axis=1)
            y_arg_test = np.argmax(y_test, axis=1)
            # acc = np.sum(y_pred == y_arg).astype(np.float32) / y_pred.shape[0]
            # testAcc = np.sum(y_pred_test == y_arg_test).astype(np.float32) / y_pred_test.shape[0]
            acc = np.round(accuracy_score(y_arg, y_pred), 5)
            testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)

            nmi = np.round(normalized_mutual_info_score(y_arg, y_pred), 5)
            nmi_test = np.round(normalized_mutual_info_score(y_arg_test, y_pred_test), 5)
            ari = np.round(adjusted_rand_score(y_arg, y_pred), 5)
            ari_test = np.round(adjusted_rand_score(y_arg_test, y_pred_test), 5)
            print('====================')
            print('====================')
            print('====================')
            print('====================')
            print('Train accuracy')
            print(acc)
            print('Test accuracy')
            print(testAcc)

            print('NMI')
            print(nmi)
            print('ARI')
            print(ari)
            print('====================')
            print('====================')
            print('====================')
            print('====================')
            num_examples_test = len(self.x_test)
            return  _,num_examples_test, {"accuracy": testAcc}

    # Start Flower client
    fl.client.start_numpy_client("172.18.80.1:8080", client=CifarClient())