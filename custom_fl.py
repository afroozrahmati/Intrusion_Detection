#reference to this intresting artile to create a simple custom FL model
#https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import sys
from data_processing import *

def load_processed_data(file_path_normal,file_path_abnormal):
    data_process= data_processing()
    x_train,y_train,x_test,y_test = data_process.load_data(file_path_normal,file_path_abnormal)

    print("train shape: ", np.shape(x_train))
    print("test shape: ", np.shape(x_test))
    print("train label shape: ", y_train.shape)
    print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    return x_train,y_train,x_test, y_test


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


file_path_normal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]
x_train, y_train, x_test, y_test = load_processed_data(file_path_normal, file_path_abnormal)  # args.partition)

#create clients
clients = create_clients(x_train, y_train, num_clients=10, initial='client')

