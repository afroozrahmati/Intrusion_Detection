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
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from data_processing import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score,mean_squared_error,mutual_info_score
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.cluster import adjusted_rand_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from ClusteringLayer import *

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

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def get_model(timesteps,n_features):


    gamma = 1
    # tf.keras.backend.clear_session()
    print('Setting Up Model for training')
    print(gamma)

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
    encoder_model = Model(inputs=inputs, outputs=encoder_out)
    # kmeans.fit(encoder_model.predict(x_train))

    model = Model(inputs=inputs, outputs=[clustering, output])

    clustering_model = Model(inputs=inputs, outputs=clustering)

    # plot_model(model, show_shapes=True)
    model.summary()
    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    model.compile(loss={'clustering':  'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer=optimizer,
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    print('Model compiled.           ')
    return model

def model_training(model,x_train,y_train,epochs=1000):
    print('Training Starting:')

    print("train shape: ", np.shape(x_train))
    print("train label shape: ",np.shape( y_train.shape))

    callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max', verbose=2, patience=800,
                              restore_best_weights=True)
    batch_size = 64


    train_history = model.fit(x_train,
                              y={'clustering': y_train, 'decoder_out': x_train},
                              epochs=epochs,
                              validation_split=0.2,
                              # validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=2,
                              callbacks=callbacks
                              )
    return model

# def weight_scalling_factor(clients_trn_data, client_name):
#     client_names = list(clients_trn_data.keys())
#     #get the bs
#     #bs = list(clients_trn_data[client_name])[0][0].shape[0]
#
#     bs = np.shape(clients_trn_data[client_name][0])[0]
#
#     #first calculate the total training data points across clinets
#     global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
#     # get the total number of data points held by a client
#     local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
#     return local_count/global_count

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


file_path_normal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]
x_train, y_train, x_test, y_test = load_processed_data(file_path_normal, file_path_abnormal)  # args.partition)

x_train = np.asarray(x_train)
x_test = np.nan_to_num(x_test)
x_test = np.asarray(x_test)




#create clients
clients = create_clients(x_train, y_train, num_clients=10, initial='client')

# process and batch the training data for each client
# process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))


# process and batch the test set
#test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
timesteps = np.shape(x_train)[1]
n_features = np.shape(x_train)[2]

model = get_model(timesteps, n_features)
comms_round = 1000


# commence global training loop
for comm_round in range(comms_round):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    # randomize client data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)



    # loop through each client and create new local model
    for client in client_names:

        local_model = get_model(timesteps, n_features)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)


        #model = model_training(local_model, x_data,y_data,epochs=1)
        # fit local model with client's data
        #local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
    #     scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
    #     scaled_local_weight_list.append(scaled_weights)
    #
    #     # clear session to free memory after each communication round
    #     K.clear_session()
    #
    # # to get the average over all the local model, we simply take the sum of the scaled weights
    # average_weights = sum_scaled_weights(scaled_local_weight_list)
    #
    # # update global model
    # global_model.set_weights(average_weights)
    #
    # # test global model and print out metrics after each communications round
    # for (X_test, Y_test) in test_batched:
    #     global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
#         SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)