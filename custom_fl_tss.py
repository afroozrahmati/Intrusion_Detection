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
import sklearn.metrics.pairwise as smp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_processed_data(file_path_normal,file_path_abnormal):
    data_process= data_processing()
    x_train,y_train,x_test,y_test = data_process.load_data(file_path_normal,file_path_abnormal)

    # print("train shape: ", np.shape(x_train))
    # print("test shape: ", np.shape(x_test))
    # print("train label shape: ", y_train.shape)
    # print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    return x_train,y_train,x_test, y_test


def create_clients(x_train, y_train, num_clients=10, initial='clients'):
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

    # shard data and place at each client
    size = len(x_train) // num_clients
    client_dict={}
    for i in range(num_clients):
        client_dict[client_names[i]]= [x_train[i:i + size], y_train[i:i + size]]

    return client_dict


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
    # print('Setting Up Model for training')
    # print(gamma)

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
    #model.summary()
    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    model.compile(loss={'clustering':  'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer=optimizer,
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    # print('Model compiled.           ')
    return model

def model_training(model,x_train,y_train,epochs=1000):
    # print('Training Starting:')
    #
    # print("train shape: ", np.shape(x_train))
    # print("train label shape: ", y_train.shape)

    callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max', verbose=2, patience=800,
                              restore_best_weights=True)
    batch_size = 64


    train_history = model.fit(x_train,
                              y={'clustering': y_train, 'decoder_out': x_train},
                              epochs=epochs,
                              validation_split=0.2,
                              # validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=0,
                              callbacks=callbacks
                              )
    return model

def weight_scalling_factor(clients_trn_data, client_name):
    # client_names = list(clients_trn_data.keys())
    # #get the bs
    # #bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #
    # bs = np.shape(clients_trn_data[client_name][0])[0]
    #
    # #first calculate the total training data points across clinets
    # global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # # get the total number of data points held by a client
    # local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs

    local_count = 1
    global_count = len(clients_trn_data)
    return local_count/global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def model_evaluate(model,x_train,y_train,x_test,y_test,epochs):
    q,_ = model.predict(x_train, verbose=0)
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
    # print('====================')
    # print('====================')
    # print('====================')
    # print('====================')
    # print('Train accuracy')
    # print(acc)
    # print('Test accuracy')
    # print(testAcc)
    #
    # print('NMI')
    # print(nmi)
    # print('ARI')
    # print(ari)
    # print('====================')
    # print('====================')
    # print('====================')
    # print('====================')

    print('comm_round: {} | global_acc: {:.3%} | global_nmi: {} | global_ari: {}'.format(epochs, testAcc, nmi,ari))


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = len(grads)

    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv

# client_grads = Compute gradients from all the clients
def aggregate_gradients(client_grads):
    num_clients = len(client_grads)

    grad_len = np.array(client_grads[0][-2].data.shape).prod()

    grads = np.zeros((num_clients, grad_len))
    for i in range(len(client_grads)):
        grads[i] = np.reshape(client_grads[i][-2].data, (grad_len))

    wv = foolsgold(grads)  # Use FG

    print(wv)

    agg_grads = []
    # Iterate through each layer
    for i in range(len(client_grads[0])):
        temp = wv[0] * client_grads[0][i]
        # Aggregate gradients for a layer
        for c, client_grad in enumerate(client_grads):
            if c == 0:
                continue
            temp += wv[c] * client_grad[i]
        temp = temp / len(client_grads)
        agg_grads.append(temp)

    return agg_grads


file_path_normal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]
x_train, y_train, x_test, y_test = load_processed_data(file_path_normal, file_path_abnormal)  # args.partition)

x_train = np.asarray(x_train)
x_test = np.nan_to_num(x_test)
x_test = np.asarray(x_test)


#create clients
clients = create_clients(x_train, y_train, num_clients=10, initial='client')

# process and batch the training data for each client
client_names=[]
for (client_name, data) in clients.items():
    print("x_train", np.shape(data[0]))
    print("y_train", np.shape(data[1]))
    client_names.append(client_name)


# process and batch the test set
#test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
timesteps = np.shape(x_train)[1]
n_features = np.shape(x_train)[2]

global_model = get_model(timesteps, n_features)
comms_round = 100


# commence global training loop
for comm_round in range(comms_round):
    print("start round" ,comm_round )
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    client_grads = []
    # loop through each client and create new local model
    for (client_name, data) in clients.items():

        local_model = get_model(timesteps, n_features)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        local_model = model_training(local_model, data[0], data[1],epochs=1)
        # fit local model with client's data
        #local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        #scaling_factor = weight_scalling_factor(clients, client_name)
        #scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(local_model.get_weights())

        client_grads.append(local_model.get_weights())

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    #average_weights = sum_scaled_weights(scaled_local_weight_list)
    average_weights = aggregate_gradients(client_grads)
    # update global model
    global_model.set_weights(average_weights)

    # test global model and print out metrics after each communications round

    model_evaluate(global_model,x_train,y_train,x_test,y_test,comm_round)
