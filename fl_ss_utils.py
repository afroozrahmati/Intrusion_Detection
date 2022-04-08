import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics.pairwise as smp
from sklearn.metrics import accuracy_score, f1_score, precision_score
from fl_ss_data_processing import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This function loads processed data.  
# file_path_normal: CSV file 
# file_path_abnormal: CSV file
# returns:  4 arrays
def load_processed_data(file_path_normal,file_path_abnormal):
    data_process= data_processing()
    x_train,y_train,x_test,y_test = data_process.load_data(file_path_normal,file_path_abnormal)

    #print("train shape: ", np.shape(x_train))
    #print("test shape: ", np.shape(x_test))
    #print("train label shape: ", y_train.shape)
    #print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    return x_train,y_train,x_test, y_test


''' create_clients creates a number of 
   args:
        image_list: a list of numpy arrays of training images
        label_list:a list of binarized labels for each image
        num_client: number of fedrated members (clients)
        initials: the clients'name prefix, e.g, clients_1

   return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        
'''
def create_clients(x_train, y_train, num_clients=10, initial='clients'):
    print("Creating Clients with Data Shards \n")

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # shard data and place at each client
    size = len(x_train) // num_clients
    #print("size is ", size, "\n")
    client_dict={}
    for i in range(num_clients):
        client_dict[client_names[i]]= [x_train[i:i + size], y_train[i:i + size]]
        #print("client is ", client_names[i])

    return client_dict


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    #print("Data Shard used to be *data_shard ",data_shard)
    data, label = zip(data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def get_model(timesteps,n_features):
#def get_model(x_train):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(timesteps, n_features)))
    #model.add(LSTM(100, input_shape=(timesteps, n_features)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.25))
    model.add(LSTM(128))
    model.add(Dropout(.25))
    #model.add(Dense(2))
    model.add(Dense(2, activation='softmax'))
    #model.add(Dense(2, activation='sigmoid'))
    #model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(),'accuracy'])
    #model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_training(model,x_train,y_train,epochs=4000):
    #print('Training Starting:')
    #print("train shape: ", np.shape(x_train))
    #print("train head", x_train[:5])
    #print("train label shape: ", y_train.shape)

    callbacks = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=1000,
                              restore_best_weights=True)
   # mc = ModelCheckpoint('best_model.h5', monitor='binary_accuracy', mode='max', verbose=0, save_best_only=True)
    batch_size = 10
  
    train_history = model.fit(x_train,
                              y_train,
                              epochs=epochs,
                              validation_split=0.2,
                              #validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=0,
                              callbacks=callbacks
                              )
    #saved_model = load_model('best_model.h5')

    return model

def weight_scalling_factor(clients_trn_data, client_name):
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
    q = model.predict(x_train, verbose=0)
    q_t = model.predict(x_test, verbose=0)

    #p = target_distribution(q)

    y_pred = np.argmax(q, axis=1)
    y_arg = np.argmax(y_train, axis=1)
    y_pred_test = np.argmax(q_t, axis=1)
    y_arg_test = np.argmax(y_test, axis=1)

    m = tf.keras.metrics.binary_accuracy(y_arg, y_pred, threshold=0.5)
    #score = model.evaluate(x_test, y_test, verbose=0)
    testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
    f1 = f1_score(y_arg, y_pred)
    precision = precision_score(y_arg, y_pred)

    print('comm_round: {} | global_acc: {:.3%} | global_f1: {} | global_precision: {} | global bin {}'.format(epochs, testAcc, f1, precision, m))


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

    #print(wv)

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