from typing import List
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import scatter_nd_non_aliasing_add_eager_fallback
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM,  Bidirectional 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics.pairwise as smp
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix
from fl_ss_data_processing import *
from csv import writer
import matplotlib.pyplot as plt
#from triangle_sector_similarity import Cosine_Similarity,Euclidean_Distance,TS_SS,Pairwise_TS_SS
import math
import torch
import csv
from itertools import zip_longest
from sklearn import preprocessing
import config
from scipy.special import logit, expit
from numpy import errstate
from scipy.stats import norm
from sklearn.metrics import jaccard_score
from keras.optimizers import gradient_descent_v2
#from keras.layers.core import Dense, Dropout, Flatten
#from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
#from tensorflow.python.ops.numpy_ops import np_config



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This function loads processed data.  
# file_path_normal: CSV file 
# file_path_abnormal: CSV file
# returns:  4 arrays
def load_processed_data(file_path_normal,file_path_abnormal,path, attack, defense, log_name,num_sybils=1):
    data_process= data_processing()
    timesteps = config.IDS_TIMESTEPS
    x_train,y_train,x_test,y_test,x_trainP,y_trainP,x_testP,y_testP, x_trainDbaProto, x_testDbaProto, y_trainDbaProto, y_testDbaProto , x_trainDbaPkts, x_testDbaPkts, y_trainDbaPkts, y_testDbaPkts, x_trainDbaDport, x_testDbaDport, y_trainDbaDport, y_testDbaDport, x_trainDbaBytes, x_testDbaBytes, y_trainDbaBytes, y_testDbaBytes = data_process.load_data(file_path_normal,file_path_abnormal,config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,config.NUM_SYBILS, timesteps)

    #print("train shape: ", np.shape(x_train))
    #print("test shape: ", np.shape(x_test))
    #print("train label shape: ", y_train.shape)
    #print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    x_trainP = np.asarray(x_trainP)
    x_testP = np.nan_to_num(x_testP)
    x_testP = np.asarray(x_testP)
    x_trainDbaProto = np.asarray(x_trainDbaProto)
    x_testDbaProto = np.nan_to_num(x_testDbaProto)
    x_testDbaProto = np.asarray(x_testDbaProto)
    x_trainDbaPkts = np.asarray(x_trainDbaPkts)
    x_testDbaPkts = np.nan_to_num(x_testDbaPkts)
    x_testDbaPkts = np.asarray(x_testDbaPkts)
    x_trainDbaDport = np.asarray(x_trainDbaDport)
    x_testDbaDport = np.nan_to_num(x_testDbaDport)
    x_testDbaDport = np.asarray(x_testDbaDport)
    x_trainDbaBytes = np.asarray(x_trainDbaBytes)
    x_testDbaBytes = np.nan_to_num(x_testDbaBytes)
    x_testDbaBytes = np.asarray(x_testDbaBytes)
    return x_train,y_train,x_test,y_test,x_trainP,y_trainP,x_testP,y_testP, x_trainDbaProto, x_testDbaProto, y_trainDbaProto, y_testDbaProto , x_trainDbaPkts, x_testDbaPkts, y_trainDbaPkts, y_testDbaPkts, x_trainDbaDport, x_testDbaDport, y_trainDbaDport, y_testDbaDport, x_trainDbaBytes, x_testDbaBytes, y_trainDbaBytes, y_testDbaBytes


''' create_clients creates a number of 
   args:
        image_list: a list of numpy arrays of training images
        label_list:a list of binarized labels for each image
        num_client: number of fedrated members (clients)
        initials: the clients'name prefix, e.g, clients_1

   return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        
'''
def create_clients(path, attack, num_sybils, defense, log_name, x_train, y_train, num_clients=10, initial='clients'):
    print("Create Clients: {}\n".format(num_clients))
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write("Create Clients: {}\n".format(num_clients))
    f.close()

    # create a list of client names
    if attack == 'label' or attack == 'backdoor':
        client_names = ['{}_{}'.format(initial, i + (num_sybils+1)) for i in range(num_clients)]
    elif attack == 'dba':
        client_names = ['{}_{}'.format(initial, i + ((num_sybils*4)+1)) for i in range(num_clients)]
    # shard data and place at each client
    size = len(x_train) // num_clients
    #print("size is ", size, "\n")
    client_dict={}
    for i in range(num_clients):
        client_dict[client_names[i]]= [x_train[i:i + size], y_train[i:i + size]]
        #print("client is ", client_names[i])

    return client_dict


def create_backdoor_sybils(path, attack, defense, log_name,x_trainP, y_trainP, num_sybils=1, num_clients=1, initial='client'):
    print("Creating Backdoor sybils\nnum sybils {} and num clients {}".format(num_sybils, num_clients))
    num = num_sybils
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write("Create Backdoor Sybils: {}\n".format(num_sybils))
    f.close()


    # create a list of sybil names  i + 1
    sybil_names = ['{}_{}'.format(initial, i + 1) for i in range(num_sybils)]

    # shard data and place at each client
    size = len(x_trainP) // num_sybils
    #print("size is ", size, "\n")
    sybil_dict={}
    for i in range(num_sybils):
        sybil_dict[sybil_names[i]]= [x_trainP[i:i + size], y_trainP[i:i + size]]
        #print("client is ", client_names[i])

    return sybil_dict

def create_dba_sybils(path, attack, defense, log_name, x_trainDbaProto, y_trainDbaProto,x_trainDbaPkts, y_trainDbaPkts,x_trainDbaDport,y_trainDbaDport, x_trainDbaBytes, y_trainDbaBytes, num_sybils=1,num_clients=1, initial='client'):
    print("create_dba sybils num sybils {} and num clients {}".format(num_sybils, num_clients))
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write('\nCreate DBA Sybils {}\n'.format(num_sybils))
        f.close()
    print('Creating {} DBA Sybils with Data Shards \n'.format(num_sybils))
    sybil_dict={}
    protoDict = create_proto_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, x_trainDbaProto, y_trainDbaProto,config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
    sybil_dict.update(protoDict)
    pktsDict = create_pkts_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, x_trainDbaPkts, y_trainDbaPkts,config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
    sybil_dict.update(pktsDict)
    dportDict = create_dport_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, x_trainDbaDport, y_trainDbaDport, config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
    sybil_dict.update(dportDict)    
    bytesDict = create_bytes_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, x_trainDbaBytes, y_trainDbaBytes, config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
    sybil_dict.update(bytesDict)

    return sybil_dict


def create_proto_sybils(path, attack, defense, log_name, x_trainDbaProto, y_trainDbaProto, num_sybils,num_clients=1, initial='client'):
    print("create_proto sybils num sybils {} and num clients {}".format(num_sybils, num_clients))
    num = num_sybils
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write('\nCreate Proto DBA Sybils: {}\n'.format(num))
        f.close()
    print('Creating {} Proto DBA Sybils with Data Shards \n'.format(num))

    # create a list of sybil names
    sybil_names = ['{}_{}'.format(initial, i + 1) for i in range(num)]

    # shard data and place at each client
    sizeProto = len(x_trainDbaProto) // num_sybils
    #print("size is ", size, "\n")
    sybil_dict={}
   
    for i in range(num):
        sybil_dict[sybil_names[i]]= [x_trainDbaProto[i:i + sizeProto], y_trainDbaProto[i:i + sizeProto]]
        #print("client is ", client_names[i])

    return sybil_dict

def create_pkts_sybils(path, attack, defense, log_name, x_trainDbaPkts, y_trainDbaPkts, num_sybils,num_clients=1, initial='client'):
    print("create_pkts sybils num sybils {} and num clients {}".format(num_sybils, num_clients))
    num = num_sybils
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write('\nCreate Pkts DBA Sybils: {}\n'.format(num))
        f.close()
    print('Creating {} Pkts DBA Sybils with Data Shards \n'.format(num))
    
    if num == 1:
        iPlus = 2
    elif num == 5:
        iPlus = 6
    else:
        iPlus = 11

    # create a list of sybil names
    sybil_names = ['{}_{}'.format(initial, i + iPlus) for i in range(num)]

    # shard data and place at each client
    sizePkts = len(x_trainDbaPkts) // num_sybils
    #print("size is ", size, "\n")
    sybil_dict={}
   
    for i in range(num):
        sybil_dict[sybil_names[i]]= [x_trainDbaPkts[i:i + sizePkts], y_trainDbaPkts[i:i + sizePkts]]
        #print("client is ", client_names[i])

    return sybil_dict

def create_dport_sybils(path, attack, defense, log_name, x_trainDbaDport, y_trainDbaDport, num_sybils,num_clients=1, initial='client'):
    print("create_dport sybils num sybils {} and num clients {}".format(num_sybils, num_clients))
    num = num_sybils
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write('\nCreate Dport DBA Sybils: {}\n'.format(num))
        f.close()
    print('Creating {} Dport DBA Sybils with Data Shards \n'.format(num))
    
    if num == 1:
        iPlus = 3
    elif num == 5:
        iPlus = 11
    else:
        iPlus = 21

    # create a list of sybil names
    sybil_names = ['{}_{}'.format(initial, i + iPlus) for i in range(num)]

    # shard data and place at each client
    sizeDport = len(x_trainDbaDport) // num_sybils
    #print("size is ", size, "\n")
    sybil_dict={}
   
    for i in range(num):
        sybil_dict[sybil_names[ i]]= [x_trainDbaDport[i:i + sizeDport], y_trainDbaDport[i:i + sizeDport]]
        #print("client is ", client_names[i])

    return sybil_dict

def create_bytes_sybils(path, attack, defense, log_name, x_trainDbaBytes, y_trainDbaBytes, num_sybils,num_clients=1, initial='client'):
    print("create_bytes sybils num sybils {} and num clients {}".format(num_sybils, num_clients))
    num = num_sybils
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write('\nCreate Bytes DBA Sybils: {}\n'.format(num))
        f.close()
    print('Creating {} Bytes DBA Sybils with Data Shards \n'.format(num))
    
    if num == 1:
        iPlus = 4
    elif num == 5:
        iPlus = 16
    else:
        iPlus = 31

    # create a list of sybil names
    sybil_names = ['{}_{}'.format(initial, i + iPlus) for i in range(num)]

    # shard data and place at each client
    sizeBytes = len(x_trainDbaBytes) // num_sybils
    #print("size is ", size, "\n")
    sybil_dict={}
   
    for i in range(num):
        sybil_dict[sybil_names[i]]= [x_trainDbaBytes[i:i + sizeBytes], y_trainDbaBytes[i:i + sizeBytes]]
        #print("client is ", client_names[i])

    return sybil_dict

''' create_attackers creates a number of 
   args:
        image_list: a list of numpy arrays of training images
        label_list:a list of binarized labels for each image
        attack_dict: dict of chosen attackers from client list
        initials: the clients'name prefix, e.g, clients_1

   return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        
'''
def create_label_flip_sybils(path, attack, defense, log_name, x_train, y_train,num_sybils=1, num_clients=10, initial='clients'):

    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write("\nCreate Label Flip Sybils: {}\n".format(num_sybils))
        f.close()
    print("\nCreating Label Flip Sybils with Data Shards \n")

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_sybils)]
    if num_sybils == 1:
        num = 3
    elif num_sybils == 5:
        num = 5
    elif num_sybils == 10:
        num = 10
    # shard data and place at each client
    size = len(x_train) // num
    #print("size is ", size, "\n")
    client_dict={}
    for i in range(num_sybils):
        client_dict[client_names[i]]= [x_train[i:i + size], y_train[i:i + size]]
    
    for (client_name, data) in client_dict.items():
        data = replace_1_with_0(config.PATH, config.ATTACK, config.NUM_SYBILS, config.DEFENSE, config.LOG_NAME, data[1])
    return client_dict

### for attacking all of a data set
def replace_1_with_0(path, attack, num_sybils, defense, log_name,data):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write("\nFlipping Labels\n")
        f.close()
    print("Flipping Labels")
    #print(data[:])
    for idx in range(len(data)):
        if (data[idx] == [1]).all():
            data[idx] = 0
    #print(data[:])
    return data

def get_model(timesteps,n_features):
    # loading the saved model
    loaded_model = tf.keras.models.load_model('./IDS_Persistent_Model/persistent_model_tf')
    #then call fit
        
    '''
    sgd = gradient_descent_v2.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    
    model = Sequential()
    
    #model.add(LSTM(256, return_sequences=True, input_shape=(timesteps, n_features)))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(.2))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(.25))
    #model.add(LSTM(64))
    #model.add(Dropout(.25))
    
    model.add(Bidirectional(LSTM(29, return_sequences=True), input_shape=(timesteps, n_features)))
    model.add(Dense(29, activation='relu'))
    model.add(Dropout(.2))
    model.add(Bidirectional(LSTM(14, return_sequences=True)))
    model.add(Dense(14, activation='relu'))
    model.add(Dropout(.25))
    model.add(LSTM(7,return_sequences=False))
    model.add(Dropout(.25))
    #model.add(Dense(2, activation='softmax'))
    
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(),'accuracy'])
    model.compile(optimizer=sgd, loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\model_summary.txt','a') as f:
    #        f.write(str(model.summary()))
    #        f.close()
    #print(model.summary())
    '''
    return loaded_model
    #return model


def model_training(model,x_train,y_train,epochs=4000):
    #callbacks = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=10,
    callbacks = EarlyStopping(monitor='binary_accuracy', mode='max', verbose=0, patience=75,
                              restore_best_weights=True)
    checkpoint_filepath = './ids_epoch_models/IDS/best_model.h5'
    mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='binary_accuracy', mode='max', verbose=0, save_best_only=True)
    batch_size = 5
    X_train = x_train.copy()
    Y_train = y_train.copy()
    accuracy_callback = AccuracyCallback((X_train, Y_train))


    #use verbose = 1 or 2 to see epoch progress pbar... each step is examples / batch
    train_history = model.fit(x_train,
                              y_train,
                              epochs=epochs,
                              validation_split=0.2,
                              shuffle=False,
                              #validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=0,
                              callbacks=[callbacks,mc, accuracy_callback]
                              )
    #print("\n\nBest Training Poisoning Accuracy:\n{}".format(max(train_history.history['binary_accuracy'])))
    #with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_IDS_model_'+ config.LOG_NAME,'a') as f:
    #    f.write("\n\nBest Training IDS Accuracy:\n{}".format(max(train_history.history['binary_accuracy'])))
    #f.close()
    model = load_model(checkpoint_filepath)


    return model

def model_evaluate(path, attack, defense, log_name,model,x_train,y_train,x_test,y_test,epochs, num_sybils):
    train_pred = (model.predict(x_train, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,verbose=0) > .5).astype("int32") 
    train_labels = np.copy(y_train).astype("int32")
    test_pred = (model.predict(x_test) > .5).astype("int32") 
    test_labels = np.copy(y_test).astype("int32")
    print("predicted value:\n{}".format(test_pred))
    print("label value:\n{}".format(test_labels))
    trainAcc = accuracy_score(train_labels, train_pred)
    testAcc = accuracy_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred, zero_division=0)
    precision = precision_score(test_labels, test_pred)
    classes_report = classification_report(test_labels, test_pred)
    matrix = confusion_matrix(test_labels, test_pred, labels=[1,0])


    list_data = [epochs, testAcc,f1, precision]
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_ids_model_results.csv' ,'a',newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list_data)
        f_object.close()
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_model_'+ log_name,'a') as f:
            f.write('\n#####################         IDS         ###############################################\n')
            f.write('\n############################################################################################\n')
            f.write('\ncomm_round: {} | global_test_acc: {:.3%} | global_f1: {} | global_precision: {}\n'.format(epochs, testAcc, f1, precision))
            f.write(str(classes_report))
            f.write("\nAccuracy per class:\n{}\n{}\n".format(matrix,matrix.diagonal()/matrix.sum(axis=1)))
    f.close()
    print('\n#####################         IDS         ###############################################\n')
    print('\n############################################################################################\n')
    print('\ncomm_round: {} |global_train_acc: {:.3%}|| global_test_acc: {:.3%} | global_f1: {} | global_test_precision: {}'.format(epochs, trainAcc, testAcc, f1, precision))
    print(classes_report)
    print("\nAccuracy per class:\n{}\n{}\n".format(matrix,(matrix.diagonal()/matrix.sum(axis=1))))

classes = ['normal','attack']
class AccuracyCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_data):
        self.test_data = test_data
        self.class_history = ['normal', 'abnormal']

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_data

        correct = 0
        incorrect = 0

        x_result = self.model.predict(x_data, verbose=0)

        x_numpy = []

        for i in classes:
            self.class_history.append([])

        class_correct = [0] * len(classes)
        class_incorrect = [0] * len(classes)

        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]

            res = x_result[i]

            actual_label = np.argmax(y)
            pred_label = np.argmax(res)

            if(pred_label == actual_label):
                x_numpy.append(["cor:", str(y), str(res), str(pred_label)])     
                class_correct[actual_label] += 1   
                correct += 1
            else:
                x_numpy.append(["inc:", str(y), str(res), str(pred_label)])
                class_incorrect[actual_label] += 1
                incorrect += 1
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_log.txt','a') as f:
                    f.write("\n\tCorrect: %d" %(correct))
                    f.write("\tIncorrect: %d" %(incorrect))
                    f.close()
        #print("\n\tCorrect: %d" %(correct))
        #print("\tIncorrect: %d" %(incorrect))

        for i in range(len(classes)):
            tot = float(class_correct[i] + class_incorrect[i])
            class_acc = -1
            if (tot > 0):
                class_acc = float(class_correct[i]) / tot
            with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_log.txt','a') as f:
                        f.write("\t%s: %.3f" %(classes[i],class_acc))
                        f.close()
            #print("\t%s: %.3f" %(classes[i],class_acc)) 

        acc = float(correct) / float(correct + incorrect)  
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_log.txt','a') as f:
                    f.write("\tCurrent Network Accuracy: %.3f \n" %(acc))
                    f.close()
        #print("\tCurrent Network Accuracy: %.3f" %(acc))

############################ Addtional Similarities
def pardonWV(n_clients, maxsm, sm, prc):
    # pardoningF for sm
    for i in range(n_clients):
        for j in range(config.POISON_FEATURES - 1):
            if i == j:
                continue
            if maxsm[i] < maxsm[j]:
                sm[i][j] = (sm[i][j] * maxsm[i]) / (maxsm[j] * prc)
 
    wv = 1 - (np.max(sm, axis=1))

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    alpha = np.max(sm, axis=1)

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    with np.errstate(divide='ignore'):
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

    return wv,alpha

def std(a):
    return (np.std(a))

def std_set(a_set):
    """computes jaccard for all vectors in a set"""
    std_a_set = np.zeros(a_set.shape)    

    for x in range(0, len(a_set)):
        std_a_set[x] = std(a_set[x])

    return std_a_set

def get_std(path, attack, defense, log_name,grads, num_sybils=1):
    #### could use scipy logit(grads) here?
    n_clients = len(grads)
    #print("grads {}".format(len(grads)))
    #print("Logit Total Client Grads: {}".format(n_clients))
    #    1.  Logit
    std = std_set(grads)
    sm = 2.*(std - np.min(std))/np.ptp(std)-1
    #sm = normalized - np.eye(n_clients)
    prc = 0.05 # adjust value to improve results
    #print("ED Similarity is\n {}".format(sm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nED Similarity is\n {}\n".format(sm))
    #    f.close()
    prc = 1 
    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    prc = 1 
    maxsm = np.max(sm, axis=1)
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
        #print("ED wv is {}".format(wv))
        #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        #    f.write("\n\nED wv is {}\n".format(wv))
        #    f.close()
    return wv,alpha

def norm_dist(a):
    return (norm.pdf(a,loc=np.nanmean(a), scale = 1))

def norm_dist_a_set(a_set):
    """computes jaccard for all vectors in a set"""
    norm_dist_set = np.zeros(a_set.shape)    

    for x in range(0, len(a_set)):
        norm_dist_set[x] = norm_dist(a_set[x])

    return norm_dist_set
### shows the distrubtion of points by desnity 
### gets the 
def get_norm_dist(path, attack, defense, log_name,grads, num_sybils=1):
    #### could use scipy logit(grads) here?
    n_clients = len(grads)
    #print("grads {}".format(len(grads)))
    #print("Logit Total Client Grads: {}".format(n_clients))
    #    1.  Logit
    nd = norm_dist_a_set(grads)
    sm = 2.*(nd - np.min(nd))/np.ptp(nd)-1
    #sm = normalized - np.eye(n_clients)
    prc = 0.05 # adjust value to improve results
    #print("ED Similarity is\n {}".format(sm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nED Similarity is\n {}\n".format(sm))
    #    f.close()
    prc = 1 
    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    prc = 1 
    maxsm = np.max(sm, axis=1)
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
        #print("ED wv is {}".format(wv))
        #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        #    f.write("\n\nED wv is {}\n".format(wv))
        #    f.close()
    return wv,alpha


def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j

def jaccard_a_set(a_set):
    """computes jaccard for all vectors in a set"""
    jaccard_set = np.zeros(a_set.shape)    

    length = len(a_set) - 1
    best = 0.0
    for x in range(length):
        for y in range(length):
            if x == y:
                continue
            temp = jaccard_similarity(a_set[x], a_set[y])
            if temp > best:
                best = temp
            jaccard_set[x] = best 

    return jaccard_set

## values of 0 to 1 with 1 being similar 0 is not so we dont' want to pardon these the same way
def jaccard(path, attack, defense, log_name,grads, num_sybils=1):
    #### could use scipy logit(grads) here?
    n_clients = len(grads)
    #print("grads {}".format(len(grads)))
    #print("Logit Total Client Grads: {}".format(n_clients))
    #    1.  Logit
    sm = jaccard_a_set(grads)
   
    # pardoningF for sm
    wv = np.zeros(n_clients)
    for i in range(n_clients):
        wv[i] = np.sum(sm[i])

 
    wv = 1 - (np.max(sm, axis=1))

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    alpha = np.max(sm, axis=1)

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    with np.errstate(divide='ignore'):
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

    return wv,alpha

def softmax(a_vector):
    """Compute a logit for a vector."""
    denom = (1 + sum(np.exp(a_vector)))
    logit = np.exp(a_vector)/denom
    return logit

def inv_log_a_set(a_set):
    """computes logits for all vectors in a set"""
    softmax_set = np.zeros(a_set.shape)    

    for x in range(0, len(a_set)):
        softmax_set[x] = softmax(a_set[x])

    return softmax_set

## probability of logit 0 to 1 mapped in real numbers neg inf to inf...
## however the inverse of that 0 = negative inf and 1 = pos inf..
## no need to pardon this either?
def get_inv_logit(path, attack, defense, log_name,grads, num_sybils=1):
    #### could use scipy logit(grads) here?
    n_clients = len(grads)
    #print("grads {}".format(len(grads)))
    #print("Logit Total Client Grads: {}".format(n_clients))
    #    1.  Logit
    sm = inv_log_a_set(grads)
    prc = 1 


    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    prc = 1 
    maxsm = np.max(sm, axis=1)
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
        #print("ED wv is {}".format(wv))
        #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        #    f.write("\n\nED wv is {}\n".format(wv))
        #    f.close()
    return wv,alpha


def ed(path, attack, defense, log_name,grads, num_sybils=1):
    n_clients = len(grads)
    #print("ED Total Client Grads: {}".format(n_clients))
    #    1.  Euclidean Normalized
    distance_calc = smp.euclidean_distances(grads)
    normalized = 2.*(distance_calc - np.min(distance_calc))/np.ptp(distance_calc)-1
    sm = normalized - np.eye(n_clients)
    prc = 0.05 # adjust value to improve results
    #print("ED Similarity is\n {}".format(sm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nED Similarity is\n {}\n".format(sm))
    #    f.close()
    prc = 1 
    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
        #print("ED wv is {}".format(wv))
        #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        #    f.write("\n\nED wv is {}\n".format(wv))
        #    f.close()
    return wv,alpha


def manhattan(path, attack, defense, log_name,grads, num_sybils=1):
    n_clients = len(grads)
    #print("Manhattan Total Client Grads: {}".format(n_clients))
    #    2.  Manhattan Normalized
    distance_calc = smp.manhattan_distances(grads)
    normalized = 2.*(distance_calc - np.min(distance_calc))/np.ptp(distance_calc)-1
    sm = normalized - np.eye(n_clients)
    prc = 0.05 # adjust value to improve results
    #print("Manhattan Similarity is\n {}".format(sm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nManhattan Similarity is\n {}\n".format(sm))
    #    f.close()
    prc = 1 
    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
    #print("Manhattan wv is {}".format(wv))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\n\nManhattan wv is {}\n".format(wv))
    #    f.close()
    return wv,alpha

# Takes in grad
# Compute similarity
# Get weightings
def foolsGold(path, attack, defense, log_name,grads, num_sybils=1):
    n_clients = len(grads)
    #print("FoolsGold Total Client Grads: {}".format(n_clients))

    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    #print("CS Similarity is \n {}".format(cs))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nCS Similarity is\n {}\n".format(cs))
    #    f.close()
    maxcs = np.max(cs, axis=1)
    prc = 1
    #print("Maxcs is \n {}".format(maxcs))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxcs is \n {}\n".format(maxcs)) 
    #    f.close()

    wv, alpha = pardonWV(n_clients, maxcs, cs, prc)
    #print("FG wv sum weight % is {}".format(wvWeight))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nFG wv sum weight % is {}\n".format(wvWeight))
    #    f.close()

    return wv, alpha

def ts_ss(v, eps=1e-15, eps2=1e-4):
    # reusable compute
    v_inner = torch.mm(v, v.t())
    vs = v.norm(dim=-1, keepdim=True)
    vs_dot = vs.mm(vs.t())

    # compute triangle(v)
    v_cos = v_inner / vs_dot
    v_cos = v_cos.clamp(-1. + eps2, 1. - eps2)  # clamp to avoid backprop instability
    theta_ = torch.acos(v_cos) + math.radians(10)
    theta_rad = theta_ * math.pi / 180.
    tri = (vs_dot * torch.sin(theta_rad)) / 2.

    # compute sector(v)
    v_norm = (v ** 2).sum(-1, keepdim=True)
    euc_dist = v_norm + v_norm.t() - 2.0 * v_inner
    euc_dist = torch.sqrt(torch.abs(euc_dist) + eps)  # add epsilon to avoid srt(0.)
    magnitude_diff = (vs - vs.t()).abs()
    sec = math.pi * (euc_dist + magnitude_diff) ** 2 * theta_ / 360.

    return tri * sec


# Takes in grad
# Compute similarity
# Get weightings
def asf(path, attack, defense, log_name,grads, num_sybils=1):
    n_clients = len(grads)
    #print("ASF Total Client Grads: {}".format(n_clients))
    
    #    3.  TS-SS Triangle Area Similarity - Sector Area Similarity
    v = torch.tensor(grads)
    print("ASF shape : {}\n {}".format(v))
    # TS-SS normalized
    distance_calc =  ts_ss(v).numpy()
    normalized = 2.*(distance_calc - np.min(distance_calc))/np.ptp(distance_calc)-1
    sm = normalized - np.eye(n_clients)
    #print("ASF Similarity is\n {}".format(sm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nASF Similarity is\n {}\n".format(sm))
    #    f.close()
    prc = 0.05 
    maxsm = np.max(sm, axis=1)
    #print("Maxsm is\n {}".format(maxsm))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\nMaxsm is\n {}".format(maxsm))
    #    f.close()
  
    wv, alpha = pardonWV(n_clients, maxsm, sm, prc)
    #print("ASF wv is {}".format(wv))
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #    f.write("\n\nASF wv is {}\n".format(wv))
    #    f.close()
    return wv,alpha



def make_sim_timesteps(x_data, y_data, num_steps=3):
  X = []
  y = []
  #print("In Time Steps\nAppend num_steps:{} rows\n".format(num_steps))
  #Time steps drops length of arr - num_steps to length of arr - so add num_steps at end to get all
  x_data_plus_steps = np.copy(x_data)
  y_data_plus_steps = np.copy(y_data)
  for x in range(num_steps):
    x_data_plus_steps = np.vstack([x_data_plus_steps, np.array(x_data[x])])
  for j in range(num_steps):
    y_data_plus_steps = np.vstack([y_data_plus_steps,np.array(y_data[j])])

  #print("\nCheck Length of x/y_data: {}|{} vs x/y_data_plus_steps: {}|{}\n:::: Start Loop ::::\n".format(x_data.shape[0],y_data.shape[0],x_data_plus_steps.shape[0],y_data_plus_steps.shape[0]))
  #print("x_data_plus\n{}".format(x_data_plus_steps))
  #print("y_data_plus\n{}".format(y_data_plus_steps))
  #use the length of original array for iterations
  for i in range(x_data.shape[0]):
    #new sliding window index
    end_ix = i + num_steps
    seq_X = x_data_plus_steps[i:end_ix]
    seq_y = y_data_plus_steps[end_ix]
    seq_y = float(seq_y)
    X.append(seq_X)
    y.append(seq_y)
    #print("i:{} | end_ix:{} |\nseq_X:\n{}|\nseq_y:\n{}".format(i,end_ix,seq_X,seq_y))

  #print("Make output arrs:\nLen of X:{}\n".format(len(X)))
  x_array = np.array(X)
  y_array = np.vstack([np.array(i) for i in y])

  #print("Check outputs:\nx_array shape : {}\n{}\n\ny_array shape : {}\n{}\n".format(x_array.shape,x_array,y_array.shape,y_array))
  return x_array, y_array


# Takes in grad
# Compute similarity
# Get weightings
def sim(path, attack, defense, log_name,grads, num_sybils=1):
    #1. Get weighted vectors of ASF, FG, Manhattan, ED, and Logit
    #Get ASF WV
    wv_asf, alpha = asf(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get FG WV
    wv_fg, alpha = foolsGold(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get Manhattan WV
    wv_mn, alpha = manhattan(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get ED WV
    wv_ed, alpha = ed(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get Logits WV
    wv_lg = get_inv_logit(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get Jacard WV
    wv_jc, alpha = jaccard(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get Norm Dist T WV
    wv_nd_T,alpha = get_norm_dist(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)
    #Get std T WV
    wv_std,alpha = get_std(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)


    #Make Train Test Data sets
    poison_timesteps = config.POISON_TIMESTEPS
    if attack == 'label' or attack == 'backdoor':
        if num_sybils == 1:
            y = config.Y_25_CLIENTS_1_SYBIL
        elif num_sybils == 5:
            y = config.Y_25_CLIENTS_5_SYBIL
        else:
            y = config.Y_25_CLIENTS_10_SYBIL
    if attack == 'dba':
        if num_sybils == 1:
            y = config.Y_25_CLIENTS_4_SYBIL
        elif num_sybils == 5:
            y = config.Y_25_CLIENTS_20_SYBIL
        else:
            y = config.Y_25_CLIENTS_40_SYBIL
 
    print("\ny shape {}\n".format(y.shape))
    print(y)
    wv_asf = np.array(wv_asf)
    print("\nwv_asf shape {}\n".format(wv_asf.shape))
    print(wv_asf)
    wv_fg = np.array(wv_fg)
    print("\nwv_fg shape {}\n".format(wv_fg.shape))
    print(wv_fg)
    wv_mn = np.array(wv_mn)
    print("\nwv_mn shape {}\n".format(wv_mn.shape))
    print(wv_mn)
    wv_ed = np.array(wv_ed)
    print("\nwv_ed shape {}\n".format(wv_ed.shape))
    print(wv_ed)

    ### Fix logits
    wv_lg = np.array(wv_lg)
    #print("\nwv_lg shape {}\n".format(wv_lg.shape))
    #print(wv_lg)

    wv_lg = np.delete(wv_lg,0,axis=0)
    #print("\nwv_lg shape {}\n".format(wv_lg.shape))
    #print(wv_lg)

    wv_lg = np.transpose(wv_lg)
    print("\nwv_lg shape {}\n".format(wv_lg.shape))
    print(wv_lg)
        
    wv_jc = np.array(wv_jc)
    print("\nwv_jc shape {}\n".format(wv_jc.shape))
    print(wv_jc)
    wv_nd_T = np.array(wv_nd_T)
    print("\nwv_nd_T shape {}\n".format(wv_nd_T.shape))
    print(wv_nd_T)
    wv_std = np.array(wv_std)
    print("\nwv_std shape {}\n".format(wv_std.shape))
    print(wv_std)

    x = np.column_stack((wv_asf,wv_fg))
    xmn = np.column_stack((x,wv_mn))
    xed = np.column_stack((xmn,wv_ed))    
    xlg = np.column_stack((xed,wv_lg))
    xjc = np.column_stack((xlg,wv_jc))
    xndT = np.column_stack((xjc,wv_nd_T))
    xstd = np.column_stack((xndT,wv_std))
    xy = np.column_stack((xstd,y))

    print("\nxy shape: {}\n{}".format(xy.shape,xy))
    rows, cols = xy.shape
    xy = np.nan_to_num(xy, nan=np.nanmean(xy))   
    for i in range(rows):
        asf_val = xy[i][0]
        fg_val = xy[i][1]
        mn_val = xy[i][2]
        ed_val = xy[i][3]
        lg_val = xy[i][4]
        jc_val = xy[i][5]
        ndT_val = xy[i][6]
        std_val = xy[i][7]
        y_val = xy[i][8]
        list_data = [asf_val,fg_val,mn_val,ed_val,lg_val,jc_val,ndT_val,std_val,y_val]
        with open(config.PATH +'poison_training.csv' ,'a',newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_data)
            f_object.close()

    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_poison_model_'+ log_name,'a') as f:
        f.write("\ny shape {}\n".format(y.shape))
        f.write(str(y))
        f.write("\nwv_asf shape {}\n".format(wv_asf.shape))
        f.write(str(wv_asf))
        f.write("\nwv_fg shape {}\n".format(wv_fg.shape))
        f.write(str(wv_fg))
        f.write("\nwv_mn shape {}\n".format(wv_mn.shape))
        f.write(str(wv_mn))
        f.write("\nwv_ed shape {}\n".format(wv_ed.shape))
        f.write(str(wv_ed))
        f.write("\nwv_lg shape {}\n".format(wv_lg.shape))
        f.write(str(wv_lg))
        f.write("\nwv_jc shape {}\n".format(wv_jc.shape))
        f.write(str(wv_jc))
        f.write("\nwv_ndT shape {}\n".format(wv_nd_T.shape))
        f.write(str(wv_nd_T))
        f.write("\nwv_std shape {}\n".format(wv_std.shape))
        f.write(str(wv_std))
        f.write("\nxy shape: {}\n{}".format(xy.shape,xy))
    f.close()

    if attack == 'label' or attack == 'backdoor':
        if num_sybils == 1:
            train, test = train_test_split(xy, test_size = 8, train_size= 18)
        elif num_sybils == 5:
            train, test = train_test_split(xy, test_size = 10, train_size= 20)
        else:
            train, test = train_test_split(xy, test_size = 12, train_size= 23)
    if attack == 'dba':
        if num_sybils == 1:
            train, test = train_test_split(xy, test_size = 9, train_size= 20)
        elif num_sybils == 5:
            train, test = train_test_split(xy, test_size = 15, train_size= 30)
        else:
            train, test = train_test_split(xy, test_size = 20, train_size= 45)

    #print("train shape after tts {}".format(train.shape))
    #print(train)
    #print("test shape after tts {}".format(test.shape))
    #print(test)

    ##### to categorical if you want to use softmax
    #CLASSES = ['1.0', '0.0']
    # this is one hot encoding for binary classificaiton
    #y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
    #y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        
    #y_train = np.asarray(y_train)
    #y_test = np.asarray(y_test)

    #REMOVE LABEL FROM X remove features from Y
    #TODO ##################################################################
    #######################################################################
    # do y's one time for each feature......................................((((((()))))))
    x_train = np.delete(train,config.POISON_FEATURES,1)
    y_train_asf_rm = np.delete(train,0,1)
    y_train_fg_rm = np.delete(y_train_asf_rm,0,1)
    y_train_mn_rm = np.delete(y_train_fg_rm,0,1)
    y_train_ed_rm = np.delete(y_train_mn_rm,0,1)
    y_train_lg_rm = np.delete(y_train_ed_rm,0,1)
    y_train_jc_rm = np.delete(y_train_lg_rm,0,1)
    y_train_ndT_rm = np.delete(y_train_jc_rm,0,1)
    y_train = np.delete(y_train_ndT_rm,0,1)
    x_test = np.delete(test,config.POISON_FEATURES,1)
    y_test_asf_rm = np.delete(test,0,1)
    y_test_fg_rm = np.delete(y_test_asf_rm,0,1)
    y_test_mn_rm = np.delete(y_test_fg_rm,0,1)
    y_test_ed_rm = np.delete(y_test_mn_rm,0,1)
    y_test_lg_rm = np.delete(y_test_ed_rm,0,1)
    y_test_jc_rm = np.delete(y_test_lg_rm,0,1)
    y_test_ndT_rm = np.delete(y_test_jc_rm,0,1)
    y_test = np.delete(y_test_ndT_rm,0,1)
    #print("x_train shape after deletes {}".format(x_train.shape))
    #print(x_train)
    #print("x_test shape after deletes {}".format(x_test.shape))
    #print(x_test)
    #print("y_train shape after deletes {}".format(y_train.shape))
    #print(y_train)
    #print("y_test shape after deletes {}".format(y_test.shape))
    #print(y_test)

    # Use make timesteps for LSTM timesteps.
    #print("sending x_train and y_train to make times steps {}".format(poison_timesteps))
    x_train,y_train= make_sim_timesteps(np.array(x_train),np.array(y_train),poison_timesteps)
    #print("sending x_test and y_test to make times steps {}".format(poison_timesteps))
    x_test, y_test = make_sim_timesteps(np.array(x_test), np.array(y_test), poison_timesteps)
    #print("x_train shape after time steps {}\n".format(x_train.shape))
    #print(x_train)
    #print("y_train shape after time steps {}\n".format(y_train.shape))
    #print(y_train)
    #print("x_test shape after time steps {}\n".format(x_test.shape))
    #print(x_test)
    #print("y_test shape after time steps {}\n".format(y_test.shape))
    #print(y_test)

    #Reshape for LSTM model to take as tensors
    #x_train = x_train.reshape(x_train.shape[0], poison_timesteps, config.POISON_FEATURES)
    #x_test = x_test.reshape(x_test.shape[0], poison_timesteps, config.POISON_FEATURES)   
    #assert x_train.shape[0] == y_train.shape[0]
    #assert x_test.shape[0] == y_test.shape[0]    

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    #print("x_train shape after reshape {}".format(x_train.shape))
    #print("x_test shape  after reshape {}".format(x_test.shape))
    full_set = np.append(x_train,x_test,axis=0)
    #print("full_set shape {}".format(full_set.shape))
    # wv is the weight
    return x_train, x_test, y_train, y_test, full_set

# client_grads = Compute gradients from all the clients
def aggregate_gradients(path, attack, defense, log_name, client_grads, num_sybils=1):
    num_clients = len(client_grads)
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
        f.write("\naggregate_gradients Total Client Grads: {}\n".format(num_clients))
        f.close()
    print("Aggregating Gradients for Total of Clients: {}".format(num_clients))
    
    grad_len = np.array(client_grads[0][-2].data.shape).prod()

    grads = np.zeros((num_clients, grad_len))
    for i in range(len(client_grads)):
        grads[i] = np.reshape(client_grads[i][-2].data, (grad_len))

    if defense == 'sim':
        x_train, x_test, y_train, y_test, full_set  = sim(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,grads, config.NUM_SYBILS)

    return x_train, x_test, y_train, y_test, full_set

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


def sum_scaled_weights(path, attack, defense, log_name,scaled_weight_list, poison_factor,num_sybils=1):          
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    #scale = poison_factor.numpy()
    #print("poison_factor shape {}".format(poison_factor.shape))
    print("scaled_weight_list: Rows {} cols {}".format(len(scaled_weight_list),len(scaled_weight_list[0])))
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_poison_model_'+ config.LOG_NAME,'a') as f:
            f.write("scaled_weight_list: Rows {} cols {}".format(len(scaled_weight_list),len(scaled_weight_list[0])))
    f.close()
    honest_clients = []
    for c, client_grad in enumerate(scaled_weight_list):
        print("c is {} and poison[c] is : {}".format(c, poison_factor[c]))
        if poison_factor[c] == 1:
            print("Adding node: {} value: {} to honest_clients".format(c,poison_factor[c]))
            with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_poison_model_'+ config.LOG_NAME,'a') as f:
                    f.write("Adding node: {} value: {} to honest_clients".format(c,poison_factor[c]))
            f.close()
            honest_clients.append(client_grad)

    ##################################
    ##################################
    ##################################
    #### uncomment when you find a way to ensure one honest client
    #print("After Nodes removed: Rows {} cols {}".format(len(honest_clients),len(honest_clients[0])))
    #with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_poison_model_'+ config.LOG_NAME,'a') as f:
    #        f.write("After Nodes removed: Rows {} cols {}".format(len(honest_clients),len(honest_clients[0])))
    #f.close()


    avg_grad = []
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*honest_clients):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def baseline_sum_scaled_weights_ids(path, attack, defense, log_name,scaled_weight_list,num_sybils):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    print("scaled_weight_list: Rows {} cols {}".format(len(scaled_weight_list),len(scaled_weight_list[0])))
    avg_grad = []
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
    #for grad_list_tuple in zip(*poison_grads):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad
   
