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



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_sim_model(timesteps,n_features):
    print("sim get model")
    model = Sequential()
    model.add(LSTM(20, return_sequences=False, activation='tanh',input_shape=(timesteps, n_features)))
    model.add(Dense(20, activation='relu'))
    #model.add(Dropout(.25))
    #model.add(LSTM(16))
    model.add(Dense(units=1, activation='linear'))
    #model.add(Dropout(.25))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(),'accuracy'])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\model_summary.txt','a') as f:
    #        f.write(str(model.summary()))
    #        f.close()
    #print(model.summary())

    return model

def model_sim_training(model,x_train,y_train,x_test,y_test,epochs=1):
    print("sim training")
    callbacks = EarlyStopping(monitor='binary_accuracy', mode='max', verbose=0, patience=1000,
                              restore_best_weights=True)
    # mc = ModelCheckpoint('best_model.h5', monitor='binary_accuracy', mode='max', verbose=0, save_best_only=True)
    batch_size = 1
    X_train = x_train.copy()
    Y_train = y_train.copy()
    accuracy_callback = AccuracyCallback((X_train, Y_train))
    print("under accuracy")

    #use verbose = 1 or 2 to see epoch progress pbar... each step is examples / batch
    train_history = model.fit(x_train,
                              y_train,
                              epochs=epochs,
                              validation_split=.3,
                              #validation_data=(x_test, (y_test, x_test)),
                              #validation_data=(x_test, y_test),
                              batch_size=batch_size,
                              verbose=0,
                              callbacks=[callbacks, accuracy_callback]
                              )
    #saved_model = load_model('best_model.h5')
    print("after fitting")

    return model

def model_sim_evaluate(path, attack, defense, log_name,model,x_train,y_train,x_test,y_test,epochs, num_sybils):
    print("Sim Evaluate")
    model.compile(run_eagerly=True)
    q = model.predict(x_train, verbose=0)
    #q_t = model.predict(x_test, verbose=0)

    #convert one-hot to index
    y_pred = np.argmax(q, axis=1)
    y_arg = np.argmax(y_train, axis=1)
    #y_pred_test = np.argmax(q_t, axis=1)
    #y_arg_test = np.argmax(y_test, axis=1)

    #m = tf.keras.metrics.binary_accuracy(y_arg_test, y_pred_test, threshold=0.5)
    trainAcc = np.round(accuracy_score(y_arg, y_pred), 5)
    #testAcc = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
    #f1 = f1_score(y_arg_test, y_pred_test)
    #precision = precision_score(y_arg_test, y_pred_test)
    #m_float = float(m)
    #list_data = [epochs, testAcc, f1, precision, m_float]
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ 'results.csv','a',newline='') as f_object:
    #    writer_object = writer(f_object)
    #    writer_object.writerow(list_data)
    #    f_object.close()
    #with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
    #        f.write('\n##################### POISON             ###############################################\n')
    #        f.write('\n############################################################################################\n')
    #        f.write('\ncomm_round: {} | global_acc: {:.3%} | global_f1: {} | global_precision: {} | global bin {}\n'.format(epochs, testAcc, f1, precision, m))
    #f.close()
    #print('\ncomm_round: {} |global_train_acc: {:.3%}|| global_test_acc: {:.3%} | global_test_f1: {} | global_test_precision: {} | global test bin acc {}'.format(epochs, trainAcc, testAcc, f1, precision, m))
    print('\ncomm_round: {} |global_train_acc: {:.3%}||'.format(epochs, trainAcc))


classes = ['1.0','0.0']
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
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
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
            with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
                        f.write("\t%s: %.3f" %(classes[i],class_acc))
                        f.close()
            #print("\t%s: %.3f" %(classes[i],class_acc)) 

        acc = float(correct) / float(correct + incorrect)  
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
                    f.write("\tCurrent Network Accuracy: %.3f \n" %(acc))
                    f.close()
        #print("\tCurrent Network Accuracy: %.3f" %(acc))