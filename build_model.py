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
from datetime import datetime
import os, fnmatch
import pickle
from plots import produce_plot
from ClusteringLayer import *
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_processed_data(file_path_normal,file_path_abnormal,timesteps):
    data_process= data_processing()
    x_train,y_train,x_test,y_test = data_process.load_data(file_path_normal,file_path_abnormal,timesteps)

    # print("train shape: ", np.shape(x_train))
    # print("test shape: ", np.shape(x_test))
    # print("train label shape: ", y_train.shape)
    # print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    return x_train,y_train,x_test, y_test




def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# def get_model(timesteps,n_features):
#
#
#     gamma = 1
#     # tf.keras.backend.clear_session()
#     # print('Setting Up Model for training')
#     # print(gamma)
#
#     inputs = Input(shape=(timesteps, n_features))
#     encoder = LSTM(32, activation='tanh')(inputs)
#     encoder = Dropout(0.2)(encoder)
#     encoder = Dense(64, activation='relu')(encoder)
#     encoder = Dropout(0.2)(encoder)
#     encoder = Dense(100, activation='relu')(encoder)
#     encoder = Dropout(0.2)(encoder)
#     encoder_out = Dense(100, activation=None, name='encoder_out')(encoder)
#     clustering = ClusteringLayer(n_clusters=2, name='clustering', alpha=0.05)(encoder_out)
#     hidden = RepeatVector(timesteps, name='Hidden')(encoder_out)
#     decoder = Dense(100, activation='relu')(hidden)
#     decoder = Dense(64, activation='relu')(decoder)
#     decoder = LSTM(32, activation='tanh', return_sequences=True)(decoder)
#     output = TimeDistributed(Dense(n_features), name='decoder_out')(decoder)
#     encoder_model = Model(inputs=inputs, outputs=encoder_out)
#     # kmeans.fit(encoder_model.predict(x_train))
#
#     model = Model(inputs=inputs, outputs=[clustering, output])
#
#     clustering_model = Model(inputs=inputs, outputs=clustering)
#
#     # plot_model(model, show_shapes=True)
#     #model.summary()
#     #optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
#     optimizer = Adam(0.0001, beta_1=0.1, beta_2=0.001, amsgrad=True)
#     model.compile(loss={'clustering':  'kld', 'decoder_out': 'mse'},
#                   loss_weights=[gamma, 1], optimizer=optimizer,
#                   metrics={'clustering': 'accuracy', 'decoder_out': 'mse','Sensitivity':tf.keras.metrics.SensitivityAtSpecificity()})
#
#     # print('Model compiled.           ')
#     return model

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

file_path_normal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]

for timesteps in [1]:

    x_train, y_train, x_test, y_test = load_processed_data(file_path_normal, file_path_abnormal,timesteps)  # args.partition)

    print("x_train =",np.shape(x_train))
    print("y_train = ",np.shape(y_train))

    print("x_test = ",np.shape(x_test))
    print("y_test = ",np.shape(y_test))

    optimizer = Adam(0.0001, beta_1=0.1, beta_2=0.001, amsgrad=True)
    #optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

    n_classes = 2
    batch_size = 64
    epochs = 1000

    #gamma =4
    # callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max',
    #                               verbose=2, patience=800, restore_best_weights=True)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    model_dir = './model/'
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]
    print((timesteps, n_features))

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)




    # process and batch the test set
    #test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]

    for gamma in  [1]:

        os.chdir('D:\\UW\\RA\\Intrusion_Detection')
        now = datetime.now() # current date and time
        now =now.strftime("%m")+'_'+now.strftime("%d")+'_'+now.strftime("%Y")+'_'+now.strftime("%H")+'_'+now.strftime("%M")+'_'+now.strftime("%S")

        tf.keras.backend.clear_session()
        print('Setting Up Model for training')
        print(gamma)
        model_name = now +'_'+ 'Gamma('+str(gamma) +')-Optim('+"Adam"+')'+'_'+str(epochs)
        print(model_name)

        model = 0

        inputs=encoder=decoder=hidden=clustering=output=0

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

        #plot_model(model, show_shapes=True)
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
                      loss_weights=[gamma, 1], optimizer=optimizer,
                      metrics={'clustering': 'accuracy', 'decoder_out': 'mse'
                               })
        tf.keras.utils.plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

        print('Model compiled.')
        print('Training Starting:')
        train_history = model.fit(x_train,
                                  y={'clustering': y_train, 'decoder_out': x_train},
                                  epochs=epochs,
                                  validation_split=0.2,
                                  # validation_data=(x_test, (y_test, x_test)),
                                  batch_size=batch_size,
                                  verbose=2,
                                  callbacks=tensorboard_callback)





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

        y_test_one = np.argmax(y_test, axis=1)
        print("y_test = ", np.shape(y_test_one))
        print("y_pred_test = ", np.shape(y_pred_test))
        cm = confusion_matrix(y_test_one,y_pred_test)
        print("Confusin matrix:", cm)

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

        result = "Gamma="+str(gamma)+', timesteps='+str(timesteps)+', Epochs='+str(epochs)+ ', Lr='+'0.0001, '+'NMI='+str(nmi) +', ARI='+str(ari) +', Train accuracy='+str(acc) + ', Test accuracy='+ str(testAcc)+"\n"
        with open('result.txt', 'a') as f:
            f.write(result)

        os.chdir('D:\\UW\\RA\\Intrusion_Detection')
        saved_format = {
            'history': train_history.history,
            'gamma': gamma,
            'lr': K.eval(model.optimizer.lr),
            'batch': batch_size,
            'accuracy': acc,
            'nmi': nmi,
            'ari': ari,
            'nmi_test': nmi_test,
            'ari_test': ari_test,
            'test_accuracy': testAcc,
        }

        os.chdir(model_dir)
        pklName = model_name + '.pkl'
        # saved_format = [train_history.history, gamma, K.eval(model.optimizer.lr), batch_size]
        # with open(pklName, 'wb') as out_file:
        #     pickle.dump(train_history.history, out_file, pickle.HIGHEST_PROTOCOL)
        with open(pklName, 'wb') as out_file:
            pickle.dump(saved_format, out_file, pickle.HIGHEST_PROTOCOL)

        print('Saving model.')
        save_name = './model/' + model_name
        model.save(save_name)

        os.chdir('D:\\UW\\RA\\Intrusion_Detection')
        produce_plot(model_name, train_history.history, gamma, testAcc)