
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from data_processing import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.cluster import adjusted_rand_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from ClusteringLayer import *
from datetime import datetime

def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def get_model(x_train,y_train,x_test,y_test):
    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)

    # now = datetime.now()  # current date and time
    # now = now.strftime("%m") + '_' + now.strftime("%d") + '_' + now.strftime("%Y") + '_' + now.strftime(
    #     "%H") + '_' + now.strftime("%M") + '_' + now.strftime("%S")
    # now
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]
    gamma = 1
    # tf.keras.backend.clear_session()
    print('Setting Up Model for training')
    print(gamma)
    # model_name = now + '_' + 'Gamma(' + str(gamma) + ')-Optim(' + "Adam" + ')'
    # print(model_name)

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

    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    model.compile(loss={'clustering':           'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer=optimizer,
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    print('Model compiled.           ')
    print('Training Starting:')

    print("train shape: ", np.shape(x_train))
    print("train label shape: ", y_train.shape)

    callbacks = EarlyStopping(monitor='val_clustering_accuracy', mode='max', verbose=2, patience=800,
                              restore_best_weights=True)
    n_classes = 2
    batch_size = 64
    epochs = 100

    train_history = model.fit(x_train,
                              y={'clustering': y_train, 'decoder_out': x_train},
                              epochs=epochs,
                              validation_split=0.2,
                              # validation_data=(x_test, (y_test, x_test)),
                              batch_size=batch_size,
                              verbose=2,
                              callbacks=callbacks)
    return model,train_history


# def model_training(model,epochs,batch_size,x_train,y_train):
#
#     return train_history

def model_evaluate(model,x_train,y_train,x_test,y_test):
    q, k = model.predict(x_train, verbose=0)
    print("secnd k x",k)
    q_t, h = model.predict(x_test, verbose=0)
    print("second h test",h)
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



file_path_normal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]
data_processing= data_processing()
x_train,y_train,x_test,y_test = data_processing.load_data(file_path_normal,file_path_abnormal)

print("train shape: ", np.shape(x_train))
print("test shape: ", np.shape(x_test))
print("train label shape: ", y_train.shape)
print("test label shape: ", y_test.shape)

model,train_history= get_model(x_train,y_train,x_test,y_test)

model_evaluate(model,x_train,y_train,x_test,y_test)
# Use the last 2k training examples as a validation set
start = len(x_train) - 2000
x_val, y_val = x_train[start:len(x_train)], y_train[start:len(x_train)]
loss, accuracy = model.evaluate(x_val, y_val)
print("loss",loss,"acc= ", accuracy)
