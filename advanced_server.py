from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import os

import flwr as fl
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score,mean_squared_error,mutual_info_score
from data_processing import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def get_model(timesteps , n_features ):
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
    model.compile(loss={'clustering': 'kld', 'decoder_out': 'mse'},
                  loss_weights=[gamma, 1], optimizer='adam',
                  metrics={'clustering': 'accuracy', 'decoder_out': 'mse'})

    print('Model compiled.           ')
    return model

def load_processed_data(file_path):
    data_process= data_processing()
    x_train,y_train,x_test,y_test = data_process.load_data_total(file_path)

    print("train shape: ", np.shape(x_train))
    print("test shape: ", np.shape(x_test))
    print("train label shape: ", y_train.shape)
    print("test label shape: ", y_test.shape)

    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)
    return x_train,y_train,x_test, y_test

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model= get_model(1,23)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("192.168.1.237:8080", config={"num_rounds": 100}, strategy=strategy)




def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    file_path = 'D:\\UW\\RA\\Intrusion_Detection\\data\\df_shuffled.csv'  # sys.argv[1] #    #+ sys.argv[0]
    #file_path_abnormal = 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal.csv'  # sys.argv[2] #  #+ sys.argv[1]
    x_train, y_train, x_test, y_test = load_processed_data(file_path)  # args.partition)
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[15051:16723], y_train[15051:16723]      #x_test, y_test

    # The `evaluate` function will be called after every round
    def evaluate(
            weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters

        q_t, _ = model.predict(x_val, verbose=0)
        y_pred_test = np.argmax(q_t, axis=1)
        y_arg_test = np.argmax(y_val, axis=1)
        accuracy = np.round(accuracy_score(y_arg_test, y_pred_test), 5)
        # mse_loss = np.round(mean_squared_error(y_arg_test, y_pred_test), 5)
        kld_loss = np.round(mutual_info_score(y_arg_test, y_pred_test), 5)
        #print(len(x_test))
        print("kld_loss=",kld_loss,"accuracy=",accuracy)

        return kld_loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()