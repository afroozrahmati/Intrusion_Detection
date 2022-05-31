'''
    Program Name:  Poisoning defense for fl 
    Description: This program simulates an anomaly-based IDS using FL for IoT 
    with poisoning defense.  The poisoning defense is built upon 
    cosine-similarity...

    This program simulates the centralized server shipping the model to the 
    client, the client training a local model and uploading it back for the 
    server to aggregate it.
    
    The program begins by reading in two csv files that contain normal and
    abnormal (attack) packets.  Each file has xxxx rows of packets identified
    by the their sequence id, with xxx columns of features to include a
    binary attack or benign column.

    The program uses a utility file to load the data, pre-process the data
    for LSTM and return typical x/y_train and x/y_test arrays.  Next, the 
    program creates 10 clients.  The current implementation does not
    utilize non-IID.  TODO: implement it.

    Next a centralized or global variables are created such as the global
    model weights, LSTM timesteps, number of features, and the number of 
    communcation rounds to test the system.

    This simulation is achieved in 2 for loops with an implicit 3rd loop.  
    The outer loop, comms_round [i], gets the global weight of the global
    model and creates a scaled local weight list.

    The inner loop or client loop, creates a local model and sets the 
    weights according to the global weights.  It then proceeds to train
    the local model on the local clients data.

    This is where the implicit 3rd loop which is the local epochs conducts
    the learning to create a local weight in one epoch.  The new weigths are
    scaled and appended to the local weight list.

    Next the program moves back to the outer loop where it summs the scaled
    local weights.  Here is where the poisoning defense takes place as the 
    aggregated weights are sent to be tested for cosine-similarity.

    The cosine similarity is tested in the fools gold method where
    xyz happens.

    After determining which client wieghts to keep and pardon 
    the program updates the global model.  That completes 1 full global
    training epoch.

    The simulation runs 100 global training loops, there are 10 clients that run 
    1 epoch each.   
'''

###############################################################################
###############################################################################
#                          API Imports                                        #
###############################################################################
###############################################################################

import tensorflow as tf
from tensorflow.keras import backend as K
import time
from ..fl_ss_utils import *
import poison_config
from ..fl_ss_sim import *
from data_processing_poison import *

###############################################################################
###############################################################################
#                          Time Utility                                       #
###############################################################################
###############################################################################
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# 1. 
###############################################################################
###############################################################################
#                            Main                                             #
###############################################################################
###############################################################################

def main():
    comm_round = poison_config.COMMS_ROUND
    data_process= data_processing_poison()
    xp_train, xp_test, yp_train, yp_test = data_process.load_data(poison_config.PATH_OUT,poison_config.POISON_CSV,timesteps=80):
    global_sim_model = get_sim_model(poison_config.POISON_TIMESTEPS,poison_config.POISON_FEATURES)
    global_sim_model = model_sim_training(global_sim_model, xp_train, yp_train, xp_test,yp_test,config.POISON_EPOCHS)
    model_sim_evaluate(poison_config.PATH_OUT, poison_config.NO_ATTACK, poison_config.NO_DEFENSE, poison_config.LOG_NAME,global_sim_model,xp_train,yp_train,xp_test,yp_test,comm_round, poison_config.NUM_SYBILS)
    global_sim_model.save('./POISON_Persistent_Model/persistent_model_tf',save_format='tf')           
if __name__ == "__main__":
    main()
