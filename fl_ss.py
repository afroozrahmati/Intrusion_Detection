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

import tensorflow as tf
from tensorflow.keras import backend as K
from fl_ss_utils import *

###############################################################################
###############################################################################
#                            Main                                             #
###############################################################################
###############################################################################

# 1. Import Dataset.  Data is split into 2 files, benign normal traffic
# and attacks / abnormal traffic.
print("Starting Machine:\nLoading Data...")
file_path_normal = 'C:\\Users\\ChristianDunham\\Source\\Repos\\Intrusion_Detection\\data\\normal_ft.csv'  # sys.argv[1] #    #+ sys.argv[0]
file_path_abnormal = 'C:\\Users\\ChristianDunham\\Source\\Repos\\Intrusion_Detection\\data\\abnormal_ft.csv'  # sys.argv[2] #  #+ sys.argv[1]

# 2. Split the data into two sets, Train and Test.  Each set split features from labels
# Function takes in both normal and abnormal packet files
x_train, y_train, x_test, y_test = load_processed_data(file_path_normal, file_path_abnormal)  # args.partition)

# 3. ??? Why do WE have to do this?
x_train = np.asarray(x_train)
x_test = np.nan_to_num(x_test)
x_test = np.asarray(x_test)

# 4. create clients
clients = create_clients(x_train, y_train, num_clients=10, initial='client')

# process and batch the training data for each client
client_names=[]
for (client_name, data) in clients.items():
    client_names.append(client_name)

timesteps = np.shape(x_train)[1]
n_features = np.shape(x_train)[2]

global_model = get_model(timesteps, n_features)
comms_round = 100

# commence global training loop
for comm_round in range(comms_round):
    print("Simulate Models Sent to Clients\n")
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    
    client_grads = []
    
    # loop through each client and create new local model
    for (client_name, data) in clients.items():
        print('{} training'.format(client_name))
        local_model = get_model(timesteps, n_features)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        local_model = model_training(local_model, data[0], data[1],epochs=10)
        

        # scale the model weights and add to list
        scaled_local_weight_list.append(local_model.get_weights())
        client_grads.append(local_model.get_weights())

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = aggregate_gradients(client_grads)

    # update global model
    global_model.set_weights(average_weights)
    
    # test global model and print out metrics after each communications round
    model_evaluate(global_model,x_train,y_train,x_test,y_test,comm_round)

'''
    References:
        1) https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
'''