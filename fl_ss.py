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
from fl_ss_utils import *
import config

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

###############################################################################
###############################################################################
#                          Simulation                                         #
###############################################################################
###############################################################################

def simulation(path, path_in, log_name, table_name, comms_round, attack='label', defense='fg', num_sybils=1, num_clients=10):
    print("Test Variable Input:\nattack: {}\ndefense: {}\nnum_sybils: {}\nnum_clients: {}".format(attack,defense,num_sybils,num_clients))
    print("Test Variable Input:\npath: {}\nlog_name: {}\ntable_name: {}\ncomms_rounds: {}".format(path,log_name,table_name,comms_round))
    ''' 
    1. Import Dataset.  Data is split into 2 files, benign normal traffic
     and attacks / abnormal traffic.
    '''
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("Starting Machine:\nLoading Data...\n")
            f.close()
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Starting Machine:\nLoading Data...")
    file_path_normal = path_in + 'normal_ft.csv'  
    file_path_abnormal = path_in + 'abnormal_ft.csv' 
    
    '''
    2. Split the data into two sets, Train and Test.  Each set split features from labels
    Function takes in both normal and abnormal packet files
    '''
    x_train,y_train,x_test,y_test,x_trainP,y_trainP,x_testP,y_testP, x_trainDbaProto, x_testDbaProto, y_trainDbaProto, y_testDbaProto , x_trainDbaPkts, x_testDbaPkts, y_trainDbaPkts, y_testDbaPkts, x_trainDbaDport, x_testDbaDport, y_trainDbaDport, y_testDbaDport, x_trainDbaBytes, x_testDbaBytes, y_trainDbaBytes, y_testDbaBytes = load_processed_data(file_path_normal, file_path_abnormal,config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, config.NUM_SYBILS)  # args.partition)

    # 3.a client sets to arrays
    x_train = np.asarray(x_train)
    x_test = np.nan_to_num(x_test)
    x_test = np.asarray(x_test)

    # 3.b backdoor sets to arrays
    if attack == 'backdoor':
        x_trainP = np.asarray(x_trainP)
        x_testP = np.nan_to_num(x_testP)
        x_testP = np.asarray(x_testP)

    # 3.c dba sets to arrays
    if attack == 'dba':
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


    # 4.a create clients
    clients = create_clients(config.PATH, config.ATTACK, config.NUM_SYBILS, config.DEFENSE, config.LOG_NAME, x_train, y_train, config.NUM_CLIENTS, initial='client')   
    n_clts = len(clients)
    print("Number of total Clients {}".format(n_clts))

    print("Number of Sybils Passed in to simulation {}".format(num_sybils))
    # 4.b create label flip sybils
    if attack == 'label':
        num_clients = num_sybils
        sybils = create_label_flip_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME, x_train, y_train, config.NUM_SYBILS,config.NUM_CLIENTS, initial='client')
        clients.update(sybils)

    # 4.c create backdoor sybils
    if attack == 'backdoor':
        sybils = create_backdoor_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,x_trainP, y_trainP, config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
        clients.update(sybils)

    # 4.d create dba sybils
    if attack == 'dba':
        sybils = create_dba_sybils(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,x_trainDbaProto, y_trainDbaProto,x_trainDbaPkts, y_trainDbaPkts,x_trainDbaDport,y_trainDbaDport, x_trainDbaBytes, y_trainDbaBytes,config.NUM_SYBILS, config.NUM_CLIENTS, initial='client')
        clients.update(sybils)


    # 5. process and batch the training data for each client
    client_names=[]
    for (client_name, data) in clients.items():
        client_names.append(client_name)
    
    # 6. collect timesteps and features for global model
    timesteps = np.shape(x_train)[1]
    n_features = np.shape(x_train)[2]
    n_clients = len(client_names)
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
                f.write("\nNumber of total clients and sybils {}\n".format(n_clients))
                f.close()
    print("Number of total clients and sybils {}\n".format(n_clients))
    global_model = get_model(timesteps, n_features)
    
    # 7.
    ###############################################################################
    ###############################################################################
    #              Begin Global Training Loop                                     #
    ###############################################################################
    ###############################################################################
    for comm_round in range(config.COMMS_ROUND):
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
                f.write("\nSimulate Models Sent to Clients\n")
                f.close()
        print("\nSimulate Models Sent to Clients\n")

        # 1. get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # 2. initial lists to collect local model weights after before and scaling
        client_grads_scaled = []
        client_grads_unscaled = []

        # 4. 
        ###############################################################################
        ###############################################################################
        #              Loop through each client and conduct training                  #
        ###############################################################################
        ###############################################################################
        for (client_name, data) in clients.items():
            # 1. create initial local model for client x and set initial weights
            local_model = get_model(timesteps, n_features)
            local_model.set_weights(global_weights)

            # 2. Train local model
            local_tic = time.perf_counter()             
            local_model = model_training(local_model, data[0], data[1],epochs=1)
            local_toc = time.perf_counter() 
            with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
                f.write("\nTotal time for {} : local training was {}".format(client_name, local_toc-local_tic))
            print("Total time for {} : local training was {}".format(client_name, local_toc-local_tic))      
            
            # 3. simulate scaling (would happen globally IRW) and add to list
            scaling_factor = weight_scalling_factor(clients, client_name)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            client_grads_scaled.append(scaled_weights)
        
            # 4. get unscaled weights for defense
            client_grads_unscaled.append(local_model.get_weights())

            # clear session to free memory after each communication round
            K.clear_session()
        
        # 5. Get the poison scaling factor from defense and aggregate the local batch weights 
        num_grads_scaled = len(client_grads_scaled)
        num_grads_unscaled = len(client_grads_unscaled)
        # to get the average over all the local model, we simply take the sum of the scaled weights
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
             f.write("\n\nTotal Client Grads Scaled : {} unscaled {}\n".format(num_grads_scaled, num_grads_unscaled))
             f.close()
        print("\nTotal Client Grads Scaled : {} unscaled {}\n".format(num_grads_scaled, num_grads_unscaled))
        baseline = config.BASELINE
        if baseline == True:
            average_weights = baseline_sum_scaled_weights_ids(config.PATH, config.ATTACK, config.NO_DEFENSE, config.LOG_NAME,client_grads_scaled, config.NUM_SYBILS)
            # 6. update global model
            global_model.set_weights(average_weights)

            # 7. test global model and print out metrics after each communications round
            model_evaluate(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,global_model,x_train,y_train,x_test,y_test,comm_round, config.NUM_SYBILS)
        else:
            poison_scaling = aggregate_gradients(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,client_grads_unscaled, config.NUM_SYBILS)
            average_weights = sum_scaled_weights(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,client_grads_scaled,poison_scaling, config.NUM_SYBILS)
            # 6. update global model
            global_model.set_weights(average_weights)

            # 7. test global model and print out metrics after each communications round
            model_evaluate(config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,global_model,x_train,y_train,x_test,y_test,comm_round, config.NUM_SYBILS)
        
# 1. 
###############################################################################
###############################################################################
#                            Main                                             #
###############################################################################
###############################################################################

def main():

    '''
       To make single runs
    '''
    #simulation(config.PATH, config.PATH_IN, config.LOG_NAME, config.TABLE_NAME, config.COMMS_ROUND, config.ATTACK, config.DEFENSE, config.NUM_SYBILS, config.NUM_CLIENTS)
    
    
    '''
       To run just ids - no poison defense and poison attacks
    '''
    global_tic = time.perf_counter()
    config.BASELINE = True
    simulation(config.PATH, config.PATH_IN, config.LOG_NAME, config.TABLE_NAME, config.COMMS_ROUND, config.ATTACK, config.NO_DEFENSE, config.NUM_SYBILS, config.NUM_CLIENTS)
    '''
    for i in range(len(config.ATTACK_LIST)):
        config.ATTACK = config.ATTACK_LIST[i]
        for k in range(len(config.NUM_SYBILS_LIST)):
            config.NUM_SYBILS = config.NUM_SYBILS_LIST[k]
            print("Number of Sybils in loop {} for {} attack with a {} defense".format(config.NUM_SYBILS, config.ATTACK, config.NO_DEFENSE))
            with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.NO_DEFENSE +'_'+ config.LOG_NAME,'a') as f:
                f.write("###############################################################################")
                f.write("###############################################################################")
                f.write("#                   Begin {}_{}_sybils_{} Simulation              #".format(config.ATTACK, config.NUM_SYBILS, config.NO_DEFENSE))
                f.write("###############################################################################")
                f.write("###############################################################################")
            f.close()
            tic = time.perf_counter()
            simulation(config.PATH, config.PATH_IN, config.LOG_NAME, config.TABLE_NAME, config.COMMS_ROUND, config.ATTACK, config.NO_DEFENSE, config.NUM_SYBILS, config.NUM_CLIENTS)
            toc = time.perf_counter()
            sim_time = convert(toc-tic)
            with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.NO_DEFENSE +'_'+ config.LOG_NAME,'a') as f:
                f.write("Total time for simulation : {}_{}_sybils_{} was {}".format(config.ATTACK, config.NUM_SYBILS, config.NO_DEFENSE, sim_time))
                f.close()
            print("Total time for simulation : {}_{}_sybils_{} was {}".format(config.ATTACK, config.NUM_SYBILS, config.NO_DEFENSE, sim_time))
    global_toc = time.perf_counter()
    total_time = convert(global_toc-global_tic)
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.NO_DEFENSE +'_'+ config.LOG_NAME,'a') as f:
        f.write("###############################################################################")
        f.write("###############################################################################")
        f.write("#                      Total time for simulations was {}               #".format(total_time))
        f.write("###############################################################################")
        f.write("###############################################################################")
        f.close()
    print("###############################################################################")
    print("###############################################################################")
    print("#                      Total time for simulations was {}               #".format(total_time))
    print("###############################################################################")
    print("###############################################################################")
    '''

    '''
       To Test Poison Attacks against Poison Defenses
    '''
    '''
    global_tic = time.perf_counter()
    config.BASELINE = False
    for i in range(len(config.ATTACK_LIST)):
        for j in range(len(config.DEFENSE_LIST)):
            for k in range(len(config.NUM_SYBILS_LIST)):
                config.ATTACK = config.ATTACK_LIST[i]
                config.DEFENSE = config.DEFENSE_LIST[j]
                config.NUM_SYBILS = config.NUM_SYBILS_LIST[k]
                print("Number of Sybils in loop {} for {} attack with a {} defense".format(config.NUM_SYBILS, config.ATTACK, config.DEFENSE))
                with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_'+ config.LOG_NAME,'a') as f:
                    f.write("###############################################################################")
                    f.write("###############################################################################")
                    f.write("#                   Begin {}_{}_sybils_{} Simulation              #".format(config.ATTACK, config.NUM_SYBILS, config.DEFENSE))
                    f.write("###############################################################################")
                    f.write("###############################################################################")
                f.close()
                tic = time.perf_counter()
                simulation(config.PATH, config.PATH_IN, config.LOG_NAME, config.TABLE_NAME, config.COMMS_ROUND, config.ATTACK, config.DEFENSE, config.NUM_SYBILS, config.NUM_CLIENTS)
                toc = time.perf_counter()
                sim_time = convert(toc-tic)
                with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_'+ config.LOG_NAME,'a') as f:
                    f.write("Total time for simulation : {}_{}_sybils_{} was {}".format(config.ATTACK, config.NUM_SYBILS, config.DEFENSE, sim_time))
                    f.close()
                print("Total time for simulation : {}_{}_sybils_{} was {}".format(config.ATTACK, config.NUM_SYBILS, config.DEFENSE, sim_time))
    global_toc = time.perf_counter()
    total_time = convert(global_toc-global_tic)
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_'+ config.LOG_NAME,'a') as f:
        f.write("###############################################################################")
        f.write("###############################################################################")
        f.write("#                      Total time for simulations was {}               #".format(total_time))
        f.write("###############################################################################")
        f.write("###############################################################################")
        f.close()
    print("###############################################################################")
    print("###############################################################################")
    print("#                      Total time for simulations was {}               #".format(total_time))
    print("###############################################################################")
    print("###############################################################################")
    '''
if __name__ == "__main__":
    main()

'''
References:
    1) https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
'''