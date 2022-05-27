###############################################################################
###############################################################################
#                          Global Vars                                        #
###############################################################################
###############################################################################
import numpy as np

PATH = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\'
PATH_IN = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\input\\'
ATTACK_LIST = ['label','backdoor','dba']
ATTACK = ATTACK_LIST[1]
NO_ATTACK = 'none'
DEFENSE_LIST = ['none','fg', 'asf', 'sim']
DEFENSE = DEFENSE_LIST[3]
NO_DEFENSE = 'none'
NUM_SYBILS_LIST = [1,5,10]
NUM_SYBILS = NUM_SYBILS_LIST[1]
NO_SYBILS = 0
NUM_CLIENTS_LIST = [25] ############ if you change this you need to adjust create sybils to have numclients + 1 for names
NUM_CLIENTS = NUM_CLIENTS_LIST[0]
LOG_NAME = 'log.txt'
TABLE_NAME = 'results.csv'
COMMS_ROUND = 33
BASELINE = False
POISON_TIMESTEPS = 3 ########## Can't be more than size of test set in fl_ss_utils - sim which is min(8)
POISON_FEATURES = 4
Y_25_CLIENTS_1_SYBIL = np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_4_SYBIL = np.array([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_5_SYBIL = np.array([0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_10_SYBIL = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_20_SYBIL = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_25_SYBIL = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
Y_25_CLIENTS_40_SYBIL = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])