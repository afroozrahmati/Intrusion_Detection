###############################################################################
###############################################################################
#                          Global Vars                                        #
###############################################################################
###############################################################################

PATH = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\'
PATH_IN = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\input\\'
ATTACK_LIST = ['label','backdoor','dba']
ATTACK = ATTACK_LIST[0]
DEFENSE_LIST = ['fg', 'asf', 'sim']
DEFENSE = DEFENSE_LIST[1]
NUM_SYBILS_LIST = [1,5,10]
NUM_SYBILS = NUM_SYBILS_LIST[2]
NUM_CLIENTS_LIST = [10] ############ if you change this you need to adjust create sybils to have numclients + 1 for names
NUM_CLIENTS = NUM_CLIENTS_LIST[0]
LOG_NAME = 'log.txt'
TABLE_NAME = 'results.csv'
COMMS_ROUND = 25