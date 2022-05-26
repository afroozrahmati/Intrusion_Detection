###############################################################################
###############################################################################
#                          Global Vars                                        #
###############################################################################
###############################################################################

PATH = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\'
PATH_IN = 'C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\input\\'
ATTACK_LIST = ['label','backdoor','dba']
ATTACK = ATTACK_LIST[1]
NO_ATTACK = 'none'
DEFENSE_LIST = ['none','fg', 'asf', 'sim']
DEFENSE = DEFENSE_LIST[3]
NO_DEFENSE = 'none'
NUM_SYBILS_LIST = [1,5,10]
NUM_SYBILS = NUM_SYBILS_LIST[0]
NO_SYBILS = 0
NUM_CLIENTS_LIST = [15] ############ if you change this you need to adjust create sybils to have numclients + 1 for names
NUM_CLIENTS = NUM_CLIENTS_LIST[0]
LOG_NAME = 'log.txt'
TABLE_NAME = 'results.csv'
COMMS_ROUND = 33
BASELINE = False