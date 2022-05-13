import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import missingno
import numpy as np
import matplotlib.pyplot as plt
import config

class data_processing:
    def __init__(self):
        pass

    def normalize(self,data):
        '''
        Normalizes an array
        (subtract mean and divide by standard deviation)
        '''
        eps = 0.001
        # print(np.shape(data))
        if np.std(data) != 0:
            data = (data - np.mean(data)) / np.std(data)
        else:
            data = (data - np.mean(data)) / eps
        return data

    def normalize_dataset(self,x):
        '''
        Normalizes list of arrays
        (subtract mean and divide by standard deviation)
        '''
        normalized_dataset = []
        for data in x:
            normalized = self.normalize(data)
            normalized_dataset.append(normalized)
        return normalized_dataset

    '''
        Used to clean data with hex values
    '''
    def convertHextoDec(self, val):
        return int(val, 0)

    def convertHexData(self,df, column_name):
        values = df.loc[df[column_name].astype(str).str.startswith('0x'), column_name]
        for i in set(values):
            df.loc[df[column_name].astype(str).str.startswith(i), column_name] = self.convertHextoDec(i)
        df[column_name] = df[column_name].astype(float)
        return df
            
    # pre_processsing of the data
    # takes in the self object, and a dictionary
    def pre_processing(self, df, path, attack, defense, log_name,num_sybils=1 ):
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("\n\nPre-Processing Data")
            f.close()
        print("\nPre-Processing Data")
        
        # Removed any columns that do not contribute to anomaly based -- i.e. ip specific
        df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'category', 'subcategory '],
                axis=1, inplace=True)
        #df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'category'],
        #        axis=1, inplace=True)
        #df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco'],
        #        axis=1, inplace=True)
        # Fix strings
        d = {'e': 1, 'e s': 2, 'e d': 3, 'e *': 4, 'e g': 5, 'eU': 6, 'e &': 7, 'e   t': 8, 'e    F': 9, 'e r': 10}
        df['flgs'] = df['flgs'].map(d)
        d = {'udp': 1, 'tcp': 2, 'arp': 3, 'ipv6-icmp': 4, 'icmp': 5, 'igmp': 6, 'rarp': 7}
        df['proto'] = df['proto'].map(d)
        d = {'CON': 1, 'INT': 2, 'FIN': 3, 'NRS': 4, 'RST': 5, 'URP': 6, 'ACC': 7, 'REQ': 8}
        df['state'] = df['state'].map(d)
        #d = {'Normal': 0, 'DDoS': 1, 'DoS': 2, 'Reconnaissance': 3}
        #df['category'] = df['category'].map(d)
        #d = {'Normal': 0, 'UDP': 1, 'TCP': 2, 'Service_Scan': 3, 'OS_Fingerprint': 4, 'HTTP': 5}
        #df['subcategory '] = df['subcategory '].map(d)
        
        # Fix hex values hidden in two columns
        df = self.convertHexData(df,'sport')
        df = self.convertHexData(df,'dport')

        #  Make sure readable by ML 
        df['bytes'] = df['bytes'].astype(int)
        df['pkts'] = df['pkts'].astype(int)
        df['spkts'] = df['spkts'].astype(int)
        df['dpkts'] = df['dpkts'].astype(int)
        df['sbytes'] = df['sbytes'].astype(int)
        df['dbytes'] = df['dbytes'].astype(int)
        df['sport'] = df['sport'].astype(float)
        df['dport'] = df['dport'].astype(float)
        df['TnBPsrcIP'] = df['TnBPsrcIP'].astype(int)
        df['TnBPDstIP'] = df['TnBPDstIP'].astype(int)
        df['TnPPSrcIP'] = df['TnPPSrcIP'].astype(int)
        df['TnPPDstIP'] = df['TnPPDstIP'].astype(int)
        df['TnPPerProto'] = df['TnPPerProto'].astype(int)
        df['TnPPerDport'] = df['TnPPerDport'].astype(int)
        df['avgPD'] = df['avgPD'].astype(float)            
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write('Head of Data Frame\n')
            f.write(str(df.head()))
            f.write('\nCheck Data is Clean')
            f.write('Null Values')
            f.write(str(df.isnull().sum()))
            f.close()
        print('Head of Data Frame\n',df.head())
        print('\nCheck Data is Clean')
        print('Null Values')
        print(df.isnull().sum())
        missingno.matrix(df)
        #df.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\cleaned.csv', index=False)
        return df

    '''
    A UDF to convert input data into 3-D
    array as required for LSTM network.
    '''

    def make_timesteps(self,X, y, lookback):
        output_X = []
        output_y = []
        for i in range(len(X) - lookback - 1):
            t = []
            for j in range(1, lookback + 1):
                # Gather past records upto the lookback period
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + lookback + 1])
        return output_X, output_y
    
    # First call to load the two csv files.  one file is normal traffic,
    # other file is attack or abnormal traffic.  Timesteps are used as 
    # an LSTM parameter to help gain contextual knowledge of the different
    # data points. 
    def load_data(self,file_path_normal, file_path_abnormal,path, attack, defense, log_name,num_sybils=1, timesteps=80):
        # Create a normal dictiionary
        df_normal = pd.read_csv(file_path_normal)
        #print(df_normal.head())

        # Create an abnormal dictionary
        df_abnormal = pd.read_csv(file_path_abnormal)
        #print(df_abnormal.head())

        # Combine the two dictionaries together - ignore_index makes
        # sure the index count starts over at the concatenation
        df = pd.concat([df_normal, df_abnormal]).reset_index(drop=True)
        print(list(df.columns))
        print()
        attack_distribution = df.groupby(by='attack').size()
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("\nData Distributions:\n")
            f.close()
        print("Data Distributions:")
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("normal (0) abnormal(1)\n")
            f.write(str(attack_distribution))
            f.close()
        print("normal (0) abnormal(1)", attack_distribution)
        fig = attack_distribution.plot(kind='bar', figsize=(20,16), fontsize=14).get_figure()
        fig.savefig('attack_distr.pdf')
        proto_distribution = df.groupby(by='proto').size()
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("\nData Proto Distributions:\n")
            f.close()
        print("Data Proto Distributions:")
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("normal (0) abnormal(1)\n")
            f.write(str(proto_distribution))
            f.close()
        print("udp: 1, tcp: 2, arp: 3, ipv6-icmp: 4, icmp: 5, igmp: 6, rarp: 7", proto_distribution)
        fig = proto_distribution.plot(kind='bar', figsize=(20,16), fontsize=14).get_figure()
        fig.savefig('proto_distr.pdf')

        # Now have a dict of all data, pre-process the data
        df = self.pre_processing(df,config.PATH, config.ATTACK, config.DEFENSE, config.LOG_NAME,config.NUM_SYBILS)
        
        #want to target dos and ddos attacks - so proto tcp, port 80, bytes < 300 packets = 1
        #make a poisoned data set for sybils - can choose to target type proto TCP
        #dba are dba datasets
        if attack == 'backdoor' or attack == 'dba':
            print("\nIn backdoor or dba dataframe creation")
            poisonDF = df.copy()
            dbaPkts = poisonDF.copy(deep=True)
            dbaDport = poisonDF.copy(deep=True)
            dbaBytes = poisonDF.copy(deep=True)
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==1])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==5])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==3])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==4])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==6])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['proto']==7])
            dbaProto = poisonDF.copy(deep=True)
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['pkts']!=1])
            dbaPkts = dbaPkts.drop(dbaPkts.index[dbaPkts['pkts']!=1])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['dport']!=80])
            dbaDport = dbaDport.drop(dbaDport.index[dbaDport['dport']!=80])
            poisonDF = poisonDF.drop(poisonDF.index[poisonDF['bytes']>=201])
            dbaPkts = dbaPkts.drop(dbaPkts.index[dbaPkts['bytes']>=201])
            sizePoisonDF = poisonDF.size
            sizeDbaPkts = dbaPkts.size
            sizeDbaDport = dbaDport.size
            sizeDbaProto = dbaProto.size
            sizeDbaBytes = dbaBytes.size
            print("Backdooring DoS / DDoS, proto : TCP, pkts : 1, dport: 80, bytes >= 201 size of df is {}".format(sizePoisonDF))
            print("DBA DoS / DDoS, proto : TCP, pkts : 1, dport: 80, bytes >= 201 size of proto is {} : pkts : {} : dport : {} : bytes : {}".format(sizeDbaProto, sizeDbaPkts, sizeDbaDport, sizeDbaBytes))
            with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
                f.write("\n\nBackdooring DoS / DDoS, proto : TCP, pkts : 1, dport: 80, bytes >= 201 size of df is {}\n".format(sizePoisonDF))
                f.write("\nDBA DoS / DDoS, proto : TCP, pkts : 1, dport: 80, bytes >= 201 size of proto is {} : pkts : {} : dport : {} : bytes : {}\n".format(sizeDbaProto, sizeDbaPkts, sizeDbaDport, sizeDbaBytes))
            f.close()
            d = {1:0, 0:0}
            poisonDF['attack'] = poisonDF['attack'].map(d)
            dbaPkts['attack'] = dbaPkts['attack'].map(d)
            dbaDport['attack'] = dbaDport['attack'].map(d)
            dbaDport['attack'] = dbaDport['attack'].map(d)
            dbaBytes['attack'] = dbaBytes['attack'].map(d)
            #attack_subcat_distribution = df.groupby(by='subcategory ').size()
            #print("\nHTTP : 5 ", attack_subcat_distribution)
            with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
                f.write('\nHead of Poison Data Frame\n')
                f.write(str(poisonDF.head()))
                f.write('\nCheck Poison Data is Clean\n')
                f.write('Poison Null Values')
                f.write(str(poisonDF.isnull().sum()))

                # proto
                f.write('\n\nHead of dbaProto Data Frame\n')
                f.write(str(dbaProto.head()))
                f.write('\nCheck dbaProto Data is Clean')
                f.write('dbaProto Null Values')
                f.write(str(dbaProto.isnull().sum()))

                # dport
                f.write('\n\nHead of dbaDport Data Frame\n')
                f.write(str(dbaDport.head()))
                f.write('\nCheck dbaDport Data is Clean\n')
                f.write('\ndbaDport Null Values\n')
                f.write(str(dbaDport.isnull().sum()))

                # bytes
                f.write('\n\nHead of dbaBytes Data Frame\n')
                f.write(str(dbaBytes.head()))
                f.write('\nCheck dbaBytes Data is Clean\n')
                f.write('\ndbaBytes Null Values\n')
                f.write(str(dbaBytes.isnull().sum()))

                # pkts
                f.write('\n\nHead of dbaPkts Data Frame\n')
                f.write(str(dbaPkts.head()))
                f.write('\nCheck dbaPkts Data is Clean\n')
                f.write('\ndbaPkts Null Values\n')
                f.write(str(dbaPkts.isnull().sum()))
            f.close()
            print('\nHead of Poison Data Frame\n',poisonDF.head())
            print('\nCheck Poison Data is Clean')
            print('Poison Null Values')
            print(poisonDF.isnull().sum())
            missingno.matrix(poisonDF)
            #poisonDF.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\poisoncleaned.csv', index=False)
            #proto
            print('\nHead of dbaProto Data Frame\n',dbaProto.head())
            print('\nCheck dbaProto Data is Clean')
            print('dbaProto Null Values')
            print(dbaProto.isnull().sum())
            missingno.matrix(dbaProto)
            #dbaProto.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\dbaProtoCleaned.csv', index=False)
            #port
            print('\nHead of dbaDport Data Frame\n',dbaDport.head())
            print('\nCheck dbaDport Data is Clean')
            print('dbaDport Null Values')
            print(dbaDport.isnull().sum())
            missingno.matrix(dbaDport)
            #dbaDport.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\dbaDportCleaned.csv', index=False)
            #bytes
            print('\nHead of dbaBytes Data Frame\n',dbaBytes.head())
            print('\nCheck dbaBytes Data is Clean')
            print('dbaBytes Null Values')
            print(dbaBytes.isnull().sum())
            missingno.matrix(dbaBytes)
            #dbaBytes.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\dbaBytesCleaned.csv', index=False)
            #pkts
            print('\nHead of dbaPkts Data Frame\n',dbaPkts.head())
            print('\nCheck dbaPkts Data is Clean')
            print('dbaPkts Null Values')
            print(dbaPkts.isnull().sum())
            missingno.matrix(dbaPkts)
            #dbaPkts.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\dbaPktsCleaned.csv', index=False)

        # Next the dictionary with processed normal and attack data needs to be split.
        train, test = train_test_split(df, test_size=0.3, random_state=16)
        x_train, y_train = train.drop(columns=['attack']), train['attack']
        x_test, y_test = test.drop(columns=['attack']), test['attack']

        # Poison Next the dictionary with processed normal and attack data needs to be split.
        # create these to return empties to speed batch testing.
        x_trainP, x_testP, y_trainP, y_testP = np.empty(1),np.empty(1),np.empty(1),np.empty(1)
        if attack == 'backdoor':
            print("in backdoor train test split")
            trainP, testP = train_test_split(poisonDF, test_size=0.3, random_state=16)
            x_trainP, y_trainP = trainP.drop(columns=['attack']), train['attack']
            x_testP, y_testP = testP.drop(columns=['attack']), test['attack']

        # dbaProto Next the dictionary with processed normal and attack data needs to be split
        # create these to return empties to speed batch testing.
        x_trainDbaProto, x_testDbaProto, y_trainDbaProto, y_testDbaProto = np.empty(1),np.empty(1),np.empty(1),np.empty(1)
        x_trainDbaPkts, x_testDbaPkts, y_trainDbaPkts, y_testDbaPkts = np.empty(1),np.empty(1),np.empty(1),np.empty(1)
        x_trainDbaDport, x_testDbaDport, y_trainDbaDport, y_testDbaDport = np.empty(1),np.empty(1),np.empty(1),np.empty(1)
        x_trainDbaBytes, x_testDbaBytes, y_trainDbaBytes, y_testDbaBytes = np.empty(1),np.empty(1),np.empty(1),np.empty(1)
        if attack == 'dba':
            print("in dba train test split")
            trainDbaProto, testDbaProto = train_test_split(dbaProto, test_size=0.3, random_state=16)
            x_trainDbaProto, y_trainDbaProto = trainDbaProto.drop(columns=['attack']), train['attack']
            x_testDbaProto, y_testDbaProto = testDbaProto.drop(columns=['attack']), test['attack']

            # dbaPkts Next the dictionary with processed normal and attack data needs to be split.
            trainDbaPkts, testDbaPkts = train_test_split(dbaPkts, test_size=0.3, random_state=16)
            x_trainDbaPkts, y_trainDbaPkts = trainDbaPkts.drop(columns=['attack']), train['attack']
            x_testDbaPkts, y_testDbaPkts = testDbaPkts.drop(columns=['attack']), test['attack']

            # dbaPkts Next the dictionary with processed normal and attack data needs to be split.
            trainDbaDport, testDbaDport = train_test_split(dbaDport, test_size=0.3, random_state=16)
            x_trainDbaDport, y_trainDbaDport = trainDbaDport.drop(columns=['attack']), train['attack']
            x_testDbaDport, y_testDbaDport = testDbaDport.drop(columns=['attack']), test['attack']

            # dbaBytes Next the dictionary with processed normal and attack data needs to be split.
            trainDbaBytes, testDbaBytes = train_test_split(dbaBytes, test_size=0.3, random_state=16)
            x_trainDbaBytes, y_trainDbaBytes = trainDbaBytes.drop(columns=['attack']), train['attack']
            x_testDbaBytes, y_testDbaBytes = testDbaBytes.drop(columns=['attack']), test['attack']



        # Get features from second column, needed to reshape for LSTM 3-D array (timesteps)
        features = x_train.shape[1]
        with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_'+ log_name,'a') as f:
            f.write("\n\nData Features: {} | LSTM Timesteps: {}\n".format(features, timesteps))
        f.close()
        print("\nData Features: {} | LSTM Timesteps: {}".format(features, timesteps))

        # Make the array's numpy multi-dimension arrays to change shape for timesteps later
        # Make sure any NaNs are nums
        # Normalize data
        x_train = np.array(x_train)
        x_train = np.nan_to_num(x_train)
        x_train = self.normalize_dataset(x_train)
        x_test = np.array(x_test)
        x_test = np.nan_to_num(x_test)
        x_test = self.normalize_dataset(x_test)
        del df

        # Poison Make the array's numpy multi-dimension arrays to change shape for timesteps later
        # Make sure any NaNs are nums
        # Normalize data
        if attack == 'backdoor':
            print("in backdoor np.array and nan to num")
            x_trainP = np.array(x_trainP)
            x_trainP = np.nan_to_num(x_trainP)
            x_trainP = self.normalize_dataset(x_trainP)
            x_testP = np.array(x_testP)
            x_testP = np.nan_to_num(x_testP)
            x_testP= self.normalize_dataset(x_testP)
            del poisonDF

        # dbaProto Make the array's numpy multi-dimension arrays to change shape for timesteps later
        # Make sure any NaNs are nums
        # Normalize data
        if attack == 'dba':
            print("in dba np.array and nan to num")
            x_trainDbaProto = np.array(x_trainDbaProto)
            x_trainDbaProto = np.nan_to_num(x_trainDbaProto)
            x_trainDbaProto = self.normalize_dataset(x_trainDbaProto)
            x_testDbaProto = np.array(x_testDbaProto)
            x_testDbaProto = np.nan_to_num(x_testDbaProto)
            x_testDbaProto= self.normalize_dataset(x_testDbaProto)

            # dbaPkts Make the array's numpy multi-dimension arrays to change shape for timesteps later
            # Make sure any NaNs are nums
            # Normalize data
            x_trainDbaPkts = np.array(x_trainDbaPkts)
            x_trainDbaPkts = np.nan_to_num(x_trainDbaPkts)
            x_trainDbaPkts = self.normalize_dataset(x_trainDbaPkts)
            x_testDbaPkts = np.array(x_testDbaPkts)
            x_testDbaPkts = np.nan_to_num(x_testDbaPkts)
            x_testDbaPkts= self.normalize_dataset(x_testDbaPkts)

            # dbaDport Make the array's numpy multi-dimension arrays to change shape for timesteps later
            # Make sure any NaNs are nums
            # Normalize data
            x_trainDbaDport = np.array(x_trainDbaDport)
            x_trainDbaDport = np.nan_to_num(x_trainDbaDport)
            x_trainDbaDport = self.normalize_dataset(x_trainDbaDport)
            x_testDbaDport = np.array(x_testDbaDport)
            x_testDbaDport = np.nan_to_num(x_testDbaDport)
            x_testDbaDport= self.normalize_dataset(x_testDbaDport)

            # dbaDport Make the array's numpy multi-dimension arrays to change shape for timesteps later
            # Make sure any NaNs are nums
            # Normalize data
            x_trainDbaBytes = np.array(x_trainDbaBytes)
            x_trainDbaBytes = np.nan_to_num(x_trainDbaBytes)
            x_trainDbaBytes = self.normalize_dataset(x_trainDbaBytes)
            x_testDbaBytes = np.array(x_testDbaBytes)
            x_testDbaBytes = np.nan_to_num(x_testDbaBytes)
            x_testDbaBytes= self.normalize_dataset(x_testDbaBytes)
            del dbaProto
            del dbaPkts
            del dbaDport
            del dbaBytes

        #Create two classes and use keras.utils to create binary label arrays
        CLASSES = ['normal', 'abnormal']
        # this is one hot encoding for binary classificaiton
        y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
        y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        #Poison this is one hot encoding for binary classificaiton
        if attack == 'backdoor':
            print("in backdoor np.array 2nd time")
            y_trainP = np.array(keras.utils.to_categorical(y_trainP, len(CLASSES)))
            y_testP = np.array(keras.utils.to_categorical(y_testP, len(CLASSES)))
        
            y_trainP = np.asarray(y_trainP)
            y_testP = np.asarray(y_testP)

        #DbaProto this is one hot encoding for binary classificaiton
        if attack == 'dba':
            print("in dba np.array 2nd time")
            y_trainDbaProto = np.array(keras.utils.to_categorical(y_trainDbaProto, len(CLASSES)))
            y_testDbaProto = np.array(keras.utils.to_categorical(y_testDbaProto, len(CLASSES)))
        
            y_trainDbaProto = np.asarray(y_trainDbaProto)
            y_testDbaProto = np.asarray(y_testDbaProto)

            #DbaPkts this is one hot encoding for binary classificaiton
            y_trainDbaPkts = np.array(keras.utils.to_categorical(y_trainDbaPkts, len(CLASSES)))
            y_testDbaPkts = np.array(keras.utils.to_categorical(y_testDbaPkts, len(CLASSES)))
        
            y_trainDbaPkts = np.asarray(y_trainDbaPkts)
            y_testDbaPkts = np.asarray(y_testDbaPkts)

            #DbaDport this is one hot encoding for binary classificaiton
            y_trainDbaDport = np.array(keras.utils.to_categorical(y_trainDbaDport, len(CLASSES)))
            y_testDbaDport = np.array(keras.utils.to_categorical(y_testDbaDport, len(CLASSES)))
        
            y_trainDbaDport = np.asarray(y_trainDbaDport)
            y_testDbaDport = np.asarray(y_testDbaDport)

            #DbaBytes this is one hot encoding for binary classificaiton
            y_trainDbaBytes = np.array(keras.utils.to_categorical(y_trainDbaBytes, len(CLASSES)))
            y_testDbaBytes = np.array(keras.utils.to_categorical(y_testDbaBytes, len(CLASSES)))
        
            y_trainDbaBytes = np.asarray(y_trainDbaBytes)
            y_testDbaBytes = np.asarray(y_testDbaBytes)

        # set all arrays to numpy arrays
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        #P set all arrays to numpy arrays
        if attack == 'backdoor':
            print("in backdoor np.array 3rd time")
            x_trainP = np.array(x_trainP)
            x_testP = np.array(x_testP)
            y_trainP = np.array(y_trainP)
            y_testP = np.array(y_testP)

            #P set all arrays to numpy arrays
            x_trainP = np.array(x_trainP)
            x_testP = np.array(x_testP)
            y_trainP = np.array(y_trainP)
            y_testP = np.array(y_testP)

        if attack == 'dba':
            print("in dba np.array 3rd time")
            x_trainDbaProto = np.array(x_trainDbaProto)
            x_testDbaProto = np.array(x_testDbaProto)
            y_trainDbaProto = np.array(y_trainDbaProto)
            y_testDbaProto = np.array(y_testDbaProto)
            x_trainDbaPkts = np.array(x_trainDbaPkts)
            x_testDbaPkts = np.array(x_testDbaPkts)
            y_trainDbaPkts = np.array(y_trainDbaPkts)
            y_testDbaPkts = np.array(y_testDbaPkts)
            x_trainDbaDport = np.array(x_trainDbaDport)
            x_testDbaDport = np.array(x_testDbaDport)
            y_trainDbaDport = np.array(y_trainDbaDport)
            y_testDbaDport = np.array(y_testDbaDport)
            x_trainDbaBytes = np.array(x_trainDbaBytes)
            x_testDbaBytes = np.array(x_testDbaBytes)
            y_trainDbaBytes = np.array(y_trainDbaBytes)
            y_testDbaBytes = np.array(y_testDbaBytes)

        # Use make timesteps for LSTM timesteps.
        x_train,y_train=self.make_timesteps(np.array(x_train),np.array(y_train),timesteps)
        x_test, y_test = self.make_timesteps(np.array(x_test), np.array(y_test), timesteps)

        # P Use make timesteps for LSTM timesteps.
        if attack == 'backdoor':
            print("in backdoor timestamps")
            x_trainP,y_trainP=self.make_timesteps(np.array(x_trainP),np.array(y_trainP),timesteps)
            x_testP, y_testP = self.make_timesteps(np.array(x_testP), np.array(y_testP), timesteps)

        # dba Use make timesteps for LSTM timesteps.
        if attack == 'dba':
            print("in dba timestamps")
            x_trainDbaProto,y_trainDbaProto=self.make_timesteps(np.array(x_trainDbaProto),np.array(y_trainDbaProto),timesteps)
            x_testDbaProto, y_testDbaProto = self.make_timesteps(np.array(x_testDbaProto), np.array(y_testDbaProto), timesteps)
            x_trainDbaPkts,y_trainDbaPkts=self.make_timesteps(np.array(x_trainDbaPkts),np.array(y_trainDbaPkts),timesteps)
            x_testDbaPkts, y_testDbaPkts = self.make_timesteps(np.array(x_testDbaPkts), np.array(y_testDbaPkts), timesteps)
            x_trainDbaDport,y_trainDbaDport=self.make_timesteps(np.array(x_trainDbaDport),np.array(y_trainDbaDport),timesteps)
            x_testDbaDport, y_testDbaDport = self.make_timesteps(np.array(x_testDbaDport), np.array(y_testDbaDport), timesteps)
            x_trainDbaBytes,y_trainDbaBytes=self.make_timesteps(np.array(x_trainDbaBytes),np.array(y_trainDbaBytes),timesteps)
            x_testDbaBytes, y_testDbaBytes = self.make_timesteps(np.array(x_testDbaBytes), np.array(y_testDbaBytes), timesteps)

        # Make arrays numpy again and change x arrays for LSTM change shape
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], timesteps, features)
        x_test = x_test.reshape(x_test.shape[0], timesteps, features)

        #P Make arrays numpy again and change x arrays for LSTM change shape
        if attack == 'backdoor':
            print("in backdoor change shapes")
            x_trainP = np.array(x_trainP)
            x_testP = np.array(x_testP)
            y_trainP = np.array(y_trainP)
            y_testP = np.array(y_testP)
            x_trainP = x_trainP.reshape(x_trainP.shape[0], timesteps, features)
            x_testP = x_testP.reshape(x_testP.shape[0], timesteps, features)

        if attack == 'dba':
            print("in dba change shapes")
            x_trainDbaProto = np.array(x_trainDbaProto)
            x_testDbaProto = np.array(x_testDbaProto)
            y_trainDbaProto = np.array(y_trainDbaProto)
            y_testDbaProto = np.array(y_testDbaProto)
            x_trainDbaPkts = np.array(x_trainDbaPkts)
            x_testDbaPkts = np.array(x_testDbaPkts)
            y_trainDbaPkts = np.array(y_trainDbaPkts)
            y_testDbaPkts = np.array(y_testDbaPkts)
            x_trainDbaDport = np.array(x_trainDbaDport)
            x_testDbaDport = np.array(x_testDbaDport)
            y_trainDbaDport = np.array(y_trainDbaDport)
            y_testDbaDport = np.array(y_testDbaDport)
            x_trainDbaBytes = np.array(x_trainDbaBytes)
            x_testDbaBytes = np.array(x_testDbaBytes)
            y_trainDbaBytes = np.array(y_trainDbaBytes)
            y_testDbaBytes = np.array(y_testDbaBytes)

            x_trainDbaProto = x_trainDbaProto.reshape(x_trainDbaProto.shape[0], timesteps, features)
            x_testDbaProto = x_testDbaProto.reshape(x_testDbaProto.shape[0], timesteps, features)
            x_trainDbaPkts = x_trainDbaPkts.reshape(x_trainDbaPkts.shape[0], timesteps, features)
            x_testDbaPkts = x_testDbaPkts.reshape(x_testDbaPkts.shape[0], timesteps, features)
            x_trainDbaDport = x_trainDbaDport.reshape(x_trainDbaDport.shape[0], timesteps, features)
            x_testDbaDport = x_testDbaDport.reshape(x_testDbaDport.shape[0], timesteps, features)
            x_trainDbaBytes = x_trainDbaBytes.reshape(x_trainDbaBytes.shape[0], timesteps, features)
            x_testDbaBytes = x_testDbaBytes.reshape(x_testDbaBytes.shape[0], timesteps, features)

        return x_train,y_train,x_test,y_test,x_trainP,y_trainP,x_testP,y_testP, x_trainDbaProto, x_testDbaProto, y_trainDbaProto, y_testDbaProto , x_trainDbaPkts, x_testDbaPkts, y_trainDbaPkts, y_testDbaPkts, x_trainDbaDport, x_testDbaDport, y_trainDbaDport, y_testDbaDport, x_trainDbaBytes, x_testDbaBytes, y_trainDbaBytes, y_testDbaBytes