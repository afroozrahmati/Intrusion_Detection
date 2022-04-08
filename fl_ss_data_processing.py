import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import missingno
import numpy as np

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
    def pre_processing(self, df ):
        print("Pre-Processing Data")
        
        # Removed any columns that do not contribute to anomaly based -- i.e. ip specific
        df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'category'],
                axis=1, inplace=True)

        # Fix strings
        d = {'e': 1, 'e s': 2, 'e d': 3, 'e *': 4, 'e g': 5, 'eU': 6, 'e &': 7, 'e   t': 8, 'e    F': 9}
        df['flgs'] = df['flgs'].map(d)
        d = {'udp': 1, 'tcp': 2, 'arp': 3, 'ipv6-icmp': 4, 'icmp': 5, 'igmp': 6, 'rarp': 7}
        df['proto'] = df['proto'].map(d)
        d = {'CON': 1, 'INT': 2, 'FIN': 3, 'NRS': 4, 'RST': 5, 'URP': 6, 'ACC': 7, 'REQ': 8}
        df['state'] = df['state'].map(d)
        d = {'Normal': 0, 'UDP': 1, 'TCP': 2, 'Service_Scan': 3, 'OS_Fingerprint': 4, 'HTTP': 5}
        df['subcategory '] = df['subcategory '].map(d)
        
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

        print('Head of Data Frame\n',df.head())
        print('Check Data is Clean')
        print('\nNull Values')
        print(df.isnull().sum())
        missingno.matrix(df)
        df.to_csv(r'C:\Users\ChristianDunham\Source\Repos\Intrusion_Detection\data\cleaned.csv', index=False)
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
    def load_data(self,file_path_normal, file_path_abnormal,timesteps=80):
        # Create a normal dictiionary
        df_normal = pd.read_csv(file_path_normal)
        #print(df_normal.head())

        # Create an abnormal dictionary
        df_abnormal = pd.read_csv(file_path_abnormal)
        #print(df_abnormal.head())

        # Combine the two dictionaries together - ignore_index makes
        # sure the index count starts over at the concatenation
        df = pd.concat([df_normal, df_abnormal]).reset_index(drop=True)
       
        # Now have a dict of all data, pre-process the data
        df = self.pre_processing(df)

        # Next the dictionary with processed normal and attack data needs to be split.
        train, test = train_test_split(df, test_size=0.3, random_state=16)

        # For the training data we don't want the attack label
        x_train, y_train = train.drop(columns=['attack']), train['attack']
        #print("x_train is:\n ", str(np.shape(x_train)))
        #print(x_train.head())
        #print("y_train is: \n", str(np.shape(y_train)))
        #print(y_train.head())

        # TODO what is this?  thought these were supposed to be labels
        x_test, y_test = test.drop(columns=['attack']), test['attack']
        #print("x_test.head() ", x_test.head())
        #print("y_test.head() ", y_test.head())

        # Get features from second column, needed to reshape for LSTM 3-D array (timesteps)
        features = x_train.shape[1]
        print("Data Features: {} | LSTM Timesteps: {}".format(features, timesteps))

        # Make the array's numpy multi-dimension arrays to change shape for timesteps later
        # Make sure any NaNs are nums
        # Normalize data
        x_train = np.array(x_train)
        x_train = np.nan_to_num(x_train)
        x_train = self.normalize_dataset(x_train)
        x_test = np.array(x_test)
        x_test = np.nan_to_num(x_test)
        x_test = self.normalize_dataset(x_test)
        
        # Why are we deleteing the object?
        del df

        #Create two classes and use keras.utils to create binary label arrays
        CLASSES = ['normal', 'abnormal']
        # this is one hot encoding for binary classificaiton - ????? why? 
        y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
        y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        #encoder = LabelEncoder()
        #encoder.fit(y_train)
        #y_train = encoder.transform(y_train)
        #encoder2 = LabelEncoder()
        #encoder2.fit(y_test)
        #y_test = encoder2.transform(y_test)
        
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # set all arrays to numpy arrays
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Use make timesteps for LSTM timesteps.
        x_train,y_train=self.make_timesteps(np.array(x_train),np.array(y_train),timesteps)
        x_test, y_test = self.make_timesteps(np.array(x_test), np.array(y_test), timesteps)

        # Make arrays numpy again and change x arrays for LSTM change shape
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], timesteps, features)
        x_test = x_test.reshape(x_test.shape[0], timesteps, features)

        return x_train,y_train,x_test,y_test