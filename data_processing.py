import pandas as pd
from ClusteringLayer import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

class data_processing:
    def __init__(self):
        pass

    def normalize(self,img):
        '''
        Normalizes an array
        (subtract mean and divide by standard deviation)
        '''
        eps = 0.001
        # print(np.shape(img))
        if np.std(img) != 0:
            img = (img - np.mean(img)) / np.std(img)
        else:
            img = (img - np.mean(img)) / eps
        return img

    def normalize_dataset(self,x):
        '''
        Normalizes list of arrays
        (subtract mean and divide by standard deviation)
        '''
        normalized_dataset = []
        for img in x:
            normalized = self.normalize(img)
            normalized_dataset.append(normalized)
        return normalized_dataset

    def pre_processing(self, df ):
        df.drop(columns=['saddr', 'daddr', 'ltime', 'stime', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'category'],
                axis=1, inplace=True)
        d = {'e': 1, 'e s': 2, 'e d': 3, 'e *': 4, 'e g': 5, 'eU': 6, 'e &': 7, 'e   t': 8, 'e    F': 9}
        df['flgs'] = df['flgs'].map(d)

        d = {'udp': 1, 'tcp': 2, 'arp': 3, 'ipv6-icmp': 4, 'icmp': 5, 'igmp': 6, 'rarp': 7}
        df['proto'] = df['proto'].map(d)

        df['bytes'] = df['bytes'].astype(int)
        df['pkts'] = df['pkts'].astype(int)
        df['spkts'] = df['spkts'].astype(int)
        df['dpkts'] = df['dpkts'].astype(int)
        df['sbytes'] = df['sbytes'].astype(int)
        df['dbytes'] = df['dbytes'].astype(int)
        df['sport'] = df.sport.fillna(value=0)
        df['dport'] = df.sport.fillna(value=0)
        # TODO: need to apply it on any hex values ( functon require)
        df.loc[df.sport.astype(str).str.startswith('0x0303'), "sport"] = 771
        df.loc[df.dport.astype(str).str.startswith('0x0303'), "dport"] = 771
        df['sport'] = df['sport'].astype(float)
        df['dport'] = df['dport'].astype(float)
        d = {'CON': 1, 'INT': 2, 'FIN': 3, 'NRS': 4, 'RST': 5, 'URP': 6}
        df['state'] = df['state'].map(d)
        d = {'Normal': 0, 'UDP': 1, 'TCP': 2, 'Service_Scan': 3, 'OS_Fingerprint': 4, 'HTTP': 5}
        df['subcategory '] = df['subcategory '].map(d)
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

    def load_data(self,file_path_normal, file_path_abnormal,timesteps):
        df_normal = pd.read_csv(file_path_normal)
        df_abnormal = pd.read_csv(file_path_abnormal)
        df = pd.concat([df_normal, df_abnormal], ignore_index=True)
        df = self.pre_processing(df)


        train, test = train_test_split(df, test_size=0.3, random_state=16)

        x_train, y_train = train.drop(columns=['attack']), train['attack']
        x_test, y_test = test.drop(columns=['attack']), test['attack']

        #timesteps = 3
        features = x_train.shape[1]

      #  x_train = x_train.values.reshape((x_train.shape[0], timesteps, x_train.shape[1]))
      #  x_test = x_test.values.reshape((x_test.shape[0], timesteps, x_test.shape[1]))

        x_train = np.array(x_train)
        x_train = np.nan_to_num(x_train)
        x_train = self.normalize_dataset(x_train)
        x_test = np.array(x_test)
        x_test = np.nan_to_num(x_test)
        x_test = self.normalize_dataset(x_test)
        del df
        CLASSES = ['normal', 'abnormal']
        y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
        y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        x_train,y_train=self.make_timesteps(np.array(x_train),np.array(y_train),timesteps)
        x_test, y_test = self.make_timesteps(np.array(x_test), np.array(y_test), timesteps)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], timesteps, features)
        x_test = x_test.reshape(x_test.shape[0], timesteps, features)



        return x_train,y_train,x_test,y_test

    def load_data_total(self,file_path):
        df = pd.read_csv(file_path)
        df = self.pre_processing(df)


        train, test = train_test_split(df, test_size=0.3, random_state=16)

        x_train, y_train = train.drop(columns=['attack']), train['attack']
        x_test, y_test = test.drop(columns=['attack']), test['attack']

        #timesteps = 3
        features = x_train.shape[1]

      #  x_train = x_train.values.reshape((x_train.shape[0], timesteps, x_train.shape[1]))
      #  x_test = x_test.values.reshape((x_test.shape[0], timesteps, x_test.shape[1]))

        x_train = np.array(x_train)
        x_train = np.nan_to_num(x_train)
        x_train = self.normalize_dataset(x_train)
        x_test = np.array(x_test)
        x_test = np.nan_to_num(x_test)
        x_test = self.normalize_dataset(x_test)
        del df
        CLASSES = ['normal', 'abnormal']
        y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
        y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        x_train,y_train=self.make_timesteps(np.array(x_train),np.array(y_train),1)
        x_test, y_test = self.make_timesteps(np.array(x_test), np.array(y_test), 1)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], 1, features)
        x_test = x_test.reshape(x_test.shape[0], 1, features)



        return x_train,y_train,x_test,y_test