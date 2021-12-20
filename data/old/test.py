import os
import pandas as pd


file_path_normal= 'D:\\UW\\RA\\Intrusion_Detection\\data\\normal_1.csv'   #+ sys.argv[0]
file_path_abnormal= 'D:\\UW\\RA\\Intrusion_Detection\\data\\abnormal_1.csv'  #+ sys.argv[1]

df_normal = pd.read_csv(file_path_normal, usecols=['pkSeqID', 'flgs'])

df_normal.set_index('pkSeqID')

df_normal.drop(columns=["flgs"], axis=1, inplace=True)
print(df_normal.dtypes.value_counts())