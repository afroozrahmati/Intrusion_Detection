import pandas as pd

df_normal = pd.read_csv('./data/normal.csv')
df_abnormal = pd.read_csv('./data/abnormal.csv')

print("normal data shape: ", df_normal.shape)
print("abnormal data shape: ", df_abnormal.shape)


df = pd.concat([df_normal,df_abnormal]).reset_index(drop=True)
df= df.drop(columns=['saddr','daddr','ltime','stime','smac','dmac','soui','doui','sco','dco'])

d = {'e':1, 'e s':2 , 'e d':3, 'e *':4, 'e g':5 ,'eU':6,'e &':7,'e   t':8, 'e    F':9}
df['flgs'] = df['flgs'].map(d)

d = {'udp':1, 'tcp':2 , 'arp':3, 'ipv6-icmp':4, 'icmp':5 ,'igmp':6,'rarp':7}
df['proto'] = df['proto'].map(d)

df['bytes']=df['bytes'].astype(int)
df['pkts']=df['pkts'].astype(int)
df['spkts']=df['spkts'].astype(int)
df['dpkts']=df['dpkts'].astype(int)
df['sbytes']=df['sbytes'].astype(int)
df['dbytes']=df['dbytes'].astype(int)
df['sport'] = df.sport.fillna(value=0)
df['dport'] = df.sport.fillna(value=0)

df.loc[df.sport.astype(str).str.startswith('0x') , "sport"]
df.loc[df.sport.astype(str).str.startswith('0x0303') , "sport"]=771
df.loc[df.dport.astype(str).str.startswith('0x') , "dport"]
df.loc[df.dport.astype(str).str.startswith('0x0303') , "dport"]=771
df['sport']=df['sport'].astype(float)
df['dport']=df['dport'].astype(float)
d = {'CON':1, 'INT':2 , 'FIN':3, 'NRS':4, 'RST':5 ,'URP':6}
df['state'] = df['state'].map(d)
d = {'Normal':0, 'UDP':1 , 'TCP':2, 'Service_Scan':3, 'OS_Fingerprint':4 ,'HTTP':5}
df['subcategory '] = df['subcategory '].map(d)
df= df.drop(columns=['category'])
