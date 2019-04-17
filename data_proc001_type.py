# @author:abanger
# blog: https://abanger.github.io
# github: https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles

# #  数据整合，未处理

import pandas as pd
import os
from tqdm import *
path="./"
data_list = os.listdir(path+'data_train/')


def add_device_type(devicetype,lenbb):
    chadd = pd.DataFrame(columns = ['ZVe44', 'ZV573',  'ZV63d', 'ZVfd4',  'ZVa9c',  'ZVa78',  'ZV252'])
    for i in range(lenbb): #插入一行
        chadd.loc[i] = [0  for n in range(7)]    
    chadd[devicetype]=1
    #print(devicetype)
    return chadd


file_name='data/data_all_n2_type.csv'
df = pd.read_csv(path+'data_train/'+ data_list[0])
lenbb=len(df)
devicetype=df.loc[0]['设备类型']
df = pd.concat( (df,add_device_type(devicetype,lenbb) ),axis=1)
df['sample_file_name'] = data_list[0]

df.to_csv(file_name, index=False,encoding='utf-8')

for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_train/' + data_list[i])
        lenbb=len(df)
        devicetype=df.loc[0]['设备类型']
        df = pd.concat( (df,add_device_type(devicetype,lenbb) ),axis=1)
        df['sample_file_name'] = data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue


test_data_list = os.listdir(path+'data_test/')


for i in tqdm(range(len(test_data_list))):
    if test_data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_test/' + test_data_list[i])
        lenbb=len(df)
        devicetype=df.loc[0]['设备类型']
        df = pd.concat( (df,add_device_type(devicetype,lenbb) ),axis=1)        
        df['sample_file_name'] = test_data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue