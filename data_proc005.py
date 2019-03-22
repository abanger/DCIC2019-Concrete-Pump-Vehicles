# 20190222修改
# 删除异常数据 
## 20190222添加特征数据

import pandas as pd
import os
from tqdm import *
path="./"
data_list = os.listdir(path+'data_train/')


file_name='data/data_all_n2_d5.csv'
df = pd.read_csv(path+'data_train/'+ data_list[0])

df.drop(df[df['活塞工作时长']>150].index)  ##50
df.drop(df[ (df['发动机转速']>10000) ].index)
df.drop(df[  (df['油泵转速']>15000) ].index)
#df.drop(df[ df['泵送压力']>450 ].index)  ##>0
df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
df['t2']=df['发动机转速']*df['泵送压力'] #？？
df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
df['t6']=df['排量电流'] /df['流量档位']    ###每?
df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏

df['sample_file_name'] = data_list[0]
df.to_csv(file_name, index=False,encoding='utf-8')

for i in tqdm(range(1, len(data_list))):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_train/' + data_list[i])

        df.drop(df[df['活塞工作时长']>150].index)  ##50
        df.drop(df[ (df['发动机转速']>10000) ].index)
        df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
        df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力'] #？？
        df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['排量电流'] /df['流量档位']    ###每?
        df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏
  
        df['sample_file_name'] = data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue


test_data_list = os.listdir(path+'data_test/')


for i in tqdm(range(len(test_data_list))):
    if test_data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_test/' + test_data_list[i])

        df.drop(df[df['活塞工作时长']>150].index)  ##50
        df.drop(df[ (df['发动机转速']>10000) ].index)
        df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df.drop(df[ df['排量电流']>50000 ].index)  ##由训练数据图表观察到异常      
        df['t1']=df['发动机转速']*df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力'] #？？
        df['t3']=df['搅拌超压信号']/df['活塞工作时长']  ##出错机率
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['排量电流'] /df['流量档位']    ###每?
        df['t7']=[ 0    if x<85 and x>0  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏 
            
        df['sample_file_name'] = test_data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue
