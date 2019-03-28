# 20190222修改
# 删除异常数据 
## 20190222添加特征数据
## 搅拌超压信号异常在训练集与测试集
import pandas as pd
import os
#from tqdm import *
path="./"
data_list = os.listdir(path+'data_train/')


file_name='data/data_all_d20.csv'
df = pd.read_csv(path+'data_train/'+ data_list[0])

df = df.drop(df[df['活塞工作时长']>120].index)  ##50
df = df.drop(df[ (df['发动机转速']>10000) ].index)
df = df.drop(df[  (df['油泵转速']>15000) ].index)
#df.drop(df[ df['泵送压力']>450 ].index)  ##>0
df = df.drop(df[ df['排量电流']>15000 ].index)  ##由训练数据图表观察到异常      
df['t0']=df['油泵转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
df['t1']=df['发动机转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
df['t2']=df['发动机转速']*df['泵送压力']/df['流量档位'] #？？
df['t3']=df['油泵转速']*df['泵送压力']/df['流量档位']
df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
df['t6']=df['发动机转速']/df['排量电流']     ###每?
df['t7']=[ 0    if x>105 or x<10  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏
df['t8']=[ 1    if x>35 and x<85  else  0  for x in df['液压油温'] ]   #正常工作温度
df['t9']=[ 1    if x<2000  else  0  for x in df['排量电流'] ]   #正常电流  
df['engine_int']=[ 1   if x<7500 and x>500  else  0  for x in df['发动机转速'] ]  ##正常转速范围
df['pump_speed_int']=[ 1    if x<8500 and x>500  else  0  for x in df['油泵转速'] ]  ##正常转速范围
df['t10']=df['搅拌超压信号']*df['活塞工作时长']  ##出错机率
df['t11']=df['油泵转速']/df['排量电流']     ###每?

#df['flow_int']=df['流量档位'].astype(int)  ##档位取整


df['sample_file_name'] = data_list[0]
df.to_csv(file_name, index=False,encoding='utf-8')

#for i in tqdm(range(1, len(data_list))):
for i in range(1, len(data_list)):
    if data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_train/' + data_list[i])

        df = df.drop(df[df['活塞工作时长']>120].index)  ##50
        df = df.drop(df[ (df['发动机转速']>10000) ].index)
        df = df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df = df.drop(df[ df['排量电流']>15000 ].index)  ##由训练数据图表观察到异常      
        df['t0']=df['油泵转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t1']=df['发动机转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力']/df['流量档位'] #？？
        df['t3']=df['油泵转速']*df['泵送压力']/df['流量档位']
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['发动机转速']/df['排量电流']     ###每?
        df['t7']=[ 0    if x>105 or x<10  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏
        df['t8']=[ 1    if x>35 and x<85  else  0  for x in df['液压油温'] ]   #正常工作温度
        df['t9']=[ 1    if x<2000  else  0  for x in df['排量电流'] ]   #正常电流  
        df['engine_int']=[ 1   if x<7500 and x>500  else  0  for x in df['发动机转速'] ]  ##正常转速范围
        df['pump_speed_int']=[ 1    if x<8500 and x>500  else  0  for x in df['油泵转速'] ]  ##正常转速范围
        df['t10']=df['搅拌超压信号']*df['活塞工作时长']  ##出错机率
        df['t11']=df['油泵转速']/df['排量电流']     ###每?

  
        df['sample_file_name'] = data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue


test_data_list = os.listdir(path+'data_test/')


#for i in tqdm(range(len(test_data_list))):
for i in range(len(test_data_list)):
    if test_data_list[i].split('.')[-1] == 'csv':
        df = pd.read_csv(path+'data_test/' + test_data_list[i])

        df = df.drop(df[df['活塞工作时长']>120].index)  ##50
        df = df.drop(df[ (df['发动机转速']>10000) ].index)
        df = df.drop(df[  (df['油泵转速']>15000) ].index)
        #df.drop(df[ df['泵送压力']>450 ].index)  ##>0
        df = df.drop(df[ df['排量电流']>15000 ].index)  ##由训练数据图表观察到异常      
        df['t0']=df['油泵转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t1']=df['发动机转速']/df['分配压力']  #（暂时为出口流量？）电机的功率也将越大
        df['t2']=df['发动机转速']*df['泵送压力']/df['流量档位'] #？？
        df['t3']=df['油泵转速']*df['泵送压力']/df['流量档位']
        df['t4']=df['分配压力']/df['泵送压力']  ##两者什么关系？
        df['t5']=df['油泵转速']*df['泵送压力']  ##qt=nV 式中n-一液压油泵的转速;V一一液压油泵的排量
        df['t6']=df['发动机转速']/df['排量电流']     ###每?
        df['t7']=[ 0    if x>105 or x<10  else  1  for x in df['液压油温'] ]   #60度左右，一般不会超过85度，坏
        df['t8']=[ 1    if x>35 and x<85  else  0  for x in df['液压油温'] ]   #正常工作温度
        df['t9']=[ 1    if x<2000  else  0  for x in df['排量电流'] ]   #正常电流  
        df['engine_int']=[ 1   if x<7500 and x>500  else  0  for x in df['发动机转速'] ]  ##正常转速范围
        df['pump_speed_int']=[ 1    if x<8500 and x>500  else  0  for x in df['油泵转速'] ]  ##正常转速范围
        df['t10']=df['搅拌超压信号']*df['活塞工作时长']  ##出错机率
        df['t11']=df['油泵转速']/df['排量电流']     ###每?

            
        df['sample_file_name'] = test_data_list[i]
        df.to_csv(file_name, index=False, header=False, mode='a+',encoding='utf-8')
    else:
        continue
