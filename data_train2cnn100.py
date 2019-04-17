# @author:abanger
# blog: https://abanger.github.io
# github: https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles

#类似图像处理情况（处理数据）

import os
#from PIL import Image
import numpy as np
import pandas as pd
#from tqdm import *
#活塞工作时长,发动机转速,油泵转速,泵送压力,液压油温,流量档位,分配压力,排量电流,低压开关,高压开关,搅拌超压信号,正泵,反泵,设备类型
def add_device_type(devicetype,lenbb):
    chadd = pd.DataFrame(columns = ['ZVe44', 'ZV573',  'ZV63d', 'ZVfd4',  'ZVa9c',  'ZVa78',  'ZV252'])
    for i in range(lenbb): #插入一行
        chadd.loc[i] = [0  for n in range(7)]    
    chadd[devicetype]=1
    #print(devicetype)
    return chadd


path="./"
    
##训练数据
#直接每个文件平均数据
#path=r"e:\notebook\2019CCF-cd3\data_train"
fm=os.listdir(path+'data_train/') 


'''
doclen=0
for f in fm: 
    #ff= os.listdir(path+'\\'+f+'\\') 
    td=pd.read_csv(path+'/data_train/'+f )
    tlen=td.shape[0]
    if tlen%20!=0:
        tlen=tlen+20-tlen%20
    doclen+=tlen
print(doclen)
'''
doclen=10460160//20
i=0
chunks = []
train_data=[]
train_lable=pd.read_csv(path+"data/train_labels.csv")

data = np.empty((doclen,1,20,20),dtype="float32")
label = np.empty((doclen,))

#for f in tqdm(fm):  
for f in fm: 
    #ff= os.listdir(path+'\\'+f+'\\')  
    td=pd.read_csv(path+'/data_train/'+f )
    #删除异常数据
    td.drop(td[td['活塞工作时长']>150].index)  ##50
    td.drop(td[ (td['发动机转速']>10000) ].index)
    td.drop(td[  (td['油泵转速']>15000) ].index)
    #td.drop(td[ td['泵送压力']>450 ].index)  ##>0
    td.drop(td[ td['排量电流']>5000 ].index)  ##由训练数据图表观察到异常      

    td["活塞工作时长"]=np.log1p(td["活塞工作时长"]) *25.5
    td["发动机转速"]=np.log1p(td["发动机转速"]) *25.5
    td["油泵转速"]=  np.log1p(td["油泵转速"]) *25.5
    #td["pump_pressure"]=np.log1p(td["pump_pressure"])
    td["液压油温"]=np.log1p(td["液压油温"]+70) *25.5
    #td["pressure"]=np.log1p(td["pressure"])
    td["排量电流"]=np.log1p(td["排量电流"]) *25.5


    devicetype=td.loc[0]['设备类型']
    ff=train_lable[train_lable['sample_file_name']==f ]['label']
    #td.insert(0,'label',ff.values[0])    
    #td.insert(0,'sample_file_name',f)
    bb=td.drop(['设备类型'], axis=1)
    lenbb=bb.shape[0]
    a=bb.mean(axis=0)  
    aa=pd.DataFrame(a)
    aa=aa.T
    #aa.insert(0,'sample_file_name',f)
    bb=pd.concat( (bb,add_device_type(devicetype,lenbb) ),axis=1)
    aa=pd.concat( (aa,add_device_type(devicetype,1) ),axis=1)
    if lenbb<20:
        for ii in range(lenbb,20):
            bb=bb.append(aa,ignore_index=True)
        data[i,:,:,:]  =  bb.values
        label[i] =ff.values[0]
        i+=1
    else:
        #cc=bb
        a=lenbb//20
        j=0
        for j in range(1,a+1): 
            dd=bb[(j-1)*20:j*20]
            data[i,:,:,:]  =  dd.values
            label[i] =ff.values[0]
            i+=1
        cc=bb[j*20:lenbb]
        if (lenbb%20!=0):
            for k in  range(lenbb,lenbb+20-lenbb%20):
                cc=cc.append(aa,ignore_index=True)
            data[i,:,:,:]  =  cc.values
            label[i] =ff.values[0]
            i+=1    
        #bb=cc
    #train_data.append(bb)

    if i%500==0:
        print('read:'+str(i))
    #    break;
    #data[i,:,:,:]  =  bb.values
    #label[i] =ff.values[0]

        
#train_data = pd.concat(train_data, ignore_index=True)

#train_data.to_csv('train_data_2cnn100.csv',index=False,encoding='utf-8')
np.save('data_train_cnn100.npy', data)
np.save('label_train_cnn100.npy', label)

