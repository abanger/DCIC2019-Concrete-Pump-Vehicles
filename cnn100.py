# @author:abanger
# blog: https://abanger.github.io
# github: https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles

## 处理data_train2cnn100.py 生成数据，过程
import os
import sys
from keras import backend as K
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Dropout
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
#coding:utf-8
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""    ###这两句代码用来关闭CUDA加速
from keras.models import Input,Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D,GlobalAveragePooling2D
from keras.layers import BatchNormalization,Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import glob
import random
import numpy as np
img_h,img_w,n_channels = 20,20,1 ####图像高，宽，通道数
classes = 2 ###类别
train_batch_size = 40  
test_batch_size= 20   

import pickle
import gc
#import matplotlib.pyplot as plt
#import seaborn as sns
#import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2018)
print("load_data")


train_data=np.load('data_train_cnn100.npy')
train_lable=np.load('label_train_cnn100.npy')

##CNN输入的数据预处理
'''
lent=train_data.shape[0]
ddtrain_data =np.empty((lent,1,20,20),dtype="float32")
for i in range(0,lent):
    ddtrain_data[i,0,:,:]  = train_data[i,0,:,: ]-np.mean(train_data[i,0,:,: ], axis = 0)
    if i%8000==0:
        print(str(i))
'''

x_train,x_val,y_train,y_val = train_test_split(train_data,train_lable)
#x_train,x_val,y_train,y_val = train_test_split(ddtrain_data,train_lable)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
train_data = x_train /255
test_data=x_val/255


# dimensions of our images.
img_width, img_height = 20, 20
K.set_image_dim_ordering('th') 

input_data = Input(shape=(n_channels,img_h,img_w))
out = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(input_data)
out = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(out)
out = BatchNormalization(axis=3)(out)

####out = Dense(units=512,activation='relu')(out) ###数据集比较小，可不加这一层
##out = Dense(units=5,activation='softmax')(out)
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#out = Dense(units=64,activation='relu')(out)
#out = Dropout(0.5)(out)
#out = Flatten()(out)
out = GlobalAveragePooling2D()(out)
out = Dense(units=1,activation='sigmoid')(out)
model = Model(inputs=input_data,outputs=out)
model.summary()

model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy',  metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=128, epochs=200,  verbose=1, validation_data=(x_val, y_val))
model.save('model_cnn119-129.h5')

score1 = model.evaluate(x_val, y_val, verbose=0)
print(score1)
y_val_pred=model.predict(x_val)
rmse=mean_squared_error(y_val,y_val_pred)**0.5
score=1.0/(1.0+rmse)
print('score:',score)
score_all=f1_score(y_val,y_val_pred>0.5 )
print('f1_score:',score_all)