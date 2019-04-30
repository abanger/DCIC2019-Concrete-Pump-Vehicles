# @author:abanger
# blog: https://abanger.github.io
# github: https://github.com/abanger/DCIC2019-Concrete-Pump-Vehicles

import pandas as pd
import pickle
import gc
#import matplotlib.pyplot as plt
#import seaborn as sns
#import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import f1_score


from sklearn import metrics
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2018)
print("load_data")

df_train=pd.read_csv("data/train_labels.csv")
df_test=pd.read_csv("data/submit_example.csv")


data_feature=pd.read_csv("data/data_all_d20.csv",header=0,names=[
                                                            "work_time","engine_speed","pump_speed","pump_pressure",
                                                            "temperature","flow","pressure","output_current","low_on",
                                                            "high_on","signal","positive_pump","anti_pump","unit_type",
                                                            't0','t1','t2','t3','t4', 't5', 't6', 't7','t8','t9',
                                                            'engine_int','pump_speed_int','t10','t11',"sample_file_name"
                                                            ])




data_feature = pd.get_dummies(data_feature,columns=['unit_type'])


#众数   scipy.stats.modes
aggs = {
    "sample_file_name": ["size"],
    "work_time": ['max','min','sum','mean'],
    "engine_speed":['max','min','sum','mean','std','median',np.ptp],
    "pump_speed":['max','min','sum','mean','std','median',np.ptp],
    "pump_pressure":['max','min','sum','mean','std','median',np.ptp],
    "temperature":['max','min','sum','mean','std','median',np.ptp],
    "flow":['max','min','sum','mean','std','median',np.ptp],
    "pressure":['max','min','sum','mean','std','median',np.ptp],
    "output_current":['max','min','sum','mean','std','median',np.ptp],  
    
    "low_on":['sum',"mean"],
    "high_on":['sum',"mean"],
    "signal":['sum',"mean"],
    "positive_pump":['sum',"mean"],
    "anti_pump":['sum',"mean"],
    'unit_type_ZVe44':['sum',"mean"], 
    'unit_type_ZV573':['sum',"mean"], 
    'unit_type_ZV63d':['sum',"mean"], 
    'unit_type_ZVfd4':['sum',"mean"], 
    'unit_type_ZVa9c':['sum',"mean"], 
    'unit_type_ZVa78':['sum',"mean"], 
    'unit_type_ZV252':['sum',"mean"],
    "t0":['max','min','sum','mean'],    
    "t1":['max','min','sum','mean'],
    "t2":['max','min','sum','mean'],
    "t3":['max','min','sum','mean'],   
    "t4":['max','min','sum','mean'],
    "t5":['max','min','sum','mean'],
    "t6":['max','min','sum','mean'],
    "t7":['max','min','sum'],
    "t8":['sum','mean'],
    "t9":['sum','mean'],
    "engine_int":['sum','mean'],
    "pump_speed_int":['sum','mean'], 
    "t10":['max','min','sum','mean','std'],
    "t11":['max','min','sum','mean','std'],
  

}


def make_feature(data,aggs,name):

    agg_df = data.groupby('sample_file_name').agg(aggs)
    agg_df.columns = agg_df.columns = ['_'.join(col).strip()+name for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)
    
    agg_df["f0"]=agg_df["engine_speed_mean_tsf"]-agg_df["pump_speed_mean_tsf"]
    agg_df["f1"] = agg_df["work_time_max" + name] / agg_df["sample_file_name_size" + name]
    agg_df["f2"] = agg_df["positive_pump_sum" + name] * agg_df["sample_file_name_size" + name]
    agg_df["f3"] = agg_df["temperature_mean" + name] /agg_df["work_time_max" + name]
    agg_df["f4"] = agg_df["pump_pressure_mean" + name]/agg_df["work_time_max" + name]
    agg_df["f5"] = agg_df["work_time_max" + name] * agg_df["sample_file_name_size" + name]
    agg_df["f6"] = (agg_df["positive_pump_sum" + name]+agg_df["anti_pump_sum" + name])/agg_df["sample_file_name_size" + name]
    agg_df["f7"]=agg_df["f6"]*agg_df["work_time_max" + name]
    return agg_df

agg_df=make_feature(data_feature,aggs,"_tsf")
#agg_df.to_csv('agg_xgb_d20_100.csv', index=False, encoding='utf-8')

print(agg_df.head(2))
print(agg_df.columns)
del data_feature
gc.collect()




##直接使用xgb生成数据
print("agg_xgb_d20_100...")
#agg_df=pd.read_csv("agg_xgb_d20_100.csv",encoding='utf-8')


def my_scorer(y_true, y_predicted,X_test):
    loss_train = np.sum((y_true - y_predicted)**2, axis=0) / (X_test.shape[0])  #RMSE
    loss_train = loss_train **0.5
    score = 1/(1+loss_train)
    return loss_train

n_folds = 4
 
features = df_train.merge(agg_df,on='sample_file_name',how='left')
test_features = df_test.merge(agg_df,on='sample_file_name',how='left')

df_train=features
df_test=test_features
# Extract the ids
train_ids = features['sample_file_name']
test_ids = test_features['sample_file_name']

# Extract the labels for training
labels = features['label']

# Remove the ids and target
features = features.drop(columns = ['sample_file_name', 'label'])
test_features = test_features.drop(columns = ['sample_file_name', 'label'])


# Extract feature names
feature_names = list(features.columns)

# Convert to np arrays
features = np.array(features)
test_features = np.array(test_features)

# Create the kfold object
k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))

# Empty array for test predictions
test_predictions = np.zeros(test_features.shape[0])
train_predictions = np.zeros(features.shape[0])

# Empty array for out of fold validation predictions
out_of_fold = np.zeros(features.shape[0])

# Lists for recording validation and training scores
#valid_scores = []
train_scores = []

# Iterate through each fold
for train_indices, valid_indices in k_fold.split(features):

    # Training data for the fold
    train_features, train_labels = features[train_indices], labels[train_indices]
    # Validation data for the fold
    valid_features, valid_labels = features[valid_indices], labels[valid_indices]

    # Create the model
    #'''  效果好点，发给lgb要XGBClassifier效果好
    model = xgb.XGBRegressor(objective = 'reg:linear',n_estimators=16000,min_child_weight=1,num_leaves=20,
                               learning_rate = 0.01, max_depth=6,n_jobs=20,
                               subsample = 0.6, colsample_bytree = 0.4, colsample_bylevel = 1)
    '''
    model = xgb.XGBRegressor(objective = 'binary:logistic',n_estimators=16000,min_child_weight=1,num_leaves=20,
                               learning_rate = 0.01, max_depth=6,n_jobs=20,
                               subsample = 0.6, colsample_bytree = 0.4, colsample_bylevel = 1)

    model = xgb.XGBClassifier(objective = 'binary:logistic',n_estimators=16000,min_child_weight=1,num_leaves=20,
                               learning_rate = 0.01, max_depth=6,n_jobs=20,
                               subsample = 0.6, colsample_bytree = 0.4, colsample_bylevel = 1)
                             
    # Train the model  LGBMClassifier  XGBClassifier
    
    model.fit(train_features, train_labels,
              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
              early_stopping_rounds = 300, verbose = 600)
    '''
    model.fit(train_features, train_labels,
              eval_set = [(train_features, train_labels),(valid_features, valid_labels)],
              eval_metric=["error", "logloss"],
              early_stopping_rounds = 300, verbose = 600)

    # Record the best iteration
    best_iteration = 16000

    # Record the feature importances
    feature_importance_values += model.feature_importances_ / k_fold.n_splits

    # Make predictions
    test_predictions += model.predict(test_features)/ k_fold.n_splits
    train_predictions += model.predict(features)/ k_fold.n_splits
    # Record the out of fold predictions
    out_of_fold      = model.predict(valid_features)/ k_fold.n_splits

    # Record the best score
    train_score =  my_scorer(valid_labels,out_of_fold,valid_features)

    # valid_scores.append(valid_score)
    train_scores.append(train_score)

    # Clean up memory
    gc.enable()
    del model, train_features, valid_features
    gc.collect()

# Make the submission dataframe
submission = pd.DataFrame({'sample_file_name': test_ids, 'label': test_predictions})
train_sub = pd.DataFrame({'sample_file_name': train_ids, 'label': train_predictions})
# Make the feature importance dataframe
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

# Overall validation score
#valid_auc = roc_auc_score(labels, out_of_fold)

# Add the overall scores to the metrics
#valid_scores.append(valid_auc)
train_scores.append(np.mean(train_scores))

# Needed for creating dataframe of validation scores
fold_names = list(range(n_folds))
fold_names.append('overall')

# Dataframe of validation scores
metric = pd.DataFrame({'fold': fold_names,
                        'train': train_scores,
                        })



print('Baseline metrics')
print(metric)


#submission["label"] = [np.round(x) for x in submission["label"]]
#submission["label"]=submission["label"].astype("int")
# 阈值0.47，由于时间没有进一步测试
submission["label"] = [ 1    if x>0.47  else  0  for x in  submission["label"]]   
submission.to_csv("sub/xgb_use_xgb_d20_105.csv", index=False)


##feature_importances.to_csv("sub/feature_importances_xgb_d20_105.csv", index=False)
##feature_importances_columns = [c for c in feature_importances.columns if c not in ['sample_file_name', 'label']]
##print(feature_importances_columns)




