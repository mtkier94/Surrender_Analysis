import numpy as np
import os, sys, copy
import pickle5 as pickle

import sklearn
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import  RandomUnderSampler

import tensorflow as tf


import functions.sub_surrender_models as sub_surrender_models

sys.modules['sub_surrender_models'] = sub_surrender_models # avoid path conflict when pickle-loading object that has references to sub_surrender_models

def load_ANN_boost_object(path):
    '''
    pickle load ANN_boost object. 
    Note: tf.keras.model.Sequential() objects as in object.model_base[i] have been replaced by their respective parameters. 
    The functional object need to be restored. For this we provide the object function .restore_learners(). The architecture is provided by the object. 
    '''

    with open(path, 'rb') as file:
        # throws error unless sys.models['sub_surrender_models'] is manually provided
        # source of error: see https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
        ANN_boost_object = pickle.load(file) 
        ANN_boost_object.restore_learners()
        print('NN boosting object loaded successfully!')
    return ANN_boost_object


def exp_decay_scheduler(epoch, lr):
    if epoch%5==0:
        return lr*0.9
    else:
        return lr

def keras_count_nontrainable_params(model, trainable = False):
    
    '''
    Returns the number of nontrainable parameters in a tf.keras model, as displayed in the tf.keras summary() function.
    
    Inputs:
    -------
        model: tf.keras model
        trainable: Boolean. Use False to count nontrainable params and True to count trainable params
        
    Outputs:
    --------
        desired number of nontrainable (if trainable == False; otherwise trainable) params 
    '''
    
    count = 0
    for l in model.layers:
        if l.trainable == trainable:
            for i in l.get_weights():
                count += len(i.flatten())
    return count


def resample_and_shuffle(X_train, y_train, resample_type: str):

    '''
    Resample and shuffle data X, y according to the resampling type 'resample_type'.
    '''

    X,y = copy.copy(X_train), copy.copy(y_train)

    if resample_type == 'undersampling':
        X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X, y)
        X,y = sklearn.utils.shuffle(X,y)
    elif resample_type == 'SMOTE':
        X,y = SMOTE().fit_resample(X, y)
        X,y = sklearn.utils.shuffle(X,y)
    elif resample_type == 'None':
        X,y = sklearn.utils.shuffle(X,y)
    else:
        ValueError('Unknown value for resample_type!')

    return X, y


def create_tf_dataset(X,y, val_share, batch_size):
    '''
    Split numpy "training" data X, y into training and validation data and transform them into a tf.data.Dataset object, e.g. to avoid sharding data.
    '''

    if type(X) != type(np.array([1])):
        try:
            X = X.values
        except:
            raise ValueError('invalide type(X): ', type(X))


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_share, random_state=42)
    train_data, val_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)), tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    return train_data, val_data



def reshape_model_input(df_input, degrees_lst = [1,1,1]):
    
    '''
    Create additional feature input for higher degrees. Relevant for Logit-Model and adaptive feature selection.
    
    df_input: DataFrame with original data. Order of columns specifies order with which polynomial degrees are applied.
    degrees_lst: List with degrees of features in altered Dataframe
    
    Output: New DataFrame with higher order polynomial features
    '''
    
    df_new = df_input.copy()
    
    for i in range(len(degrees_lst)):
        for j in range(2,degrees_lst[i]+1):
            # Note: We do not feature-engineer binary encoded, categorical features as is only increases complexity w/o benefit
            if df_input.columns[i] not in ['Premium_freq_0', 'Premium_freq_1']:
                df_new[df_input.columns[i]+ '^' +str(j)] = df_new[df_input.columns[i]]**j
            
    
    return df_new 

