import numpy as np
import pandas as pd
import pickle5 as pickle
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


from functions.sub_surrender_profiles import get_risk_drivers
from global_vars import path_tables, getDataPath

def reviewDataBalance(surrender_profile):

    path_data = getDataPath(surrender_profile)

    # import Training and Test Data
    X_train = pd.read_csv(os.path.join(path_data,r'X_train.csv'), index_col= 0) 
    X_test = pd.read_csv(os.path.join(path_data, r'X_test.csv'), index_col= 0) 
    y_train = pd.read_csv(os.path.join(path_data, r'y_train.csv'), index_col= 0).values.flatten()
    y_test = pd.read_csv(os.path.join(path_data, r'y_test.csv'), index_col= 0 ).values.flatten()

    # restrict data to relevant features -> assume proper exploratory data analysis
    cache = get_risk_drivers(profile=surrender_profile)
    features_profile_lst = []
    for el in cache:
        if el != 'Premium_freq': 
            features_profile_lst.append(el)
        else:
            features_profile_lst.append('Premium_freq_0')
            features_profile_lst.append('Premium_freq_1')

    X_train, X_test = X_train[[el for el in features_profile_lst]], X_test[[el for el in features_profile_lst]]

    # Load Scaling range used in 'Lapse_data_preparation' for later visualization
    with open(os.path.join(path_data,'dict_range_scale_{}.pkl'.format(surrender_profile)), 'rb') as f:
        dict_range_scale =  pickle.load(f)
    # Load beta0 of latent surrender model for later visualization
    with open(os.path.join(path_data,r'beta0.pkl'), 'rb') as f:
        beta0 = pickle.load(f)

    # RUS resampling
    _, y_train_rus = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X_train,y_train)
    # SMOTE resampling
    _, y_train_smote = SMOTE().fit_resample(X_train,y_train)

    # analyze balance of data
    dict_data = {'numb_train': len(y_train), 'imb_train': np.round_(sum(y_train)/len(y_train),4), 'numb_rus': len(y_train_rus), 'numb_smote': len(y_train_smote), 'numb_test': len(y_test), 'imb_test': np.round_(sum(y_test)/len(y_test),4)}
    
    # save result for profile i
    with open(os.path.join(path_tables, r'{}_DataCounts.pkl'.format(surrender_profile)), 'wb') as f:
        pickle.dump(dict_data, f, pickle.HIGHEST_PROTOCOL)
    print('Profile ', surrender_profile, ' analyzed.')

def createSummaryOfDataBalance():

    data = {}
    for i in range(4):
        if os.path.exists(os.path.join(path_tables, r'{}_DataCounts.pkl'.format(i))):
            with open(os.path.join(path_tables, r'{}_DataCounts.pkl'.format(i)), 'rb') as f:
                    data[i] = pickle.load(f)

    df_data = pd.DataFrame.from_dict(data, orient = 'index')
    print(df_data)

    with open(os.path.join(path_tables,r'Data_review.tex'),'w') as f:
        f.write(df_data.to_latex())


if __name__ == '__main__':

    for i in [0,1,2,3]:
        # analyze data of profile i
        reviewDataBalance(i)

    # create LaTeX-table for data-balance of all four surrender-profiles
    createSummaryOfDataBalance()

    