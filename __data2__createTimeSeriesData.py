import numpy as np
import seaborn as sns
import pandas as pd
# import scipy as sc

import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
# from scipy.stats import norm, skewnorm

from sklearn.model_selection import train_test_split
import pickle5 as pickle
import copy, os.path


# import functions
from functions.sub_surrender_profiles import get_risk_drivers, visualize_surrender_profiles, get_model_input
from functions.sub_simulate_events import simulate_surrender_time_series, visualize_time_series_lapse, create_summary_for_lapse_events
from functions.sub_preparation_and_exploration_of_data import scale_DATA, compare_data_sets, transform_TS_portfolio, determine_iteration_for_training_test_split
from global_vars import path_plots, path_portfolio, getDataPath


def createTimeSeriesData(surrender_profile, bool_plot = False):

    path_data = getDataPath(surrender_profile)
    # Load portfolio data
    Portfolio = pd.read_csv(os.path.join(path_portfolio, r'Portfolio.csv'), index_col=0 )


    # Assumptions: Risk drivers
    features_surrender_profile_lst = get_risk_drivers(surrender_profile)
    print('Features in Profile {}: '.format(surrender_profile) + str(features_surrender_profile_lst))
    # Assumption: Structure of Surrender
    modeling_interval = 1
    target_surrender_rate = (1+0.025)**modeling_interval -1

    # Simulate lapses (surrender, maturity, death) for given portfolio over time (until all contracts have been lapsed)
    TS_portfolio, beta0 = simulate_surrender_time_series(df = Portfolio, target_rate = target_surrender_rate, 
                                                        profile_nr = surrender_profile, modeling_time_step = modeling_interval, 
                                                        time_period_max= 15,
                                                        option_new_business=True, rate_new_business=0.06)


    with open(os.path.join(path_data, r'beta0.pkl'), 'wb') as f:
        # save baseline hazard for later comparison of models
        pickle.dump(beta0, f, pickle.HIGHEST_PROTOCOL)


    if bool_plot:
        events_profile = visualize_time_series_lapse(TS_portfolio, modeling_time_step= modeling_interval, 
                                                    option_view= 'lapse_decomp',#'abs_view', 
                                                    option_maturity= False, 
                                                    option_annualized_rates=True, zoom_factor= 1, 
                                                    title= 'Surrender Profile {}'.format(surrender_profile), 
                                                    fig_size= (14,4))

        create_summary_for_lapse_events(events_profile, profile_name= str(surrender_profile))

    ### Data Preparation
    # print('Summary of DataFrames')
    # print('----------------------------------------------------------------------------------------------------------')
    # print('\t TS_portfolio: \t Simulated Portfolio with Lapsed (0/1/2/3) indication')
    # print('\t \t \t Time Series format, i.e. list of DataFrames \n')

    # print('\t TS_portfolio_scaled: Simulated Portfolio with Lapsed (0/1/2/3) indication')
    # print('\t \t \t Scaled version of TS_portfolio')
    # print('\t \t \t Scaling by scale_TS_DATA() \n')

    # print('\t df_events:\t Simulated Portfolio with Lapsed (0/1/2/3) indication')
    # print('\t \t \t DataFrame with event times per contract (for profile x) \n')

    # print('\t DATA: \t\t Simulated Portfolio with Lapsed (0/1) indication')
    # print('\t \t \t DataFrame version of TS_portfolio with dummy version of "Premium_freq"')
    # print('\t \t \t Transformation by transform_TS_portfolio()')
    # print('\t \t \t Basis for X_train_raw, X_test_raw \n')

    # print('\t DATA_scaled: \t Simulated Portfolio with Lapsed (0/1) indication')
    # print('\t \t \t Scaled version of DATA, scaling range given in dict_range_scale')
    # print('\t \t \t Scaling by scale_DATA()')
    # print('\t \t \t Basis for X_train, X_test, y_train, y_test')
    # print('----------------------------------------------------------------------------------------------------------')

    # Write time series (list format) as a single array
    DATA = transform_TS_portfolio(TS_portfolio)
    # exclude times and target-values from scaling
    X_raw, y_raw = copy.deepcopy(DATA.drop(['Lapsed', 'Time'], axis=1)), copy.deepcopy(DATA['Lapsed'])
    df_rem = pd.DataFrame(data=X_raw['Duration']-X_raw['Duration_elapsed'], columns=['Duration_remain'])
    X_raw = pd.concat([X_raw,df_rem],axis=1,)

    # train-test split
    train_share = 0.7
    N_train, _ = determine_iteration_for_training_test_split(TS_portfolio, train_share)

    dict_range_scale = {}
    scaler_min, scaler_max = X_raw.loc[0:N_train].min(axis=0), X_raw.loc[0:N_train].max(axis=0) # extract range from training-data
    for i in range(len(X_raw.columns)):
        dict_range_scale[X_raw.columns[i]] = {'min': scaler_min[i], 'max': scaler_max[i]}
    with open(os.path.join(path_data, 'dict_range_scale_{}.pkl'.format(surrender_profile)), 'wb') as f:
        pickle.dump(dict_range_scale, f, pickle.HIGHEST_PROTOCOL)

    X, y = scale_DATA(X_raw, dict_range_scale=dict_range_scale), y_raw
    X_raw, X = pd.concat([DATA['Time'], X_raw], axis=1), pd.concat([DATA['Time'], X], axis=1)
    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= N_train, shuffle=False)#random_state=42)
    X_train_raw, X_test_raw, _, _ = train_test_split(X_raw, y_raw, train_size= N_train, shuffle=False)


    if bool_plot:
        # check for validity of data
        compare_data_sets(x_train= X_train, x_test= X_test, fig_size= (14,4), n_cols= 3,
                        x_train_raw= X_train_raw, x_test_raw=X_test_raw, bool_dist=True)


    print('------------------------------------------')
    print('Surrender event in training data: ' + str( np.round(sum(y_train)/y_train.shape[0]*100, decimals = 4))+ '%')
    print('Surrender event in test data: ' + str( np.round(sum(y_test)/y_test.shape[0]*100, decimals = 3))+ '%')
    print('------------------------------------------')

    # Export Data
    X_train.to_csv(os.path.join(path_data, r'X_train.csv'), header=True)
    X_train_raw.to_csv(os.path.join(path_data, r'X_train_raw.csv'), header=True)
    X_test.to_csv(os.path.join(path_data, r'X_test.csv'), header=True)
    X_test_raw.to_csv(os.path.join(path_data, r'X_test_raw.csv'), header=True)
    y_train.to_csv(os.path.join(path_data, r'y_train.csv'), header=True)
    y_test.to_csv(os.path.join(path_data, r'y_test.csv'), header=True)

    print('Train and test data saved!')


if __name__ == '__main__':


    bool_simulate = False
    bool_plot = True

    if bool_simulate:
        for i in [0,1,2,3]:
            # simulation of ts-data for profile i
            createTimeSeriesData(surrender_profile = i, bool_plot = bool_plot)
            # save plot of surrender profile visualization for profile i
            visualize_surrender_profiles(profile_nr= i, fig_size=(12,2), path=os.path.join(os.path.dirname(os.path.realpath(__file__)), r'Plots'))
            

            
