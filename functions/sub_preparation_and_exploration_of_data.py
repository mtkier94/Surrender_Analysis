import pandas as pd
import numpy as np 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from functions.sub_simulate_events import get_surrender_prob



# # Prepare Data for Modeling

def transform_TS_portfolio(TS_portfolio_lst, column_names_lst = ['Age', 'Face_amount', 'Duration', 'Duration_elapsed', 
                               'Premium_freq', 'Premium', 'Premium_annual']):
    '''
    Transform the time series (list of DataFrames) to a single DataFrame 
    starting point: list where each entry is a dataframe for one point in time 

    Input
    \t TS_portfolio_lst: List of DataFrames with columns given in column_names_lst
    \t column_names_lst: List of names of columns to be appended to unifying DataFrame
    
    Output
    \t Single DataFrame where list elements have been appended into a single and 'Premium_freq' is tranformed into dummy vars 
    '''   
   
    times= []
    # Values at initial time 0
    data = copy.deepcopy(TS_portfolio_lst[0][column_names_lst])
    times+=[0]*len(TS_portfolio_lst[0])

    targets = (TS_portfolio_lst[0]['Lapsed'] == 1).astype('int')
    
    # Append all points in time to existing DataFrame
    for i in range(1,len(TS_portfolio_lst)):
        data = data.append(TS_portfolio_lst[i][column_names_lst], ignore_index = True)
        times+=[i]*len(TS_portfolio_lst[i])
        # Save events {0,1,2,3} binarily
        targets = targets.append((TS_portfolio_lst[i]['Lapsed'] == 1).astype('int'))
    
    
    DATA = pd.DataFrame(data)
    DATA['Lapsed'] = targets.values
    DATA['Time'] = times
    ## accomplished: Transfer data from list (with dataframes as entries) into a single dataframe 'DATA'
    ## Next: Prepare DATA
    
    # create dummies categorical variable 'Premium_freq' (Note: Use Premium_freq_12 as baseline, i.e. monthly payment)
    prem_freq_dummies = pd.get_dummies(DATA.Premium_freq, prefix = 'Premium_freq').iloc[:,0:-1]
    # Combine data
    DATA_combined = pd.DataFrame(data = pd.concat([DATA.drop('Premium_freq', axis = 1),#, 'Lapsed']),#.loc[:, (DATA.columns != 'Premium_freq')&(DATA.columns != 'Lapsed')], 
                               prem_freq_dummies], axis = 1))
    
    return DATA_combined    

def scale_DATA(df, dict_range_scale = {'Age': {'min': 0, 'max': 100}, 'Face_amount': {'min': 0, 'max': 26000}, 
                                       'Duration': {'min': 0, 'max': 33}, 'Duration_elapsed': {'min': 0, 'max': 33},
                                       'Premium': {'min': 0, 'max': 19000}, 
                                       'Premium_annual': {'min': 0, 'max': 19000}},
              reverse_bool = False):
    
    '''
    Aim: Scale (selected) columns of DataFrame

    Input
    \t df: \t\t Dataframe to scale
    \t dict_range_scale: \t dictionary with column names and corresponding 'min', 'max' values for (linear) scaling
    \t reverse_bool: \t boolean True: Convert raw-> scaled, False: Convert scaled -> raw
    
    
    Output
    \t Scaled/ converted DataFrame
    '''
    
    DATA_scaled = pd.DataFrame.copy(df)
    
    if reverse_bool == False: # Convert raw data into scaled version
        for feature in dict_range_scale:
            if feature in df.columns:
                DATA_scaled[feature] = 2*(copy.copy(df[feature])-dict_range_scale[feature]['min'])/(dict_range_scale[feature]['max']-dict_range_scale[feature]['min']) -1
    else: # Convert scaled data into raw version
        for feature in dict_range_scale:
            if feature in df.columns:
                DATA_scaled[feature] = ((copy.copy(df[feature])+1)/2)*(dict_range_scale[feature]['max']-dict_range_scale[feature]['min'])+dict_range_scale[feature]['min']
        if 'Time' in df.columns:
            DATA_scaled['Time'] = (copy.copy(df['Time'])+1)/2*15
    
    return DATA_scaled

def scale_TS_DATA(TS_lst, 
                  dict_range_scale = {'Age': {'min': 0, 'max': 100}, 'Face_amount': {'min': 0, 'max': 26000}, 
                                      'Duration': {'min': 0, 'max': 33}, 'Duration_elapsed': {'min': 0, 'max': 33}, 
                                      'Premium': {'min': 0, 'max': 19000}, 
                                       'Premium_annual': {'min': 0, 'max': 19000}}):
    
    '''
    Aim: Scale (MinMax) data of time series for every single point in time (i.e. entries of list)
    
    Inputs:
    -------
        TS_portfolio_lst:   list where each entry 'i' corresponds to a dataframe with all active contracts at time 'i'
        dict_range_scale:   dictionary with column names and corresponding 'min', 'max' values for (linear) scaling
    
    Outputs:
    --------
        Timeseries/ List of DataFrames with scaled values
    '''
    
    
    # Create DEEP COPY, meaning a copy for all nested elements (avoid using the same references!!)
    TS_scaled = copy.deepcopy(TS_lst) 
    # Start scaling procedure for all elements/ points in time 'i'
    # For time 'i': scaled relevant columns of nested (!) dataframe
    for i in range(len(TS_lst)):
        for feature in dict_range_scale:
            TS_scaled[i][feature] = 2*(TS_lst[i][feature]-dict_range_scale[feature]['min'])/(dict_range_scale[feature]['max']-dict_range_scale[feature]['min']) -1
    
    return TS_scaled


def determine_iteration_for_training_test_split(TS_data_lst, target_share):
    '''
    Aim: Determine the number of m-thly iterations of training data until a target percentage of data has been seen

    Inputs:
    -------
        TS_data_lst: time series of data in a list format; entriy 'i' of list correspond to portfolio decomposition at time 'i' 
        target_share: target share of data to be included in training data

    Outputs:
    --------
        i: m-thly iteration after which target_share of data has been seen
        count_cum[i]: number of data points that corresponds to i
    '''
    count = {}
    for i in range(len(TS_data_lst)):
        count[i]=TS_data_lst[i].shape[0]

    count_cum = np.array(list(count.values())).cumsum()

    # determine entry number and m-thly iteration 
    for i in range(len(TS_data_lst)):
        if count_cum[i]/count_cum[-1] > target_share:
            # Note: index count starts at 0 (-> subtract 1)
            return count_cum[i], i 



def compare_data_sets(x_train, x_test, x_train_raw = None, x_test_raw = None, bool_dist = False,
                      fig_size = (12,8), labels = ['Training Data', 'Test Data'], n_cols = 4,
                      beta0 = 0, profile = 0):
    
    '''
    Compare distribution of features in training (x_train) and test set (x_test)
    '''
    
    features = x_train.columns
    _, ax = plt.subplots(ncols = n_cols, nrows = int(np.ceil(len(features)/n_cols)), figsize = fig_size)
    ax = ax.flatten()
    
    for i in range(len(features)):
        sns.distplot(a = x_train[features[i]], color= 'green', kde= False, norm_hist= True,
                     ax = ax[i], label = labels[0] )
        sns.distplot(a = x_test[features[i]], color= 'orange', kde= False, norm_hist= True,
                     ax = ax[i], label = labels[1] )
        ax[i].set_xlabel(features[i])
        if (i%n_cols) == 0:
            ax[i].set_ylabel('Densitiy')
    
    ax[0].legend()
    plt.tight_layout()
    plt.show()
    
    if bool_dist:
        
        _, ax = plt.subplots(ncols = 2, nrows = 1, figsize = fig_size)
        ax = ax.flatten()
        
        # Note: get_surrender_prob() works with raw/ non-scaled features and 'Premium_freq' \in {0,1,12}
        prob_test = get_surrender_prob(df = pd.concat([x_test_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(x_test['Premium_freq_0']*0+ (x_test['Premium_freq_1']==1)+\
                                                               12*(x_test['Premium_freq_0']+ x_test['Premium_freq_1']<=0), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                                       #features_lst = features_names_lst,
                                   profile_nr= profile, beta0= beta0, rand_noise_var= 0)
        
        # Note: get_surrender_prob() works with raw/ non-scaled features and 'Premium_freq' \in {0,1,12}
        prob_train = get_surrender_prob(df = pd.concat([x_train_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(x_train['Premium_freq_0']*0+ (x_train['Premium_freq_1']==1)+\
                                                               12*(x_train['Premium_freq_0']+ x_train['Premium_freq_1']<=0), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                                    #features_lst = features_names_lst,
                                   profile_nr= profile, beta0= beta0, rand_noise_var= 0)
        sns.distplot(a = prob_train, color= 'green', kde= True, norm_hist= True, hist = False, 
                     ax = ax[0], label = labels[0])
        ax[0].set_xlabel(r'$p_{Y|X}(1|x)$')
        ax[0].set_ylabel('Kernel density estimate')
        ax[0].legend()
        
        sns.distplot(a = prob_test, color= 'orange', kde= True, norm_hist= True, hist = False, 
                     ax = ax[1], label = labels[1])
        ax[1].legend()
        ax[1].set_xlabel(r'$p_{Y|X}(1|x)$')
        plt.show()



def display_confusion_matrix(matrix):
    return pd.DataFrame(matrix, columns= [r'pred.: 0', r'pred.: 1'], index= [r'obs.: 0', r'obs.: 1'])


def display_evaluation_curves(x, y, predictors_lst, predictors_name_lst, curve_type = 'ROC', figsize = (8,4)):
    
    '''
    Evaluate models w.r.t. different threshold for prediction.
    
    Input
    \t x: \t \t Data
    \t y: \t \t True Values
    \t predictors_lst: \t list with models to evaluate
    \t predictors_name_lst: \t list of names for legend of models
    \t curve_type: \t ''ROC' (Display ROC), 'RP' (Recall vs Precision Curve) or 'both'    
    
    Output
    \t Plot
    '''
    
    n_models = len(predictors_lst)
    
    # false positive rate, true positive rate
    fpr, tpr = dict(), dict()
    recall, precision = dict(), dict()
    auc_ROC, auc_prec_rec = dict(), dict()
    
    for i in range(n_models):
        # Note: Predict probabilities and use only positive outcome
        
        #if ('weights' in dir(predictors_lst[i])) == False: # 'weights' property not existent for Logistic Model
        # Use -1 in predict_proba(x)[:,-1] to avoid case-switch for Logit and ANN
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true= y, y_score= predictors_lst[i].predict_proba(x)[:,-1])
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_true=y, 
                                                                probas_pred = predictors_lst[i].predict_proba(x)[:,-1])
            
        auc_ROC[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        auc_prec_rec[i] = sklearn.metrics.auc(recall[i],precision[i])
        
    if curve_type == 'both':
        _, ax = plt.subplots(nrows= 1, ncols= 2, figsize= figsize)
        
        for i in range(n_models):
            ax[0].plot(fpr[i], tpr[i], 
                       label = predictors_name_lst[i]+': AUC = {}'.format(np.round(auc_ROC[i], decimals = 4)))
            ax[1].plot(recall[i], precision[i], 
                       label = predictors_name_lst[i] +': AUC = {}'.format(np.round(auc_prec_rec[i], decimals=4)))
            
        # Plot Layout: ROC Plot
        ax[0].plot([0,1], [0,1], linestyle = '--', label = 'Blind guess')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].legend(loc = 'lower right')
        
        # Plot Layout: Recall-Precision Curve
        ax[1].plot([0,1], [sum(y[y==1])/y.shape[0],sum(y[y==1])/y.shape[0]], linestyle = '--', 
         label = 'Blind guess')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
    else: 
        _, ax = plt.subplots(nrows= 1, ncols= 1, figsize= figsize)
        if curve_type == 'ROC':
            for i in range(n_models):
                ax.plot(fpr[i], tpr[i], 
                        label = predictors_name_lst[i]+': AUC = {}'.format(np.round(auc_ROC[i], decimals = 4)))
            # Plot Layout: ROC Plot
            ax.plot([0,1], [0,1], linestyle = '--', label = 'Blind guess')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc = 'lower right')
            ax.set_title('ROC', fontsize = 'large')
            plt.show()
        
        elif curve_type == 'RP':
            for i in range(n_models):
                ax.plot(recall[i], precision[i], 
                        label = predictors_name_lst[i] +': AUC = {}'.format(np.round(auc_prec_rec[i], decimals=4)))
            # Plot Layout: Recall-Precision Curve
            ax.plot([0,1], [sum(y[y==1])/y.shape[0],sum(y[y==1])/y.shape[0]], linestyle = '--', 
             label = 'Blind guess')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend()
            plt.show()
        else:
            print('Error: curve_type unknown!')
    
