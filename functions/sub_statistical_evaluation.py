import numpy as np 
import matplotlib
matplotlib.use('Agg') # avoid Error 'Tcl_AsyncDelete: async handler deleted by the wrong thread'
from matplotlib import pyplot as plt

import sklearn
import pandas as pd 
import seaborn as sns 
from scipy import stats
import lightgbm as lgb

from functions.sub_surrender_profiles import get_age_coeff, get_duration_coeff, get_duration_elapsed_coeff, get_duration_remain_coeff, \
                                    get_prem_freq_coeff, get_prem_annual_coeff, get_time_coeff, get_risk_drivers
from functions.sub_simulate_events import get_surrender_prob


def display_confusion_matrix(matrix):

    '''
    Display confusion matrix sklear.metrics.confusion_matrix()
    '''
    return pd.DataFrame(matrix, columns= [r'pred.: 0', r'pred.: 1'], index= [r'obs.: 0', r'obs.: 1'])



def display_evaluation_curves(x, y, predictors_lst, predictors_name_lst, curve_type = 'ROC', figsize = None, path=None, bool_plot=False):
    
    '''
    curve_type: Display ROC, Recall vs Precision Curve (or both)

    Inputs:
    -------
        x: data
        y: target values
        predictors_lst: list with models that take model.predict_proba()
        curve_type: 'ROC' or 'RP' or 'both'
    Outputs:
    --------
        plots
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
        y_pred = predictors_lst[i].predict_proba(x)[:,-1]

        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true= y, y_score= y_pred)
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_true=y, probas_pred = y_pred)
            
            
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
        if path != None:
                plt.savefig(path+r'.png', bbox_inches='tight')
                plt.savefig(path+r'.eps', bbox_inches='tight')
                plt.savefig(path+r'.pdf', bbox_inches='tight')
        if bool_plot:
            plt.show()
        else:
            plt.close()
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
            if path != None:
                plt.savefig(path+r'.png', bbox_inches='tight')
                plt.savefig(path+r'.eps', bbox_inches='tight')
            if bool_plot:
                plt.show()
            else:
                plt.close()
        
        elif curve_type == 'RP':
            for i in range(n_models):
                ax.plot(recall[i], precision[i], 
                        label = predictors_name_lst[i] +': AUC = {}'.format(np.round(auc_prec_rec[i], decimals=4)))

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend()
            if path != None:
                plt.savefig(path+r'.png', bbox_inches='tight')
                plt.savefig(path+r'.eps', bbox_inches='tight')
            if bool_plot:
                plt.show()
            else:
                plt.close()
    
        else:
            print('Error: curve_type unknown!')



def function_exp_optional(val, bool_log):

    '''
    Function that returns the exponential of an array, unless bool_log == True
    auxilary function for display_implied_surrender_profile()
    '''
    if bool_log:
        return val
    else:
        return np.exp(val)


def single_eval(y_true, prob_pred, prob_true, pred_threshold):
        return [sklearn.metrics.accuracy_score(y_true = y_true, y_pred = (prob_pred>pred_threshold)),
                                 sklearn.metrics.f1_score(y_true = y_true, y_pred = (prob_pred>pred_threshold)),
                                 sklearn.metrics.log_loss(y_true = y_true, y_pred= prob_pred),
                                 #np.mean(np.abs(prob_true-prob_pred)),
                                 sklearn.metrics.mean_absolute_error(y_true=prob_true, y_pred=prob_pred),
                                 np.var(np.abs(prob_true-prob_pred))]

def model_evaluation(X_train, X_test, X_train_raw, X_test_raw,  y_train, y_test, model_lst, model_names_lst, beta0_true, 
                     dict_range_scale, pred_threshold = 0.5, profile = 0):
    
    '''
    Provide a table summary for performance of several models.
    Evaluation w.r.t. accuracy, F1-score, Brier/MSE, Bin. crossentropy
    
    Inputs:
    -------
        x:  Input Data
        y:  Target values
        model_lst:  list of models to evaluate
        model_names_lst: list of respective models' names
        beta0_true: true beta0 coeff of meta-model (LR); allows to compute the true surrender prob
        dict_range_scale: required to obtain raw X-values, which are used to compute the true surrender prob
        pred_threshold: threshold to transform predicted prob. into label
        profile:    Surrender profile to obtain true distribution from

    Outputs:
    --------
        DataFrame with 'Acc.', 'F1', 'Brier', 'Crossentropy', r'$E_X(Lapses)$', 'Lapses', 'MAE(p,q)' measures for train- and test-set for all provided models.
    '''
    
    df = pd.DataFrame(data = None, columns= [['Train.','Train.','Train.','Train.', 'Train.',#,'Train.',
                                              'Test','Test','Test','Test','Test'],#,'Test'],
                                             ['acc.', r'F_1', 'crossentropy', #'KL Div.', 'KL',  r'$E(lapses)$', 'lapses',  # Note: E(Lapses) vs actual lapses is evaluated by mean-surrender confidence plots
                                             r'mae$(p,\hat{p})$', r'Var$(p-\hat{p})$',
                                              'acc.', r'F_1', 'crossentropy', #'KL Div.','KL',r'$E(lapses)$', 'lapses',
                                              r'mae$(p,\hat{p})$', r'Var$(p-\hat{p})$']]) 
    df.columns.names = ['Data', 'Metric'] 
    
    # Latent surrender model works with raw/ un-scaled data
    # X_train_raw = scale_DATA(df = X_train, dict_range_scale=dict_range_scale, reverse_bool=True)
    # X_test_raw = scale_DATA(df = X_test, dict_range_scale=dict_range_scale, reverse_bool=True)
    

    # Get true prob./ distribution
    # Note: Here we have to work with raw input features, i.e. non-scaled versions
    # Also: Re-transform dummy variable Premium_freq_{x}
    prob_true_train = get_surrender_prob(df = pd.concat([X_train_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(X_train['Premium_freq_0']*0+ (X_train['Premium_freq_1']==1).astype('int')+\
                                                               12*((X_train['Premium_freq_0']==-1)& (X_train['Premium_freq_1']==-1)), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                                   profile_nr= profile, beta0= beta0_true, rand_noise_var= 0)  
    prob_true_test = get_surrender_prob(df = pd.concat([X_test_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(X_test['Premium_freq_0']*0+(X_test['Premium_freq_1']==1).astype('int')+\
                                                               12*((X_test['Premium_freq_0']==-1)&(X_test['Premium_freq_1']==-1)), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                                   profile_nr= profile, beta0= beta0_true, rand_noise_var= 0) 


    
    for i in range(len(model_lst)):
        if isinstance(model_lst[i], lgb.basic.Booster) == False:
            prob_train = model_lst[i].predict_proba(X_train)[:,-1]
            prob_test = model_lst[i].predict_proba(X_test)[:,-1]
        else:
            prob_train = model_lst[i].predict(X_train)
            prob_test = model_lst[i].predict(X_test)        
        

        df.loc[model_names_lst[i],:] = single_eval(y_true = y_train, prob_pred = prob_train, 
                                                  prob_true = prob_true_train, pred_threshold = pred_threshold) + \
                                        single_eval(y_true = y_test, prob_pred = prob_test, 
                                                  prob_true = prob_true_test, pred_threshold = pred_threshold)
        
    # Replace 'inf' by N.A., e.g. for KL-div which is inf if we predict zero probability
    df = df.applymap(lambda x: 'N.A.' if x == np.inf else x )
    
    return df



def pq_plot(x_scal, x_raw, model_lst, model_names_lst, beta0,
            bool_correction_resampling = False, correction_rate = None,
            profile = 0, fig_size = (12,5), path=None, bool_plot=False, bool_resampling = False):
    
    '''
    Create a p-q-plot of each classifier w.r.t. the latent, true surrender model
    
    Parameters:
    -----------
        x:        Input Data
        y:        Target values
        model_lst: list of models to evaluate
        model_names_lst: list of respective models' names
        resampling_correction_factor: Factor to compensate for distortion of P(Y=1) by resampling
        bool_correction_resampling: Correction for altered baseline P(Y=1) != P^S(Y=1), see 'The foundation of cost-sensitive Learning' (2001). C. Elkan
        correction_rate: Target baseline P(Y=1) to be corrected to, if bool_correction_sampling is True. Resampling changes baseline hazard to P^S(Y=1)=0.5
        profile:  Surrender profile to obtain true distribution from
        
    Return:
    -------
        Nothing; generates and displays plot
    '''

    parameters = {'axes.labelsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14}
    plt.rcParams.update(parameters)

    col_lst = ["green", "blue", "orange", "red", "purple", 'gray', 'cyan']



    # Get true prob./ distribution
    # Note: Here we have to work with raw input features, i.e. non-scaled versions
    #       And we have to convert the binary Premium frequency dummies back to its raw version with vals \in {0,1,12}
    prob_true = get_surrender_prob(df = pd.concat([x_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(x_scal['Premium_freq_0']*0+ (x_scal['Premium_freq_1']+1)/2+\
                                                               12*((x_scal['Premium_freq_0']==-1)&(x_scal['Premium_freq_1']==-1)), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                                   profile_nr= profile, beta0= beta0, rand_noise_var= 0) 
    
    # transform pandas.series object to numpy.array
    prob_true = prob_true.values
    # record max value incl. buffer for display
    max_prob_true = min(prob_true.max()*1.25,1.03) # 1.03 to allow for some white space in plot
    max_prob_est = 0.5 # to be updated iteratively for all classifiers below

    #print('In pqplot fct.: prob_true.shape = ', prob_true.shape)
    ncols = 3    
    _, ax = plt.subplots(nrows=int(np.ceil(len(model_lst)/ncols)), ncols = ncols, sharex=True, sharey=True, figsize= fig_size)
    ax=ax.flatten()    
  
    for i in range(len(model_lst)):
        # if isinstance(model_lst[i], lgb.basic.Booster):
        #     prob = model_lst[i].predict(x_scal) 
        # else:
        prob = model_lst[i].predict_proba(x_scal)[:,-1]

        max_prob_est = max(prob.max(), max_prob_est)

        # Optional correction
        if bool_correction_resampling:
            prob = correction_rate/(correction_rate+(1-prob)*(1-correction_rate)/prob)
        
        # Display the cond. prob. for all models in one plot
        ax[i].plot([0,1],[0,1], color = 'black', linestyle = ':')
        
        #print('In pqplot fct.: prob.shape = ', prob.shape)


        # Create a p-q-plot for each individual model
        # for .png-format
        sns.scatterplot(x=prob_true, y = prob,  s = 10, color= col_lst[i], alpha = 0.3, #0.2, # alpha val cannot be stored for .eps-format
                        label = model_names_lst[i], ax = ax[i], marker = "+") 

        # for .eps-format
        # color_adj = colorAlpha_to_rgb(colors= col_lst[i], alpha = 0.05)
        # print(color_adj)
        # sns.scatterplot(x=prob_true, y = prob,  s = 10, color= color_adj[0], #0.2, # alpha val cannot be stored for .eps-format
        #                 label = model_names_lst[i], ax = ax[i], marker = "+") 
        
        # check mae value
        # print('shapes, prob: ', prob.shape)
        # print(type(model_lst[i]), ' - mae: ', np.mean(np.abs(prob_true-prob)))

        # if type(model_lst[i] == ANN_boost):
        #     sns.jointplot(x = prob_true, y = prob, s = 10, color= col_lst[i], alpha = 0.1, # alpha val cannot be stored for .eps-format
        #                 label = model_names_lst[i], ax = ax[i])
        # else:
        #     sns.scatterplot(x=prob_true, y = prob,  s = 10, color= col_lst[i], alpha = 0.1, # alpha val cannot be stored for .eps-format
        #                 label = model_names_lst[i], ax = ax[i], marker = "+") 
        
        if (i>=len(model_lst)-ncols): # Note: 0-indexing of ax[i]
            ax[i].set_xlabel(r'$p(1|x)$')
        if i%ncols ==0:
            if bool_resampling:
                ax[i].set_ylabel(r'$\hat{p}^{S}(1|x)$')
            else:
                ax[i].set_ylabel(r'$\hat{p}(1|x)$')

    # adjust range of xlabeling in accordance with the observed range of predictions
    max_prob_est = min(max_prob_est*1.25, 1.03) # as some white space for nicer display
    for i in range(len(model_lst)):
        ax[i].set_xlim((-0.03, max_prob_true))
        ax[i].set_ylim((-0.03,max_prob_est))
        # turn on legend per axis
        ax[i].legend()
    
    
    if len(model_lst)%ncols != 0:
        # remove white box of empty legend
        # ax[-1].get_legend().remove() # Note: Error: 'NoneType' object has no attribute remove; Yet white box still displayed ..
        # remove empty axis
        ax[-1].set_axis_off()
        
    # Note: manually set xticks of last plot in 1st row for nicer visual presentation
    ax[ncols-1].xaxis.set_tick_params(which='both', labelbottom=True)
    

    #plt.legend(fontsize = 'large') # turn on legend per axis -> avoid empty legend for empty plots
    #plt.tight_layout()

    if path != None:
        plt.savefig(path+r'.png', bbox_inches='tight', dpi = 400)
        # plt.savefig(path+r'.eps', bbox_inches='tight') # note: eps does not allow for transparency, which implicitely allows us to indicate the distribution of the data
        # plt.savefig(path+r'.pdf', bbox_inches='tight')
    if bool_plot:
        plt.show()
    else:
        plt.close()
            


def evaluate_surrender_rate(times, X, y, model_lst, model_names_lst, data_split, do_label_pred = False, bool_ci = True, 
                            ci_alpha=0.05, path=None, bool_plot = False, bool_logscale = False):
    '''
    Display the mean surrender rate over time, i.e. how many contracts surrender in year t.
    Use Lindeberg-Levy CLT to provide a (1-ci_alpha)% CI.
    Note: The surrender event of each contract is Bernoulli with p_x depending on the contract settings x.
            It is trivial to prove the Lindeberg condition. Then holds
            
            frac{1}{n} sum_{i=1}^{n} Y_i -> mathcal{N}(1/n sum p_i, sum (p_i*(1-p_i)/n)^2)

    Inputs:
    -------
        times: Provides time index for all contracts X; used to infer how many contracts are active at t in times
        X:  Contracts
        y:  Observed (binary) target values
        model_lst: list contraining calibrated models
        model_names_lst: names for labeling model types
        data_split: times index at which train set ends
        bool_ci: boolean whether to provide a 95%-CI for each model at each time t in times

    '''
    # t_count indicates the number of contracts active per time period
    t_count = [0]+list(np.cumsum(np.unique(times, return_counts=True)[1]))
    pred = []
    col_lst = ['brown', 'green', 'blue', 'orange', 'red', 'purple', 'grey', 'c']

    fontsize_axis = 14
    fontsize_label = 16

    # track range for later train-test-split vlines()
    y_lim = [min([y[t_count[j-1]:t_count[j]].mean() for j in range(1,len(t_count))]),max([y[t_count[j-1]:t_count[j]].mean() for j in range(1,len(t_count))])]
    # true, observable surrender rates over calendar-time
    plt.plot([y[t_count[j-1]:t_count[j]].mean() for j in range(1,len(t_count))], color='black', marker= 'x', linestyle='None', label='true rate')
    plt.xlabel('time t', fontsize = fontsize_label)
    plt.ylabel('surrender rate', fontsize = fontsize_label)
    plt.xticks(fontsize=fontsize_axis)
    plt.yticks(fontsize=fontsize_axis)

    # Compute probability estimates per model for all time periods
    for model in (model_lst):
        if isinstance(model, lgb.basic.Booster):
            pred.append([model.predict(X[t_count[j-1]:t_count[j]]) for j in range(1,len(t_count))])
        else:
            pred.append([model.predict_proba(X[t_count[j-1]:t_count[j]])[:,-1] for j in range(1,len(t_count))])
            
        
    for i in range(len(model_lst)):
        if do_label_pred:
            mu = np.array([(t_pred>0.5).mean() for t_pred in pred[i]])
        else:
            mu = np.array([t_pred.mean() for t_pred in pred[i]])
        y_lim = [min(y_lim[0], min(mu)), max(y_lim[1],max(mu))]
        plt.plot(mu, color=col_lst[i], linestyle='-', label=model_names_lst[i])
        if bool_ci & (do_label_pred==False):
            # 1-ci-alpha - confidence intervals
            var = np.array([sum(t_pred*(1-t_pred))/(len(t_pred.flatten())*(len(t_pred.flatten())-1)) for t_pred in pred[i]]) # note: unbiased sample variance with correction n/n-1
            # std. normal quantile, two-sided confidence intervals (!)
            quantile = stats.norm.ppf(1-ci_alpha/2)
            #print(model_names_lst[i], 2*np.sqrt(var))
            plt.plot(mu+quantile*np.sqrt(var), color=col_lst[i], linestyle='', marker='1')
            plt.plot(mu-quantile*np.sqrt(var), color=col_lst[i], linestyle='', marker='2')

    # replot true values for visual display
    plt.plot([y[t_count[j-1]:t_count[j]].mean() for j in range(1,len(t_count))], color='black', marker= 'x', linestyle='None')
    # indicate train-test-split of data
    plt.vlines(x = data_split+0.5,ymin=.9*y_lim[0], ymax= 1.1*y_lim[1], colors='grey', linestyles='--')
    if bool_logscale:
        plt.yscale('log')
    plt.legend(fontsize = 10)
    if path != None:
                plt.savefig(path+r'.png', bbox_inches='tight', dpi = 400)
                plt.savefig(path+r'.eps', bbox_inches='tight', dpi = 400)
    if bool_plot:
        plt.show()
    else:
        plt.close()




##########################
##### LEGACY CODE ########


def compare_target_distribution(x_scal, x_raw, model_lst, model_names_lst, beta0,
                                profile = 0, fig_size = (12,5), bool_plot=False):
    
    '''
    Visualize the underlying true distribution of surrender profile {x} vs distribution(s) implied by model choices
    
    Input
    \t x: \t\t Input Data
    \t y: \t\t Target values
    \t model_lst: \t list of models to evaluate
    \t model_names_lst: list of respective models' names
    \t resampling_correction_factor: Factor to compensate for distortion of P(Y=1) by resampling
    \t profile: \t \t Surrender profile to obtain true distribution from
    '''
    
    col_lst = ['green', 'blue', 'orange', 'red', 'purple', 'grey', 'c']
    
    
    
    # Get true prob./ distribution
    # Note: Here we have to work with raw input features, i.e. non-scaled versions
    #       And we have to convert the binary Premium frequency dummies back to its raw version with vals \in {0,1,12}
    prob_true = get_surrender_prob(df = pd.concat([x_raw.drop(['Premium_freq_0', 'Premium_freq_1'],axis=1),
                                                  pd.DataFrame(x_scal['Premium_freq_0']*0+\
                                                               x_scal['Premium_freq_1']+\
                                                               12*(x_scal['Premium_freq_0']+\
                                                               x_scal['Premium_freq_1']==0), 
                                                               columns = ['Premium_freq'])],
                                                  axis = 1), 
                               
                                   profile_nr= profile, beta0= beta0, rand_noise_var= 0) 
    
    _, ax = plt.subplots(nrows=1, ncols = 1, figsize= fig_size)    
    ax.set_xlim((0,1))
    sns.distplot(prob_true, norm_hist=True, hist=False, color='black', label = 'True value', ax = ax)
    
    
    for i in range(len(model_lst)):
        prob = model_lst[i].predict_proba(x_scal)[:,-1]
        # Display the cond. prob. for all models in one plot
        sns.distplot(prob, norm_hist=True, hist=False, color= col_lst[i], label = model_names_lst[i], ax = ax) 

    ax.set_xlabel(r'$p_{Y|X}(1|x)$')
    ax.set_ylabel('Kernel density estimate')
    if bool_plot:
        plt.show()
    else:
        plt.close()


def display_implied_surrender_profile(model_lst, model_names_lst, beta0_true = 0,
                                      model_predictors_lst = ['Age', 'Face_amount', 'Ins_dur', 'Ins_dur_remain',
                                                              'Premium_annual', 'Premium_freq_0', 'Premium_freq_1'],
                                      scaling_range_dict = None, log_odd_ratio = True,
                                      bool_rug_plot = False, df_rug_data = None, x_train=None, y_train=None,
                                      true_profile = 0, n_columns = 3,
                                     fig_size = (10,8)):
    
    '''
    ##! depreciated
    Aim: Analyse the underlying effect of features on surrender activity, implied by some model (eg logit, ANN,...)

    Inputs:
    -------
        model_lst:  list of models for which to predict surrender probability
        model_type:     String, encoding model types as e.g. 'Logit', 'ANN'
        scaling_range_dict:     Dictionary with range of features, allows to rescale scaled feature values to an absolute scale
        log_odd_ratio:   Boolean, display either 'odd-ratios' or log-value
        bool_rug_plot:  Boolean. Indicate marginal distribution of training data as rug plot
        (x_train,y_train): Training data to indicate marginal distributions in rugplot        
        true_profile:   Integer, indicating true surrender profile  

    Outputs:
    --------
        plot

    '''
    
    features_profile_lst = get_risk_drivers(true_profile)
    col_lst = ['green', 'blue', 'orange', 'red', 'purple', 'grey', 'c', 'brown', 'purple']
    
    N_simulations = 50
    x_axis_scaled = np.linspace(start = -1, stop=1, num = N_simulations )
    N_features = len(model_predictors_lst)
    N_profile_features = len(features_profile_lst)
    # record baseline hazards of models in model_lst
    beta0 = {}
    
    # setup canvas for plot
    _, ax = plt.subplots(nrows= int(np.ceil(N_profile_features/n_columns)), ncols = n_columns, figsize= fig_size)
    ax = ax.flatten()
    
    count = -1
    for model in model_lst:
        # count the index of the current model
        count+= 1
        # Note: [:,-1] Notation works for Logit and ANN
        p0 = model.predict_proba(pd.DataFrame(data = np.zeros(shape = (1,N_features))-1, columns= model_predictors_lst))[:,-1] # matrix with '-1'-elements
        
        beta0[count] = np.log(p0/(1-p0))
        # Compute baseline beta0
        # Note: DecisionTreeClassifier does not posses interpretable baseline hazard
        #if type(model) != sklearn.tree._classes.DecisionTreeClassifier:
        #    beta0[count] = np.log(p0/(1-p0))
        #else:
        #    beta0[count] = 0

        count_ax = -1
        for feature in features_profile_lst:
            # count which axis to draw plot in
            count_ax +=1
            # set all features to their baseline
            model_input = pd.DataFrame(data = np.zeros(shape = (N_simulations,N_features)), 
                                       columns= model_predictors_lst)
            if feature != 'Premium_freq':
                # adjust the respective, single feature to extract its influence on the predicted probability
                model_input[feature] = x_axis_scaled
                # raw values for plotting
                x_axis = (x_axis_scaled+1)/2*(scaling_range_dict[feature]['max']-scaling_range_dict[feature]['min'])+   scaling_range_dict[feature]['min']
                # model prediction
                prob_val = model.predict_proba(model_input)[:,-1]
                
                # Plot Odd-ratio
                # display true (known) surrender profile; only once, not for every iteration of models_lst
                if count == 0:
                    if bool_rug_plot:
                        sns.rugplot(x_train.loc[y_train==1,feature], alpha = .1, ax = ax[count_ax])
                    if feature == 'Age':
                        ax[count_ax].step(x_axis,function_exp_optional(get_age_coeff(x_axis, profile = true_profile),
                                                                      bool_log = log_odd_ratio), color = 'black', label = 'true value')
                    elif feature == 'Duration':
                        ax[count_ax].step(x_axis,function_exp_optional(get_duration_coeff(x_axis, profile = true_profile),
                                                                      bool_log = log_odd_ratio), color = 'black', label = 'true value')
                    elif feature == 'Duration_elapsed':
                        ax[count_ax].step(x_axis,function_exp_optional(get_duration_elapsed_coeff(x_axis, profile = true_profile),
                                                                      bool_log = log_odd_ratio), color = 'black', label = 'true value')
                    elif feature == 'Duration_remaining':
                        ax[count_ax].step(x_axis,function_exp_optional(get_duration_remain_coeff(x_axis, profile = true_profile),
                                                                      bool_log = log_odd_ratio), color = 'black', label = 'true value')
                    elif feature == 'Premium_annual':
                        ax[count_ax].step(x_axis,function_exp_optional(get_prem_annual_coeff(x_axis, profile = true_profile),
                                                                      bool_log = log_odd_ratio), color = 'black', label = 'true value')
                    elif feature == 'Time':
                        ax[count_ax].step(x_axis,function_exp_optional(get_time_coeff(x_axis, profile = true_profile),
                                                bool_log = log_odd_ratio),  color = 'black', label = 'true value')
                    else:
                        print('Feature {} unknown!'.format(feature))
                        
                # Exclude effect of intercept to make this comparable to simulated lapse profiles
                # log(p/1-p) = beta0+ beta'X = beta0 +beta_i*x_i if x_j=0 for all j!=i
                # -> beta_i*x_i = log(p(1-p)) - beta0
                # Note: Calculation of p is independent of the model in use
                ax[count_ax].plot(x_axis, function_exp_optional(np.log(prob_val/(1-prob_val))-beta0[count],
                                                               bool_log= log_odd_ratio), 
                                  color = col_lst[count%len(col_lst)], label = model_names_lst[count])                       
                
                ax[count_ax].set_ylabel((1-log_odd_ratio)*'Odd-ratio'+ log_odd_ratio*r'$\beta_i x_i$')
                ax[count_ax].set_xlabel(feature)

                if bool_rug_plot:
                    sns.rugplot(x_train.loc[y_train==1,feature], alpha = .1, ax = ax[count_ax]) 
                
                # Plot legend only once
                if count_ax == (n_columns-1):
                    ax[count_ax].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                
            else: # case 'Premium_freq'
                #odd ratios
                or_mthly = 1*(log_odd_ratio==False) # baseline
                or_ann = -1
                or_single = -1


                for i in range(2):
                    model_input[feature+'_{}'.format(i)] = 1
                    prob_val = model.predict_proba(model_input)[0,-1]
                    if i == 0:
                        or_single = function_exp_optional(np.log(prob_val/(1-prob_val))-beta0[count],
                                                               bool_log= log_odd_ratio)
                    else:
                        or_ann = function_exp_optional(np.log(prob_val/(1-prob_val))-beta0[count],
                                                               bool_log= log_odd_ratio)
                    #reset input    
                    model_input[feature+'_{}'.format(i)] = -1
                    
                # plot true profile only once (not every time we evaluate a new prediction model)
                if count == 0: 
                    ax[count_ax].plot(['single', 'annual', 'monthly'],
                             function_exp_optional(get_prem_freq_coeff(prem_freq= np.array([0,1,12]), 
                                                                       profile = true_profile),
                                                  bool_log= log_odd_ratio),'o', 
                             color = 'black', label = 'true value')
                # plot premium freq profile of prediction model (with a set color)
                ax[count_ax].plot(['single', 'annual', 'monthly'], 
                                  [or_single, or_ann, or_mthly], '+', markersize = 8, 
                                  color = col_lst[count%len(col_lst)], label = model_names_lst[count])

                ax[count_ax].set_xlabel('Premium Payment')
                ax[count_ax].set_ylabel((1-log_odd_ratio)*'Odd-ratio'+ log_odd_ratio*r'$\beta_i x_i$')
                if count_ax == (n_columns-1):
                    ax[count_ax].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
    # plot baseline hazards beta0
    ax[len(features_profile_lst)].bar(x=0, height = beta0_true, color= 'black', label = 'True value') #.hist(beta0,color = col_lst[0:len(model_lst)])
    for i in range(len(model_lst)):
        ax[len(features_profile_lst)].bar(x=i+1, height = beta0[i], color= col_lst[i%len(col_lst)], 
                                                              label = model_names_lst[i])
    ax[len(features_profile_lst)].set_ylabel(r'$\beta_0$')
    ax[len(features_profile_lst)].set_xlabel('Model type')
    #ax[len(features_surrender_profile_lst)].set_xticklabels([])
    ax[len(features_profile_lst)].tick_params(axis='x',which='both', 
                                                        bottom=False,top=False,labelbottom=False)
    
    # Set non-used axis as not-visible
    if (len(features_profile_lst)%n_columns)==1:
        ax[len(features_profile_lst)+1].axis('off')     
    
    plt.tight_layout()
    plt.show()

