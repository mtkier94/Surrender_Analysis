import numpy as np
import seaborn as sns
import pandas as pd
import pickle5 as pickle
import os.path, joblib, time


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# import functions
from functions.sub_surrender_models import Naive_Classifier, ANN_bagging, Logit_model, hpsearch_boost_ann, resample_and_shuffle
from functions.sub_utils import load_ANN_boost_object
from functions.sub_statistical_evaluation import model_evaluation, pq_plot, display_evaluation_curves, evaluate_surrender_rate 
from functions.sub_surrender_profiles import get_model_input
from functions.sub_sklearn_hyperopt import map_and_clean_hparams, hyperopt_get_best_boost_ann
from HPSearch_sklearn import poly_degree_max
from global_vars import path_plots, path_tables

#---------------------
import warnings
warnings.filterwarnings("ignore") # supress eps-PostScript warnings about transparency
#---------------------

pd.set_option('precision', 4)
sns.set()
sns.set_style('ticks')
sns.set_context('paper')

N_epochs = 2000 # at the beginning of script for better visibility
val_share = 0.3 # will be imported from other script to make sure val_share is used consistently

def run_main(surrender_profile: int, bool_load_models: bool, bool_update_plots = True, resampling = 'None'):

    '''
    Run resp. load all models -> create plots and statistics.
    '''

    assert surrender_profile <4
    assert resampling in ['None', 'SMOTE', 'undersampling']

    print(f'Applying {resampling} resampling!')

    # training boolean-values
    bool_load_ann = True # If true: ANN-Boost model will be loaded
    bool_load_ann_boost = True
    bool_load_LR = True
    bool_save_results = True

    # general bagging & boosting hyperparams
    N_bagging = 5 # for ANN
    N_bagging_logit = 10 # for LogisticRegression
    
    # ANN config
    tf_strategy = tf.distribute.MirroredStrategy()
    early_stopping = EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 25, restore_best_weights= True)

    # path variables
    cwd = os.path.dirname(os.path.realpath(__file__))
    path_hparams = os.path.join(cwd,r'profile_{}'.format(surrender_profile))
    path_save_models = os.path.join(os.path.join(cwd,r'profile_{}'.format(surrender_profile)), r'models')
    path_save_models_boosting = os.path.join(path_save_models,r'Boosting')
    path_save_models_bagging = os.path.join(path_save_models,r'Bagging')
    path_data = os.path.join(cwd,r'profile_{}'.format(surrender_profile))

    # adjust path is resampling is applied (optional)
    if resampling == 'SMOTE':
        path_save_models = os.path.join(path_save_models, r'SMOTE')
        path_save_models_bagging, path_save_models_boosting = path_save_models, path_save_models
    elif resampling == 'undersampling':
        path_save_models = os.path.join(path_save_models, r'Undersampling')
        path_save_models_bagging, path_save_models_boosting = path_save_models, path_save_models
    else:
        pass


    # import Training and Test Data
    X_train = pd.read_csv(os.path.join(path_data,'X_train.csv'), index_col= 0)
    X_test = pd.read_csv(os.path.join(path_data,'X_test.csv'), index_col= 0)
    y_train = pd.read_csv(os.path.join(path_data, 'y_train.csv'), index_col= 0).values.flatten()
    y_test = pd.read_csv(os.path.join(path_data,'y_test.csv'), index_col= 0 ).values.flatten()
    X_train_raw = pd.read_csv(os.path.join(path_data,'X_train_raw.csv'), index_col= 0)
    X_test_raw = pd.read_csv(os.path.join(path_data,'X_test_raw.csv'), index_col= 0)

    if surrender_profile == 3:
        # manually check whether time-feature has been scaled
        # Note: although this represents non-stationary noise, its scale might prevent iterative-solvers from converging
        assert np.min(X_train['Time'])>=-1 and np.max(X_train['Time'])<=1, 'X_train["Time"] has not been scaled properly'

    # restrict data to relevant features -> assume proper exploratory data analysis
    features_profile_lst = get_model_input(surrender_profile)
    X_train, X_test = X_train[[el for el in features_profile_lst]], X_test[[el for el in features_profile_lst]]

    # record times, e.g. for mean lapse ratio lateron
    # Note: We keep 'Time' feature in the data set X_raw, as it does not make a difference; raw-data are not used for modeling
    times_train, times_test = X_train_raw['Time'], X_test_raw['Time']
    
    # Load Scaling range used in 'Lapse_data_preparation' for later visualization
    with open(os.path.join(cwd,'dict_range_scale_{}.pkl'.format(surrender_profile)), 'rb') as f:
        dict_range_scale=  pickle.load(f)

        
    # Load beta0 of latent surrender model for later visualization
    with open(os.path.join(cwd,'profile_{}/beta0.pkl'.format(surrender_profile)), 'rb') as f:
        beta0 = pickle.load(f)
        print('Implied beta0: ', beta0)


    # # Construction of Models
    # n_input = X_train.shape[1]
    # Baseline Model - Constant surrender probability
    Baseline = Naive_Classifier(rate=sum(y_train)/len(y_train))
    print('Naive Classifier (Baseline) constructed.')

    ## Improved models using bagging and boosting (No resampling)
    #### model setup

    ###-------------------------  Polynomial LR  ##-------------------------
    print('\n Constructing bagged LR estimator ...')
    if os.path.exists(os.path.join(path_save_models_bagging,r'LR.pkl'))&bool_load_models&bool_load_LR:
        with open(os.path.join(path_save_models_bagging,r'LR.pkl'), 'rb') as file:
            LR_poly_bag = pickle.load(file)
            print('\t LR estimator (bag) loaded.')
    else:
        trials = joblib.load(os.path.join(path_hparams, r'hyperopt_logit.pkl'))
        params = map_and_clean_hparams(trials, 'logit')
        time_LR = time.time()
        LR_poly_bag = Logit_model(params = params, poly_degrees = [poly_degree_max]*X_train.shape[1], N_bag = N_bagging_logit, resampler= resampling).fit(X_train, y_train)
        time_LR = time.time()-time_LR
        print('Training time of Logit Bagging: ', time_LR)
        print('__________________________________________________________________________', '\n')
        with open(os.path.join(path_save_models_bagging,r'LR.pkl'), 'wb') as file:
            pickle.dump(LR_poly_bag, file)


    ###------------------------- Random Forest ##-------------------------
    print('\n Constructing Random Forest estimator ...')
    if os.path.exists(os.path.join(path_save_models_bagging,r'RF.pkl'))&bool_load_models:    
        RF = pickle.load(open(os.path.join(path_save_models_bagging,r'RF.pkl'), 'rb'))
        print('\t ... RF model loaded')
    else:
        trials = joblib.load(os.path.join(path_hparams, r'hyperopt_rf.pkl'))
        params = map_and_clean_hparams(trials, 'rf')
        time_RF = time.time()
        RF = RandomForestClassifier(**params)
        if resampling != 'None':
            X_res, y_res = resample_and_shuffle(X_train, y_train, resample_type= resampling)
        else:
            X_res, y_res = X_train, y_train
        RF.fit(X_res, y_res)
        time_RF = time.time()-time_RF
        print('Training time RF: ', time_RF) 
        pickle.dump(RF, open(os.path.join(path_save_models_bagging,r'RF.pkl'),'wb'))
        print('__________________________________________________________________________', '\n')


    ###------------------------- XGBoost classifier  ##-------------------------
    print('\n Constructing XGB estimator ...')
    if (os.path.exists(os.path.join(path_save_models_boosting,r'xgb.model'))&bool_load_models):
        trials = joblib.load(os.path.join(path_hparams, r'hyperopt_xgboost.pkl'))
        params = map_and_clean_hparams(trials, 'xgboost')
        xgb = XGBClassifier(**params)
        xgb.load_model(os.path.join(path_save_models_boosting,r'xgb.model'))
        print('\t ... XGB loaded')
    else:
        trials = joblib.load(os.path.join(path_hparams, r'hyperopt_xgboost.pkl'))
        params = map_and_clean_hparams(trials, 'xgboost')
        time_xgb = time.time()
        xgb = XGBClassifier(**params)
        if resampling != 'None':
            X_res, y_res = resample_and_shuffle(X_train, y_train, resample_type= resampling)
        else:
            X_res, y_res = X_train, y_train
        xgb.fit(X_res, y_res)
        time_xgb = time.time()-time_xgb
        print('Training time of XGB: ', time_xgb)
        xgb.save_model(os.path.join(path_save_models_boosting,r'xgb.model'))
        print('__________________________________________________________________________', '\n')

    
    ##------------------------- Neural Network - binary crossentropy ##-------------------------
    print('\n Constructing bagged NN (bc) estimator ...')
    trials = joblib.load(os.path.join(path_hparams, r'hyperopt_ann.pkl'))
    params = map_and_clean_hparams(trials, 'ann', surrender_profile= surrender_profile)
    print('\t hyperparams: ', params)
    NN_bc_bag = ANN_bagging(N_models = N_bagging, hparams=params, tf_dist_strat= tf_strategy, resampler = resampling)
    if os.path.exists(os.path.join(path_save_models_bagging,r'NN_bc_bag_0.h5'))&bool_load_ann:
        for i in range(N_bagging):
            # load existing config
            NN_bc_bag.model[i] = load_model(os.path.join(path_save_models_bagging,r'NN_bc_bag_{}.h5'.format(i)), compile=False)
            NN_bc_bag.model[i].compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'] )
            # important: link ensemble-object to new parametrization
            NN_bc_bag.re_init_ensemble()
        print('\t NN (bc) loaded!')
    else:    
        time_NN_bag = time.time()
        # note: resampling-scheme implicitely set at initialization
        NN_bc_bag.fit(X_train = X_train, y_train=y_train,  callbacks = [early_stopping], val_share = val_share, N_epochs = N_epochs)
        time_NN_bag=time.time()-time_NN_bag
        print('Training time of NN_bag: ', time_NN_bag)
        # Save NN-model
        for i in range(N_bagging):
            NN_bc_bag.model[i].save(os.path.join(path_save_models_bagging,r'NN_bc_bag_{}.h5'.format(i)))


    # Perform boosting of ANN
    print('\n Constructing boosted NN (bc) estimator ...')
    if os.path.exists(os.path.join(path_save_models_boosting,r'NN_bc_boost.pkl')) & bool_load_ann_boost:
        with tf_strategy.scope():
            NN_bc_boost= load_ANN_boost_object(path=os.path.join(path_save_models_boosting,r'NN_bc_boost.pkl'))
            print('\t ... NN boosting object loaded!')
    else:
        if resampling == 'None':
            # extract values from HPSearch
            try:
                with tf_strategy.scope():
                    NN_bc_boost, time_NN_boost = hyperopt_get_best_boost_ann(path_hparams)
                    NN_bc_boost.restore_learners() # helper-function which allows ANN-object to be pickled and then to be restored
                    print('\t ... loaded best ANN_boost from the hypersearch within tf-distribution.')
            except:
                NN_bc_boost, time_NN_boost = hyperopt_get_best_boost_ann(path_hparams)
                NN_bc_boost.restore_learners() # helper-function which allows ANN-object to be pickled and then to be restored
                print('\t ... loaded best ANN_boost from the hypersearch.')
            NN_bc_boost.save_object(path=os.path.join(path_save_models_boosting,r'NN_bc_boost.pkl'))
        else:
            # actually train model, based on non-resampling HParams (computational necessity and otherwise HPSearch very seed dependent)
            trials = joblib.load(os.path.join(path_hparams, r'hyperopt_boost_ann.pkl'))
            params = map_and_clean_hparams(trials, 'boost_ann', surrender_profile= surrender_profile)
            print('\t hyperparams: ', params)

            # init model, incl. resampling-scheme
            NN_bc_boost = hpsearch_boost_ann(resampler = resampling, tf_dist_strat= tf_strategy, **params)

            tic = time.time()
            NN_bc_boost.fit(x=X_train, y = y_train, N_epochs=N_epochs, N_batch=params['batch_size'], val_share=val_share, callbacks=[early_stopping], correction_freq=params['n_boosting']+1)
            time_NN_boost = time.time()-tic
            NN_bc_boost.save_object(path=os.path.join(path_save_models_boosting,r'NN_bc_boost.pkl'))

    
    if bool_update_plots:

        if resampling != 'None':
            tag = '_' + resampling
        else:
            tag = ''

        t_start_eval = time.time()
        print('\n Plotting mean surrender rate incl. CIs:')
        evaluate_surrender_rate(times = times_train.append(times_test), X=X_train.append(X_test), y=np.concatenate((y_train,y_test)), data_split = len(np.unique(times_train)),
                                    model_lst = [Baseline, LR_poly_bag, RF, xgb, #lgb_model, 
                                                NN_bc_bag, NN_bc_boost], 
                                    model_names_lst = ['Baseline', 'Logist. Regr.', 'Random Forest',  'XGBoost', #'LGB', 
                                                'NN - bagging','NN - boosting'],
                                    path = os.path.join(path_plots, r'{}_msr_boost{}'.format(surrender_profile, tag)))
        print('Time msr calculation: ', time.time()-t_start_eval)

        t_start_eval = time.time()
        print('\n', 'pq-plot of predicted vs. actual lapse probability:')
        pq_plot(x_scal=X_test, x_raw = X_test_raw,
                                model_lst = [LR_poly_bag, RF, xgb, #lgb_model, 
                                            NN_bc_bag, NN_bc_boost],
                                model_names_lst = ['Logist. Regr.', 'Random Forest',  'XGBoost', #'LGB', 
                                                'NN - bagging','NN - boosting'],
                                beta0=beta0, profile = surrender_profile, path= os.path.join(path_plots, r'{}_pq_boost{}'.format(surrender_profile, tag)))
        print('Time pq calculation: ', time.time()-t_start_eval)

        t_start_eval = time.time()
        print('\n','ROC and RP curve for vanilla estimators')
        display_evaluation_curves(x=X_test, y=y_test, 
                                predictors_lst= [LR_poly_bag, RF, xgb, #lgb_model, 
                                                NN_bc_bag, NN_bc_boost],#, NN_bc], 
                                predictors_name_lst= ['Logist. Regr.', 'Random Forest',  'XGBoost', #'LGB', 
                                                'NN - bagging','NN - boosting'],
                                curve_type= 'both', figsize= (8,3), 
                                path= os.path.join(path_plots, r'{}_roc_boost{}'.format(surrender_profile,tag)))
        print('Time ROC calculation: ', time.time()-t_start_eval)    
        
        t_start_eval = time.time()
        print('Computing statistics for Boosted model')
        eval_boost = model_evaluation(X_train=X_train, X_test=X_test,  X_train_raw= X_train_raw, X_test_raw = X_test_raw,
                                        y_train=y_train, y_test=y_test,
                                        model_lst = [Baseline, LR_poly_bag, RF, xgb, #lgb_model, 
                                                    NN_bc_bag, NN_bc_boost], 
                                        model_names_lst = ['Baseline', 'Logist. Regr.', 'Random Forest',  'XGBoost', #'LGB', 
                                                'NN - bagging','NN - boosting'],
                                        beta0_true= beta0, dict_range_scale=dict_range_scale,
                                        profile = surrender_profile)
        print(eval_boost)
        print('Time df-stats calculation: ', time.time()-t_start_eval)

        if bool_save_results:
            with open(os.path.join(path_tables,r'{}_stats_boost{}.tex'.format(surrender_profile, tag)),'w') as f:
                f.write(eval_boost.to_latex())


    # Create, Update and/or save training times
    # Note: new training times will be updated starting line 453
    try:
        # if file exists and is not empty
        with open(os.path.join(path_tables, r'{}_training_times{}.pkl'.format(surrender_profile,tag)), 'rb') as f:
            dict_times = pickle.load(f)
        print('Training times could be loaded!')
    except:
        dict_times = {'LR': None, 'RF': None, 'XGB': None, 'NN_bag': None, 'NN_boost': None}
        print('Training times had to be re-initialized!')

    # check whether new training times have been recorded
    try: 
        dict_times['RF'] = time_RF
        print('Training time of RF updated!')
    except: pass

    try: 
        dict_times['XGB'] = time_xgb
        print('Training time of XGBoost updated!')
    except: pass

    try: 
        dict_times['LR'] = time_LR
        print('Training time of LR updated!')
    except: pass

    try:
        dict_times['NN_boost'] = time_NN_boost
        print('Training time of NN-boost updated!')
    except: pass  

    try:
        dict_times['NN_bag'] = time_NN_bag
        print('Training time of NN-bag updated!')
    except: pass  

    
    with open(os.path.join(path_tables, r'{}_training_times{}.pkl'.format(surrender_profile, tag)), 'wb') as f:
        pickle.dump(dict_times, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
 
    updatePlots = False
    loadExistingModels = True

    for res_type in ['None', 'SMOTE', 'undersampling']:
        for i in [0,1,2,3]:
            run_main(surrender_profile=i, bool_load_models=loadExistingModels, bool_update_plots=updatePlots, resampling= res_type)