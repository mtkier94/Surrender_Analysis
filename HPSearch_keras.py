import pandas as pd
import time, os.path, joblib, sys

from hyperopt import fmin, tpe, STATUS_OK, Trials, plotting
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


# import functions
from functions.sub_surrender_models import hpsearch_ann, hpsearch_boost_ann, create_tf_dataset
from functions.sub_surrender_profiles import get_model_input
from functions.sub_sklearn_hyperopt import get_search_space
from global_vars import getDataPath


# global variable
trial_count = 0 


def run_hpsearch(surrender_profile: int, bool_boost: bool, resampling = 'None'):
    '''
    Load data -> define objective -> get hp_search_space -> run hpsearch for ann

    Depending on bool_boost: HPSearch either for standard ann (bool_boost=False) or a boosted ensemble of anns (bool_boost = True)
    '''

    # ANN config
    tf_strategy = tf.distribute.MirroredStrategy()
    early_stopping = EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 25, restore_best_weights= True)
    N_GPUs = tf_strategy.num_replicas_in_sync # later: scale N_batch by number of GPUs

    val_share = 0.3  # Note: prev. val_share = 0.2
    N_epochs = 1000

    # path variables
    path_data = getDataPath(surrender_profile)
    path_save_models = getDataPath(surrender_profile) # will potentially be adjusted in the following (hence, duplicate to path_data)

    # adjust path if resampling is applied (optional)
    if resampling == 'SMOTE':
        path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'SMOTE')
    elif resampling == 'undersampling':
        path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'Undersampling')
    else:
        pass

    # import Training Data
    X_train = pd.read_csv(os.path.join(path_data, r'X_train.csv'), index_col= 0)
    y_train = pd.read_csv(os.path.join(path_data, r'y_train.csv'), index_col= 0).values.flatten()    
    

    if resampling == 'SMOTE':
        X_train,y_train = SMOTE().fit_resample(X_train,y_train)
        X_train,y_train = shuffle(X_train,y_train)
    elif resampling == 'undersampling':
        X_train,y_train = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X_train,y_train)
        X_train,y_train = shuffle(X_train,y_train)
    elif resampling == 'None':
        pass
    else:
        raise ValueError('Input not compatible!')

    # restrict data to relevant features -> assume proper exploratory data analysis
    features_profile_lst = get_model_input(surrender_profile)
    X_train = X_train[[el for el in features_profile_lst]]


    # # Construction of Models
    n_input = X_train.shape[1]
    X_train = X_train.values

    def objective(params):
        '''
        Define the objective of the hparam-search. We use the neg_log_loss (alias bin. cross_entropy) and K=3 fold CF
        '''
        global trial_count

        classifier_type = params['type']
        del params['type']
        print(f'Trial {trial_count}, setting: ', params)
        tic = time.time()
        if (classifier_type == 'ann'):
            try: # check if GPU-devices available for tf-distributed training
                with tf_strategy.scope():
                    clf = hpsearch_ann(**params)
            except:
                clf = hpsearch_ann(**params)

            # Note: for boosting we reduce patience of EarlyStopping as later weak-learners can still compensate
            if N_GPUs > 0:
                train, val = create_tf_dataset(X=X_train, y=y_train, batch_size=params['batch_size']*N_GPUs, val_share=val_share)
                clf.fit(x=train, validation_data = val, epochs = N_epochs, callbacks= [EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 10, restore_best_weights= True)], verbose =2)
            else:
                clf.fit(X_train, y_train, batch_size= params['batch_size']*N_GPUs, epochs = N_epochs, validation_split = val_share, 
                callbacks= [EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 5, restore_best_weights= True)], verbose =2)
            print('\t one round of fitting completed!')

            
            try:
                entropy, acc = clf.evaluate(x=val, batch_size = 1024, verbose = 0)
            except:
                n_val = int((1-val_share)*len(X_train))
                entropy, acc = clf.evaluate(x=X_train[n_val:], y=y_train[n_val:], batch_size = 1024, verbose = 0)
            print('values: loss= ', entropy, ' acc= ', acc)
        elif classifier_type == 'boost_ann':

            try:
                with tf_strategy.scope():
                    clf = hpsearch_boost_ann(**params)
            except:
                clf = hpsearch_boost_ann(**params)

            clf.fit(X_train, y_train, tf_dist = tf_strategy, N_batch= params['batch_size']*N_GPUs, N_epochs = N_epochs, 
                    correction_freq= 100, val_share = val_share, callbacks= [early_stopping]) # Note: corrective step turned off for hpsearch
            print('\t one round of fitting completed!')
            #entropy = cross_val_score(clf, X_train, y_train, cv = cross_val_obj, scoring = 'neg_log_loss', n_jobs=-1).mean()
            n_val = int((1-val_share)*len(X_train))
            entropy = log_loss(y_true=y_train[n_val:], y_pred= clf.predict(X_train[n_val:]))
            print('values: loss= ', entropy)
            
            # use save_object-interface of ANN_Boost object
            clf.save_object(path=os.path.join(path_save_models, r'hps_boost_ann_trial_{}'.format(trial_count)))
            trial_count += 1
        else:
            raise ValueError('Unknown classifier_type!')

        # Note: fmin() tries to minimize the objective.
        return {'loss': entropy, 'status': STATUS_OK, 'eval_time': time.time()-tic}


    trials = Trials() # start logging information
    
    if bool_boost:
        search_space = get_search_space('boost_ann', n_input= n_input)
        eval_nums = 28
    else:
        search_space = get_search_space('ann', n_input= n_input)
        eval_nums = 28

    # start the hparam-search
    _ = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals= eval_nums, # lower number of trials for ann, compared to logit, random_forest and xgboost due to runtime
        trials= trials)


    # plotting.main_plot_history(trials, title= f'Loss History (ann), boost={bool_boost}')
    # plotting.main_plot_vars(trials)

    # save hp-tuning result
    if bool_boost:
        joblib.dump(trials, os.path.join(path_save_models, r'hyperopt_{}.pkl'.format('boost_ann')))
    else:
        joblib.dump(trials, os.path.join(path_save_models, r'hyperopt_{}.pkl'.format('ann')))


if __name__ == '__main__':

    # choose resampling type: cmd-line input argument
    try:
        res_type = sys.argv[1] # automated user input to be run with python 'filename' profile_nr
        if res_type not in ['None', 'SMOTE', 'undersampling']:
            raise ValueError('Unknown type of resampling. User input not compatible.')
    except:
        res_type = 'None'

    for i in range(4):
        run_hpsearch(surrender_profile=i, bool_boost=True, resampling = res_type)