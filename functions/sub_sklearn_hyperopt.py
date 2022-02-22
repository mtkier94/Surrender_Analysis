# hp tuning with sklearn
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
import  hpsklearn
import joblib, os
import numpy as np

from functions.sub_surrender_profiles import get_model_input

def get_search_space(name: str, n_input = None, resampling = None):

    '''
    Return the search space for a hyper-parameter search using hyperopt.

    Inputs:
    -------
        name: either 'logit', 'rf' or 'xgboost' indicating the model for which hyper-parameters have to be created.
        n_input: size of input; only required for ANN

    Outputs:
    --------
        dictionary with the respective hyperparams and ranges for hyperparam-search
    '''

    if name == 'logit':
        # No hpsklearn default values accessible
        hps = {
                'type': 'logit',
                'penalty': hp.choice('penalty', ['l1', 'l2']), # 'lbfgs' does not support 'l1' penalty, 'liblinear' does not support 'none'
                'C': hp.lognormal('C', 0, 1.0),
                'max_iter': 300, 
                #'max_iter':  hp.choice('max_iter', [200]),
                'solver': 'liblinear', # hp.choice('solver', ['liblinear']), # Note: 'lbfgs' or 'saga' fail to converge; 
                'random_state': hp.randint('random_state', 5)
                #'class_weight': hp.choice('class_weight', [None, 'balanced'])
                #'n_jobs': -1, # Note: liblinear does not support n_jobs>1
            }
    elif name == 'rf':
        # hpsklearn default values, see https://github.com/hyperopt/hyperopt-sklearn/blob/master/hpsklearn/components.py -> _trees_hp_space and sub-methods
        hps = hpsklearn.components._trees_hp_space(lambda x: x)
        hps['type'] = 'rf' # add type-indicator, in order to use for all models a single objective function with a simple case-switch
        # slightly alter some default settings of sklearn
        hps['max_depth'] = scope.int(hp.quniform('max_depth', 2, 20, 1))
        hps['max_features'] = hp.pchoice('max_features', [
                                (0.2, 'sqrt'),  # most common choice.
                                (0.1, 'log2'),  # less common choice.
                                (0.1, None),  # all features, less common choice.
                                # a few selected shares of features -> explicitely list them w/c hp.uniform(..) to avoid unknown input-argument, as e.g. max_features.frac
                                (0.1, 0.2),
                                (0.1, 0.3),
                                (0.1, 0.4),
                                (0.1, 0.5),
                                (0.1, 0.6),
                                (0.1, 0.7)
                            ])
        hps['min_samples_leaf'] = 1 #hp.randint('min_samples_leaf', 1, 40, 1)
        hps['random_state'] = hp.randint('random_state', 5)
        try:
            del hps['rstate'] # unknown keyword in default hpsklearn param. space
        except:
            pass
        hps['n_jobs'] = -1 # allow for parallel processing
    elif name == 'xgboost':
        # hpsklearn default values, see https://github.com/hyperopt/hyperopt-sklearn/blob/master/hpsklearn/components.py -> _xgboost_hp_space and sub-methods
        hps = hpsklearn.components._xgboost_hp_space(lambda x: x)
        hps['type'] = 'xgboost'
        hps['max_depth'] = scope.int(hp.quniform('max_depth', 1, 11,1))
        hps['n_estimators'] = scope.int(hp.quniform('n_estimators', 100, 1000, 20))
        hps['random_state'] = hp.randint('random_state', 5)
        try:
            del hps['rstate'] # unknown keyword in default hpsklearn param. space
        except:
            pass
        hps['n_jobs'] = -1 # allow for parallel processing (default)
    elif name == 'ann':
        if type(n_input)==type(None):
            raise ValueError('n_input has to be specified!')

        if type(resampling) == type(None):
            depth_lst = [1,2,3]
        else:
            # include prior information for resampling to reduce run-times
            depth_lst = [2,3]
        hps =  {
            'type': 'ann',
            'n_input': n_input,
            #'epochs': 10, #2000, # Note: set in-script and subject to early stopping
            #'val_share': 0.2, 
            'width_1': scope.int(hp.quniform('width_1', 10, 50, 5)),
            'width_2': scope.int(hp.quniform('width_2', 10, 50, 5)),
            'width_3': scope.int(hp.quniform('width_3', 10, 50, 5)),
            'depth': hp.choice('depth', depth_lst),
            'actv': hp.choice('actv', ['relu', 'sigmoid']),
            'dropout': hp.uniform('dropout', 0, 0.5),
            'lrate': 10**hp.quniform('lrate', -4, -2, 0.5),
            'batch_size': hp.choice('batch_size', [64, 128, 256])
        }  
    elif name == 'boost_ann':
        if type(n_input)==type(None):
            raise ValueError('n_input has to be specified!')
        hps =  {
            'type': 'boost_ann',
            'n_input': n_input,
            'n_boosting': 10, #scope.int(hp.quniform('n_boosting', 10, 25, 5)),
            'width': scope.int(hp.quniform('width', 10, 40, 10)),
            'actv': 'relu', #hp.choice('actv', ['relu', 'sigmoid']),
            'lrate': 10**hp.quniform('lrate', -2, -0.5, 0.5), # compensate for high batch sizes
            'batch_size': hp.choice('batch_size', [64, 128, 256])
        }   
    else:
        raise ValueError('Unknown name input!')

    return hps


def map_and_clean_hparams(trials, clf_type: str, surrender_profile=None):

    '''
    Take the trial object resulting from the hyperopt hyperparam-search, map the numeric (potentially indicator) values to the original hparam-values and clean the values.

    Inputs:
    -------
        trials: Trials-object from hyperopt
        clf_type: string, either 'rf', 'logit',  'xgboost' or 'ann'

    Outputs:
    --------
        dictionary of params, which can be used in the line of create_model(**params) for RandomForestClassifier, LogisticRegression and XGBoostClassifier.
    '''

    # initial input-check
    assert clf_type in ['rf', 'logit',  'xgboost', 'ann', 'boost_ann']
    # initial input-check
    if (clf_type == 'ann') or (clf_type == 'boost_ann'):
        assert type(surrender_profile) == type(0)

    # Note: ann-type requires n_input argument
    if (clf_type == 'ann') or (clf_type == 'boost_ann'):
        n_input = len(get_model_input(surrender_profile))
    else:
        n_input = None

    params = trials.best_trial['misc']['vals']
    # print(params)
    for hps in params.keys():
        params[hps] = params[hps][0] # remove list format from hparam
    search_space = get_search_space(clf_type, n_input)
    del search_space['type'] # only required for objective function, not for creating classifiers
    params = space_eval(search_space, params) # maps indicators for hp.choice back to actual value, e.g. string
    #print('mapped params: ', params, '\n')

    # delete some redundant information, i.e. constant hyper-params

    #general paramters
    try: del params['random_state'] # subject to hp-search, to check for seed dependency of results
    except: pass

    try: del params['n_jobs'] # using -1
    except: pass

    try: del params['seed'] 
    except: pass

    # random forest - constant hparams
    try: del params['verbose'] # default at False
    except: pass

    try: del params['oob_score'] # default at False
    except: pass

    # xboost - constant hparams
    try: del params['base_score'] # default at 0.5
    except: pass

    try: del params['max_delta_step'] # default 0 -> no constraint
    except: pass

    try: del params['scale_pos_weight'] # default at 1
    except: pass

    return params

def hyperopt_get_best_boost_ann(path_profile):
    '''
    We have a summary of the hparam-tuning in the hyperopt-object. Additionally, we saved the ANN_boost objects for each trial, to reduce effort for re-training models.
    '''

    # 1) get trial object
    trials = joblib.load(os.path.join(path_profile, r'hyperopt_{}.pkl'.format('boost_ann')))

    vals = trials.results # list of trial-values; dictionaries with keys loss, status and eval_time
    best_id = np.argmin([ vals[i]['loss'] for i in range(len(vals))])

    # load best boost_ann model
    clf_best = joblib.load(os.path.join(path_profile, r'hps_boost_ann_trial_{}.pkl'.format(best_id)))

    # save model
    joblib.dump(clf_best, os.path.join(os.path.join(path_profile, r'models'), r'NN_bc_boost.pkl'))

    # return ANN_bost object and its training time
    return clf_best, vals[best_id]['eval_time']