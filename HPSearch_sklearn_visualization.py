import pandas as pd
from hyperopt import plotting
import os.path, joblib, sys
from functions.sub_sklearn_hyperopt import  map_and_clean_hparams
from global_vars import path_tables


def get_hparams_info(classifier: str, resampling = 'None'):
    '''
    Print the obtained choice of hyper-params for all profiles of a classifier.
    '''

    # initialize variable for storing individual hparams
    hparam_all = {}

    print('Classifier: ', classifier)
    print('----------------------------')
    for surrender_profile in range(4):
        cwd = os.path.dirname(os.path.realpath(__file__))
        path_save_models = os.path.join(cwd,r'profile_{}'.format(surrender_profile))


        # adjust path is resampling is applied (optional)
        if resampling == 'SMOTE':
            path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'SMOTE')
            name_tag = '_'+resampling
        elif resampling == 'undersampling':
            path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'Undersampling')
            name_tag = '_' + resampling
        elif resampling == 'None':
            name_tag = ''
        else:
            raise ValueError('resampling type unknown!')

        classifier_type = classifier
        try:
            trials = joblib.load(os.path.join(path_save_models, r'hyperopt_{}.pkl'.format(classifier_type)))
            params = map_and_clean_hparams(trials, classifier_type, surrender_profile)
            print('\t \t runtime: ', trials.best_trial['result']['eval_time'], '\n')

            # store values
            hparam_all[surrender_profile] = params
        except:
            print(f'No HPSearch-object for type {classifier} and surrender_profile {surrender_profile} available yet.')

    df = pd.DataFrame(hparam_all).transpose()
    with open(os.path.join(path_tables,r'hparams_{}{}.tex'.format(classifier_type, name_tag)),'w') as f:
        f.write(df.to_latex())


def plot_hparams_info(classifier: str, resampling='None'):
    '''
    Plot for all profiles of a classifier how the hyper-params populate the search-space.
    '''

    print('Classifier: ', classifier)
    print('----------------------------')
    for surrender_profile in range(4):
        cwd = os.path.dirname(os.path.realpath(__file__))
        path_save_models = os.path.join(cwd,r'profile_{}'.format(surrender_profile))

        # adjust path is resampling is applied (optional)
        if resampling == 'SMOTE':
            path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'SMOTE')
            name_tag = '_'+resampling
        elif resampling == 'undersampling':
            path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'Undersampling')
            name_tag = '_' + resampling
        elif resampling == 'None':
            name_tag = ''
        else:
            raise ValueError('resampling type unknown!')

        classifier_type = classifier
        try:
            trials = joblib.load(os.path.join(path_save_models, r'hyperopt_{}.pkl'.format(classifier_type)))  
            plotting.main_plot_history(trials, title= f'Loss History ({classifier_type+name_tag})')
            # plotting.main_plot_vars(trials)
        except:
            print(f'No HPSearch-object for type {classifier} and surrender_profile {surrender_profile} available yet.')



if __name__ == '__main__':

    # choose resampling type: cmd-line input argument
    try:
        res_type = sys.argv[1] # automated user input to be run with python 'filename' profile_nr
        if res_type not in ['None', 'SMOTE', 'undersampling']:
            raise ValueError('Unknown type of resampling. User input not compatible.')
    except:
        res_type = 'None'

    # look up hparams and respective CV-loss values
    for clf in ['boost_ann', 'ann', 'logit', 'rf', 'xgboost']:
        get_hparams_info(classifier= clf, resampling=res_type)
    
    print('Displaying hparam search process and loss')
    # check how hparams populate feature space and stability of loss-values
    for clf in ['boost_ann', 'ann', 'logit', 'rf', 'xgboost']:
        plot_hparams_info(classifier= clf, resampling=res_type)
    