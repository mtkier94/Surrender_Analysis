import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# hp tuning with sklearn
from sklearn.model_selection import cross_val_score, KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from hyperopt import fmin, tpe, STATUS_OK, Trials
import os.path, joblib, sys, time, mlflow

# import functions
from functions.sub_surrender_models import reshape_model_input
from functions.sub_surrender_profiles import get_model_input
from functions.sub_sklearn_hyperopt import get_search_space

poly_degree_max = 4 # see literature references in paper (>4 leads to numerical instability and high multicolinearity)


def run_hpsearch(surrender_profile: int, resampling = 'None'):
    '''
    Load data -> define objective -> get hp_search_space -> run hpsearch for logit, random_forest and xgboost
    '''
    # path variables
    cwd = os.path.dirname(os.path.realpath(__file__))
    path_save_models = os.path.join(cwd,r'profile_{}'.format(surrender_profile))
    path_data = os.path.join(cwd,r'profile_{}'.format(surrender_profile))

    # adjust path is resampling is applied (optional)
    if resampling == 'SMOTE':
        path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'SMOTE')
    elif resampling == 'undersampling':
        path_save_models = os.path.join(os.path.join(path_save_models, r'models'), r'Undersampling')
    elif resampling == 'None':
            pass
    else:
        raise ValueError('resampling type unknown!')


    # import Training and Test Data
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
    X_train= X_train[[el for el in features_profile_lst]]





    # # Construction of Models
    n_input = X_train.shape[1]
    X_train_logit = reshape_model_input(X_train, degrees_lst = [poly_degree_max]*n_input)


    def objective(params):
        '''
        Define the objective of the hparam-search. We use the neg_log_loss (alias bin. cross_entropy) and K=3 fold CF
        '''
        classifier_type = params['type']
        del params['type']
        print(params)
        cross_val_obj = KFold(n_splits=3) # Note: shuffle=True introduces rstate as hparam which then would also be tuned
        tic = time.time()
        if classifier_type == 'rf':
            clf = RandomForestClassifier(**params)
            entropy = cross_val_score(clf, X_train, y_train, cv = cross_val_obj, scoring = 'neg_log_loss', n_jobs=-1).mean()
        elif classifier_type == 'logit':
            clf = LogisticRegression(**params)
            entropy = cross_val_score(clf, X_train_logit, y_train, cv = cross_val_obj, scoring = 'neg_log_loss', n_jobs=-1).mean()
        elif classifier_type == 'xgboost':
            clf = XGBClassifier(**params)
            entropy = cross_val_score(clf, X_train, y_train, cv = cross_val_obj, scoring = 'neg_log_loss', n_jobs=-1).mean()
        else:
            raise ValueError('Unknown classifier_type!')

        # Note: fmin() tries to minimize the objective.        
        # scoring 'neg_log_loss' already returns entropy with negative sign -> has to be cancelled out for minimization
        return {'loss': -entropy, 'status': STATUS_OK, 'eval_time': time.time()-tic}


    
    for name in ['logit', 'rf',  'xgboost']:
        trials = Trials() # start logging information
        search_space = get_search_space(name)

        if (name == 'logit') and (resampling == 'SMOTE'):
            search_space['penalty']= 'l2' # l1 computationally inefficient
            eval_number = 32
            try:
                del search_space['random_state']
            except:
                pass
        else:
            eval_number = 128
        # start the hparam-search
        with mlflow.start_run():
            _ = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest, 
                max_evals= eval_number,
                trials= trials)

        # save hp-tuning result
        joblib.dump(trials, os.path.join(path_save_models, r'hyperopt_{}.pkl'.format(name)))


if __name__ == '__main__':

    # choose resampling type: cmd-line input argument
    try:
        res_type = sys.argv[1] # automated user input to be run with python 'filename' profile_nr
        if res_type not in ['None', 'SMOTE', 'undersampling']:
            raise ValueError('Unknown type of resampling. User input not compatible.')
    except:
        res_type = 'undersampling'

    for i in range(4):
        run_hpsearch(surrender_profile=i, resampling = res_type)