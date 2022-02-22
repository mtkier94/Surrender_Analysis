import numpy as np
import pandas as pd
import time, copy
import pickle as pickle 

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import expit
import matplotlib.pyplot as plt


from sklearn.ensemble import AdaBoostClassifier 
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.eager.context import num_gpus

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import  RandomUnderSampler

from sub_utils import exp_decay_scheduler, keras_count_nontrainable_params, resample_and_shuffle, create_tf_dataset, reshape_model_input


class Naive_Classifier:
    
    '''
    Create naive baseline classifier, that assigns a constant surrender rate, regardsless of the feature configuration.
    
    Parameters
    ----------
        rate: Constant probability prediction
    '''
    
    def __init__(self, rate, ):
        self.rate = rate
        
    def predict_proba(self, X):
        pred = np.zeros(shape=(len(X),2))
        pred[:,0] = 1-self.rate
        pred[:,1]= self.rate
        return pred
    
    def predict(self, X):
        return self.predict_proba(X)
    
    def predict_class(self, X, threshold=0.5):
        return self.predict_proba(X)>threshold


def create_ann(widths: list, actv: list, dropout: float, n_input: int, lrate: float):
    '''
    Create individual ANNs for ANN_bagging.
    '''
    model = Sequential()
    for j in range(len(widths)):
        if j==0: # Specify input size for first layer
            model.add(Dense(units = widths[j], activation = actv[j], input_dim = n_input))
        else:
            model.add(Dense(units = widths[j], activation = actv[j]))
        if j<(len(widths)-1): # No dropout after output layer
            model.add(Dropout(rate = dropout))

    model.compile(loss = 'binary_crossentropy', metrics= ['acc'], optimizer=Adam(lr=lrate))
    return model


def hpsearch_ann(**params):
    '''
    Use params obtained via a hpsearch to create an ann.
    This function is a helper function, to simplify the varying notation.
    '''

    widths = [params['width_{}'.format(1+i)] for i in range(params['depth'])]+[1]
    actv = params['depth']*[params['actv']]+['sigmoid']
    dropout = params['dropout']
    n_input = params['n_input']
    lrate = params['lrate']
    model = create_ann(widths=widths, actv=actv, dropout=dropout, n_input= n_input, lrate = lrate)
    return model


def hpsearch_boost_ann(resampler ='None', tf_dist_strat = None, **params):
    '''
    Helper function to map params to ANN_boost object initialization.
    '''

    N_boosting = params['n_boosting']
    n_input = params['n_input']
    boost_width = params['width']
    actv = params['actv']
    lrate = params['lrate']

    return ANN_boost(N_models = N_boosting, N_input = n_input, width=boost_width, act_fct=actv, lr = lrate, resampler = resampler, tf_dist_strat=tf_dist_strat)



class Logit_model:

    '''
    A bagged version of the sklearn LogisticRegression model.
    '''

    def __init__(self, params, poly_degrees, N_bag = 5, resampler = 'None'):
        self.poly_degrees = poly_degrees
        self.resampler = resampler
        self.N_bag = N_bag
        try:
            del params['random_state']
        except:
            pass
        
        self.models = [LogisticRegression(**params) for _ in range(self.N_bag)]

    def fit(self, X_train, y_train):
        '''
        Fit all individual models independently for data X, y.
        '''
        
        for i in range(self.N_bag):
            # optional resampling
            if self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            else:
                X,y = X_train, y_train
                X,y = sklearn.utils.shuffle(X,y)

            # polynomial feature engineering
            X_logit, y_logit = reshape_model_input(X, degrees_lst = self.poly_degrees), y

            # fit model
            self.models[i].fit(X_logit, y_logit)

        # [self.models[i].fit(*shuffle(X_logit, y_logit, random_state=i)) for i in range(self.N_bag)]

        return self # allow for one-line notation of creating and fitting the model

    def predict_proba(self, X):
        '''
        Predict probabilities using the full ensembles of self.N_bag individual models.
        '''

        X_logit = reshape_model_input(X, degrees_lst = self.poly_degrees)

        return np.sum(np.array([self.models[i].predict_proba(X_logit) for i in range(self.N_bag)]), axis = 0)/self.N_bag

    def predict_proba_running_avg(self, X):
        '''
        Predict probabilities for all individual logit-models and report rolling average results, i.e. the benefit of adding more individual models to the ensemble.
        '''

        X_logit = reshape_model_input(X, degrees_lst = self.poly_degrees)
        return np.cumsum(np.array([self.models[i].predict_proba(X_logit) for i in range(self.N_bag)]), axis = 0)/np.arange(1, self.N_bag+1).reshape((-1,1,1))

    def predict_proba_individual(self, X): 
        '''
        Predict probabilities for all individual logit-models and report them as an array of shape (N_bag, len(X), 2).
        '''

        X_logit = reshape_model_input(X, degrees_lst = self.poly_degrees)
        return np.array([self.models[i].predict_proba(X_logit) for i in range(self.N_bag)])    


class ANN_bagging:
    
    """
    Purpose: Build multiple ANN models, use the bagged predictor in combination with an optional resampling procedure to reduce the variance of a predictor.
    New version - compatible with hpsklearn optimized parameter values as input

    Initialize the architecture of all individual models in the bagging procedure.
            
            
            Inputs:
            -------
                N_models: Number of models to be included in bagging procedure
                N_input: Number of input nodes
                width_lst: List containing the width for all layers, and hence implicitely also the depth of the network
                act_fct_lst: List containing the activation function for all layers
                dropout_rate: Dropout rate applied to all layers (except output layer)
                                dropout_rate = 0 will effectively disable dropout
                resampler: 'None': No resampling
                            'SMOTE': SMOTE resampling
                            'undersampling': RandomUndersampling
                loss: loss function which the model will be compiled with. Standard option: 'binary_crossentropy'
                optimizer: loss function which the model will be compiled with. Standard option: 'adam'
            
            Outputs:
            --------
                None. Creates self.model object with type(object) = dict
    """
    
    def __init__(self, N_models: int, hparams:dict, tf_dist_strat, resampler = 'None'):

        self.resampler = resampler
        self.model = {}
        self.hparams = hparams
        self.lr = hparams['lrate']
        self.tf_dist_strat = tf_dist_strat
        for i in range(N_models):
            # create model i
            try:
                with self.tf_dist_strat.scope():
                    self.model[i] = hpsearch_ann(**hparams)     
            except:
                self.model[i] = hpsearch_ann(**hparams) 
        # set ensemble model
        try:
            with self.tf_dist_strat.scope():
                INPUT = Input(shape = (self.hparams['n_input'],))
                self.ensemble = Model(inputs=INPUT, outputs = tf.keras.layers.Average()([self.model[i](INPUT) for i in range(len(self.model))]))
                # reduce learning rate for final fine-tuning of collective bagged model                
                self.ensemble.compile(optimizer = Adam(learning_rate=self.lr/2), loss = 'binary_crossentropy', metrics = ['acc'])    
        except:
            INPUT = Input(shape = (self.hparams['n_input'],))
            self.ensemble = Model(inputs=INPUT, outputs = tf.keras.layers.Average()([self.model[i](INPUT) for i in range(len(self.model))]))
            # reduce learning rate for final fine-tuning of collective bagged model                
            self.ensemble.compile(optimizer = Adam(learning_rate=self.lr/2), loss = 'binary_crossentropy', metrics = ['acc'])   

    def re_init_ensemble(self):
        '''
        Note: If we load old parametrizations by setting self.model[i] = value, the self.ensemble does not update automatically. 
        Hence, we need this value for consistently loading old values.
        '''

        # re-set ensemble model
        try:
            with self.tf_dist_strat.scope():
                INPUT = Input(shape = (self.hparams['n_input'],))
                self.ensemble = Model(inputs=INPUT, outputs = tf.keras.layers.Average()([self.model[i](INPUT) for i in range(len(self.model))]))
                # reduce learning rate for final fine-tuning of collective bagged model                
                self.ensemble.compile(optimizer = Adam(learning_rate=self.lr/2), loss = 'binary_crossentropy', metrics = ['acc'])    
        except:
            INPUT = Input(shape = (self.hparams['n_input'],))
            self.ensemble = Model(inputs=INPUT, outputs = tf.keras.layers.Average()([self.model[i](INPUT) for i in range(len(self.model))]))
            # reduce learning rate for final fine-tuning of collective bagged model                
            self.ensemble.compile(optimizer = Adam(learning_rate=self.lr/2), loss = 'binary_crossentropy', metrics = ['acc']) 


    def fit(self, X_train, y_train, callbacks = [], val_share = 0.3, N_epochs = 200):
        
        """
        Purpose: Train all model instances in the bagging procedure.
        
        output:
        \t None. Updates parameters of all models in self.model
        input
        \t X_train, y_train: \t Training data
        \t resampling_option: \t 'None': No resampling is performed
        \t                   \t 'undersampling': random undersampling of the majority class
        \t                     \t 'SMOTE': SMOTE methodology applied
        \t callbacks: \t callbacks for training
        \t val_share, N_epochs, N_batch: \t Additional arguments for training
        """

        # handle pandas-datatype
        if type(X_train)==type(pd.DataFrame([1])):
            X_train=X_train.values
        if type(y_train) == type(pd.DataFrame([1])):
            y_train=y_train.values

        # check if GPUs are available
        try:
            N_GPUs = self.tf_dist_strat.num_replicas_in_sync()
        except:
            N_GPUs = 1
        
        for i in range(len(self.model)):
            # utilze concept of resampling
            X,y = resample_and_shuffle(X_train, y_train, self.resampler)

            # transform into tf.data.Dataset
            try:
                train_data, val_data = create_tf_dataset(X, y, val_share, self.hparams['batch_size']*num_gpus())
            except:
                # go on with regular, numpy-data-type
                print('tf.data.Dataset could not be constructed. Continuing with numpy-data.')
                pass

            
            if len(self.model)==1:
                try:
                    self.model[i].fit(x=train_data, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                                    validation_data = val_data, verbose = 2, callbacks=callbacks)
                except:
                    print('using non-tf.data-format')
                    self.model[i].fit(x=X, y = y, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                        validation_split= val_share, verbose = 2, callbacks=callbacks)
            else:
                if i==0:               
                    # More compact view on models' training progress
                    print('Data of shape {} '.format(X.shape) + 'and balance factor {}'.format(sum(y)/len(y)))

                # Start training of model  
                print('Training Model {}'.format(i))
                t_start = time.time()
                try:
                    self.model[i].fit(x=train_data, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                            validation_data= val_data, verbose = 2, callbacks=callbacks+[LearningRateScheduler(exp_decay_scheduler)])
                except:
                    print('using non-tf.data-format')
                    self.model[i].fit(x=X, y = y, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                            validation_split= val_share, verbose = 2, callbacks=callbacks+[LearningRateScheduler(exp_decay_scheduler)])

                n_epochs_trained = len(self.model[i].history.history['loss'])
                print('\t ...  {} epochs'.format(n_epochs_trained))

                # plt.plot(self.model[i].history.history['loss'], label='loss')
                # plt.plot(self.model[i].history.history['val_loss'], label='val_loss')
                # plt.legend()
                # plt.show()
                
                for _ in range(3):
                    print('\t ... Fine tuning')
                    # reduce learning rate
                    self.model[i].optimizer.learning_rate = self.model[i].optimizer.learning_rate/2
                    try:
                        self.model[i].fit(x=train_data, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                                validation_data= val_data, verbose = 2, callbacks=callbacks+[LearningRateScheduler(exp_decay_scheduler)])#, initial_epoch= n_epochs_trained)
                    except:
                        print('using non-tf.data-format')
                        self.model[i].fit(x=X, y = y, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, 
                                validation_split= val_share, verbose = 2, callbacks=callbacks+[LearningRateScheduler(exp_decay_scheduler)])#, initial_epoch= n_epochs_trained)
                    # print(self.model[i].history.history)
                    # n_epochs_trained += len(self.model[i].history.history['loss'])

                print('\t ... Overall time: {} sec.'.format(time.time()-t_start))
                print('\t ... Done!')

                # plt.plot(self.model[i].history.history['loss'], label='loss')
                # plt.plot(self.model[i].history.history['val_loss'], label='val_loss')
                # plt.legend()
                # plt.show()



        print('Final fine tuning of whole bagged estimator:')
        t_start = time.time() 
        try:
            self.ensemble.fit(x=train_data, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, validation_data= val_data, verbose = 0, callbacks=callbacks)
        except:
            print('using non-tf.data-format')
            self.ensemble.fit(x=X, y = y, batch_size= N_GPUs*self.hparams['batch_size'], epochs = N_epochs, validation_split= val_share, verbose = 0, callbacks=callbacks)
        print('\t ...  {} epochs'.format(len(self.ensemble.history.history['val_loss'])))
        print('\t ... {} sec.'.format(time.time()-t_start))
        print('\t ... Done!')
        
        # Return object to allow for shorter/ single-line notation, i.e. ANN_bagging().fit()
        return self
        
    def predict(self, X):  
        
        """
        Purpose: Predict event probability for data
        
        Inputs:
        -------
        \t X: \t Input data        
        
        Outputs:
        --------
        \t Predictions for all input data
        """

        # handle pandas-datatype
        if type(X)==type(pd.DataFrame([1])):
            X=X.values
        return self.ensemble.predict(X)
    

    def predict_proba(self, X):
        
        """
        Purpose: Predict event probability for data
        
        Replicate predict_proba method of Sequential() or Model() class to unify notation.
        See documentation of self.predict() method.
        """
        # handle pandas-datatype
        if type(X)==type(pd.DataFrame([1])):
            X=X.values
        return self.predict(X)
    

    def predict_classes(self, X, threshold = 0.5):
        
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of Sequential() or Model() class to unify notation.
        """
        # handle pandas-datatype
        if type(X)==type(pd.DataFrame([1])):
            X=X.values
        return (self.predict(X)>= threshold)


class ANN_boost:
    
    '''
    Create a boosting instance with neural networks as weak learner instances.
    As we add a new weak learner it will train primarily on errors of previous models. Boost rate equal 1, i.e. weak learners added by summation.  
    For the purpose of binary classification we impose a binary_crossentropy loss.
    
    '''
    
    def __init__(self, N_models, N_input, width: int, act_fct: str, lr = 0.001, tf_dist_strat = None, resampler = 'None'):
        
        """
        Initialize the architecture of all individual models in the bagging procedure.
        Model style of weak learner: input->hidden_layer-> actv_fct-> single output (incl linear actv) -> sigmoid actv (to be carved off when combining multiple weak learners)
        
        
        Inputs:
        -------
            N_models: Number of models to be included in bagging procedure
            N_input: Number of input nodes
            width_lst: List containing the width for all layers, and hence implicitely also the depth of the network
            act_fct_lst: List containing the activation function for all layers. 
                            Last entry should be linear, as boosting models add a final sigmoid activation to the added weak learners to ensure a proper probability distribution.
            dropout_rate: Dropout rate applied to all layers (except output layer)
                            dropout_rate = 0 will effectively disable dropout
            loss: loss function which the model will be compiled with. Standard option: 'binary_crossentropy'
            optimizer: loss function which the model will be compiled with. Standard option: 'adam'
        
        Outputs:
        --------
            None. Creates self.model_base objects with type(object) = dict
        """        
        
        self.N_models = N_models
        self.loss = 'binary_crossentropy'
        self.N_input = N_input
        self.width = width
        self.act_fct = act_fct
        self.tf_dist = tf_dist_strat
        # self.dropout_rate = dropout_rate # canceled; not useful with only one hidden layer of which we tune its width
        self.lr_init = lr
        self.optimizer = Adam(learning_rate=self.lr_init)
        self.resampler = resampler
        self.history_val = []
        self.history_train = []
        self.training_steps = 0
        
            
        # boosted models will be assigned during fitting procedure
        #self.model_boost = [None]*self.N_models # depreciated version
        self.model_boost = None # Save memory by reusing file-space, i.e. not saving each intermediate boosting step separately as they are recorded by self.model_base
        # Create list of weak learner instances (compilation happens in creating functions)
        # try:
        #     with self.tf_dist.scope():
        #         self.model_base = [self.create_model_prior()]+[self.create_model_learner() for _ in range(self.N_models-1)]
        # except Exception as e:
        #     print('Leaners not created within tf-distribution-strategy due to:')
        #     print(e)
        self.model_base = [self.create_model_prior()]+[self.create_model_learner() for _ in range(self.N_models-1)]
                    
            
    def fit(self, x, y,  callbacks = [], val_share = 0.3, N_epochs = 200, N_batch = 64, correction_freq = 5):
        
        '''
        Fitting procedure for the ANN_boost object.
        
        Inputs:
        -------
            x: Input Data
            y: Targets
            callbacks: list of tf.keras.callbacks objects, e.g. earlyStopping
            val_share: share of (x,y) used for validation of the model during training and for potential callback options
            N_epochs: number of epochs for training
            N_batch: batch size for training
            correction_freq: frequency in which a corrective step is performed, e.g. 0: never, 1: every epoch, 5: every 5 epochs, ...
        '''

        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost.fit: x values changed from pandas.DataFrame to numpy.array')
        if type(y) == type(pd.DataFrame([1])):
            y=y.values
            #print('ANN_boost.fit: y values changed from pandas.DataFrame to numpy.array')


        # optional resampling
        x,y = resample_and_shuffle(x, y, self.resampler)

        # transform into tf.data.Dataset (important: transformation after optional resampling)
        try:
            train_data, val_data = create_tf_dataset(x,y,val_share, N_batch*num_gpus())
        except:
            # go on with regular, numpy-data-type
            print('tf.data.Dataset could not be constructed. Continuing with numpy-data.')
            pass
        

        
        if self.N_input != x.shape[1]:
            raise ValueError('Error: Invalid input shape. Expected ({},) but given ({},)'.format(self.N_input, x.shape[1]))

        
        # iterate over number of weak learners included in boosting
        INPUT = Input(shape= (self.N_input,)) # re-use this input layer to avoid more cache-intensiv multi-inputs models
        for n in range(1,self.N_models+1):
            
            try:
                with self.tf_dist.scope():
                    if n == 1:
                        # Note: Average Layer expects >= 2 inputs
                        # Add final sigmoid Activation for classification
                        self.model_boost = Model(inputs = INPUT, outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(self.model_base[0](INPUT)))
                    else:
                        self.model_boost = Model(inputs = INPUT,#[self.model_base[i].input for i in range(n)], 
                                                    # Note: Average() needs list as input; use .output, not .outputs (-> list of lists)
                                                    outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(
                                                        tf.keras.layers.Add()(
                                                            [self.model_base[i](INPUT) for i in range(n)]# .output for i in range(n)]
                                                        )
                                                    )
                                                )
                    # set trainable = True for newly added weak learner (relevant if we retrain model)
                    self.model_base[n-1].trainable = True
                    # compile model
                    self.model_boost.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])

            except Exception as e:
                print('Booster not created within distribution strategy due to:')
                print(e)
                if n == 1:
                    # Note: Average Layer expects >= 2 inputs
                    # Add final sigmoid Activation for classification
                    self.model_boost = Model(inputs = INPUT, outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(self.model_base[0](INPUT)))#.output))
                else:
                    self.model_boost = Model(inputs = INPUT,#[self.model_base[i].input for i in range(n)], 
                                                # Note: Average() needs list as input; use .output, not .outputs (-> list of lists)
                                                outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(
                                                    tf.keras.layers.Add()(
                                                        [self.model_base[i](INPUT) for i in range(n)]# .output for i in range(n)]
                                                    )
                                                )
                                                )
                # set trainable = True for newly added weak learner (relevant if we retrain model)
                self.model_base[n-1].trainable = True
                # compile model
                self.model_boost.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])        


            # train boosting model
            print('Training Model {}'.format(n))
            print('\t trainable params: '+ str(keras_count_nontrainable_params(self.model_boost, trainable=True)))
            print('\t nontrainable params: '+ str(keras_count_nontrainable_params(self.model_boost, trainable=False)))

            t_start = time.time()
            if (n==1):
                # set weights = 0 and bias = sigmoid^-1(baseline_hazard)
                try:
                    with self.tf_dist.scope():
                        self.model_boost.layers[1].set_weights([np.array([0]*self.N_input).reshape((-1,1)), np.array([np.log(y.mean()/(1-y.mean()))])])
                except Exception as e:
                    print('Setting weights of baseline-learner not performed within tf-distribution-strategy due to:')
                    print(e)
                    self.model_boost.layers[1].set_weights([np.array([0]*self.N_input).reshape((-1,1)), np.array([np.log(y.mean()/(1-y.mean()))])])
            else:
                try:
                # if data in tf.data.Dataset format available
                    print('\t .. training on tf.data.Dataset')
                    self.model_boost.fit(x=train_data, validation_data = val_data, epochs = N_epochs, verbose = 2, callbacks=callbacks)
                except Exception as e:
                    print('Leaners not created within tf-distribution-strategy due to:')
                    print(e)
                    self.model_boost.fit(x=x, y = y, batch_size= N_batch, epochs = N_epochs, validation_split= val_share, verbose = 0, callbacks=callbacks)
                self.history_val += self.model_boost.history.history['val_loss']
                self.history_train += self.model_boost.history.history['loss']
                
                # evolutionary fitting of boosting model
                #self.fit_evolutionary(x=x, y=y, batch_size=N_batch, epochs=N_epochs, epochs_per_it=25, validation_split=val_share, callbacks=callbacks)
                
                print('\t ...  {} epochs'.format(len(self.history_val)-self.training_steps))              
                self.training_steps = len(self.history_val)

            print('\t ... {} sec.'.format(time.time()-t_start))
            #print('\t ... eval.: ', self.model_boost.evaluate(x,y, verbose=0)) # optional: display to observe progress of training; however, slows down training.
            print('\t ... Done!')

            # decaying influence of weak learners
            #self.optimizer.lr = self.lr_init*0.9**n

            
            # corrective step: set all parameters as trainable and update them using SGD
            if n>1:
                if (correction_freq > 0) & (n%correction_freq ==0):
                    self.corrective_step(model = self.model_boost, x=x, y=y, callbacks=callbacks, 
                                        val_share=val_share, N_epochs = N_epochs, N_batch= N_batch)
            
            
            
            # set trainable = False for weak learner that has been included in the boosting model 
            self.model_base[n-1].trainable = False


    def fit_evolutionary(self, x, y, batch_size, epochs, epochs_per_it, validation_split, callbacks):
        '''
        Customized training scheme, using early stopping/ callbacks and a iterative reduction of the initial learning rate.
        ## DEPRECIATED as not very affective in the given scenario
        '''
        
        self.model_boost.fit(x=x, y = y, batch_size= batch_size, epochs = epochs_per_it, validation_split=validation_split, verbose = 0, callbacks=callbacks)
        self.history_train += self.model_boost.history.history['loss']
        self.history_val += self.model_boost.history.history['val_loss']
        #print(self.history_train)
        #print(type(self.history_train))
        val_loss = min(self.history_val)
        #print('minimum val_loss: ', val_loss)


        evol_patience = 0
        for ep in range(epochs//epochs_per_it):
            self.optimizer.lr= self.lr_init*1.2**(1+ep%4)
            # compile to effectively update lr
            self.model_boost.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
            print(' \t Fine tuning step ', ep, '...', ' (val_loss: ', np.round_(val_loss,4), ')')
            self.model_boost.fit(x=x, y = y, batch_size=batch_size, epochs = epochs_per_it, validation_split=validation_split, verbose = 0, callbacks=callbacks)
            # record training/ validation history
            self.history_train += self.model_boost.history.history['loss']
            self.history_val += self.model_boost.history.history['val_loss']
            
            if min(self.history_val) < val_loss*0.99:
                val_loss = min(self.history_val)
            else:
                evol_patience += 1
                if evol_patience > 3:
                    break            
    
    def corrective_step(self, model, x, y, callbacks = [], val_share = 0.3, N_epochs = 200, N_batch = 64):
        '''
        Perform a corrective step by updating all parameters of boosting model, i.e. all included weak learners.
        '''
        
        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost.fit: x values changed from pandas.DataFrame to numpy.array')
        if type(y) == type(pd.DataFrame([1])):
            y=y.values
            #print('ANN_boost.fit: y values changed from pandas.DataFrame to numpy.array')


        # transform into tf.data.Dataset
        try:
            train_data, val_data = create_tf_dataset(x,y,val_share, N_batch*num_gpus())
        except:
            # go on with regular, numpy-data-type
            print('tf.data.Dataset could not be constructed. Continuing with numpy-data.')
            pass

        # allow updating of all parameters
        try:
            with self.tf_dist.scope():
                model.trainable = True
                model.compile(optimizer = Adam(lr=self.lr_init/2), loss = self.loss, metrics = ['acc'])
        except Exception as e:
            print('Leaners not created within tf-distribution-strategy due to:')
            print(e)
            model.trainable = True
            model.compile(optimizer = Adam(lr=self.lr_init/2), loss = self.loss, metrics = ['acc'])
        
        print('Corrective Step ... ')
        print('\t trainable params: '+ str(keras_count_nontrainable_params(model, trainable=True)))
        print('\t nontrainable params: '+ str(keras_count_nontrainable_params(model, trainable=False)))

        t_start = time.time()
        
        #self.fit_evolutionary(x=x, y=y, batch_size=N_batch, epochs=N_epochs, epochs_per_it=25, validation_split=val_share, callbacks=callbacks)
        try:
            # train with tf.data.dataset; explicitly indicate val_data; batch_size indicated in tf.data.dataset
            model.fit(x=train_data, epochs = N_epochs,  validation_data= val_data, verbose = 2, callbacks=callbacks)
        except Exception as e:
            print('Model not created within tf-distribution-strategy due to:')
            print(e)
            model.fit(x=x, y = y, batch_size= N_batch, epochs = N_epochs,  validation_split= val_share, verbose = 2, callbacks=callbacks)



        print('\t ...  {} epochs'.format(len(model.history.history['val_loss'])))
        run_time = time.time()-t_start
        print('\t ... {} sec.'.format(run_time))
        print('\t ... Correction performed!')
              
        # Lock updates
        model.trainable = False      

        return run_time
        
    def save_object(self, path):
        '''
        Function to save the ANN_boost object. 
        Required, as e.g. Sequential()-Object in self.model_base[i] cannot be pickled or dilled. 
        Hence, we save only the respective weights and provide a function load_object to restore the fully functional ANN_boost object.
        Note: load_ANN_boost_object is no ANN_boost object function. However, the loaded ANN_boost object uses object.restore_learners() to restore learners and boosted models.
        '''
        # save weights of learners
        #self.model_base = [self.model_base[i].get_weights() for i in range(self.N_models)]
        # delete boosted models temporarily for pickling; can be restored with weights of (trained) learners
        #cache = clone_model(self.model_boost)
        #cache.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
        
        model_backup = ANN_boost(N_models= self.N_models, N_input= self.N_input, width = self.width, act_fct = self.act_fct)
        model_backup.model_base = [sub_model.get_weights() for sub_model in self.model_base] # save only weights -> to be restored in self.restore_learners()
        # Note: Adam-object cannot be pickled in tf 2.4.
        # workaround: switch to string-information and restore full optimizer (incl. learning_rate) in restore_learners
        model_backup.optimizer = 'adam' 

        #self.model_boost = None#*self.N_models
        with open( path, "wb" ) as file:
            pickle.dump(model_backup, file)
            print('ANN object dumped to ', path)

        #self.model_boost = cache

    def restore_learners(self):
        '''
        Restore the full Sequential() architecture of self.model_base[i] and self.model_boost[i] which were replaced by their weights to pickle dump the object.
        '''
        weights = copy.copy(self.model_base)
        self.model_base = [self.create_model_prior()]+[self.create_model_learner() for _ in range(1,self.N_models)]
        [self.model_base[i].set_weights(weights[i]) for i in range(self.N_models)]
        #print(self.model_base)
        # iterate over number of weak learners included in boosting
        for n in range(1,self.N_models+1):
            INPUT = Input(shape= (self.N_input,))
            if n == 1:
                # Note: Average Layer expects >= 2 inputs
                # Add final sigmoid Activation for classification
                #self.model_boost[n-1] = Model(inputs = self.model_base[0].input, 
                #                              outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(self.model_base[0].output))
                self.model_boost = Model(inputs = INPUT,#self.model_base[0].input, 
                                        outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(self.model_base[0](INPUT)))#.output))
            else:
                #self.model_boost[n-1] 
                self.model_boost = Model(inputs = INPUT,#[self.model_base[i].input for i in range(n)], 
                                              # Note: Average() needs list as input; use .output, not .outputs (-> list of lists)
                                              outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(
                                                  tf.keras.layers.Add()(
                                                      [self.model_base[i](INPUT) for i in range(n)]# .output for i in range(n)]
                                                  )
                                              )
                                             )

            # set trainable = True for newly added weak learner (relevant if we retrain model)
            self.model_base[n-1].trainable = True
            # compile model
            self.model_boost.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])

    def create_model_prior(self):
        '''
        Base model 0 in boosting structure; expresses a prior estimate (here constant rate) that will be improved by subsequent model created by create_model_learner.
        '''
        model = Sequential()
        model.add(Dense(1, activation= 'linear', input_dim = self.N_input))
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
        return model

    def create_model_learner(self):
        '''
        Create architecture for weak learners in boosting strategy.
        '''
        model = Sequential()
        # Hidden layer
        try:
            model.add(Dense(units = self.width, activation = self.act_fct, input_dim = self.N_input))
        except:
            # old implementation
            model.add(Dense(units = self.width_lst[0], activation = self.act_fct_lst[0], input_dim = self.N_input))
            print('sub_surrender_models, create_model_learner(): atributes width_lst and act_fct_lst depreciated!')
        # Output layer
        model.add(Dense(units = 1, activation = 'linear'))
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
        return model

    def prune_booster(self, n_learners:int):
        '''
        Take user input how many weak learners should be utilized. The rest will be discarded.
        '''

        assert n_learners<= self.N_models
        assert n_learners > 1

        INPUT = Input(shape= (self.N_input,)) # re-use this input layer to avoid more cache-intensiv multi-inputs models
        self.model_boost = Model(inputs = INPUT,#[self.model_base[i].input for i in range(n)], 
                                                    # Note: Average() needs list as input; use .output, not .outputs (-> list of lists)
                                                    outputs = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(
                                                        tf.keras.layers.Add()(
                                                            [self.model_base[i](INPUT) for i in range(n_learners)]# .output for i in range(n)]
                                                        )
                                                    )
                                                )
        # compile model
        self.model_boost.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
        

    def evaluate(self, x, y=None):
        try:
            # x is tf.data.Dataset
            return self.model_boost.evaluate(x, verbose=0)
        except:
            return self.model_boost.evaluate(x,y, verbose=0)

    def predict_proba(self, x):
        
        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """

        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost.fit: x values changed from pandas.DataFrame to numpy.array')

        
        # Use last iteration of boosting procedure
        # Note: tf.keras.models.Model() does not posses .predict_proba(), but only .predict()
        return self.model_boost.predict(x)
    
    
    def predict(self, x):

        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """

        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost.fit: x values changed from pandas.DataFrame to numpy.array')

        
        # Use last iteration of boosting procedure
        # Note: tf.keras.models.Model() does not posses .predict_proba(), but only .predict()
        return self.model_boost.predict(x)
    

    def predict_classes(self, x, threshold = 0.5):
        
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of Sequential() or Model() class to unify notation.
        """

        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost.fit: x values changed from pandas.DataFrame to numpy.array')
        return (self.predict(x)> threshold)
    

def analyze_ensemble(model, x, y, profile: int, poly_degrees_max = None):
    '''
    Check for different model types, i.e. Logit-ensemble, ANN-ensemble and ANN-booster, by how much an additional learners improves the performance.
    Goal: Determine a reasonable number of the depth of boosting/ no. of weak lerners to work together, to limit computational effort

    Inputs:
    -------
        model:  model(s) to evaluate; either list of models or single model
        x:      input data, typically validation data
        y:      target data, typically validation data

    Outputs:
    --------
        None; a plot with performance over number of learners is produced.
    '''

    if type(model) == type([]):
        pass
    else:
        model = [model]

    for m in model:
        if type(m) == ANN_bagging:
            try: x_val = x.values
            except: pass
            # learners = model.model # dictionary of learners
            pred = [l.predict(x_val) for l in m.model.values()]
            # respect avering effect of bagging-ensemble
            pred = np.cumsum(np.array(pred), axis = 0)/np.arange(1, len(pred)+1).reshape((-1,1,1))
            entropy = [log_loss(y_true = y, y_pred=p) for p in pred]
            plt.plot(range(1, len(pred)+1), entropy, label = 'NN (bag)')
        elif type(m) == ANN_boost:
            try: x_val = x.values
            except: pass
            # learners = model.model_base # list of models
            pred = [l.predict(x_val) for l in m.model_base]
            # Note: do not forget final sigmoid function to form boosted-ensemble-prediction
            pred = expit(np.cumsum(np.array(pred), axis = 0))
            entropy = [log_loss(y_true = y, y_pred=pred[i]) for i in range(len(pred))]
            plt.plot(range(1, len(pred)+1), entropy, label = 'NN (boost)')
        elif type(m) == Logit_model:
            assert type(poly_degrees_max) != type(None)
            # learners = model.models # list of models
            pred = [l.predict_proba(reshape_model_input(x, degrees_lst=[poly_degrees_max]*x.shape[1]))[:,-1] for l in m.models]
            # respect avering effect of bagging-ensemble
            pred = np.cumsum(np.array(pred), axis = 0)/np.arange(1, len(pred)+1).reshape((-1,1))
            entropy = [log_loss(y_true = y, y_pred=p) for p in pred]
            plt.plot(range(1, len(pred)+1), entropy, label = 'Logist. Regr.')
        else:
            raise ValueError('Model type not compatible with method!')

    plt.ylabel('entropy loss')
    plt.xlabel('# of learners')
    plt.yscale('log')
    plt.legend()
    plt.title(f'ensemble models of profile {profile}')
    plt.show()



########################################
########### LEGACY CODE ################
########################################

# Note: These classes are either not used, e.g. since Logit_boosting showed poor performance, 
# or have been updated to a later version, e.g. Logit_model_old (with integrated feature preprocessing) -> Logit_model (a priori feature-preprocessing and K-Fold)

class Logit_model_old:
    '''
    Create a logistic model from either sklearn or statsmodels (significance analysis in summary included).
    Further, we allow for adding higher degrees of the input variables without having to change the input data.
    
    Requirements: 
    \t import statsmodels.api as sm
    \t from sklearn.models import LogisticRegression
    '''
    
    def __init__(self, package='sklearn', polynomial_degrees = [1,1,1], resampler = 'None',
                 X = None, y = None):
        self.package = package
        self.poly_degrees = polynomial_degrees
        # Polynomial degrees of features selected after feed-forward fitting process
        self.poly_selected = [1]*len(polynomial_degrees)
        
        self.resampler = resampler
        if package == 'sklearn':
            self.model = LogisticRegression(solver = 'liblinear', penalty = 'l2')
        elif package == 'statsmodels':
            
            if self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X, y=y)
            elif self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X, y=y)
            elif self.resampler == 'None':
                # Do nothing
                pass
            else:
                print('Error: Resampler not recognized!')
                
            self.model = sm.Logit(endog = y, exog = reshape_model_input(sm.add_constant(X),
                                                                          degrees_lst =self.poly_degrees))
        else:
            print('Error: Package not valid!')
            
            
    def fit(self, X=None, y=None, val_share = 0.2):
        
        '''
        Fit a Logistic Regression object. 
        Add higher-order polynomials of the input features in a feed-forward manner up to the max input-polynomial degree.
        E.g. poly_degrees = [3,3,3] would result in checking [1,1,1], [2,1,1], .., [2,2,2], [3,2,2], ... [3,3,3]
        Degree is only increased if validation error increases.
        
        Parameters
        ----------
            X: Features
            y: Targets
            var_share: Share of (X,y) used for validation during the forward-selection of features
        '''
        
        if self.package == 'sklearn':
            
            if self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X, y=y)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X, y=y)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'None':
                # Do nothing
                pass
            else:
                print('Error: Resampler not recognized!')
            
            
            
            
            # forward selection
            n = len(self.poly_degrees)
            bool_degree_increase = [True]*n
            degrees_start = [1]*n
            best_model = self.model.fit(X = reshape_model_input(X[0:int((1-val_share)*len(X))], 
                                                      degrees_lst =degrees_start), y = y[0:int((1-val_share)*len(y))])
            
            best_model_eval = sklearn.metrics.log_loss(y_true = y[int((1-val_share)*len(y)):],
                               y_pred=best_model.predict_proba(X = reshape_model_input(X[int((1-val_share)*len(X)):], 
                                                                                          degrees_lst =degrees_start))[:,-1])
            
            # 
            for _ in range(1,max(self.poly_degrees)):
                # increase all elements/ orders in list stepwise each by magnitude of 1
                for i in range(n):
                    # check if degree permitted by self.poly_degrees
                    if (degrees_start[i]+1 <= self.poly_degrees[i])& bool_degree_increase[i]:
                        degrees_start[i]+=1
                        
                        #print(degrees_start)
                        
                        model_new = self.model.fit(X = reshape_model_input(X[0:int((1-val_share)*len(X))], 
                                                      degrees_lst =degrees_start), y = y[0:int((1-val_share)*len(y))])
                        model_new_eval = sklearn.metrics.log_loss(y_true = y[int((1-val_share)*len(y)):],
                               y_pred = model_new.predict_proba(X = reshape_model_input(X[int((1-val_share)*len(X)):], 
                                                                                          degrees_lst =degrees_start)))
                        
                        
                        # compare validation error
                        if model_new_eval< best_model_eval:
                            # save new, best model reference
                            best_model_eval = copy.copy(model_new_eval)
                            best_model = copy.copy(model_new)
                        else:
                            # reverse increase of polynomial order and stop fwd-selection of feature i
                            degrees_start[i]-=1
                            #print('Validation error increased for feature {}.'.format(i))
                            bool_degree_increase[i] = False   
                            
                    #print(str(model_new.coef_.shape)+ '(new) vs. (best) ' + str(best_model.coef_.shape))
                    #print('\n')       
                
            self.poly_selected = degrees_start
            self.model = best_model
            # fit cross-validated model on selected poly.-degrees for all data
            #self.model = self.model.fit(X = reshape_model_input(X, degrees_lst =self.poly_selected), y = y)
            
            print('Logistic model built successfully; pruned to polynomial features with degrees {}'.format(self.poly_selected))
            
        if self.package == 'statsmodels':
            self.model = self.model.fit(method='bfgs', maxiter=100)
            print('Note: Fitting data for statsmodel provided at initialization.')
            print('Note: Forward selection of model features not implemented for "statsmodels-package".')
            
        return self
          
        
        
    def predict(self, X):
        
        """
        Purpose: Predict class for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """
        
        if self.package == 'sklearn':
            return self.model.predict(X = reshape_model_input(df_input = X, degrees_lst =self.poly_selected))
        elif self.package == 'statsmodels':
            return self.model.predict(exog=reshape_model_input(df_input = sm.add_constant(X),
                                                                          degrees_lst =self.poly_selected))
        else:
            print('Error: Package unknown!') 
            
            

    def predict_proba(self, X):
        """
        Purpose: Predict event probability for data
        
        Replicate predict_proba method of other model-classes to unify notation.
        See documentation of self.predict() method.
        """
        if self.package == 'sklearn':
            return self.model.predict_proba(X = reshape_model_input(df_input = X, degrees_lst =self.poly_selected))
        elif self.package == 'statsmodels':
            return self.model.predict(exog=reshape_model_input(df_input = sm.add_constant(X),
                                                                          degrees_lst =self.poly_selected))
        else:
            print('Error: Package unknown!')
            
 
    
    def predict_classes(self, X, threshold = 0.5):
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of other model-classes to unify notation.
        """            
        return self.predict_proba(X)>threshold
    
 
    
    def summary(self, X=None):
        
        '''
        Provide brief summary of coefficients, values and significance (for statsmodels only).
        '''
        if self.package == 'sklearn':
            
            df = pd.DataFrame(data = None,
                               columns = ['const.']+list(reshape_model_input(df_input = X.loc[0:1,:], 
                                                                        degrees_lst =self.poly_selected).columns))
            df.loc['',:] = [self.model.intercept_[0]]+self.model.coef_.flatten().tolist()
            
            print(df)
            
        elif self.package == 'statsmodels':
            # Use summary() of statsmodels.api.Logit object
            print(self.model.summary())

# preliminary - class not functional yet
class ANN_boost_grad:
    
    '''
    Create a gradient boosting instance with neural networks as weak learner instances.
    As we add a new weak learner it will train primarily on errors of previous models. Boost rate initialized with 1, but eventually adapted in corrective step.  
    For the purpose of binary classification we impose a binary_crossentropy loss.
    
    '''
    
    def __init__(self, N_models, N_input, width_lst = [], act_fct_lst = [], dropout_rate = 0,  optimizer = 'adam'):
        
        """
        Initialize the architecture of all individual models in the bagging procedure.
        
        
        Inputs:
        -------
            N_models: Number of models to be included in bagging procedure
            N_input: Number of input nodes
            width_lst: List containing the width for all layers, and hence implicitely also the depth of the network
            act_fct_lst: List containing the activation function for all layers. 
                            Last entry should be sigmoid, as gradient boosting models add probability outputs of weak learners.
            dropout_rate: Dropout rate applied to all layers (except output layer)
                            dropout_rate = 0 will effectively disable dropout
            loss: loss function which the model will be compiled with. Standard option: 'binary_crossentropy'
            optimizer: loss function which the model will be compiled with. Standard option: 'adam'
        
        Outputs:
        --------
            None. Creates self.model_base objects with type(object) = dict
        """        
        
        self.N_boost = N_models
        self.optimizer = optimizer
        self.loss = 'binary_crossentropy'
        self.N_input = N_input
        
        if act_fct_lst[-1] != 'sigmoid':
                raise Exception('Gradient boosting models adds probability outputs of weak learners. Final activation should be sigmoid!')
            
        # boosted models will be assigned during fitting procedure
        self.model_boost = {}
        # Create weak learner instances
        self.model_base = {}
        for i in range(N_models):
            # Create first model to to capture baseline hazard
            if i == 0:
                self.model_base[i] = Sequential()
                self.model_base[i].add(Dense(1, activation= 'sigmoid', input_dim = N_input))
                
            else:
                self.model_base[i] = Sequential()
                for j in range(len(width_lst)):
                    if j==0: # Specify input size for first layer
                        self.model_base[i].add(Dense(units = width_lst[j], activation = act_fct_lst[j], input_dim = N_input))
                    else:
                        self.model_base[i].add(Dense(units = width_lst[j], activation = act_fct_lst[j]))
                    if j<(len(width_lst)-1): # No dropout after output layer
                        self.model_base[i].add(Dropout(rate = dropout_rate))
                    
            # compile base models       
            self.model_base[i].compile(optimizer = self.optimizer, loss = self.loss, metrics = ['acc'])
            
            
            
    def fit(self, x, y,  callbacks = [], val_share = 0.2, N_epochs = 200, N_batch = 64, correction_freq = 1):
        
        '''
        Fitting procedure for the ANN_boost_grad object.
        
        Inputs:
        -------
            x: Input Data
            y: Targets
            callbacks: list of tf.keras.callbacks objects, e.g. earlyStopping
            val_share: share of (x,y) used for validation of the model during training and for potential callback options
            N_epochs: number of epochs for training
            N_batch: batch size for training
            correction_freq: frequency in which a corrective step is performed, e.g. 0: never, 1: every epoch, 5: every 5 epochs, ...
        '''
        
        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost_grad.fit: x values changed from pandas.DataFrame to numpy.array')
        if type(y) == type(pd.DataFrame([1])):
            y=y.values
            #print('ANN_boost_grad.fit: y values changed from pandas.DataFrame to numpy.array')

        if self.N_input== x.shape[1]:
            pass
        else:
            print('Error: Invalid input shape. Expected ({},) but given ({},)'.format(self.N_input, x.shape[1]))
            exit() 
            
        if type(y) != type(np.array([1])):
            # transform pd.series to np.array format -> required for tf.keras model and sample_weight
            y = y.values.reshape((-1,1))
            
            
        
        # iterate over number of weak learners included in boosting
        for n in range(1,self.N_boost+1):
            
            # train weak learners conditionally
            print('Training weak learner {}'.format(n))
            print('\t trainable params: '+ str(keras_count_nontrainable_params(self.model_base[n-1], trainable=True)))
            #print('\t nontrainable params: '+ str(keras_count_nontrainable_params(self.model_boost[n-1], trainable=False)))

            t_start = time.process_time()
            if n==1:
                # set weights = 0 and bias = sigmoid^-1(baseline_hazard)
                self.model_base[n-1].layers[-1].set_weights([np.array([0]*self.N_input).reshape((-1,1)),
                                                          np.array([-np.log((1-y.mean())/y.mean())])])
            else:                
                
                # compute new targets based of 2nd order taylor approx of binary-crossentropy loss
                pred = self.model_boost[n-2].predict([x]*(n-1))
                g = (pred-y)/(pred*(1-pred)) # 1st order
                h = (-2*pred**2+y*(1+pred))/(pred*(1-pred))**2 # 2nd order
                #print('type(g): ' +str(type(g)))
                #print('type(h): ' +str(type(h)))
                #print('type(g/h): ' +str(type(g/h)))
                #print('\n')
                #print('g.shape: ' +str(g.shape))
                #print('h.shape: ' +str(h.shape))
                #print('g/h.shape: ' +str((g/h).shape))
                #print('y.shape: ' +str(y.shape))

                # train weak learner w.r.t. mse loss for new target; for faster convergence, normalize sample_weights
                self.model_base[n-1].fit(x=x, y = -g/h, sample_weight = h.flatten()/h.sum(), batch_size= N_batch, epochs = N_epochs, 
                        validation_split= val_share, verbose = 0, callbacks=callbacks)
                print('\t ...  {} epochs'.format(len(self.model_base[n-1].history.history['val_loss'])))
                
            print('\t ... {} sec.'.format(time.process_time()-t_start))
            print('\t ... Done!')
            
            # add newly trained weak learner to boosting model
            if n == 1:
                # Note: Add Layer expects >= 2 inputs
                self.model_boost[n-1] = self.model_base[n-1]
            else:
                self.model_boost[n-1] = Model(inputs = [self.model_base[i].input for i in range(n)], 
                                              # Note: Add() needs list as input; use .output, not .outputs (-> list of lists)
                                              outputs = tf.keras.layers.Add()(
                                                      [self.model_base[i].output for i in range(n)]
                                              )
                                             )
            self.model_boost[n-1].compile(loss = 'binary_crossentropy', optimizer = self.optimizer)
                
            # corrective step: set all parameters as trainable and update them using SGD
            if n>1:
                if (correction_freq > 0) & (n%correction_freq ==0):
                    self.corrective_step(model = self.model_boost[n-1], x=x, y=y, callbacks=callbacks, 
                                        val_share=val_share, N_epochs = N_epochs, N_batch= N_batch)       

               
    
    def corrective_step(self, model, x, y,  callbacks = [], val_share = 0.2, N_epochs = 200, N_batch = 64):
        '''
        Perform a corrective step by updating all parameters of boosting model, i.e. all included weak learners.
        '''
        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost_grad.corrective_step: x values changed from pandas.DataFrame to numpy.array')
        if type(y) == type(pd.DataFrame([1])):
            y=y.values
            #print('ANN_boost_grad.corrective_step: y values changed from pandas.DataFrame to numpy.array')

        # allow updating of all parameters
        model.trainable = True
        
        print('Corrective Step ... ')
        print('\t trainable params: '+ str(keras_count_nontrainable_params(model, trainable=True)))
        print('\t nontrainable params: '+ str(keras_count_nontrainable_params(model, trainable=False)))

        t_start = time.process_time()
        model.fit(x=[x]*len(model.inputs), y = y, batch_size= N_batch, epochs = N_epochs, 
                    validation_split= val_share, verbose = 0, callbacks=callbacks)

        print('\t ...  {} epochs'.format(len(model.history.history['val_loss'])))
        print('\t ... {} sec.'.format(time.process_time()-t_start))
        print('\t ... Correction performed!')
              
        # Lock updates
        model.trainable = False      
        
    
    def predict_proba(self, x):
        
        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """
    
        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost_grad.predict_proba: x values changed from pandas.DataFrame to numpy.array')


        # Use last iteration of boosting procedure
        # Note: tf.keras.models.Model() does not posses .predict_proba(), but only .predict()
        return self.model_boost[self.N_boost-1].predict([x]*self.N_boost)
    
    def predict(self, x):

        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """
        
        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost_grad.predict: x values changed from pandas.DataFrame to numpy.array')

        # Use last iteration of boosting procedure
        # Note: tf.keras.models.Model() does not posses .predict_proba(), but only .predict()
        return self.model_boost[self.N_boost-1].predict([x]*self.N_boost)
    
    
    def predict_classes(self, x, threshold = 0.5):
        
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of Sequential() or Model() class to unify notation.
        """

        # handle pandas-datatype
        if type(x)==type(pd.DataFrame([1])):
            x=x.values
            #print('ANN_boost_grad.predict_classes: x values changed from pandas.DataFrame to numpy.array')

        return (self.predict([x]*self.N_boost)> threshold)
    

class Logit_boosting:
    
    '''
    Build a bagging procedure for Logistic models (from either the 'sklearn' or 'statsmodels' package) including an optional resampling procedure.
    '''
    
    def __init__(self, N_models, polynomial_degrees = [1,1,1], bool_ada_boost =True, resampler = 'None',
                 package='sklearn', X = None, y = None):
        self.resampler = resampler
        self.polynomial_degrees = polynomial_degrees
        self.bool_ada_boost = bool_ada_boost
        if self.bool_ada_boost:
            self.model = AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators=N_models)
        else:
            raise ValueError('logitBoost not implemented')
            # self.model = logitboost.LogitBoost(LogisticRegression(), n_estimators=N_models, random_state=0)
            # print('Note: LogitBoost model only works for regressors as weak learners are fitted on residuals, i.e. crossentropy loss fails.')
            # print('Abording action in Logit_boosting.__init__ in sub_surrender_models.py')
            # exit()
                
    def fit(self, X_train, y_train, val_share = 0.2):
        
        """
        Purpose: Train all model instances in the boosting procedure.
        

        Inputs:
        -------
        \t X_train, y_train: \t Training data
        \t resampling_option: \t 'None': No resampling is performed
        \t                   \t 'undersampling': random undersampling of the majority class
        \t                     \t 'SMOTE': SMOTE methodology applied
        \t callbacks: \t callbacks for training
        \t val_share, N_epochs, N_batch: \t Additional arguments for training

        Outputs:
        --------
        \t None. Updates parameters of all models in self.model
        """

        # transform input X to higher degrees of features


        if self.bool_ada_boost:
            # utilze concept of resampling
            if self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            else:
                X,y = X_train, y_train
                #X,y = sklearn.utils.shuffle(X,y)
            
            # include higher polynomial-degrees of input features
            X = reshape_model_input(X, degrees_lst =self.polynomial_degrees)
            # utilize AdaBoostClassifier object
            self.model.fit(X,y)
            return self
        else:
            # utilze concept of resampling
            if self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            else:
                X,y = X_train, y_train
                #X,y = sklearn.utils.shuffle(X,y)

            X = reshape_model_input(X, degrees_lst =self.polynomial_degrees)

            self.model.fit(X = X, y = y)
            
            # Return model(s) to allow for shorter/ single-line notation, i.e. Logit_bagging().fit()
            return self

        
    def predict_proba(self, X):  
        
        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """
        #pred = sum([self.model[i].predict_proba(X) for i in range(len(self.model))])/len(self.model)

        return self.model.predict_proba(X = reshape_model_input(X, degrees_lst =self.polynomial_degrees))
    

    def predict(self, X):
        
        """
        Purpose: Predict label for data
        
        Replicate predict_proba method of Sequential() or Model() class to unify notation.
        See documentation of self.predict() method.
        """
        return self.model.predict(X = reshape_model_input(X, degrees_lst =self.polynomial_degrees))
    

    def predict_classes(self, X, threshold = 0.5):
        
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of Sequential() or Model() class to unify notation.
        """
        return self.model.predict(X = reshape_model_input(X, degrees_lst =self.polynomial_degrees))        


class Logit_bagging:
    
    '''
    Build a bagging procedure for Logistic models (from either the 'sklearn' or 'statsmodels' package) including an optional resampling procedure.
    '''
    
    def __init__(self, N_models, package='sklearn', polynomial_degrees = [1,1,1], resampler = 'None',
                 X = None, y = None):
        self.resampler = resampler
        self.model = {}
        for i in range(N_models):
            # create model i
            self.model[i] = Logit_model(package=package, polynomial_degrees = polynomial_degrees,
                                        resampler = resampler, X = X, y = y)
    
    def fit(self, X_train, y_train, val_share = 0.2):
        
        """
        Purpose: Train all model instances in the bagging procedure.
        

        Inputs:
        -------
        \t X_train, y_train: \t Training data
        \t resampling_option: \t 'None': No resampling is performed
        \t                   \t 'undersampling': random undersampling of the majority class
        \t                     \t 'SMOTE': SMOTE methodology applied
        \t callbacks: \t callbacks for training
        \t val_share, N_epochs, N_batch: \t Additional arguments for training

        Outputs:
        --------
        \t None. Updates parameters of all models in self.model
        """
        
        for i in range(len(self.model)):
            # utilze concept of resampling
            if self.resampler == 'undersampling':
                X,y = RandomUnderSampler(sampling_strategy= 'majority').fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            elif self.resampler == 'SMOTE':
                X,y = SMOTE().fit_resample(X=X_train, y=y_train)
                # shuffle data, otherwise all oversampled data are appended
                X,y = sklearn.utils.shuffle(X,y)
            else:
                X,y = X_train, y_train
                X,y = sklearn.utils.shuffle(X,y)
            
            self.model[i].fit(X=X, y = y, val_share = val_share)
        
        # Return model(s) to allow for shorter/ single-line notation, i.e. Logit_bagging().fit()
        return self

        
    def predict_proba(self, X):  
        
        """
        Purpose: Predict event probability for data
        
        output:
        \t Predictions for all input data
        input:
        \t X: \t Input data
        """
        pred = self.model[0].predict_proba(X)

        for i in range(1,len(self.model)):
            pred+=self.model[i].predict_proba(X)

        return pred/len(self.model)
    

    def predict(self, X):
        
        """
        Purpose: Predict label for data
        
        Replicate predict_proba method of Sequential() or Model() class to unify notation.
        See documentation of self.predict() method.
        """
        return self.predict_classes(X)
    

    def predict_classes(self, X, threshold = 0.5):
        
        """
        Purpose: Predict class memberships/ labels for data
        
        Replicate predict_classes method of Sequential() or Model() class to unify notation.
        """
        return (self.predict_proba(X)> threshold)        


class Tree_Classifier:
    
    '''
    Build a tree based classifier. Fitting is based on pruning w.r.t. binary crossentropy, i.e. log_loss().
    For the RandomForestClassifier option we prune our tree automatically at a max_depth=5
    
        criterion:   Method for binary splits of tree fitting procedure, {'gini, 'entropy'}
        bool_forest: Boolean to decide whether a DecisionTreeClassifier (False) or a RandomForestClassifier (True) will be built.
        resampling:  Indicates if a resampling strategy is used {'SMOTE', 'undersampling'}, or not {'None'}    
    '''
    
    def __init__(self, criterion = 'gini', bool_cv = False, bool_forest = False, N_trees = 1, alpha = 0,
                 resampling = 'None'):
        
        if bool_forest == False:
            self.model = sklearn.tree.DecisionTreeClassifier(criterion= criterion, ccp_alpha = alpha)
        else:
            self.N_trees = N_trees
            self.model = sklearn.ensemble.RandomForestClassifier(criterion = criterion, n_estimators=self.N_trees,
                                                   max_depth= 5, ccp_alpha = alpha )
        self.criterion = criterion
        self.resampling = resampling
        self.bool_forest = bool_forest
        self.bool_cv = bool_cv
        
    def fit(self, X, y, val_share = 0.2, max_depth = 10):
        
        '''
        Fit classifier, including a pruning procedure.
        Pruning is performed w.r.t. binary_crossentropy evaluated on a validation set and up to a maximal depth.
        
        
        Parameters:
        -----------
            val_share: Determines share of training data used for validation
            max_depth: Maximum depth considered in pruning procedure
        '''
        
        if self.resampling == 'SMOTE':
            X,y = SMOTE().fit_resample(X,y)
            # shuffle data, otherwise all oversampled data are appended
            X,y = sklearn.utils.shuffle(X,y)
        elif self.resampling == 'undersampling':
            X,y = RandomUnderSampler().fit_resample(X,y)
            # shuffle data, otherwise all oversampled data are appended
            X,y = sklearn.utils.shuffle(X,y)
        elif self.resampling == 'None':   
            pass # do nothing
        else:
            print('Error: Resampling Option is not yet implemented!')
        
        
        if self.bool_forest == False:
            # Perform pruning for DecisionTreeClassifier
            
            if self.bool_cv:
                model_cv = sklearn.model_selection.GridSearchCV(estimator=self.model, 
                                               param_grid= {'ccp_alpha':[0, 0.001, 0.0001, 0.00001, 0.000001],
                                                           'criterion':['gini', 'entropy']})
                model_cv.fit(X,y)
                self.model =  model_cv.best_estimator_
                
            else:
                # pruning purely wrt max_depth and validated entropy-loss
                classifier = {}
                classifier_eval = {}
                for i in range(1,max_depth+1):
                    # Build models up to max_depth
                    classifier[i] = sklearn.tree.DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf = 20,
                                                        max_depth=i).fit(X=X[0:int((1-val_share)*len(X))], 
                                                                          y=y[0:int((1-val_share)*len(y))])
                    # Evaluate log_loss of models
                    classifier_eval[i] = sklearn.metrics.log_loss(y_true = y[int((1-val_share)*len(y)):],
                                                   y_pred = classifier[i].predict_proba(X=X[int((1-val_share)*len(X)):])[:,-1])

                best = 1+np.argmin(list(classifier_eval.values()))
                #plt.plot([i for i in range(1,max_depth+1)], [classifier_eval[i] for i in range(1,max_depth+1)])

                print('Note: Pruning of tree classifier sucessful with max_depth = {}'.format(best))

                # Build model w.r.t. optimal depth
                self.model = classifier[best]
        else:
            # Build RandomForestClassifier with a imposed max_depth=5
            self.model.fit(X, y)
            
        # return object to allow for compact notation of e.g. Tree_classifier().fit()
        return self
        
        
    def predict_proba(self, X):
        '''
        Predict event probability of data X
        '''
        return self.model.predict_proba(X)
        
        
    def predict(self, X):
        '''
        Predict class membership of data X
        '''
        return self.model.predict(X)
        
        
    def predict_classes(self, X):
        '''
        Predict class membership of data X
        '''
        
        pred = np.zeros(shape=(len(X),2))
        pred_class = self.model.predict(X)
        pred[:,1] = pred_class
        pred[:,0] = 1- pred_class
        
        return pred