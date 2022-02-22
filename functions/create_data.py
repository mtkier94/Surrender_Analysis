import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from functions.sub_surrender_profiles import get_risk_drivers, get_age_coeff, get_duration_coeff, get_duration_elapsed_coeff, \
                                    get_prem_freq_coeff, get_prem_annual_coeff, get_time_coeff



def get_integer_death_times(ages_PH, mortality_params):
    
    '''
    Get death times (integer age) based on integer valued ages of policy holders
    We can either work with these integer valued ages or use them as refined starting values for a non-integer solution
    by Newton's method (-> Avoid problem of approx. zero-valued derivative for poor initial value)
    '''
    
    
    # Extract info about Number of contracts
    N_individuals = ages_PH.shape[0]
    death_times = np.zeros(shape = (N_individuals,))
    
    # Simulate Death Times, i.e. potentially censoring times
    np.random.seed(42)
    death_times_prob = np.random.uniform(low=0, high=1, size= N_individuals)
    
    for i in range(N_individuals):
        for t in range(mortality_params['age_max']-int(ages_PH[i])+1):
            # if survival prob < simulated death prob -> death
            if np.exp(-mortality_params['A']*t-mortality_params['B']/np.log(mortality_params['c'])*mortality_params['c']**int(ages_PH[i])*(mortality_params['c']**t-1)) < death_times_prob[i]:
                death_times[i] = int(ages_PH[i])+t
                break
    
    return death_times


def get_noninteger_death_times(ages_PH, mortality_params):
    
    '''
    Get death times (exact)
    To avoid failure of Newton's method due to zero-derivative and poor starting values we use integer-approx. death times
    as starting points
    '''
    
    # Extract info about Number of contracts
    N_individuals = ages_PH.shape[0]
    death_times = np.zeros(shape = (N_individuals,))
    
    death_times_int = get_integer_death_times(ages_PH, mortality_params)
    
    # Simulate Death Times, i.e. potentially censoring times.
    # Note: Seed is identical to seed for death_times_prob in get_integer_death_times (!!!)
    np.random.seed(42)
    death_times_prob = np.random.uniform(low=0, high=1, size= N_individuals)
    
    for i in range(N_individuals):
        death_times[i] = ages_PH[i] + sc.optimize.newton( lambda t: np.exp(-mortality_params['A']*t-mortality_params['B']                                                   /np.log(mortality_params['c'])*mortality_params['c']**ages_PH[i]*                                                   (mortality_params['c']**t-1))-death_times_prob[i],
                   x0=death_times_int[i]-ages_PH[i])
    
    return death_times


def surv_prob_SUSM(x, t, params):
    
    '''
    get SUSM survival probability
    x: current age
    t: period of time for survival
    params: dictionary with parameters A, B, c of SUSM model
    '''
    return np.exp(-params['A']*t-params['B']/np.log(params['c'])*params['c']**(x)*(params['c']**t-1))


def geom_sum(x,n,m):
    
    '''
    Compute n-1-th partial sum sum_{i=0}^{n*m-1} x^(i/m) [= (x^{nm/n} -1)/(x^(1/m)-1)]
    x: value, i.e. discount factor
    n: duration
    m: frequency, i.e. m=12: monthly payments
    '''   

    if m == 0: # lump sum payment
        return (x**n-1)/(x-1)
    else: # (intra-) yearly payments
        return (x**(n) -1)/(x**(1/m)-1)


def get_annuity(x, duration, mortality_params, management_params, m = 1):
    
    '''
    Calculate annuity of WL insurance
    x: current age
    duration: period of time for annuity payments
    interest: actuarial interest rate for accounting
    params: dictionary with parameters A, B, c of SUSM model
    m: frequency of (premium) payments, i.e. m= 12 (monthly), m = 1 (annually), m=0 (one time payment)
    '''
    
    v = 1/(1+management_params['int_rate'])
    ann = 0
    # Note: duration has to be positive. Premiums are only paid up to retirement
    # i.e.: Contracts set up for PH older than retirement have a lump sum payment
    for k in np.linspace(start=0, stop= int(np.ceil(duration*m)), num= int(max(np.ceil(duration*m),0)),endpoint=False):
        
        if x+k/m < management_params['age_retire']:
            ann+= 1/m*v**(k/m)*surv_prob_SUSM(x, t=k/m, params=mortality_params)  
    
    # Catch special case where m=0 (single premium) or 
    # duration = 0 (e.g. PH makes only one premium payment before retirement)
    if duration*m==0:
        # lump sum payment and no discounting (by interest or survival probability)
        # note: for m==0 is np.linspace(..) = [] empty
        ann = 1 
    
    return ann



def get_APV_ben(x, duration, mortality_params,management_params, survival_benefit = True):
    
    '''
    Calculate apv of benefits for WL insurance
    x: current age
    duration: period of time for annuity payments
    interest: actuarial interest rate for accounting
    params: dictionary with parameters A, B, c of SUSM model

    No frequency parameter m!
    Assumption that all benefits get paid at the end of the corresponding year (not month or similar) of death
    '''
    
    v = 1/(1+management_params['int_rate'])
    apv = 0
    # Term Life Benefit
    for k in np.linspace(start=0, stop= int(duration), num= int(duration),endpoint=False):
        apv += v**(k+1)*surv_prob_SUSM(x, t=k, params=mortality_params)*(1-surv_prob_SUSM(x=x+k,t=1,params=mortality_params))
    
    # Endowment Survival Benefit
    if survival_benefit:
        apv += v**(duration)*surv_prob_SUSM(x, t=duration, params=mortality_params)
    
    return apv


def get_fair_premium(age_init,Sum_ins,duration, prem_freq, mortality_params, expenses_params, management_params):
    
    
    '''
    Calculate premium for WL insurance
    age init: age at the start of the contract
    Sum_ins: Face Amount
    duration: period of time for annuity payments
    m: frequency of premium payments, i.e. m= 12 (monthly), m = 1 (annually), m=0 (one time payment)
    interest: actuarial interest rate for accounting
    params: dictionary with parameters A, B, c of SUSM model

    Formula: P*m*ä^(m)_x:t = S*A_x:n + alpha*(t*m*P) + (gamma*S*ä_x:n+beta*m*P*ä^(m)_x:t
             n: duration of contract; 
             t: duration of premium payments, i.e. up to retirement (if no lump sum payment (m=0))
    Note: P is m-thly, S is yearly (assumption: Sum insured always paid at the end of the year)
          gamma charge: administrative (general) w.r.t. sum insured 
                         -> yearly up to end of contract
          beta charge: administrative (specific) w.r.t. payment structure 
                         -> m-thly up to end of prem. payments
    Also: Remember that annuity ä^(m)_x:n = sum_{k=0}^{m*n-1} 1/m*v^{k/m}* {}_{k/m}p_x
          -> Faktor 1/m scales yearly sums/ benefits, i.e. m-thly premium P need to be rescaled as m*P

    Final Formula: P = S(A_x:n + gamma*ä_x:n) / [m*(ä^(m)_x:t*(1-beta) - alpha*t)]

    Special case m= 0 (Premium payment upfront): P = S(A_x:n+gamma*ä_x:n)/(1-beta-alpha)
    '''
    
    if prem_freq > 0:
        # remaining time [yrs] for premium payments
        # Note: non-integer values -> flooring of t*m in APV and annuity calcultion
        t = min(management_params['age_retire'] - age_init, duration)
        
        
        return Sum_ins/prem_freq*(get_APV_ben(x=age_init, duration= duration, management_params=management_params, 
                                              mortality_params=mortality_params)+\
                        expenses_params['gamma']*get_annuity(x=age_init, duration=duration, 
                                                             management_params=management_params, 
                                                             mortality_params = mortality_params,m = 1))/\
                        (get_annuity(x=age_init, duration=t, management_params=management_params, 
                                   mortality_params = mortality_params, m = prem_freq)*(1-expenses_params['beta'])-\
                        expenses_params['alpha']*t)
    elif prem_freq == 0:
        
        return Sum_ins*(get_APV_ben(x=age_init, duration= duration, management_params=management_params,
                                    mortality_params=mortality_params)+ \
                        expenses_params['gamma']* get_annuity(x=age_init, duration=duration, 
                                                              management_params= management_params, 
                                                              mortality_params =mortality_params, m=1))/\
                        (1-expenses_params['beta']- expenses_params['alpha'])
        
    else:
        print('Error: Invalid payment structure "m"')
        return


def get_premiums_portfolio(portfolio_df, mortality_params, expenses_params, management_params):
    
    '''
    Apply get_fair_premium to the whole insurance portfolio
    Includes e.g. a premium loading (-> management_params)    
    
    Input
    \t portfolio_df: \t DataFrame with columns 'Age', 'Face_amount', 'Duration', 'Premium_freq'
    \t mortality_params: \t information on raw mortality assumption; 
    \t                  \t dictionary with 'A', 'B', 'c' (see SUSM model), 'age_max'
    \t expenses_params: \t information on expenses; dictionary with 'alpha', 'beta', 'gamma'
    \t management_params: \t information on management assumptions; 
    \t                   \t dictionary with 'int_rate', 'premium_loading', 'age_retire'
    
    Output
    \t Series with (loaded) premiums
    '''
    
    N = len(portfolio_df.index)
    Premium = np.zeros(N)
    
    for i in range(N):
        Premium[i] = get_fair_premium(age_init=portfolio_df['Age_init'][i],Sum_ins=portfolio_df['Face_amount'][i],
                                     # Duration of Whole Life encoded by max_age = 125
                                     # Adapt duration to remaining time up to max_age
                                     duration=portfolio_df['Duration'][i],#max(portfolio_df['Ins_dur'][i], 
                                                 # np.ceil(mortality_params['age_max']-portfolio_df['Age'][i])),
                                     prem_freq= portfolio_df['Premium_freq'][i], mortality_params= mortality_params, 
                                     expenses_params= expenses_params, management_params = management_params)*\
                     (1+management_params['prem_loading'])
                    #Note: Fixed loading included
    
    return Premium


# ## Simulate Surrender

def get_surrender_coeff(df,  profile =0, bool_errors = True):
    
    '''
    Combine all individual features' coefficients to obtain \beta^T*x
    
    Parameter
    ----------
        df:         DataFrame with columns that match names in features_lst
        profile:    integer, representing some lapse profile
                    0: Fitted Logit Model in Milhaud 2011
                    1: Empirical effects in Data in Milhaud 2011
        bool_error: Boolean, display error messages when feature no risk driver in profile
        
    '''
    #\t features_lst: \t list of features to be considered in true surrender model
    #\t               \t Options: 'Age', 'Duration', 'Duration_elapsed', 'Premium_freq', 'Premium_annual', 'Face_amount'
    
    
    coeff = np.zeros(shape=(df.shape[0],))
    
    risk_drivers = get_risk_drivers(profile)
    if bool_errors:
        print('Risk drivers in profile {} are restricted to {}'.format(profile, risk_drivers))
    
    for feature in risk_drivers: #features_lst:
        if feature == 'Age':
            coeff += get_age_coeff(df[feature], profile=profile)
        elif feature == 'Duration':
            coeff += get_duration_coeff(df[feature], profile=profile)
        elif feature == 'Duration_elapsed':
            coeff += get_duration_elapsed_coeff(dur= df[feature], profile = profile)
        elif feature == 'Duration_remain':
            coeff += get_duration_elapsed_coeff(df[feature], profile=profile)
        elif feature == 'Premium_freq':
            coeff += get_prem_freq_coeff(df[feature], profile=profile)
        elif feature == 'Premium_annual':
            coeff += get_prem_annual_coeff(df[feature], profile=profile)
        elif feature == 'Time':
            coeff += get_time_coeff(df[feature],profile=profile)
        #elif feature == 'Face_amount':
        #    print('Note: Face_amount has no implemented coefficient')
        
    
    return coeff


def get_surrender_prob(df, profile_nr = 0, adjust_baseline = False, target_surrender = None, beta0 = 0,
                      rand_noise_var = 0, bool_errors = True):
    
    ''' 
    Combine coefficients for age, duration, premium frequency and annual premium amount in a logit model
    To obtain a reasonable average surrender rate: adapt by introducing a baseline factor beta0

    Parameter:
    ----------
    \t df: \t \t DataFrame with columns that match names in features_lst
    \t features_lst: \t list of features to be considered in true surrender model
    \t               \t Options: 'Age', 'Duration', 'Duration_elapsed', 'Premium_freq', 'Premium_annual', 'Face_amount'
    \t profile_nr: \t integer value indicating a predefined surrender profile for the simulation
    \t          \t 0: Fitted Logit Model in Milhaud 2011
    \t          \t 1: Empirical effects in Data in Milhaud 2011
    \t adjust baseline: \t Boolean whether baseline to be adjusted
    \t target_surrender: \t target value for mean surrender of portfolio. Used to adjust beta0 if adjust_baseline == True
    \t beta0: \t baseline surrender factor beta0 (optional user input)
    \t rand_noise_var: \t variance of a centered, gaussian noise that is added to the logit-model for surrender probs
    \t bool_error:       Boolean, display error messages when feature no risk driver in profile
    
    Outputs
    -------
    \t array of surrender activity
    \t underlying beta0 coeff (optional, if adjust_baseline==True)
    '''
    
    odd_ratio = np.exp(get_surrender_coeff(df=df, profile =profile_nr, bool_errors=bool_errors))
     
    
    if adjust_baseline:
        beta0 = sc.optimize.newton(func= lambda x: target_surrender - (np.exp(x)*odd_ratio/(1+np.exp(x)*odd_ratio)).mean(),
                                   x0 = -1)
        
    # Adjust odd_ratio by baseline factor beta0 (either user_input or determined to fit target_surrender rate)    
    odd_ratio = odd_ratio*np.exp(beta0)

        
    # Add random noise
    # Note standard modeling assumption: log(p/1-p) = betas*X + noise
    odd_ratio = odd_ratio*np.exp(np.random.normal(loc = 0, scale = rand_noise_var,size = odd_ratio.size))
    
    if adjust_baseline: # Also return adjusted beta0 coefficient (-> to replicate results)
        return (odd_ratio/(1+odd_ratio), beta0)
    else: # return purely probabilites
        return odd_ratio/(1+odd_ratio)



def simulate_surrender(df, profile_nr, adjust_baseline, target_rate = None, beta0 = 0, simulation_noise_var = 0, bool_errors = True):
    
    '''
    Simulate a 1-year surrender in the portfolio with given surrender profile(s)

    Inputs
    \t df: \t \t DataFrame with columns that match names in features_lst & 'Lapsed' column
    \t features_lst: \t list of features to be considered in true surrender model
    \t               \t Options: 'Age', 'Duration', 'Duration_elapsed', 'Premium_freq', 'Premium_annual', 'Face_amount'
    \t profile_nr: \t integer value indicating a predefined surrender profile for the simulation
    \t          \t 0: Fitted Logit Model in Milhaud 2011
    \t          \t 1: Empirical effects in Data in Milhaud 2011
    \t target_surrender: \t target value for mean surrender of portfolio. Used to adjust beta0 (if adjust_baseline == True)
    \t adjust_baseline: \t Boolean whether baseline factor beta0 should be determined to match some target_surrender rate
    \t beta0: \t baseline surrender factor beta0. Default = 0.
    \t simulation_noise_var: \t variance of a centered, gaussian noise that is added to the logit-model for surrender probs    
    \t bool_error:           Boolean, display error messages when feature no risk driver in profile
    '''
    
    
    N = df['Age'].size
    
    #np.random.seed(8) # set seed in simulate_surrender_time_series()
    # simulated probs to compare with surrender probs later obtained by get_surrender_prob()
    sim_probs = np.random.uniform(size = N)
    

    # surrender probs
    # if adjust_baseline == True: val[0]: surrender_probs, val[1]: adjusted beta0
    # if adjust_baseline == False: val: surrender_probs
    val = get_surrender_prob(df=df,
                             profile_nr= profile_nr, 
                             adjust_baseline= adjust_baseline, beta0= beta0,
                             target_surrender= target_rate, rand_noise_var = simulation_noise_var, 
                             bool_errors= bool_errors)

    if adjust_baseline: # also return corresponding beta0 coefficient (for later comparison)
        # return 0: active (after period), 1: surrender (within period)
        return ((sim_probs < val[0]).astype('int'), val[1])
    else:
        return (sim_probs < val).astype('int')



def simulate_surrender_time_series(df, target_rate, profile_nr, modeling_time_step = 1/12, 
                                   simulation_noise_var = 0, rnd_seed = 0):
    
    '''
    Simulate the portfolio decomposition over time, i.e. iteratively apply simulate_surrender()
    Important: Fix the baseline hazard beta0 determined at initial time 0 and apply it for consecutive points in time
    
    Parameter
    ---------
    \t df: \t \t DataFrame with columns that match names in features_lst & 'Lapsed' column
    \t features_lst: depreciated
                     \t list of features to be considered in true surrender model
    \t               \t Options: 'Age', 'Duration', 'Duration_elapsed', 'Premium_freq', 'Premium_annual', 'Face_amount'
    \t target_surrender: \t target value for mean surrender of portfolio. Used to adjust beta0 (adjust_baseline == True default)
    \t profile_nr: \t integer value indicating a predefined surrender profile for the simulation
    \t          \t 0: Fitted Logit Model in Milhaud 2011
    \t          \t 1: Empirical effects in Data in Milhaud 2011
    \t modeling_time_step: \t frequency in which we iteratively apply the modelling of surrender, i.e. 1/12=monthly, 1=annually 
    \t simulation_noise_var: \t variance of a centered, gaussian noise that is added to the logit-model for surrender probs    
    '''
    
    # set seed
    np.random.seed(rnd_seed)
    
    TS_length = int(df['Duration'].max()/modeling_time_step) +1 # Note: simulate monthly surrender
    
    # Initialize time series
    TS_portfolio = [None]*TS_length
    #Initial value
    TS_portfolio[0] = pd.DataFrame.copy(df) 
    
    
    # Dataframe to record time of event and type of event (lapse) for each contract
    df_events = pd.DataFrame(data = None, columns = ['event_time', 'event_type'], index = range(df.shape[0]))
    # Maturity Event (potentially censored by death of surrender)
    df_events.loc[:,'event_type'] = 2
    df_events.loc[:,'event_time'] = df['Duration']
    # Death Events (potentially censored by surrender)
    df_events.loc[df['Death']<df['Age']+df['Duration'],'event_type'] = 3
    df_events.loc[df['Death']<df['Age']+df['Duration'],'event_time'] = df['Death']-df['Age']
    
    ####### Indicate lapse due to maturity (level 2) ###########
    # This can be overwritten by lapse due to death or surrender
    TS_portfolio[0].loc[(TS_portfolio[0]['Duration_remain']-modeling_time_step)<0,
                        'Lapsed'] = 2
    
    ######## Indicate lapse due to death (level 3) ###########
    # Can censor simulated surrender
    TS_portfolio[0].loc[(TS_portfolio[0]['Age']+modeling_time_step>TS_portfolio[0]['Death']) & (TS_portfolio[0]['Death']<TS_portfolio[0]['Age']+TS_portfolio[0]['Duration']),'Lapsed']= 3

    ######## Indicate surrender (level 1) ##############
    surrender = simulate_surrender(df = TS_portfolio[0], #features_lst= features_lst,
                                   profile_nr= profile_nr, adjust_baseline = True, 
                                   target_rate= target_rate, simulation_noise_var= simulation_noise_var)    
    
    TS_portfolio[0].loc[surrender[0]==1,'Lapsed'] = 1
    beta0 = surrender[1]
    # prospective, dataframe
    df_events.loc[TS_portfolio[0].loc[(TS_portfolio[0]['Lapsed']== True),:].index,'event_type'] = 1
    df_events.loc[TS_portfolio[0].loc[(TS_portfolio[0]['Lapsed']== True),:].index,
                  'event_time'] = modeling_time_step    
  
    for i in range(1,TS_length):
        
        # Check if there are still active contracts
        if sum(TS_portfolio[i-1]['Lapsed']==0)>0:
            # Advance time for TS, drop contracts that lapsed at the previous time step
            TS_portfolio[i] = pd.DataFrame(data = TS_portfolio[i-1][TS_portfolio[i-1]['Lapsed']==0])
            
            # Adjust Features by advance of time
            TS_portfolio[i]['Age'] += modeling_time_step
            TS_portfolio[i]['Time'] += modeling_time_step
            TS_portfolio[i]['Duration_elapsed'] += modeling_time_step
            TS_portfolio[i]['Duration_remain'] -= modeling_time_step
            
            
        ###########  Indicate lapse due to maturity (level 2) ############
        # Note: Don't use 0 but 10**(-10) as threshold to compensate for rounding errors of e.g. 1/12 steps
            TS_portfolio[i].loc[((TS_portfolio[i]['Duration_remain']-modeling_time_step+10**(-10))<0),#&index_bool,
                                'Lapsed'] = 2
            
        ############# Indicate lapse due to death (level 3) ###############
            TS_portfolio[i].loc[((TS_portfolio[i]['Age']+modeling_time_step)>TS_portfolio[i]['Death'])&  (TS_portfolio[i]['Death']<TS_portfolio[i]['Age']+TS_portfolio[i]['Duration_remain']), 'Lapsed'] = 3              
            
        ########### Indicate surrender (level 1) #############
            # Note: Keep beta0 fixed (determined at initial time 0 to match some empirical target surrender rate) 
            surrender = simulate_surrender(df = TS_portfolio[i-1], #features_lst= features_lst,
                                           profile_nr= profile_nr, 
                                           adjust_baseline = False, target_rate= target_rate, 
                                           beta0 = beta0, 
                                           simulation_noise_var= simulation_noise_var,
                                          bool_errors=False) # Dont display messages for feature that are no risk driver
            TS_portfolio[i].loc[surrender==1,'Lapsed']=1
            
            # index of non-surendered contracts
            # Implicit assumption: surrender happens bevore maturity or death
            index_bool = (TS_portfolio[i].loc[:,'Lapsed']==1)
                                                           
        #### Update prospective, dataframe ####
            df_events.loc[TS_portfolio[i].loc[index_bool,:].index,'event_type'] = 1
            df_events.loc[TS_portfolio[i].loc[index_bool,:].index,'event_time'] = (i+1)*modeling_time_step
        else:
            # return TS_portfolio prematurely (due to the lack of active contracts)
            return (TS_portfolio[0:i], beta0, df_events)
    
    return TS_portfolio, beta0, df_events


def visualize_time_series_lapse(lst_of_df, modeling_time_step = 1/12, zoom_factor = 1, fig_size = (12,6), 
                                option_view= 'lapse_decomp', option_maturity = True, 
                                option_annualized_rates = True, title = ""):
    
    '''
    Visualize lapse (i.e. surrender, death and maturation of contracts) 
    Type of lapse is recorded in lst_of_df for each time t in column 'Lapsed' 
    Encoding 0: active (at the end of period), 1: surrender (during period), 2: maturity, 3: death (during period)
    Relate this to the respective at-risk-sets at time t
    Also: Illustrate the at-risk-set as a countplot
    
    Inputs:
    \t lst_of_df: \t time series (list of DataFrames) created by def simulate_surrender_time_series()
    \t zoom_factor: \t when plottling Surrender, Death and Maturity Rates zoom in to ignore large e.g. 100% maturity rate at the end
    \t modeling_time_step: \t frequency in which we iteratively apply the modelling of surrender, i.e. 1/12=monthly, 1=annually 
    \t option: \t indicate type of 2nd plot. Either 'lapse_decomp', i.e. decomposition of lapse activity over time,
                                       or 'abs_view', i.e. absolute numbers of lapse components    
    '''
    
    
    N_ts = len(lst_of_df)
    
    ### Step 1: Compute Statistics
    stats = pd.DataFrame(data = np.zeros(shape= (N_ts,4)), columns = ['Surrender', 'Maturity','Death', 'AtRisk'])
    
    for i in range(N_ts):
        stats.loc[i, 'AtRisk'] = lst_of_df[i].shape[0]
        stats.loc[i, 'Surrender'] = sum(lst_of_df[i]['Lapsed']==1)
        stats.loc[i, 'Maturity'] = sum(lst_of_df[i]['Lapsed']==2)
        stats.loc[i, 'Death'] = sum(lst_of_df[i]['Lapsed']==3)
     
    ### Step 2: Plot exposure and lapse rates
    
    # Set up x-values for plot
    x_grid = np.linspace(0,(stats.shape[0]-1)*modeling_time_step,stats.shape[0])
    x_cut = int(zoom_factor*len(x_grid))
    
    # create canvas for plots
    fig, ax = plt.subplots(1,2, figsize = fig_size)
    ax2 = ax[0].twinx()
    ax2.set_ylabel('Rate' +option_annualized_rates*' (p.a.)')#, fontsize = 'large')
    ax2.tick_params(axis='y') 
    ax[0].set_xlabel((modeling_time_step==1)*'Year'+(modeling_time_step ==1/12)*'Month'+  (modeling_time_step ==1/4)*'Quarter')
    ax[0].set_ylabel('Active Contracts')
    
    # Plot number of active contracts
    ax[0].bar(x = x_grid[0:x_cut], height = (stats['AtRisk'][0:x_cut]), color = 'grey', 
              alpha = .8, width = modeling_time_step)
    # Plot Surrender, Death and Maturity Rates
    if option_annualized_rates:
        ax2.step(x = x_grid[0:x_cut], 
                 y = (1+stats['Surrender'][0:x_cut]/stats['AtRisk'][0:x_cut])**(1/modeling_time_step)-1, 
                 color = 'blue', label = 'Surrender')
        ax2.step(x = x_grid[0:x_cut], 
                 y = (1+stats['Death'][0:x_cut]/stats['AtRisk'][0:x_cut])**(1/modeling_time_step)-1, 
                 color = 'orange', label = 'Death')
        if option_maturity:
            ax2.step(x = x_grid[0:x_cut], 
                     y = (1+stats['Maturity'][0:x_cut]/stats['AtRisk'][0:x_cut])**(1/12)-1, 
                     color = 'green', label = 'Maturity')
    else:
        ax2.step(x = x_grid[0:x_cut], y = stats['Surrender'][0:x_cut]/stats['AtRisk'][0:x_cut], 
                 color = 'blue', label = 'Surrender')
        ax2.step(x = x_grid[0:x_cut], y = stats['Death'][0:x_cut]/stats['AtRisk'][0:x_cut], 
                 color = 'orange', label = 'Death')
        if option_maturity:
            ax2.step(x = x_grid[0:x_cut], y = stats['Maturity'][0:x_cut]/stats['AtRisk'][0:x_cut], 
                     color = 'green', label = 'Maturity')
    ax2.legend(loc = 'upper center')#loc = (0.2,0.8-0.1*(title != "")))
    fig.suptitle(title, fontsize = 'large')
    
    
    # Step 3: Plot decomposition (%-wise) of surrender over time
    
    if option_view == 'lapse_decomp':
        stats_lapses = stats['Surrender']+stats['Maturity'] +stats['Death']
        # Avoid NA by deviding by 0
        stats_lapses[stats_lapses==0] = 1

        # initialize dataframe
        stats_lapsed_decomp = pd.DataFrame(data = None, columns = ['Surrender', 'Maturity','Death', 'Active'])
        # fill in new values
        stats_lapsed_decomp['Surrender'] = stats['Surrender']/stats_lapses
        stats_lapsed_decomp['Maturity'] = stats['Maturity']/stats_lapses
        stats_lapsed_decomp['Death'] = stats['Death']/stats_lapses
        stats_lapsed_decomp['Active'] = 1-(stats['Surrender']+stats['Maturity']+stats['Death'])/stats_lapses

        # plot data
        ax[1].fill_between(x=x_grid, 
                             y1=stats_lapsed_decomp['Surrender']+stats_lapsed_decomp['Maturity']+stats_lapsed_decomp['Death'],
                             y2=stats_lapsed_decomp['Surrender']+stats_lapsed_decomp['Maturity'], 
                             color = 'orange', label = 'Death') 
        ax[1].fill_between(x=x_grid, y1 = stats_lapsed_decomp['Surrender']+stats_lapsed_decomp['Maturity'], 
                         y2=stats_lapsed_decomp['Surrender'], alpha =.6, color = 'green', label = 'Maturity')


        ax[1].fill_between(x=x_grid, y1 = 0, y2=stats_lapsed_decomp['Surrender'], color = 'blue', label = 'Surrender')
        # create white label for 'No Lapse', i.e. times when we obsere no lapses at all
        ax[1].fill_between(x=x_grid,y1=-1, y2=-1,color = 'white', label= 'No Lapse')



        #plt.fill_between(x=x_grid, y1 = 0, y2=1, color = 'blue', interpolate= True)

        ax[1].set_ylim((-0.05,1.05))
        ax[1].set_ylabel('Decomposition of Lapse Activity')
        ax[1].set_xlabel((modeling_time_step==1)*'Year'+(modeling_time_step ==1/12)*'Month'+                         (modeling_time_step ==1/4)*'Quarter')
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow = False, edgecolor = 'black')
        
    else: # plot absolute numbers
        ax[1].plot(x_grid, stats.Surrender, label = 'Surrender', color = 'blue')
        ax[1].plot(x_grid, stats.Maturity, label = 'Maturity', color = 'green')
        ax[1].plot(x_grid, stats.Death, label = 'Death', color = 'orange')
        ax[1].legend()
        ax[1].set_xlabel((modeling_time_step==1)*'Year'+(modeling_time_step ==1/12)*'Month'+                         (modeling_time_step ==1/4)*'Quarter')        
        ax[1].set_ylabel('No. of observations')
        
    plt.tight_layout(rect = (0,0,.95,.95))
    plt.show()
        

    
    return stats#, stats_lapsed_decomp


def create_summary_for_lapse_events(events_df, profile_name = ''):
    
    '''
    Print summary of events including surrender, maturity and death activity
    
    Input
    events_df: DataFrame with columns 'Surrender', 'Maturity', 'Death', 'AtRisk'
        rows relate to points in time         
        events type dataframe is output of visualize_time_series_lapse()
    '''
    
    N_contracts = int(events_df.AtRisk[0])
    
    print('Overview for Surrender Profile '+ profile_name)
    print('---------------------------------------------------')

    print(events_df.head())
    print('... \t\t ... \t\t ...')

    print('---------------------------------------------------')
    print('\n Overall number of contracts: {} \n'.format(N_contracts))
    print( '\t \t contracts lapsed due to surrender: '+str(int(events_df.Surrender.sum())) + '  ('           + str(np.round(events_df.Surrender.sum()/N_contracts*100,decimals =2))+'%)')
    print( '\t \t contracts lapsed due to maturity: '+str(int(events_df.Maturity.sum())) + '  ('           + str(np.round(events_df.Maturity.sum()/N_contracts*100,decimals =2))+'%)')
    print( '\t \t contracts lapsed due to death: '+str(int(events_df.Death.sum())) + '  ('           + str(np.round(events_df.Death.sum()/N_contracts*100,decimals =2))+'%)')
    print('\n Overall number of datapoints: ' + str(int(events_df.AtRisk.sum())))
    print('\t\t share of surrender events: ' +           str(np.round(events_df.Surrender.sum()/events_df.AtRisk.sum()*100, decimals = 2))+'%')
    print('\t\t share of maturity events: ' +           str(np.round(events_df.Maturity.sum()/events_df.AtRisk.sum()*100, decimals = 2))+'%')
    print('\t\t share of death events: ' +           str(np.round(events_df.Death.sum()/events_df.AtRisk.sum()*100, decimals = 2))+'%')


