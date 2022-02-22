import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd

from functions.sub_surrender_profiles import get_risk_drivers, get_age_coeff, get_duration_coeff, get_duration_elapsed_coeff, get_duration_remain_coeff, \
                                    get_prem_freq_coeff, get_prem_annual_coeff, get_time_coeff, get_risk_drivers
from functions.sub_actuarial_functions import get_premiums_portfolio



def simulate_contracts(N = 30000, option_new_business = False):


    '''
    Simulate endowment insurance contracts.

    Inputs:
    --------
        N: number of contracts
        options_new_business: True: Simulate only new business; False: Simulate new and existing business. (Default: False)

    Outputs:
    --------
        DataFrame with N columns and rows 
    '''

    # Parameters
    N_contracts = N
    mortality_params = {'A': 0.00022, 'B': 2.7*10**(-6), 'c': 1.124, 'age_max': 125} # SUSM Dickson, Hardy, Waters
    expenses_params = {'alpha': 0.025,'beta': 0.03, 'gamma': 0.005} 
    # alpha: percentage of sum of ALL premiums (acquisition)
    # beta: percentage of annual premium amount (administrative)
    # gamma: percentage of sum insured, annual fee (administrative)
    # expenses chosen in line with Aleandri, Modelling Dynamic PH Behaviour through ML (p.20)
    # note legal requirements, e.g. alpha \leq 0.25 or recommended 'Höchstrechnungszins' of 0.005 by aktuar.de
    management_params = {'int_rate': 0.005, 'prem_loading': 0.15,# 'profit_sharing': 0.9, 'return_guaranteed': 0.01, 
                        'age_retire': 67}


    ######################### Underwriting age x  #################################
    # Simulate underwriting age
    age_max = 85

    np.random.seed(0)
    ages = np.random.gamma(shape = 5.5, scale = 6.8, size = N_contracts)
    ages[(ages>age_max)] = age_max*np.random.uniform(low = 0, high = 1, size = sum((ages<0)|(ages>age_max)))


    ###################### Premium Payments (Frequency m)  #######################
    # Simulate Premium Frequency: 0 - lump sum (m=0), 1 - annual (m=1), 12 - monthly (m=12)
    # Compare Milhaud, 'Lapse Tables [..]': supra-annual: 15.19%, annual: 23.44%, infra-annual: 61.37%
    np.random.seed(1)
    premium_freq = np.random.uniform(size=N_contracts)
    premium_freq[premium_freq <= 0.15] = 0
    premium_freq[premium_freq > 0.4] = 12 # Note: Order important (assign level 12 before level 1 (>0.4))
    premium_freq[(premium_freq>0.15)&(premium_freq <= 0.4)] = 1
    premium_freq = premium_freq.astype('int')

    # For all contracts where PH at start of contract > 67 premiums we assume single premiums
    premium_freq[premium_freq>0] = premium_freq[premium_freq>0]*(ages[premium_freq>0] + 1/(premium_freq[premium_freq>0]) < 67)


    ############################# Death times  #####################################
    death_times = get_noninteger_death_times(ages_PH= ages, mortality_params= mortality_params)

    ###################### Durations (of Endowments)  ####################################
    # Mean: 20 years
    np.random.seed(3)
    # assume right-skewed distr. of durations
    duration = 5+(12*np.random.gamma(shape = 5, scale = 1.5, size = N_contracts)).astype('int')/12


    # ## Elapsed duration
    if option_new_business == False:
        duration_elapsed = duration - duration*(np.random.rand(N_contracts))
        # Note: Make sure that the resulting Age_init is not <0 -> condition
        duration_elapsed = duration_elapsed*(duration_elapsed<ages)
    else:
        duration_elapsed = duration*0


    ################# Face Amounts (S)  ######################

    # Face Amount
    # Choice arbitrary -> Backtest by looking and resulting premiums and compare range and variance to Milhaud's paper
    np.random.seed(2)
    face_amounts = 5000+ np.random.gamma(shape = 4, scale = 2000, size = N_contracts)#/10#np.random.normal(loc = 800000, scale = 200000, size = N_contracts)

    # Combine Data
    Portfolio = pd.DataFrame(data = {'Time': 0,
                                    'Age': ages, 'Age_init': ages-duration_elapsed,
                                    'Face_amount': face_amounts, 
                                    'Duration': duration,
                                    'Duration_elapsed': duration_elapsed,
                                    'Duration_remain': duration - duration_elapsed,
                                    'Premium_freq': premium_freq, 'Premium': [None]*N_contracts,
                                    'Premium_annual': [None]*N_contracts, 
                                    'Lapsed': [0]*N_contracts,
                                    'Death': death_times})


    # ## Compute Premiums (P)
    Portfolio.Premium = get_premiums_portfolio(portfolio_df= Portfolio, mortality_params= mortality_params, 
                                    expenses_params= expenses_params, management_params = management_params)

    ###### Annualize Premiums (P_ann)  #####################

    # Use ceil since payments are made up to age of retirement
    # meaning: an monthly (m=12) premium payment for indivual with underwriting age 66.95 is effectively a single premium
    # since we assume no premium payments after the age of retirement (67)
    # Similarly, monthly premium payments for ind. with underwriting age 66.8 means that there will be 3 premium payments
    # Note: For contracts setup at underwriting age > 67, the premium time is indicated as negative

    premium_time = pd.DataFrame(data = [(management_params['age_retire']-Portfolio.Age),
                                        Portfolio.Duration]).apply(lambda x: np.min(x)**(np.min(x)>0), axis = 0)

    Portfolio.Premium_annual = (Portfolio.Premium_freq>0)*Portfolio.Premium*np.ceil(premium_time*Portfolio.Premium_freq)/\
                                np.ceil(premium_time)+ (Portfolio.Premium_freq==0)*Portfolio.Premium/np.ceil(premium_time**(premium_time>0))

    return Portfolio


def get_integer_death_times(ages_PH, mortality_params, seed = 42):
    
    '''
    Get death times (integer age) based on integer valued ages of policy holders
    We can either work with these integer valued ages or use them as refined starting values for a non-integer solution
    by Newton's method (-> Avoid problem of approx. zero-valued derivative for poor initial value)
    '''
        
    # Extract info about Number of contracts
    N_individuals = ages_PH.shape[0]
    death_times = np.zeros(shape = (N_individuals,))
    
    # Simulate Death Times, i.e. potentially censoring times
    np.random.seed(seed)
    death_times_prob = np.random.uniform(low=0, high=1, size= N_individuals)
    
    for i in range(N_individuals):
        for t in range(mortality_params['age_max']-int(ages_PH[i])+1):
            # if survival prob < simulated death prob -> death
            if np.exp(-mortality_params['A']*t-mortality_params['B']/np.log(mortality_params['c'])*mortality_params['c']**int(ages_PH[i])*(mortality_params['c']**t-1)) < death_times_prob[i]:
                death_times[i] = int(ages_PH[i])+t
                break
    
    return death_times


def get_noninteger_death_times(ages_PH, mortality_params, seed=42):
    
    '''
    Get death times (exact)
    To avoid failure of Newton's method due to zero-derivative and poor starting values we use integer-approx. death times
    as starting points
    '''
    
    # Extract info about Number of contracts
    N_individuals = ages_PH.shape[0]
    death_times = np.zeros(shape = (N_individuals,))
    
    death_times_int = get_integer_death_times(ages_PH, mortality_params, seed=seed)
    
    # Simulate Death Times, i.e. potentially censoring times.
    # Note: Seed is identical to seed for death_times_prob in get_integer_death_times (!!!)
    np.random.seed(seed)
    death_times_prob = np.random.uniform(low=0, high=1, size= N_individuals)
    
    for i in range(N_individuals):
        death_times[i] = ages_PH[i] + sc.optimize.newton( lambda t: np.exp(-mortality_params['A']*t-mortality_params['B']                                                   /np.log(mortality_params['c'])*mortality_params['c']**ages_PH[i]*                                                   (mortality_params['c']**t-1))-death_times_prob[i],
                   x0=death_times_int[i]-ages_PH[i])
    
    return death_times


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
    
    coeff = np.zeros(shape=(df.shape[0],))
    
    risk_drivers = get_risk_drivers(profile)
    if bool_errors:
        print('Risk drivers in profile {} are restricted to {}'.format(profile, risk_drivers))
    
    for feature in risk_drivers:
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
        else:
            print('Note that in sub_simulate_events, l 231: coefficient "', coeff, '" of surrender profile ignored!')
            print('Abording computation!')
            exit()

        
    
    return coeff


def get_surrender_prob(df, profile_nr = 0, adjust_baseline = False, target_surrender = None, beta0 = 0,
                      rand_noise_var = 0, bool_errors = True):
    
    ''' 
    Combine coefficients for age, duration, premium frequency and annual premium amount in a logit model
    To obtain a reasonable average surrender rate: adapt by introducing a baseline factor beta0

    Parameter:
    ----------
        df:         DataFrame with columns that match names in surrender profiles
        profile_nr: integer value indicating a predefined surrender profile for the simulation
                    0: Fitted Logit Model in Milhaud 2011
                    1: Empirical effects in Data in Milhaud 2011
                    2 & 3: Eling and Cerchiari
        adjust baseline: Boolean whether baseline to be adjusted
        target_surrender: target value for mean surrender of portfolio. Used to adjust beta0 if adjust_baseline == True
        beta0:      baseline surrender factor beta0 (optional user input)
        rand_noise_var: variance of a centered, gaussian noise that is added to the logit-model for surrender probs
        bool_error:     Boolean, display error messages when feature no risk driver in profile
    
    Outputs
    -------
        array of surrender activity
        underlying beta0 coeff (optional, if adjust_baseline==True)
    '''
    
    odd_ratio = np.exp(get_surrender_coeff(df=df, profile =profile_nr, bool_errors=bool_errors))
     
    
    if adjust_baseline:
        beta0 = sc.optimize.newton(func= lambda x: target_surrender - (np.exp(x)*odd_ratio/(1+np.exp(x)*odd_ratio)).mean(), x0 = -1)
        
    # Adjust odd_ratio by baseline factor beta0 (either user_input or determined to fit target_surrender rate)    
    odd_ratio = odd_ratio*np.exp(beta0)

        
    # Add random noise
    # Note standard modeling assumption: log(p/1-p) = betas*X + noise
    odd_ratio = odd_ratio*np.exp(np.random.normal(loc = 0, scale = rand_noise_var,size = odd_ratio.size))
    
    if adjust_baseline: # Also return adjusted beta0 coefficient (-> to replicate results)
        return (odd_ratio/(1+odd_ratio), beta0)
    else: # return purely probabilites
        return odd_ratio/(1+odd_ratio)



def simulate_surrender(df, profile_nr, adjust_baseline, 
                       target_rate = None, beta0 = 0, simulation_noise_var = 0,
                      bool_errors = True):
    
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
    
    np.random.seed(8) # set seed in simulate_surrender_time_series()
    # simulated probs to compare with surrender probs later obtained by get_surrender_prob()
    sim_probs = np.random.uniform(size = N)
    

    # surrender probs
    # if adjust_baseline == True: val[0]: surrender_probs, val[1]: adjusted beta0
    # if adjust_baseline == False: val: surrender_probs
    val = get_surrender_prob(df=df, profile_nr= profile_nr, 
                             adjust_baseline= adjust_baseline, beta0= beta0,
                             target_surrender= target_rate, rand_noise_var = simulation_noise_var, 
                             bool_errors= bool_errors)

    if adjust_baseline: # also return corresponding beta0 coefficient (for later comparison)
        # return 0: active (after period), 1: surrender (within period)
        return ((sim_probs < val[0]).astype('int'), val[1])
    else:
        return (sim_probs < val).astype('int')


def simulate_surrender_time_series(df, target_rate, profile_nr, modeling_time_step = 1/12, time_period_max = 10,
                                    option_new_business = True, rate_new_business = 0.06,
                                   simulation_noise_var = 0):
    
    '''
    Simulate the portfolio decomposition over time, i.e. iteratively apply simulate_surrender()
    Important: Fix the baseline hazard beta0 determined at initial time 0 and apply it for consecutive points in time
    
    Parameter
    ---------
        df:                 DataFrame with columns that match names in features_lst & 'Lapsed' column
        target_surrender:   target value for mean surrender of portfolio. Used to adjust beta0 (adjust_baseline == True default)
        profile_nr:         integer value indicating a predefined surrender profile for the simulation
                            0: Fitted Logit Model in Milhaud 2011
                            1: Empirical effects in Data in Milhaud 2011
                            2 & 3: effects from Eling 2012 - Figure 4c and Cerchiari
        modeling_time_step: frequency in which we iteratively apply the modelling of surrender, i.e. 1/12=monthly, 1=annually 
        time_period_max:    length of oberservation in years
        simulation_noise_var:   variance of a centered, gaussian noise that is added to the logit-model for surrender probs    
    '''
    
    N_contracts = df.shape[0]

    # set seed
    np.random.seed(0)    
    TS_length = min(time_period_max, int(df['Duration'].max()/modeling_time_step) +1)
    
    # Initialize time series
    TS_portfolio = [None]*TS_length
    #Initial value
    TS_portfolio[0] = pd.DataFrame.copy(df)
    
    ####### Indicate lapse due to maturity (level 2) ###########
    # This can be overwritten by lapse due to death or surrender
    TS_portfolio[0].loc[(TS_portfolio[0]['Duration_remain']-modeling_time_step)<0, 'Lapsed'] = 2
    
    ######## Indicate lapse due to death (level 3) ###########
    # Can censor simulated surrender
    TS_portfolio[0].loc[(TS_portfolio[0]['Age']+modeling_time_step>TS_portfolio[0]['Death']) & (TS_portfolio[0]['Death']<TS_portfolio[0]['Age']+TS_portfolio[0]['Duration']),'Lapsed']= 3

    ######## Indicate surrender (level 1) ##############
    surrender, beta0 = simulate_surrender(df = TS_portfolio[0], #features_lst= features_lst,
                                   profile_nr= profile_nr, adjust_baseline = True, 
                                   target_rate= target_rate, simulation_noise_var= simulation_noise_var)    
    

    # select contracts with multiple events
    conflict_death = (surrender == 1) & (TS_portfolio[0]['Lapsed'] == 3)
    conflict_maturity = (surrender == 1) & (TS_portfolio[0]['Lapsed'] == 2)
    if sum(conflict_death)>0 | sum(conflict_maturity)>0:
        print('\t ', 'Conflicting events: ', str(sum(conflict_death)+sum(conflict_maturity)))
        time_to_death = TS_portfolio[0].loc[conflict_death,'Death']-TS_portfolio[0].loc[conflict_death,'Age']
        time_to_maturity = TS_portfolio[0].loc[conflict_maturity,'Duration_remain']

        # conflict: surrender (1) - death (3)
        np.random.seed(42)
        sim_death = np.random.uniform(size=sum(conflict_death))
        surrender[conflict_death] += (sim_death<time_to_death)*2
        # conflict: surrender (1) - maturity (2)
        np.random.seed(42)
        sim_maturity = np.random.uniform(size=sum(conflict_maturity))
        surrender[conflict_maturity] += (sim_maturity<time_to_maturity)*2

    TS_portfolio[0].loc[surrender==1,'Lapsed'] = 1
    #beta0 = surrender[1]
  

    # iterate over time
    for i in range(1,TS_length):
        
        if sum(TS_portfolio[i-1]['Lapsed']==0)>0: # still active contracts in portfolio
            # Advance time for TS, drop contracts that lapsed at the previous time step
            TS_portfolio[i] = pd.DataFrame(data = TS_portfolio[i-1][TS_portfolio[i-1]['Lapsed']==0])
            # Adjust Features by advance of time
            TS_portfolio[i]['Age'] += modeling_time_step
            TS_portfolio[i]['Time'] += modeling_time_step
            TS_portfolio[i]['Duration_elapsed'] += modeling_time_step
            TS_portfolio[i]['Duration_remain'] -= modeling_time_step

            # add new business
            if option_new_business:
                N_contracts_new = int(len(TS_portfolio[i])*rate_new_business)
                new_business = simulate_contracts(N=N_contracts_new, option_new_business=True)
                new_business.index = range(N_contracts, N_contracts+N_contracts_new)
                N_contracts += N_contracts_new
                TS_portfolio[i] = TS_portfolio[i].append(new_business)

            ###########  Indicate lapse due to maturity (level 2) ############
            # Note: Don't use 0 but 10**(-10) as threshold to compensate for rounding errors of e.g. 1/12 steps
            TS_portfolio[i].loc[((TS_portfolio[i]['Duration_remain']-modeling_time_step+10**(-10))<0), 'Lapsed'] = 2
            
            ############# Indicate lapse due to death (level 3) ###############
            TS_portfolio[i].loc[((TS_portfolio[i]['Age']+modeling_time_step)>TS_portfolio[i]['Death'])&  (TS_portfolio[i]['Death']<TS_portfolio[i]['Age']+TS_portfolio[i]['Duration_remain']), 'Lapsed'] = 3              
            
            ########### Indicate surrender (level 1) #############
            # Note: Keep beta0 fixed (determined at initial time 0 to match some empirical target surrender rate) 
            surrender = simulate_surrender(df = TS_portfolio[i], #features_lst= features_lst,
                                           profile_nr= profile_nr, 
                                           adjust_baseline = False, target_rate= target_rate, 
                                           beta0 = beta0, 
                                           simulation_noise_var= simulation_noise_var,
                                          bool_errors=False) # Dont display messages for feature that are no risk driver


            # select contracts with multiple events
            conflict_death = (surrender == 1) & (TS_portfolio[i]['Lapsed'] == 3)
            conflict_maturity = (surrender == 1) & (TS_portfolio[i]['Lapsed'] == 2)
            if sum(conflict_death)>0 | sum(conflict_maturity)>0:
                print('\t ', 'Conflicting events: ', str(sum(conflict_death)+sum(conflict_maturity)))
                time_to_death = TS_portfolio[i].loc[conflict_death,'Death']-TS_portfolio[i].loc[conflict_death,'Age']
                time_to_maturity = TS_portfolio[i].loc[conflict_maturity,'Duration_remain']

                # conflict: surrender (1) - death (3)
                np.random.seed(42)
                sim_death = np.random.uniform(size=sum(conflict_death))
                surrender[conflict_death] += (sim_death<time_to_death)*2
                # conflict: surrender (1) - maturity (2)
                np.random.seed(42)
                sim_maturity = np.random.uniform(size=sum(conflict_maturity))
                surrender[conflict_maturity] += (sim_maturity<time_to_maturity)*2


            TS_portfolio[i].loc[surrender==1,'Lapsed'] = 1
            
            # index of non-surendered contracts
            # Implicit assumption: surrender happens bevore maturity or death
            #index_bool_lapse = (TS_portfolio[i].loc[:,'Lapsed']==1)

        else:
            # return TS_portfolio prematurely (due to the lack of active contracts)
            return (TS_portfolio[0:i], beta0)#, df_events)
    
    return (TS_portfolio, beta0)#), df_events


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
        ax[1].set_xlabel((modeling_time_step==1)*'Year'+(modeling_time_step ==1/12)*'Month'+ (modeling_time_step ==1/4)*'Quarter')
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow = False, edgecolor = 'black')
        
    else: # plot absolute numbers
        ax[1].plot(x_grid, stats.Surrender, label = 'Surrender', color = 'blue')
        ax[1].plot(x_grid, stats.Maturity, label = 'Maturity', color = 'green')
        ax[1].plot(x_grid, stats.Death, label = 'Death', color = 'orange')
        ax[1].legend()
        ax[1].set_xlabel((modeling_time_step==1)*'Year'+(modeling_time_step ==1/12)*'Month'+ (modeling_time_step ==1/4)*'Quarter')        
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


