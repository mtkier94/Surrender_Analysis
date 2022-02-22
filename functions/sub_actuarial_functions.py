import numpy as np

# functions to compute actuarial quantities as the premium

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
