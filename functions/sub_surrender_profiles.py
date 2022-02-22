import numpy as np
import matplotlib.pyplot as plt
import os

parameters = {'axes.labelsize': 16, 'xtick.labelsize':14, 'ytick.labelsize': 14, 'legend.fontsize': 14}
plt.rcParams.update(parameters)

# functions that characterice the force of driving factors for surrender (meta-model)

def get_model_input(profile = 0):
    '''
    Helper function which takes all risk drivers from get_risk_drivers and one-hot-encodes the features premium-freq to obtain the true input.

    Outputs:
    --------
        List of names of input-features
    '''

    cache = get_risk_drivers(profile)
    features_profile_lst = []
    for el in cache:
        if el != 'Premium_freq': 
            features_profile_lst.append(el)
        else:
            features_profile_lst.append('Premium_freq_0')
            features_profile_lst.append('Premium_freq_1')
    return features_profile_lst

def get_risk_drivers(profile=0):
    '''
    Return list of strings containing risk drivers in the respective profile.
    '''
    ans = []
    ans += get_age_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_duration_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_duration_elapsed_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_duration_remain_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_prem_freq_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_prem_annual_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    ans += get_time_coeff(0, profile=profile, bool_errors= False, bool_report_features=True)
    return ans 


def get_age_coeff(age, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of age from true surrender model
    
    Parameter
    ---------
        age:     integer or Series with raw values
        profile: count variable for profile of choice
                 0: Fitted Logit Model in Milhaud 2011
                 1: Empirical effects in Data in Milhaud 2011
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
        integer or series with respective coefficients
    '''   
    dummy=False
    
    if profile == 0:
        # Use Logit model coefficients by Milhaud 2011:
        # Baseline: age <20 -> coeff 0
        # Age groups of [30,50[: coefficients not significantly different from 0 -> include them in baseline
        age_coeff = ((20<=age) & (age<30))*0.3-((50<=age) & (age<60))*0.4-((60<=age) & (age<70))*0.65-(70<=age)*0.75
        dummy=True
        
    elif profile ==1:
        # Use empirical odds ratios in Milhaud 2011 to model influence of age
        # Baseline: age <20 -> coeff 0
        age_coeff = ((20<=age) & (age<30))*np.log(1.16)+((50<=age) & (age<60))*np.log(1.63)+ ((60<=age) & (age<70))*np.log(2.67)+(70<=age)*np.log(3.28)
        dummy=True
    elif (profile ==2)|(profile == 3):
        # Cerchiara 2008, Graph 8
        age_coeff = (age<40)*(0.22)+((40<=age) & (age<60))*(0.09)+  ((60<=age) & (age<80))*(0)+(80<=age)*(0.14)
        dummy=True
    else:
        # Create 0's with shape of age
        age_coeff = (0>age) & (age>0)
        if bool_errors:
            print('Note: Elapsed duration not included as Risk Driver in profile {}!'.format(profile))
    
    
    if bool_report_features == False:
        # return coefficients
        return age_coeff
    else:
        # return if feature included
        if dummy:
            return ['Age']
        else:
            return []


def get_duration_coeff(dur, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of duration from true surrender model.
    Note: Effect of this coefficient is static for each contract, i.e. encodes some level long-term planning intended by the PH by taking the contract.
    
    Parameter
    ---------
        dur:     integer or Series with raw values
        profile: count variable for profile of choice
                 0: Fitted Logit Model in Milhaud 2011
                 1: Empirical effects in Data in Milhaud 2011
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    dummy=False
    
    if (profile == 2)|(profile == 3):
        # Cerchiari 2008, Graph 5
        dur_coeff = (dur<1)*(-0.78)+ ((1<=dur) & (dur<2))*(0) + ((2<=dur) & (dur<=3))*(0.5) + ((3<=dur) & (dur<4))*(0.44) +\
                     ((4<=dur) & (dur<5))*(0.26) + ((5<=dur) & (dur<6))*(0.73) + ((6<=dur) & (dur<7))*(0.7) + ((7<=dur) & (dur<8))*(0.36) +   \
                     ((8<=dur) & (dur<9))*(0.22) + ((9<=dur) & (dur<10))*(0.21) + ((10<=dur) & (dur<11))*(-0.21) + (dur>=11)*(-0.7)
        dummy=True
    else:
        # Create 0's with shape of age
        dur_coeff = (0>dur) & (dur>0)
        if bool_errors:
            print('Note: Duration not included as Risk Driver in profile {}!'.format(profile))

    
    if bool_report_features == False:
        # return coefficients
        return dur_coeff
    else:
        # return if feature included
        if dummy:
            return ['Duration']
        else:
            return []


def get_duration_elapsed_coeff(dur, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of elapse duration from true surrender model.
    Note: Elapsed Duration is implicitely linked to the contractual duration.
    
    Parameter
    ---------
        dur:     integer or Series with raw values
        profile: count variable for profile of choice
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    dummy=False
    
    if profile == 0:
        # Use Logit model coefficients by Milhaud 2011:
        # Baseline: dur <20 -> coeff 0
        dur_coeff = ((1<dur) & (dur<=1.5))*np.log(0.27)+((1.5<dur) & (dur<=2))*np.log(0.07)+((2<dur) & (dur<=2.5))*np.log(0.06)+\
                    ((2.5<dur) & (dur<=3))*np.log(0.05) + ((3<dur) & (dur<=3.5))*np.log(0.03)- ((3.5<dur) & (dur<=4))*np.log(0.02) +\
                    ((4<dur) & (dur<=4.5))*np.log(0.02) + (4.5<dur)*np.log(0.004)
        dummy=True
        
    elif (profile == 1) | (profile == 2) | (profile == 3):
        # Use empirical odds ratios in Milhaud 2011
        dur_coeff = ((1<dur) & (dur<=1.5))*np.log(10.56)+ ((1.5<dur) & (dur<=2))*np.log(2.89) + ((2<dur) & (dur<=2.5))*np.log(2.69) + ((2.5<dur) & (dur<=3))*np.log(1.82) +                     ((3<dur) & (dur<=3.5))*np.log(1.16) + ((3.5<dur) & (dur<=4))*np.log(0.96) +                     ((4<dur) & (dur<=4.5))*np.log(0.68) + (4.5<dur)*np.log(0.19)
        dummy=True

    else:
        # Create 0's with shape of age
        dur_coeff = (0>dur) & (dur>0)
        if bool_errors:
            print('Note: Duration not included as Risk Driver in profile {}!'.format(profile))

    
    if bool_report_features == False:
        # return coefficients
        return dur_coeff
    else:
        # return if feature included
        if dummy:
            return ['Duration_elapsed']
        else:
            return []


def get_duration_remain_coeff(dur_rem, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of the passed duration from true surrender model
    
    Parameter
    ---------
        dur:     integer or Series with raw values
        profile: count variable for profile of choice
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    
    dummy=False
    
    if (profile == 2)|(profile == 3):
        # Eling 2012 - Figure 4c
        dur_rem_coeff = (dur_rem>=20)*(dur_rem-20)*.8/26+ ((dur_rem<20)&(dur_rem>=6))*(dur_rem-19)*(0.5)/13 +((dur_rem<6)&(dur_rem>=2))*(-0.1) +(dur_rem<2)*(-0.3)
        
        dummy=True
    else:
        # Create 0's with shape of age
        dur_rem_coeff = (0>dur_rem) & (dur_rem>0)
        if bool_errors:
            print('Note: Remaining duration not included as Risk Driver in profile {}!'.format(profile))

    if bool_report_features == False:
        # return coefficients
        return dur_rem_coeff
    else:
        # return if feature included
        if dummy:
            return ['Duration_remain']
        else:
            return []


def get_prem_freq_coeff(prem_freq, profile = 0, bool_errors=True, bool_report_features = False):
    
    '''
    Get coefficient of premium frequency from true surrender model
    
    Parameter
    ---------
        prem_freq: integer or Series with raw values
        profile: count variable for profile of choice
                 0: Fitted Logit Model in Milhaud 2011
                 1: Empirical effects in Data in Milhaud 2011
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    dummy = False
    
    if profile == 0:
        # Use Logit model coefficients by Milhaud 2011:
        # Baseline: monthly (m=12) -> coeff 0    
        # Note: Coefficient for prem_freq == 0 not significant (p-val = 0.455)
        prem_freq_coeff = (prem_freq==1)*0.43 - (prem_freq==0)*0.28
        dummy =True
        
    elif profile == 1:
        # Use empirical odds ratios in Milhaud 2011
        prem_freq_coeff = (prem_freq==1)*np.log(2.39) + (prem_freq==0)*np.log(1.60)
        dummy=True
        
    elif (profile == 2) | (profile == 3):
        # Eling&Kiesling 2011 - Table 4 (cont.)
        prem_freq_coeff = (prem_freq==0)*(-2.23)
        dummy=True

    else:
        # Create 0's with shape of age
        prem_freq_coeff = (0>prem_freq) & (prem_freq>0)
        if bool_errors:
            print('Note: Premium frequency not included as Risk Driver in profile {}!'.format(profile))
   
    if bool_report_features == False:
        # return coefficients
        return prem_freq_coeff
    else:
        # return if feature included
        if dummy:
            return ['Premium_freq']
        else:
            return []


# get coefficient of annual premium to be used categorically in a logit model
# serveral surrender profiles available, i.e. prem_profile some integer number

def get_prem_annual_coeff(prem_ann, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of annual premium from true surrender model
    
    Parameter
    ---------
        prem_annual: integer or Series with raw values
        profile: count variable for profile of choice
                 0: Fitted Logit Model in Milhaud 2011
                 1: Empirical effects in Data in Milhaud 2011
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    dummy = False
    
    if (profile == 0):
        # Use qualitative findings of e.g. Aleandri (2017): Premium_Amount positively correlated with surrender
        # and combine this with odd-ratios stated in Milhaud (2011) (where categories are unknown)
        # baseline: Premium_annual < 1000 -> coeff 0
        prem_coeff = ((prem_ann>1000)& (prem_ann<=2000))*0.9+ (prem_ann>2000)*1.3
        dummy = True   
    elif( profile ==1)|(profile ==2)|(profile ==3):
        # Use empirical odds ratios in Milhaud 2011 (with arbitrary border 1000 and 2000)
        prem_coeff = ((prem_ann>1000)& (prem_ann<=2000))*np.log(1.9) + (prem_ann>2000)*np.log(2.1)
        dummy=True
        
        #elif (profile ==2)|(profile ==3):
            # Use empirical odds ratios in Milhaud 2011 (with arbitrary border 1000 and 2000)
        #    prem_coeff = (prem_ann>1500)*(0.5)
        #    dummy=True   
    else:
        if bool_errors:
            print('Note: Annual Premium not included as Risk Driver in profile {}!'.format(profile))
        else:
            pass # do nothing
        
    if bool_report_features == False:
        # return coefficients
        return prem_coeff
    else:
        # return if feature included
        if dummy:
            return ['Premium_annual']
        else:
            return []
    

def get_time_coeff(time, profile = 0, bool_errors = True, bool_report_features = False):
    
    '''
    Get coefficient of time component from true surrender model.
    This implicitely captures a varying economic environment and trends.
    
    Parameter
    ---------
        time: integer or Series with raw values
        profile: count variable for profile of choice
                 0: Fitted Logit Model in Milhaud 2011
                 1: Empirical effects in Data in Milhaud 2011
        bool_errors: Boolean, display error messages.
        bool_report_features: report only the features used in the respective profile
    
    Output
    -------
    \t integer or series with respective coefficients
    '''
    dummy = False
    
    if profile == 3:
        time_coeff = (time<1)*(-0.14)+((time>=1)&(time<2))*(0.07)+((time>=2)&(time<3))*(-0.11) + ((time>=3)&(time<4))*(-0.21)+((time>=4)&(time<5))*(-0.2) +((time>=5)&(time<6))*(-0.24) +  ((time>6)&(time<7))*(-0.36)+((time>=7)&(time<8))*(-0.36) +((time>=8)&(time<9))*(-0.2) +((time>=9)&(time<10))*(-0.2)+((time>=10)&(time<11))*(-0.23) +((time>=11)&(time<12))*(-0.15) +((time>=12)&(time<13))*(0) +((time>=13)&(time<14))*(-0.15) +((time>=14)&(time<15))*(-0.02) +((time>=15)&(time<16))*(0.25) 
        dummy = True
    else:
        if bool_errors:
            print('Note: Time is not included as Risk Driver in profile {}!'.format(profile))
        else:
            pass # do nothing
        
    if bool_report_features == False:
        # return coefficients
        return time_coeff
    else:
        # return if feature included
        if dummy:
            return ['Time']
        else:
            return []
    

def visualize_surrender_profiles(profile_nr = 0, fig_size = (16,6), path=None, bool_plot=False):
    
    '''
    Visualize all effects (in terms of odd-ratios) for the features
    (current) age, duration, frequency of premium payments, premium amount (annual)
    
    Parameter
    ---------
        profile_nr: integer, representing some lapse profile
                    0: Fitted Logit Model in Milhaud 2011
                    1: Empirical effects in Data in Milhaud 2011
    
    Output
    ------
        Show Plot
    '''
    
    if (path==None) & (bool_plot==False):
        assert('Irrational user-input: Nothing will be plotted or saved.')

    im_per_col = 4

    features = get_risk_drivers(profile_nr)
    # resize figure such that all columns have the same scale
    fig_size = (fig_size[0],int(np.ceil(len(features)/im_per_col))*fig_size[1])
    #print(fig_size)
    
       
    _, ax = plt.subplots(nrows= int(np.ceil(len(features)/im_per_col)), ncols= im_per_col, #len(features), 
                            figsize = fig_size)
    ax = ax.flatten()

    if len(features)%im_per_col >0:
        for i in range(im_per_col-(len(features)%im_per_col)):
            ax[-i-1].set_axis_off()

    i=0
    for feat in features:
        
        if feat == 'Age':
            # Age 
            age_range = np.linspace(start = 0, stop = 90, num = 200)
            ax[i].plot(age_range, [0]*len(age_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].step(age_range,(get_age_coeff(age= age_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ current age')
            if i%im_per_col == 0:
                ax[i].set_ylabel(r'$\beta^{(i)}_x x^{(i)}$')
            i+=1
        elif feat == 'Duration':
        
            # Duration 
            dur_range = np.linspace(start = 0, stop = 25, num = 100)
            ax[i].plot(dur_range, [0]*len(dur_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].step(dur_range,(get_duration_coeff(dur= dur_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ duration')
            i+=1

        elif feat == 'Duration_elapsed':
            
            # Remaining Duration
            dur_el_range = np.linspace(start = 0, stop = 25, num = 100)
            ax[i].plot(dur_el_range, [0]*len(dur_el_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].step(dur_el_range,(get_duration_elapsed_coeff(dur = dur_el_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ duration (elapsed)') 
            i+=1

        elif feat == 'Duration_remain':
            
            # Remaining Duration
            dur_rem_range = np.linspace(start = 0, stop = 25, num = 100)
            ax[i].plot(dur_rem_range, [0]*len(dur_rem_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].step(dur_rem_range,(get_duration_remain_coeff(dur_rem= dur_rem_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ duration (remaining)') 
            i+=1
            
        elif feat == 'Premium_freq':
    
            # Premium Freq 
            prem_freq_range = np.array([0,1,12]) 
            ax[i].plot(['single', 'annual', 'monthly'], [0]*len(prem_freq_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].plot(['single', 'annual', 'monthly'],(get_prem_freq_coeff(prem_freq= prem_freq_range, profile = profile_nr)), 
                          'o', color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ premium (freq.)')
            if i%im_per_col == 0:
                ax[i].set_ylabel(r'$\beta^{(i)}_x x^{(i)}$')
            i+=1
        
        elif feat == 'Premium_annual':
            
            # Premium amount (annual) 
            prem_annual_range = np.linspace(start = 0, stop = 4000, num = 4001)
            ax[i].plot(prem_annual_range, [0]*len(prem_annual_range), linestyle = '--', color = 'grey', alpha = .8)
            ax[i].step(prem_annual_range,(get_prem_annual_coeff(prem_ann= prem_annual_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ premium (p.a.)')
            i+=1
            
        elif feat == 'Time':
            # Time Component
            time_range = np.linspace(start = 0, stop = 25, num = 75)
            ax[i].plot(time_range, [0]*len(time_range), '--', color='grey', alpha = .8)
            ax[i].step(time_range,(get_time_coeff(time= time_range, profile = profile_nr)), color='black')
            ax[i].set_xlabel(r'$x^{(i)}:$ time') 
            i+=1
        else:
            print('ValError: Feature ', feat, ' not included in risk profile!')
            print('Abording computation in sub_surrender_profiles.py line 453')
            exit()
    
    #fig.suptitle('Odd-ratios of features - Profile {}'.format(profile_nr), fontsize = 'x-large')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()
    if path != None:
                plt.savefig(os.path.join(path,r'surr_profile_{}.png'.format(profile_nr)), bbox_inches='tight', dpi = 400)
                plt.savefig(os.path.join(path,r'surr_profile_{}.eps'.format(profile_nr)), bbox_inches='tight', dpi = 400)
    if bool_plot:
        plt.show()
    else:
        plt.close()