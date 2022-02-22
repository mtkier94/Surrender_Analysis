import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import os

from functions.sub_simulate_events import simulate_contracts
from global_vars import path_plots, path_portfolio


def run_simulation():

    N_contracts = 30000
    Portfolio = simulate_contracts( N=N_contracts, option_new_business=False)
    Portfolio.to_csv(os.path.join(path_portfolio, r'Portfolio.csv'), header=True)



def visualize_portfolio():

    try:
        Portfolio = pd.read_csv(os.path.join(path_portfolio,r'Portfolio.csv'), index_col=0)
    except:
        raise ValueError('Portfolio cannot be loaded!')


    ## marginal distributions of relevant contract features 
    df = Portfolio.drop(['Death', 'Time', 'Lapsed', 'Age_init', 'Duration_remain', 'Premium'], axis = 1)
    fig, axes = plt.subplots(ncols=len(df.columns), figsize = (2*len(df.columns),3))
    for ax, col in zip(axes, df.columns):
        if col == 'Premium_freq':
            ls = sum(df[col]==0)/len(df)
            a = sum(df[col]==1)/len(df)
            m = sum(df[col]==12)/len(df)
            ax.bar(['up-front', 'annual', 'monthly'], [ls,a,m], width = 0.1, color='grey')
            ax.set_xticklabels(['up-front', 'annual', 'monthly'], Rotation=20)
            ax.set_xlabel('premium (freq.)')
        else:
            sns.distplot(df[col], ax=ax, color='grey')
            if col == 'Age':
                ax.set_xlabel('current age')
            elif col == 'Face_amount':
                ax.set_xlabel('face amount')
            elif col == 'Duration':
                ax.set_xlabel('duration')
            elif col == 'Duration_elapsed':
                ax.set_xlabel('duration (elapsed)')
            elif col == 'Premium_annual':
                ax.set_xlabel('premium (p.a.)')
            else:
                print('Unknown feature ', col)
                raise ValueError
                

    plt.tight_layout()
    plt.savefig(os.path.join(path_plots,r'portfolio_dist.png'), bbox_inches='tight')
    plt.savefig(os.path.join(path_plots,r'portfolio_dist.eps'), bbox_inches='tight')
    plt.close()


    ### additional visualizations
    ### note: experimental EDA on portfolio at time t=0
    pd.set_option('precision', 4)
    sns.set_style('ticks')
    sns.set_context('paper')

    ages = Portfolio['Age']
    premium_freq = Portfolio['Premium_freq']
    N_contracts = len(ages)

    # Visualize underwriting age
    # Compare to Milhaud, 'Lapse Tables [..]': 47.46% in [0,34], 34.4% in [35,54], 18.5% in [55,84]
    plt.hist(ages,bins= 100,  density= True)
    plt.vlines(x = 34.5, ymin = 0, ymax = 0.04, linestyle = '--', color = 'red')
    plt.vlines(x = 54.5, ymin = 0, ymax = 0.04, linestyle = '--', color = 'red')
    share_1 = sum(ages<34)/N_contracts
    share_2 = sum((ages>34)*(ages<54))/N_contracts
    share_3 = 1- share_1-share_2
    plt.text(x = 15, y = 0.03, s = str(round(share_1*100,ndigits=2))+'%', color = 'red')
    plt.text(x = 40, y = 0.03, s = str(round(share_2*100,ndigits=2))+'%', color = 'red')
    plt.text(x = 60, y = 0.03, s = str(round(share_3*100,ndigits=2))+'%', color = 'red')
    plt.show()

    # ## Durations (of Endowments)
    sns.axes_style('white')
    sns.distplot(Portfolio['Duration'], bins = 150, norm_hist= True)
    plt.title('Distribution: Duration of Endowments ')
    plt.show()

    # ## Elapsed duration
    sns.distplot(Portfolio['Duration_elapsed'], bins = 150)
    plt.show()

    # ## Face Amounts (S)
    # Choice arbitrary -> Backtest by looking and resulting premiums and compare range and variance to Milhaud's paper
    plt.hist(Portfolio['Face_amount'], bins = 100, density= True)
    plt.vlines(x= np.median(Portfolio['Face_amount']), ymin= 0, ymax= 0.00032, color = 'red', linestyles= '--')
    plt.show()
    sc.stats.describe(Portfolio['Face_amount'])

    # ## Annualize Premiums (P_ann)
    print('Median premium: \t ' +str(np.median(Portfolio.Premium_annual)))
    print('Mean premium: \t\t' + str(sc.stats.describe(Portfolio.Premium_annual).mean))
    print('Variance of premiums: \t' + str(np.sqrt(sc.stats.describe(Portfolio.Premium_annual).variance)))

    sns.distplot(Portfolio.Premium_annual[Portfolio.Premium_annual<Portfolio.Premium_annual.quantile(q=0.99)])
    plt.vlines(x=np.median(Portfolio.Premium_annual), ymin= 0, ymax= 0.0009, colors= 'red', linestyles= ':')
    plt.show()

    # Correlation of Features
    sns.heatmap(Portfolio.corr(),cmap= 'coolwarm', annot = True, annot_kws= {'size': 10})
    plt.show()


    sns.distplot(Portfolio['Age'], label='Age')
    sns.distplot(Portfolio['Age_init'], label = 'Age_init')
    plt.xlabel('Age [years]')
    plt.legend()

    fig, ax = plt.subplots(nrows=len(Portfolio.columns)//5+1, ncols= 5, figsize = (16,8) )
    ax=ax.flatten()
    
    for i, feature in enumerate(Portfolio.columns):
        sns.distplot(Portfolio[feature], ax = ax[i])
        ax[i].set_xlabel(feature)        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    boolSimulate = False
    boolVisualize = True

    if boolSimulate:
        # create portfolio for time t=0
        run_simulation()
    if boolVisualize:
        # visualize distribution of data
        visualize_portfolio()