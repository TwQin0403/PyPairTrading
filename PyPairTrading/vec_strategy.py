import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from collections import OrderedDict
from pykalman import KalmanFilter


def generate_spread_ls(series_1,series_2,n):
    '''
    Generate the spread based on the ls method
    '''
    series_1_train = series_1.head(n)
    series_2_train = series_2.head(n)
    S1 = sm.add_constant(series_1_train[series_1_train.columns[1]])
    results = sm.OLS(series_2_train[series_2_train.columns[1]],S1).fit()
    series_1_test = series_1.iloc[n+1:]
    series_2_test = series_2.iloc[n+1:]
    Z = series_1_test[['Date']]
    print(results.params)
    Z['Spread'] = series_2_test[series_2_test.columns[1]] - results.params[series_1_test.columns[1]] * series_1_test[series_1_test.columns[1]] 
    Z['Spread'] = Z['Spread'] - results.params['const']
    return Z


def generate_spread_kalman(series_1,series_2,n):
    '''
    Generate the spread based on the KalmanFilter
    '''

    #Kalman filter for estimating mu and gamma
    series_1_train = series_1.head(n)
    series_2_train = series_2.head(n)
    
    observation_matrices = np.vstack((np.ones(len(series_1_train)),series_1_train[series_1_train.columns[1]])).T
    shape = observation_matrices.shape
    observation_matrices = observation_matrices.reshape(shape[0],1,shape[1])

    series_1_test = series_1.iloc[n+1:]
    series_2_test = series_2.iloc[n+1:]
    kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            transition_matrices=np.eye(2),
            observation_matrices=observation_matrices
            )
    kf.em(series_2_train[series_2_train.columns[1]])
    filter_mean,filter_cov = kf.filter(series_2_train[series_2_train.columns[1]])

    parameter_list = []    
    for i in range(len(series_2_test)):
        observation_matrix = np.array([[1,series_1_test[series_1_test.columns[1]].values[i]]])
        observation = series_2_test[series_2_test.columns[1]].values[i]

        next_filter_mean,next_filter_cov = kf.filter_update(
                filtered_state_mean = filter_mean[-1],
                filtered_state_covariance = filter_cov[-1],
                observation=observation,
                observation_matrix=observation_matrix 
                )

        parameter_list.append(next_filter_mean)
    mu_list = [parameter[0] for parameter in parameter_list]
    gamma_list = [parameter[1] for parameter in parameter_list]
    # use the mu_list and gamma_list to generate spread
    Z_train = series_1_train[['Date']] 
    Z_train['mu'] = pd.Series(filter_mean[:,0])
    Z_train['gamma'] = pd.Series(filter_mean[:,1])
    Z_train['Spread'] = series_2_train[series_2_train.columns[1]]-Z_train['mu'] - Z_train['gamma'] * series_1_train[series_1_train.columns[1]]
    Z_train['Spread'].plot()
    Z = series_1_test[['Date']]
    Z['mu'] = mu_list
    Z['gamma'] = gamma_list
    Z['Spread'] = series_2_test[series_2_test.columns[1]] -Z['mu'] - Z['gamma'] * series_1_test[series_1_test.columns[1]]
    return Z

def pair_single(spread,threshold):
    '''
    translate the spread dataframe into the spread trading signals 
    '''
    signals = spread[['Date']]
    signals['Buy'] = spread['Spread'] < -threshold
    signals['Short'] = spread['Spread'] > threshold
    signals['Sell'] = (spread['Spread'].shift() < 0) & (spread['Spread'] > 0)
    signals['Cover'] = (spread['Spread'].shift() > 0) & (spread['Spread'] < 0)
    return signals

def spread_to_asset_signals(spread_signals):
    '''
    Transform the spread trading signals into the pair trading signals
    '''
    signals_2 = spread_signals.copy()
    signals_1 = spread_signals[['Date']]
    signals_1['Buy'] = signals_2['Short']
    signals_1['Short'] = signals_2['Buy']
    signals_1['Sell'] = signals_2['Cover']
    signals_1['Cover'] = signals_2['Sell']
    signals_1.index = signals_1['Date']
    signals_2.index = signals_2['Date']
    signals_1.drop('Date',axis=1,inplace=True)
    signals_2.drop('Date',axis=1,inplace=True)
    return signals_1,signals_2

    
def strategy_1(pair_dict,n,s,ty='LS'):
    '''
    Implementation of strategy_1
    '''
    name_list = []
    p_list = []
    for pair_name in pair_dict.keys():
        pair_df = pair_dict[pair_name]
        pair_df = pair_df.head(n)
        name_1 = pair_df.columns[1]
        name_2 = pair_df.columns[2]
        stat, p, cp = coint(pair_df[name_1],pair_df[name_2])
        name_list.append(pair_name)
        p_list.append(p)
    #print(min(p_list))
    pair_candiate = name_list[p_list.index(min(p_list))]
    print(pair_candiate)
    series_1 = pair_dict[pair_candiate][['Date',pair_dict[pair_candiate].columns[1]]]
    series_2 = pair_dict[pair_candiate][['Date',pair_dict[pair_candiate].columns[2]]]
    if ty == 'LS':
        spread = generate_spread_ls(series_1,series_2,n)
    elif ty =='kalman':
        spread = generate_spread_kalman(series_1,series_2,n)
    signals = pair_single(spread,s)
    signals_1,signals_2 = spread_to_asset_signals(signals)
    if ty == 'LS':
        return signals_1,signals_2,series_1.iloc[n+1:],series_2.iloc[n+1:]
    elif ty == 'kalman':
        return signals_1,signals_2,series_1.iloc[n+1:],series_2.iloc[n+1:],spread

def generate_tradeable(pair_df,n,sig_level=0.1):
    '''
    Compute the rolling parameters and cointegration test for pair AUD/EUR
    '''
    result = []
    mu_list = []
    gamma_list = []
    std_list = []
    for i in range(len(pair_df.iloc[3501:])):
        use_df = pair_df.iloc[3501+i-n:3501+i]
        name_1 = pair_df.columns[1]
        name_2 = pair_df.columns[2]
        stat, p, cp = coint(use_df[name_1],use_df[name_2])
        series_1 = use_df[[use_df.columns[1]]]
        series_2 = use_df[[use_df.columns[2]]]
        S1 = sm.add_constant(series_1)
        ls_result = sm.OLS(series_2,S1).fit()
        mu_list.append(ls_result.params['const'])
        gamma_list.append(ls_result.params[series_1.columns[0]])
        spread = use_df[['Date']]
        spread['Spread'] = series_2[series_2.columns[0]] - ls_result.params[series_1.columns[0]] * series_1[series_1.columns[0]]
        spread['Spread'] = spread['Spread'] - ls_result.params['const']
        std_list.append(spread['Spread'].std())
        
        if p < sig_level:
            result.append(True)

        else:
            result.append(False)
    return result,mu_list,gamma_list,std_list


def strategy_2(pair_dict,n,ty):
    '''
    Implementation of strategy_2
    '''
    pair_df = pair_dict['AUD/EUR']
    series_1 = pair_df[['Date','AUD']]
    series_2 = pair_df[['Date','EUR']]
    result,mu_list,gamma_list,std_list = generate_tradeable(pair_dict['AUD/EUR'],n)
    series_1_test = series_1.iloc[3501:]
    series_2_test = series_2.iloc[3501:]
    Z = series_1.iloc[3501:][['Date']]
    Z['mu'] = mu_list
    Z['gamma'] = gamma_list
    Z['std'] = std_list
    Z['tradable'] = result
    Z['Spread'] = series_2_test[series_2_test.columns[1]] - Z['gamma'] * series_1_test[series_1_test.columns[1]] - Z['mu']
    Z['Spread'] = Z['Spread']/Z['std']
    if ty == 'stop_loss':
        signals = Z[['Date']]
        signals['Buy'] = (Z['Spread'] < -1) & (Z['tradable'])
        signals['Short'] = (Z['Spread'] > 1) & (Z['tradable'])
        signals['Sell'] = ((Z['Spread'].shift() < 0) & (Z['Spread'] > 0)) | (~Z['tradable'])
        signals['Cover'] = ((Z['Spread'].shift() > 0) & (Z['Spread'] < 0)) | (~Z['tradable'])
    elif ty == 'rolling':
        signals = pair_single(Z[['Date','Spread']],1)
    signals_1,signals_2 = spread_to_asset_signals(signals)
    return signals_1,signals_2,series_1_test,series_2_test,Z
