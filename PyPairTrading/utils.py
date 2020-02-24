import pandas as pd
from functools import reduce
import itertools

def loading_test_data():
    AUD = pd.read_csv('data/AUD.csv')
    EUR = pd.read_csv('data/EUR.csv')
    GBP = pd.read_csv('data/GBP.csv')
    NZD = pd.read_csv('data/NZD.csv')
    AUD.columns = ['Date', 'AUD']
    EUR.columns = ['Date', 'EUR']
    GBP.columns = ['Date', 'GBP']
    NZD.columns = ['Date', 'NZD']
    prices = reduce(lambda x,y: pd.merge(x,y, on='Date'), [AUD, EUR, GBP,NZD])
    data = {'AUD':prices[['Date','AUD']],'EUR':prices[['Date','EUR']],'GBP':prices[['Date','GBP']],'NZD':prices[['Date','NZD']]}
    return data 

def generate_pair_dict():
    data = loading_test_data()
    pair_dict = {}
    combinations = []
    for subset in itertools.combinations(list(data.keys()), 2):
        combinations.append(subset)
    for pair in combinations:
        pair_name = pair[0] + '/' + pair[1]
        series_1 = data[pair[0]]
        series_2 = data[pair[1]]
        joint_series = pd.merge(series_1,series_2,on='Date')
        pair_dict.update({pair_name:joint_series})
    return pair_dict