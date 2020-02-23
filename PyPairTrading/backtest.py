import pandas 
import numpy
import data_processing
import strategy


def signals_to_positions(signals,amount=1,ty=None ,init_pos=0, mask=('Buy','Sell','Short','Cover')):
    """
    Translate signal dataframe into positions series
    """
    long_en,long_ex,short_en,short_ex = mask
    pos = init_pos
    ps = pandas.Series(0., index=signals.index)
    if ty == 'var':
        for (t,sig),bet in zip(signals.iterrows(),amount):
         # check exit signals
            if pos !=0: # if in position
                if pos > 0 and sig[long_ex]:
                    pos = 0
                elif pos <0 and sig[short_ex]:
                    pos = 0
            if pos ==0:
                if sig[long_en]:
                    pos += bet
                elif sig[short_en]:
                    pos -= bet
            ps[t] = pos

    else:
        for t,sig in signals.iterrows():
         # check exit signals
            if pos !=0: # if in position
                if pos > 0 and sig[long_ex]:
                    pos -= amount
                elif pos <0 and sig[short_ex]:
                    pos += amount
            if pos ==0:
                if sig[long_en]:
                    pos += amount
                elif sig[short_en]:
                    pos -= amount
            ps[t] = pos
    return ps[ps != ps.shift()]

def position_to_trade(positions,signals,prices):
    pos = positions.reindex(signals.index).ffill().shift().fillna(0)
    
    prices.index = signals.index
    trd = pandas.DataFrame({'pos':pos})
    trd['price'] = prices
    trd = trd.dropna()
    trd['vol'] = trd.pos.diff()
    return trd.dropna()
    

def pair_trade_to_equity(trd_1,trd_2):
    trd_1.columns = ['pos_1','price_1','vol_1']
    trd_2.columns = ['pos_2','price_2','vol_2']
    trd = pandas.merge(trd_1,trd_2,left_index=True,right_index=True)
    trd['Cum_PL'] = (trd.pos_1 * trd.price_1 + trd.pos_2 * trd.price_2) - (trd.vol_1 * trd.price_1 + trd.vol_2 * trd.price_2).cumsum()
    trd['PL'] = trd['Cum_PL'].diff().fillna(0)

    trd['daily_return'] = trd['PL'] / numpy.abs(trd['Cum_PL'].shift())
    trd = trd.dropna()
    trd = trd.replace([numpy.inf,-numpy.inf],0)
    return trd

def performance(trd):
    result = {'Date':list(trd.index),'PnL':list(trd['PL']),'Cum_PnL':list(trd['Cum_PL'])}
    result.update({'Sharpe':(trd['daily_return'].mean()/trd['daily_return'].std())*(252**0.5)})
    result.update({'Volatility':(trd['daily_return'].std())})
    try:
        holding_period = len(trd[trd.pos_1 != 0])/len(trd[trd.pos_1 != trd.pos_1.shift()])
    except:
        holding_period = 'No Sucessful Trade'
    print(holding_period)
    result.update({'holding_period':holding_period})
    return result
